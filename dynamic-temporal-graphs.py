import os
import shutil
import torch
import talib
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, current_process
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import aggr
from torch_geometric_temporal.nn.recurrent import GConvGRU

class Separator:
    def __init__(self, clusters):
        self.clusters = clusters
        pass
    
    def divide(self, returns_df):
        relative_returns = returns_df.subtract(returns_df.median(axis=1), axis=0)
        dist = self.correlDist(relative_returns.iloc[:60, :].corr())
        link = linkage(squareform(dist), 'ward')
        assignments = fcluster(link, self.clusters, criterion='maxclust')
        clusters = {}
        
        for i in range(1, self.clusters+1):
            clusters[i] = returns_df.columns[np.where(assignments == i)].tolist()
        return clusters
    
    def correlDist(self, corr):
        dist = ((1 - corr) / 2.)**.5 
        return dist


class Signaler:
    def __init__(self):
        pass
    
    def generate(self, df):
        grouped = df.groupby(level=0)
        lags = [5, 10, 20, 50]

        for lag in lags:
            df['sma' + str(lag)] = grouped['close'].transform(
                                                lambda x: talib.EMA(x, timeperiod=lag))
        
        df['ema1'] = df['sma5'] - df['sma10']
        df['ema2'] = df['sma10'] - df['sma20']
        df['ema3'] = df['sma20'] - df['sma50']

        df.drop(['sma5', 'sma10', 'sma20', 'sma50'], axis=1, inplace=True)
        histogram = grouped['close'].transform(lambda x: talib.MACD(x, 
                                                                    fastperiod=12, 
                                                                    slowperiod=26, 
                                                                    signalperiod=9)[2])
        df['histogram'] = histogram
        df['rsi'] = grouped['close'].transform(lambda x: talib.RSI(x, timeperiod=14))
        sto = grouped.apply(lambda x: talib.STOCH(x['high'], 
                                                  x['low'], 
                                                  x['close'], 
                                                  fastk_period=5, 
                                                  slowk_period=3, 
                                                  slowk_matype=0, 
                                                  slowd_period=3, 
                                                  slowd_matype=0))

        for idx in sto.index:
            df.loc[(idx, slice(None)), "stok"] = sto.loc[idx][0]
            df.loc[(idx, slice(None)), "stod"] = sto.loc[idx][1]
        
        obv = grouped.apply(lambda x: talib.OBV(x['close'], x['volume']))
        df['obv'] = obv.reset_index(level=0, drop=True)
        
        df['roc2'] = grouped['close'].transform(lambda x: talib.ROC(x, timeperiod=2))
        df['roc4'] = grouped['close'].transform(lambda x: talib.ROC(x, timeperiod=4))
        df['roc6'] = grouped['close'].transform(lambda x: talib.ROC(x, timeperiod=6))
        
        willr = grouped.apply(lambda x: talib.WILLR(x['high'], 
                                                    x['low'], 
                                                    x['close'], 
                                                    timeperiod=14))
        
        df['willr'] = willr.reset_index(level=0, drop=True)
        
        aroon = grouped.apply(lambda x: talib.AROON(x['high'], x['low'], timeperiod=14))

        for idx in sto.index:
            df.loc[(idx, slice(None)), "aroonUp"] = aroon.loc[idx][0]
            df.loc[(idx, slice(None)), "aroonDown"] = aroon.loc[idx][1]
            
        atr = grouped.apply(lambda x: talib.ATR(x['high'], 
                                                x['low'], 
                                                x['close'], 
                                                timeperiod=14))
        df['atr'] = atr.reset_index(level=0, drop=True)
        
        cols = list(df.columns)
        selections = [value for value in cols if value not in ['open', 
                                                               'close', 
                                                               'high', 
                                                               'low', 
                                                               'volume', 
                                                               'sto']]
        
        norms = {}
        for col in selections:
            norms[col] = df[col].unstack(level=0)
            
        window_size = 60
        signals = {}

        for col in selections:
            if col != 'return':
                signal = norms[col]
                result = pd.DataFrame()
                min_series = signal.rolling(window=window_size).min()
                max_series = signal.rolling(window=window_size).max()
                result = (signal - min_series) / (max_series - min_series)
                signals[col] = result
            else:
                signals[col] = norms[col]
                
        corrs = {}

        window=60

        for col in selections:
            signal = signals[col]
            corr = signal.rolling(window=window_size).corr()
            corrs[col] = corr.dropna()
            
        start_dates = []

        for col in selections:
            start_date = corrs[col].index.min()[0]
            start_dates.append(start_date)

        start_date = max(start_dates)

        for col in selections:
            d = signals[col]
            signals[col] = d[d.index >= start_date]
            d = corrs[col]
            corrs[col] = d[d.index.get_level_values(level=0) >= start_date]
        
        df_list = []
        for key, value in signals.items():
            value.name = key
            df_list.append(value)

        signal_combined = pd.concat(df_list, keys=[d.name for d in df_list])
        
        signal_combined = signal_combined.swaplevel()
        
        df_list = []
        for key, value in corrs.items():
            value.name = key
            df_list.append(value)

        corrs_combined = pd.concat(df_list, keys=[d.name for d in df_list])
        corrs_combined = corrs_combined.swaplevel(0,1)
        return signal_combined, corrs_combined


class Aggregator:
    def __init__(self, clusterOne, clusterTwo):
        self.clusterOne = clusterOne
        self.clusterTwo = clusterTwo
        
    def get_dataset(self, nodes_df, edges_df, returns_df):
        clusterOne = self.clusterOne
        clusterTwo = self.clusterTwo
        
        idx = pd.IndexSlice
        next_nodes_df = nodes_df.sort_index()
        next_edges_df = edges_df.sort_index()

        one_nodes_df = next_nodes_df[clusterOne]
        one_edges_df = next_edges_df[clusterOne].loc[(slice(None), slice(None), clusterOne)]
        two_nodes_df = next_nodes_df[clusterTwo]
        two_edges_df = next_edges_df[clusterTwo].loc[(slice(None), slice(None), clusterTwo)]
        
        one_ret = returns_df[clusterOne].mean(axis=1)
        two_ret = returns_df[clusterTwo].mean(axis=1)
        
        Ys = np.asarray(one_ret.values > two_ret.values).astype(float)
        one_node_attrs = self.get_node_attrs(one_nodes_df)
        two_node_attrs = self.get_node_attrs(two_nodes_df)        
        one_edge_indices, one_edge_attrs = self.get_edge_indices_attrs(one_edges_df)
        two_edge_indices, two_edge_attrs = self.get_edge_indices_attrs(two_edges_df)
        return self.construct(Ys, one_node_attrs, one_edge_indices, one_edge_attrs,
                              two_node_attrs, two_edge_indices, two_edge_attrs)
        
        
    def construct(self, Ys, node_attrs_1, edge_indices_1, edge_attrs_1, node_attrs_2, edge_indices_2, edge_attrs_2):
        node_1 = [ft.T for ft in node_attrs_1]
        edge_1 = [ef.T for ef in edge_attrs_1]
        
        node_2 = [ft.T for ft in node_attrs_2]
        edge_2 = [ef.T for ef in edge_attrs_2]
        index_1 = edge_indices_1[0]
        index_2 = edge_indices_2[0]
        targets = [np.asarray([y]) for y in Ys]
        return (node_1, index_1, edge_1, node_2, index_2, edge_2, targets)
        
    def get_edge_indices_attrs(self, df):
        edge_attrs = []
        edge_indices = []
        colnum = df.columns.shape[0]

        grouped = df.groupby(level=0)

        edge_idx = None

        for timestamp, data in grouped:
            N = data.columns.shape[0]
            I = data.shape[0] // N
            edges = data.values.reshape(I, N, N)
            edge_attr = None

            for i in range(edges.shape[0]):
                mask = np.isfinite(edges[i])
                source, target = np.where(mask)

                if edge_idx is None:
                    edge_idx = np.asarray([source, target])

                attr = edges[i][source, target].reshape(1, -1)

                if edge_attr is None:
                    edge_attr = attr
                else:
                    edge_attr = np.vstack((edge_attr, attr))
            edge_indices.append(edge_idx)
            edge_attrs.append(edge_attr)
        return edge_indices, edge_attrs

    def get_node_attrs(self, df):
        node_attrs = []
        grouped = df.groupby(level=0)
        for timestamp, data in grouped:
            data = data.reset_index(level=0, drop=True)
            node_attrs.append(data.values)
        return node_attrs



class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, edge_features):
        super(RecurrentGCN, self).__init__()
        self.snapshot = GATv2Conv(in_channels=-1, out_channels=16, edge_dim=edge_features)
        self.recurrent = GConvGRU(in_channels=16, out_channels=16, K=4)
        self.aggr = aggr.MLPAggregation(in_channels=16, 
                                        out_channels=8, 
                                        max_num_elements=8, 
                                        num_layers=2,
                                        hidden_channels=4)

    def forward(self, x, edge_index, edge_attr):
        out = self.snapshot(x=x, edge_index=edge_index, edge_attr=edge_attr)
        out = self.recurrent(X=out, edge_index=edge_index)
        idx = torch.tensor(np.asarray(range(x.shape[0]), dtype=np.int64) + 1)
        out = self.aggr(out, idx)
        return out

class GComparer(torch.nn.Module):
    def __init__(self, node_features, edge_features):
        super(GComparer, self).__init__()
        self.gcn1 = RecurrentGCN(node_features, edge_features)
        self.gcn2 = RecurrentGCN(node_features, edge_features)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(8, 16),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(16, 1))
    def forward(self, x, xi, xe, y, yi, ye):
        out1 = self.gcn1(x, xi, xe)
        out2 = self.gcn2(y, yi, ye)
        logits = self.classifier(out1.mean(dim=0) - out2.mean(dim=0))
        return logits


def train_test(inputs):
    datasets = inputs[0]
    clusterOne = inputs[1]
    clusterTwo = inputs[2]
    
    one_size = np.ones(len(clusterOne)) / len(clusterOne)
    two_size = np.ones(len(clusterTwo)) / len(clusterTwo)
    
    model = GComparer(16, 16)
    p_idx = current_process().name.split('-')[1]
    fname = f'/kaggle/working/models/weights_{p_idx}.pth'
    
    if os.path.isfile(fname):
        model.load_state_dict(torch.load(fname))
        
    weights = pd.DataFrame(columns=clusterOne+clusterTwo)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_dataset = datasets[0]
    
    model.train()
    for epoch in tqdm(range(900)):
        losses = []
        loss = 0
        N = len(train_dataset[0]) // 25
        
        for i in range(0, len(train_dataset[0])):
            x1 = torch.tensor(train_dataset[0][i],dtype=torch.float32)
            i1 = torch.tensor(train_dataset[1],dtype=torch.int64)
            e1 = torch.tensor(train_dataset[2][i],dtype=torch.float32)
            x2 = torch.tensor(train_dataset[3][i],dtype=torch.float32)
            i2 = torch.tensor(train_dataset[4],dtype=torch.int64)
            e2 = torch.tensor(train_dataset[5][i],dtype=torch.float32)
            target = torch.tensor(train_dataset[6][i],dtype=torch.float32)
            diff = model(x1, i1, e1, x2, i2, e2)
            loss += criterion(diff[0], target[0])
            
            if (i > 0 and i % N == 0) or (i == len(train_dataset[0]) - 1):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss = loss.cpu().detach().numpy()
                print(p_idx, epoch, loss)
                losses.append(loss)
                loss = 0

    torch.save(model.state_dict(), fname)
    
    test_dataset = datasets[1]
    return_dates = datasets[2]
    
    model.eval()
    for i in range(len(test_dataset[0])):
        x1 = torch.tensor(test_dataset[0][i], dtype=torch.float32)
        i1 = torch.tensor(test_dataset[1], dtype=torch.int64)
        e1 = torch.tensor(test_dataset[2][i], dtype=torch.float32)
        x2 = torch.tensor(test_dataset[3][i], dtype=torch.float32)
        i2 = torch.tensor(test_dataset[4], dtype=torch.int64)
        e2 = torch.tensor(test_dataset[5][i], dtype=torch.float32)
        target = torch.tensor(test_dataset[6][i], dtype=torch.float32)
        diff = model(x1, i1, e1, x2, i2, e2)
        pred = diff.cpu().detach().numpy()[0]
        pred = F.sigmoid(pred)
        pred = (pred > 0.5).astype(int)
        
        if i > len(test_dataset[0]) - 10:
            weights.loc[return_dates[i], clusterOne] = one_size if pred > 0 else -one_size
            weights.loc[return_dates[i], clusterTwo] = two_size if pred < 1 else -two_size
    return weights

def generate_windows(data, hours, clusterOne, clusterTwo, returns_df, window_size=300, step_size=10):
    K = hours.shape[0]
    windows = []

    for i in range(0, K - window_size, step_size):
        train_ds = data.truncate(before=hours[i]).truncate(after=hours[i+window_size-1-10]) 
        test_ds = data.truncate(before=hours[i+10]).truncate(after=hours[i+window_size-1])
        windows.append((clusterOne, clusterTwo, train_ds, test_ds, returns_df))
    return windows


def generate_features(tp):
    try:
        clusterOne = tp[0]
        clusterTwo = tp[1]
        train_ds = tp[2]
        test_ds = tp[3]
        returns_df = tp[4]

        s = Signaler()
        aggregator = Aggregator(clusterOne, clusterTwo)
        train_ds = train_ds.reset_index().set_index(['level_0', 'timestamp']).sort_index()
        node_features, edge_features = s.generate(train_ds)
        node_features.sort_index(inplace=True)
        edge_features.sort_index(inplace=True)
        return_dates = node_features.index.get_level_values(level='timestamp').sort_values().unique()
        start_idx = returns_df.index.get_loc(return_dates[0])
        end_idx = returns_df.index.get_loc(return_dates[-1])
        target_ret = returns_df.iloc[start_idx:end_idx+4]
        target_ret = target_ret.rolling(window=3).sum().dropna()
        cpu_data = aggregator.get_dataset(node_features, edge_features, target_ret)
        test_ds = test_ds.reset_index().set_index(['level_0', 'timestamp']).sort_index()
        node_features, edge_features = s.generate(test_ds)
        node_features.sort_index(inplace=True)
        edge_features.sort_index(inplace=True)
        return_dates = node_features.index.get_level_values(level='timestamp').sort_values().unique()
        start_idx = returns_df.index.get_loc(return_dates[0])
        end_idx = returns_df.index.get_loc(return_dates[-1])
        target_ret = returns_df.iloc[start_idx:end_idx+4]
        target_ret = target_ret.rolling(window=3).sum().dropna()
        test_data = aggregator.get_dataset(node_features, edge_features, target_ret)
        return (cpu_data, test_data, return_dates)
    except:
        print("Exception")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    df = pd.read_hdf('crypto.hdf')
    returns_df = df.reset_index(level=0).dropna()
    returns_df = returns_df.pivot(columns='level_0', values='return')
    dates = returns_df.index.unique()

    start_date = dates[0] 
    end_date = start_date + pd.Timedelta(hours=96)
    data = returns_df.truncate(after=end_date)
    c = Separator(2)
    clusters = c.divide(data)
    clusterOne = clusters[1]
    clusterTwo = clusters[2]

    data = df.reset_index(level=0).sort_index()
    hours = data.index.unique()

    OOS = 10
    HISTORY=275

    windows = generate_windows(data, hours, clusterOne, clusterTwo, returns_df, HISTORY, OOS)

    num_cpu = cpu_count()
    print("total cpus", num_cpu)
    with Pool(num_cpu) as p:
        results = p.map(generate_features, windows)

    folder_path = './models/'

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    inputs = [(result, clusterOne, clusterTwo) for result in results]

    with Pool(num_cpu) as p:
        outputs = p.map(train_test, inputs)
    
    weight_df = pd.concat(outputs)
    weight_df.to_csv("weights.csv")
    returns_df.to_csv("returns.csv")
