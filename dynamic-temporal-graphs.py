#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 uninstall torch torchvision torchaudio -y')
get_ipython().system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')


# In[2]:


import torch
print(torch.__version__)
print(torch.version.cuda)


# In[3]:


get_ipython().system('pip3 uninstall torch_geometric -y')
get_ipython().system('pip3 uninstall -y pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv')

get_ipython().system('pip3 install torch_geometric')
get_ipython().system('pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html')


# In[4]:


get_ipython().system('pip3 uninstall -y torch-geometric-temporal')
get_ipython().system('pip3 install torch-geometric-temporal')


# In[5]:


url = 'https://launchpad.net/~mario-mariomedina/+archive/ubuntu/talib/+files'
ext = '0.4.0-oneiric1_amd64.deb -qO'
get_ipython().system('wget $url/libta-lib0_$ext libta.deb')
get_ipython().system('wget $url/ta-lib0-dev_$ext ta.deb')
get_ipython().system('dpkg -i libta.deb ta.deb')
get_ipython().system('pip install ta-lib')


# In[6]:


import torch
from torch_geometric_temporal.signal import static_graph_temporal_signal as ts


# In[7]:


get_ipython().system('pip install pandas --upgrade')


# In[8]:


import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster

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


# In[9]:


import talib
import numpy as np
import pandas as pd


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


# In[10]:


import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric_temporal.signal import static_graph_temporal_signal as ts


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
        node_1 = [torch.tensor(ft.T, dtype=torch.float32) for ft in node_attrs_1]
        edge_1 = [torch.tensor(ef.T, dtype=torch.float32) for ef in edge_attrs_1]
        
        node_2 = [torch.tensor(ft.T, dtype=torch.float32) for ft in node_attrs_2]
        edge_2 = [torch.tensor(ef.T, dtype=torch.float32) for ef in edge_attrs_2]
        index_1 = torch.tensor(edge_indices_1[0], dtype=torch.int64)
        index_2 = torch.tensor(edge_indices_2[0], dtype=torch.int64)
        targets = [torch.tensor(np.asarray([y]), dtype=torch.float32) for y in Ys]
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


# In[11]:


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, edge_features):
        super(RecurrentGCN, self).__init__()
        self.seq = torch.nn.Sequential(torch.nn.Linear(edge_features, 32),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(32, 1),
                                       torch.nn.Sigmoid(),
                                      )
        self.recurrent = GConvGRU(node_features, 32, 2)

    def forward(self, x, edge_index, edge_attr):
        out = self.seq(edge_attr).squeeze(-1)
        out = self.recurrent(X=x, edge_index=edge_index, edge_weight=out)
        return out

class GComparer(torch.nn.Module):
    def __init__(self, node_features, edge_features):
        super(GComparer, self).__init__()
        self.gcn1 = RecurrentGCN(node_features, edge_features)
        self.gcn2 = RecurrentGCN(node_features, edge_features)
    
    def forward(self, x, xi, xe, y, yi, ye):
        out1 = self.gcn1(x, xi, xe)
        out2 = self.gcn2(y, yi, ye)
        diff = out1.median()- out2.median()
        return F.sigmoid(diff)


# In[12]:


import pandas as pd
df = pd.read_hdf('/kaggle/input/cryptohourlydata/crypto.hdf')
df.head()


# In[13]:


returns_df = df.reset_index(level=0).dropna()
print(returns_df.head())
returns_df = returns_df.pivot(columns='level_0', values='return')
dates = returns_df.index.unique()


# In[14]:


start_date = dates[0] 
end_date = start_date + pd.Timedelta(hours=96)
data = returns_df.truncate(after=end_date)
c = Separator(2)
clusters = c.divide(data)
clusterOne = clusters[1]
clusterTwo = clusters[2]
print(clusterOne, clusterTwo)


# In[ ]:


from IPython.display import clear_output
from sklearn.metrics import classification_report
from tqdm import tqdm
torch.set_default_dtype(torch.float32)

s = Signaler()
aggregator = Aggregator(clusterOne, clusterTwo)
data = df.reset_index(level=0).sort_index()
hours = data.index.unique()

one_size = np.ones(len(clusterOne)) / len(clusterOne)
two_size = np.ones(len(clusterTwo)) / len(clusterTwo)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GComparer(16, 16).to(device)
#model.load_state_dict(torch.load('/kaggle/working/model_weights.pth'))
weights = pd.DataFrame(index=hours, columns=clusterOne+clusterTwo)
for name, param in model.named_parameters():
    if 'weight' in name:
        torch.nn.init.xavier_uniform_(param)
    elif 'bias' in name:
        torch.nn.init.constant_(param, 0)
        
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.BCELoss()

OOS = 10
HISTORY=540
for N in range(HISTORY, hours.shape[0]-OOS-2, OOS):    
    train_ds = data.truncate(before=hours[N-HISTORY]).truncate(after=hours[N]).dropna()
    train_ds = train_ds.reset_index().set_index(['level_0', 'timestamp']).sort_index()
    node_features, edge_features = s.generate(train_ds)
    node_features.sort_index(inplace=True)
    edge_features.sort_index(inplace=True)
    return_dates = node_features.index.get_level_values(level='timestamp').sort_values().unique()
    print(return_dates.shape)
    start_idx = returns_df.index.get_loc(return_dates[0])
    end_idx = returns_df.index.get_loc(return_dates[-1])
    target_ret = returns_df.iloc[start_idx:end_idx+4]
    target_ret = target_ret.rolling(window=3).sum().dropna()
    cpu_data = aggregator.get_dataset(node_features, edge_features, target_ret)
    dataset = []
    
    for D in (cpu_data[0], cpu_data[2], cpu_data[3], cpu_data[5], cpu_data[6]):
        D = [d.to(device) for d in D]
        dataset.append(D)
    
    dataset.insert(1, cpu_data[1].to(device))
    dataset.insert(4, cpu_data[4].to(device))
        
    for epoch in tqdm(range(901)):
        model.train()
        losses = []
        loss = 0
        for i in range(0, len(dataset[0])):
            x1 = dataset[0][i]
            i1 = dataset[1]
            e1 = dataset[2][i]
            x2 = dataset[3][i]
            i2 = dataset[4]
            e2 = dataset[5][i]
            target = dataset[6][i]
            diff = model(x1, i1, e1, x2, i2, e2)
            loss = criterion(diff, target[0])
            del x1
            del i1
            del e1
            del x2
            del i2
            del e2
            del target
            torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = loss.cpu().detach().numpy()
            losses.append(loss)
            loss = 0

    clear_output(wait=True)
    print("timestamp", N, "epoch", epoch, "loss:", np.mean(losses))
    torch.save(model.state_dict(), '/kaggle/working/model_weights.pth')
    model = GComparer(16, 16).to(device)
    model.load_state_dict(torch.load('/kaggle/working/model_weights.pth'))
    model.eval()
    test_ds = data.truncate(before=hours[N-HISTORY+OOS]).truncate(after=hours[N+OOS])
    test_ds = test_ds.reset_index().set_index(['level_0', 'timestamp']).sort_index()
    node_features, edge_features = s.generate(test_ds)
    node_features.sort_index(inplace=True)
    edge_features.sort_index(inplace=True)
    return_dates = node_features.index.get_level_values(level='timestamp').sort_values().unique()
    start_idx = returns_df.index.get_loc(return_dates[0])
    end_idx = returns_df.index.get_loc(return_dates[-1])
    target_ret = returns_df.iloc[start_idx:end_idx+4]
    target_ret = target_ret.rolling(window=3).sum().dropna()
    dataset = aggregator.get_dataset(node_features, edge_features, target_ret)

    preds = []
    Ys = []
    for i in range(len(dataset[0])):
        x1 = dataset[0][i].to(device)
        i1 = dataset[1].to(device)
        e1 = dataset[2][i].to(device)
        x2 = dataset[3][i].to(device)
        i2 = dataset[4].to(device)
        e2 = dataset[5][i].to(device)
        target = dataset[6][i].to(device)
        diff = model(x1, i1, e1, x2, i2, e2)
        pred = diff.cpu().detach().numpy()
        pred = (pred > 0.5).astype(int)
        preds.append(pred)
        Ys.append(target.cpu().detach().numpy().flatten().astype(int)[0])
        if i > len(dataset[0]) - OOS:
            weights.loc[return_dates[i], clusterOne] = one_size if pred > 0 else -one_size
            weights.loc[return_dates[i], clusterTwo] = two_size if pred < 1 else -two_size
    print(classification_report(Ys, preds, zero_division=0))


# In[ ]:


weights = weights.dropna()
print(weights.shape)
print(weights.shape)
strategy = weights.shift(3).mul(returns_df.loc[weights.index, :])
strategy.sum(axis=1).astype(float).cumsum().plot()


# In[ ]:


weights.abs().diff().sum(axis=1).astype('float').describe()


# In[ ]:




