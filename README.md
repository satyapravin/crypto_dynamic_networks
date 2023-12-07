# Crypto Long Short Basket Trading using Temporal Graph Networks 

This is a work in progress repository and tries to do the following (note: I use a 128 core server).

1. Uses hourly OHLCV data for top 13-15 cryptocurrencies.
2. Clusters them into two baskets based on correlations.
3. Generates technical indicators for each of the cryptocurrencies.
4. Generates node and edge features and creates a graph dataset per hour for each of the baskets.
5. Uses GAT and GRU to feed into a classifier that predicts which basket will outperform the other.
6. Uses approximately 300 timesteps history to generate outperformance indicator (between baskets) for next 10 hours.
7. Results are stored in local folder.
8. The model per core is stored in models folder and if you want you can use all models for evaluation and average the logits.
