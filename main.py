# binary classification --> buy or sell
# input, volatility, stock price, we can do more though, it does not really matter
# hyperparameters, treshold t (certainty, at which stock is being bought, usually 0.5, but can make it 0.9 to avoid
# false positives, so we only buy when it is sensible
# sell after a percentage gain x, percentege loss y, or after day d
# use 1000 training data points, 1000 testing and then run for say 30 days, visualize
# goldman sachs would be quite good
# result --> 60 % accuracy or something
# result option 2: calculate profit which one would have when you would sell on the next day, then compare this to spx or something
# over say 30 day period
# can also calculate sharpe ratio

# binary classification has one output, 0 means don't buy - 1 means buy definetly
# take 20, 50 and 100 day SMA
# 23 May 2023 is the 100th trading day in 2023
# number of trading days easy to get as len(stock_data)


import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import torch
import torch.nn as nn
import yfinance as yf
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from dataset import create_labels_simple, get_stock_data_by_symbol_yf

stock = 'GS'
start_date = '2023-01-01'
end_date = '2024-01-05'

device = "cuda" if torch.cuda.is_available() else "cpu"

# inputs:
D_i, D_k, D_o = 8, 4, 1
training_period = 120

# let's use 3 hidden layers

model = nn.Sequential(
    nn.Linear(D_i, D_k),
    nn.LeakyReLU(),
    nn.Linear(D_k, D_k),
    nn.LeakyReLU(),
    nn.Linear(D_k, D_k),
    nn.LeakyReLU(),
    nn.Linear(D_k, D_o),
    nn.Sigmoid(),
).to(device)


def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_normal_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)


# initialize weights
model.apply(weights_init)
# loss criterion
# nn.BCELoss() <- BCE is better if the dataset is larger in this case
criterion = nn.MSELoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)  # faster than SDG

scheduler = StepLR(optimizer, step_size=10, gamma=0.5)


# \/ "good habit"; prevents the code below from being run if the file is imported elsewhere
if __name__ == "__main__":
    # x, y = create_labels(101)
    stock_data = get_stock_data_by_symbol_yf(stock, start_date, end_date)
    x, y = create_labels_simple(
        stock_data, training_period=training_period, max_sma_window_size=100)
    data_loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=True)

    losses = []
    for epoch in range(100):
        epoch_loss = 0.0
        # loop over batches
        for i, data in enumerate(data_loader):
            # retrieve inputs and labels for this batch
            # this is really the whole batch, so x_batch and y_batch each contain 10 arrays of input / output pairs
            x_batch, y_batch = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            pred = model(x_batch.to(device))
            print(pred.flatten().tolist())
            loss = criterion(pred, y_batch.to(device))
            # backward pass
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 10)
            # SGD update
            optimizer.step()
            # update statistics
            epoch_loss += loss.item()
        # print error
        losses.append(epoch_loss)
        print(f'Epoch {epoch:5d}, loss {epoch_loss:.10f}')
        # tell scheduler to consider updating learning rate
        scheduler.step()

    model = model.eval()

    def perform_inference():
        # inference
        # x, y = create_labels(101 + training_period)
        x, y = create_labels_simple(
            stock_data, training_period=training_period, max_sma_window_size=100+training_period)
        pred = model(x.to(device))
        pred = pred.flatten().tolist()
        y = y.tolist()
        num_of_true_positives = 0
        num_of_false_positives = 0
        num_of_true_negatives = 0
        num_of_false_negatives = 0
        print(x)
        print(pred)
        for p, l in zip(pred, y):
            if p < 0.5 and l[0] == 0.0:
                num_of_true_negatives += 1
            elif p < 0.5 and l[0] == 1.0:
                num_of_false_negatives += 1
            elif p >= 0.5 and l[0] == 0.0:
                num_of_false_positives += 1
            else:
                num_of_true_positives += 1
        print('true positives: ', num_of_true_positives, 'true negatives: ', num_of_true_negatives,
              'false positives: ', num_of_false_positives, ' false negatives: ', num_of_false_negatives)

    perform_inference()


"""
Data sample for reference (pls ignore im too lazy to put this in an extra file):
                  Open        High         Low       Close   Adj Close   Volume   (SMA, 20)   (SMA, 50)  (SMA, 100)
Date
2023-05-30  332.079987  332.529999  327.730011  330.829987  318.761322  1998600  325.407999  327.263600  340.776400
2023-05-31  326.070007  327.250000  321.820007  323.899994  314.460388  2938000  324.934499  327.551000  340.577800
2023-06-01  324.510010  324.679993  314.019989  316.399994  307.178955  3339700  324.321999  327.533400  340.261000
2023-06-02  318.239990  325.269989  317.049988  323.649994  314.217682  3987700  324.441498  327.732999  339.967500
2023-06-05  322.929993  323.489990  320.320007  321.809998  312.431305  1462900  324.180998  327.872199  339.612000
...                ...         ...         ...         ...         ...      ...         ...         ...         ...
2023-11-10  323.320007  326.059998  321.649994  325.510010  318.658630  1781400  308.934999  317.905400  325.631999
2023-11-13  324.160004  328.720001  323.529999  326.910004  320.029114  1403300  309.560999  317.895601  325.705299
2023-11-14  333.529999  341.779999  332.470001  338.720001  331.590576  3592700  311.028999  318.195001  325.945399
2023-11-15  339.899994  341.079987  335.339996  337.600006  330.494141  2534500  312.811000  318.524401  326.197799
2023-11-16  337.760010  339.750000  335.010010  336.670013  329.583710  1620400  314.685001  318.818601  326.429999
"""
