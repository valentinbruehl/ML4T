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
from dataset import create_labels_simple, get_stock_data_by_symbol
from model import SimpleModel, device

stock = 'GS'
start_date = '2023-01-01'
end_date = '2024-01-05'


# inputs:
D_i, D_k, D_o = 8, 4, 1
training_period = 120

model = SimpleModel(D_i, D_k, D_o).to(device)

# loss criterion
# nn.BCELoss() <- BCE is better if the dataset is larger in this case
criterion = nn.MSELoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)  # faster than SDG

scheduler = StepLR(optimizer, step_size=10, gamma=0.5)


# \/ "good habit"; prevents the code below from being run if the file is imported elsewhere
if __name__ == "__main__":
    # x, y = create_labels(101)
    stock_data = get_stock_data_by_symbol(stock, start_date, end_date)
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


