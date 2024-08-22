import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import torch
import os
from pathlib import Path
from typing import Tuple
import yfinance as yf
from torch.utils.data import DataLoader

DATASET_DIR = "data"

def get_trading_days(start_date, end_date, exchange='NYSE', dates_only=False):
    """Get trading hours/day for an exchange.

    Args:
        start_date (str)
        end_date (str)
        exchange (str, optional). Defaults to 'NYSE'.
        dates_only (bool, optional): Only return trading day dates. Defaults to False.

    Returns:
        pd.DataFrame: Trading hours/days
    """
    nyse = mcal.get_calendar(exchange)
    trading_days = nyse.schedule(start_date=start_date, end_date=end_date)
    dates_only_df = pd.DataFrame(trading_days.index.date)
    return dates_only_df if dates_only else trading_days


def compute_SMA(stock_data: pd.DataFrame, num_periods: int):
    """Computes simple moving average.

    Args:
        stock_data (pd.DataFrame): Input dataframe.
        num_periods (int): Size of the rolling window.

    Returns:
        pd.DataFrame: Original Dataframe with appended SMA column.
    """
    stock_data[f'SMA_{num_periods}'] = stock_data['Close'].rolling(
        window=num_periods).mean()
    return stock_data


def normalize(x: torch.Tensor, dim=1) -> torch.Tensor:
    """MinMax Norm (normalized to [0, 1] along given axis)

    Args:
        x (torch.Tensor): Input tensor.
        dim (int, optional): The axis normalized. Defaults to 1.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    x_min, _idcs = torch.min(x, dim=dim, keepdim=True)
    x_max, _idcs = torch.max(x, dim=dim, keepdim=True)
    return (x - x_min) / (x_max - x_min)


def _save_stock_data(stock_data: pd.DataFrame, file_path: Path):
    os.makedirs(Path(__file__).resolve().parent / DATASET_DIR, exist_ok=True)
    stock_data.to_csv(file_path)


def get_stock_data_by_symbol_yf(symbol: str, start_date, end_date) -> pd.DataFrame:
    """Get training data for a symbol and time period.

    Args:
        symbol (str): Ticker symbol of the stock.
        start_date (str): Start date.
        end_date (str): End date.
    Returns:
        pd.DataFrame: Stock data.
    """
    # check if data is already saved
    file_path = Path(__file__).resolve().parent / DATASET_DIR / f"{symbol}.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    # download data
    stock_data = yf.download(symbol, start_date, end_date)
    stock_data = compute_SMA(stock_data, 20)
    stock_data = compute_SMA(stock_data, 50)
    stock_data = compute_SMA(stock_data, 100)
    # save data
    _save_stock_data(stock_data, file_path)
    return stock_data

###########################################################################################################################
# Labeling functions
###########################################################################################################################

# now I need x and y, x = [open, adj close, 20sma, 50sma, 100sma, volatility, high, low]
# y: whether next day's close is higher or lower, lower = 0, higher = 1
# training period of 60 days


def create_labels_simple(stock_data: pd.DataFrame, training_period: int, max_sma_window_size=100) -> Tuple[torch.Tensor, torch.Tensor]:
    """Example for labeling data. Not (yet) finalized.

    Args:
        stock_data (pd.DataFrame): Stock data (eg. from `get_stock_data_by_symbol_yf`).
        training_period (int): Number of days to train on.
        max_sma_window_size (int, optional): Final invalid datapoint in `stock_data` (in this case the datapoint lacks a value for SMA100). Defaults to 100.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: x, Y.
    """
    days_after_start_record = max_sma_window_size + 1
    x = np.zeros((training_period, 7), dtype=np.float32)
    y = np.zeros((training_period, 1), dtype=np.float32)
    volume = torch.zeros((training_period, 1))
    part_of_data = stock_data[days_after_start_record:
                              days_after_start_record + training_period]
    # ganz anders vorgehen:
    count = 0
    price_of_last_day = 0
    for i, value in part_of_data.iterrows():
        day_data = value
        x[count] = torch.tensor([day_data['Open'], day_data['Adj Close'],
                                 day_data['SMA_20'], day_data['SMA_50'],
                                 day_data['SMA_100'],
                                 day_data['High'], day_data['Low']])
        # print(stock_data.get(i+1, {}).get('Close', 0), day_data.get('Close', 0))
        volume[count] = torch.tensor([day_data['Volume']])
        if count != 0:
            y[count - 1] = torch.tensor([1.0] if day_data['Adj Close']
                                        > price_of_last_day else [0.0])
        price_of_last_day = day_data['Adj Close']
        count += 1

    # set last item randomly to zero, as I do not know how to deal with it
    # I could run the loop one more time, and cut off last label
    y[count - 1] = 0
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)

    # norm all except volume cuz volume kinda big number
    x_tensor = normalize(x_tensor)

    # sooo yeah volume is kinda... meh: it needs to be a small, yet "independent" number that carries useful information
    # random idea: ratio that shows whether volume increased or not (there are better ways than this)
    shifted_vol = torch.cat((torch.tensor([[1]]),  # padding
                             volume[:-1]), dim=0)
    volume_ratios = volume / shifted_vol
    # insert volume tensor
    x_tensor = torch.cat(
        (x_tensor[:, :5], volume_ratios, x_tensor[:, -2:]), dim=1)

    return x_tensor, y_tensor
