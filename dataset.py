import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import torch
import os
from pathlib import Path
from typing import Tuple, List
from datetime import datetime
import yfinance as yf
from torch.utils.data import DataLoader
from scipy.signal import argrelextrema
from ibkr.api import WebAPI

DATASET_DIR = "data"

# init web api
WEBAPI = WebAPI()
WEBAPI.return_as_pandas = True
WEBAPI.DO_REQUEST_CACHING = True


def _str_to_date(date_string: str):
    return datetime.strptime(date_string, "%Y-%m-%d")


def get_trading_days(start_date, end_date, exchange="NYSE", dates_only=False):
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
    """
    Computes simple moving average.
    """
    stock_data[f"SMA_{num_periods}"] = (
        stock_data["Close"].rolling(window=num_periods).mean()
    )
    return stock_data


def compute_VWAP(stock_data: pd.DataFrame):
    # AS a timeframe, we use 10 days and get each day as a data point
    # usually, VWAP is calculated with data points every 10-15 minutes, and is primarily used for intraday trading
    # formula: sum(average price * volume) / sum(volume)
    stock_data["VWAP"] = (
        stock_data.rolling(window=10).sum(
            (stock_data["High"] + stock_data["Low"]) * stock_data["Volume"]
        )
        / stock_data["Volume"].rolling(window=10).sum()
    )
    return stock_data


def compute_EMA(stock_data: pd.DataFrame, num_periods: int):
    # similiar to SMA, but revent days are given more weight, hence recent changes have stronger influence on sma
    # weighting factor can be changed, 2 is a common choice
    # not optimal, as every day is computed 20 times

    # adjust=False means that the weights of each data point are not adjusted so they sum to 1
    # in the traditional calculation of EMA, the weights are not adjusted
    # span = number of days respected in calculation
    # alpha = 2 / (span + 1)

    stock_data['EMA'] = stock_data['Close'].ewm(
        span=num_periods, alpha=2 / (num_periods + 1), adjust=False).mean()
    return stock_data


def compute_MACD(stock_data: pd.DataFrame):
    # Moving Average Convergence/divergence
    # formula: 12 day ema - 26 day ema
    stock_data['MACD'] = compute_EMA(stock_data=stock_data, num_periods=12) - compute_EMA(
        stock_data=stock_data, num_periods=26
    )
    return stock_data


def compute_Bollinger_Bands(stock_data: pd.DataFrame):
    # returns 2 values, the upper and lower bollinger band respectively
    # it is commonly calculated with the 20-day sma line

    stock_data["Upper_Bollinger_Band"] = (
        stock_data["Close"].rolling(window=20).mean()
        + 2 * stock_data["Close"].rolling(window=20).std()
    )
    stock_data["Lower_Bollinger_Band"] = (
        stock_data["Close"].rolling(window=20).mean()
        - 2 * stock_data["Close"].rolling(window=20).std()
    )
    return stock_data


def normalize_axis(x: torch.Tensor, axis=1) -> torch.Tensor:
    """
    MinMax Norm (normalized to [0, 1] along given axis)
    """
    x_min, _idcs = torch.min(x, dim=axis, keepdim=True)
    x_max, _idcs = torch.max(x, dim=axis, keepdim=True)
    return (x - x_min) / (x_max - x_min)


def _save_stock_data(stock_data: pd.DataFrame, file_path: Path, interval: str):
    os.makedirs(Path(__file__).resolve().parent /
                DATASET_DIR / interval, exist_ok=True)
    if os.path.exists(file_path):
        # update existing data
        previous_stock_data = pd.read_csv(file_path, index_col="Time")
        previous_stock_data.index = pd.to_datetime(previous_stock_data.index)
        stock_data = pd.concat([previous_stock_data, stock_data]).sort_index()
    stock_data.to_csv(file_path)

# deprecated
def _get_stock_data_by_symbol_yf(
    symbol: str,
    start_date,
    end_date,
    interval="1d",
    sma_periods: List[int] | None = None,
) -> pd.DataFrame:
    """Get training data for a symbol and time period.

    Args:
        symbol (str): Ticker symbol of the stock.
        start_date (str): Start date.
        end_date (str): End date.
        interval (str): Frequency of stock data. Defaults to "1d".
        sma_periods (List[int]): List of SMA to compute with the given number of periods. Defaults to [20, 50, 100].
    Returns:
        pd.DataFrame: Stock data.
    """
    if sma_periods is None:
        sma_periods = [20, 50, 100]
    # check if data is already saved
    file_path = (
        Path(__file__).resolve().parent /
        DATASET_DIR / interval / f"{symbol}.csv"
    )
    if os.path.exists(file_path):
        stock_data = pd.read_csv(file_path, index_col="Time")
        stock_data.index = pd.to_datetime(stock_data.index)
        # check if requested period is contained in file (assuming that yf.download does return data INCLUDING the end_date, this should work)
        if (stock_data.index[0] <= _str_to_date(start_date)) and (
            stock_data.index[-1] >= _str_to_date(end_date)
        ):
            mask = (stock_data.index >= _str_to_date(start_date)) & (
                stock_data.index <= _str_to_date(end_date)
            )
            # compute ema
            stock_data = compute_EMA(stock_data=stock_data, num_periods=20)
            # compute Bollinger Bands
            stock_data = compute_Bollinger_Bands(stock_data)
            # compute MACD (moving average convergence divergence)
            stock_data = compute_MACD(stock_data)
            # compute VWAP
            stock_data = compute_VWAP(stock_data)
            # compute sma
            for sma in sma_periods:
                stock_data = compute_SMA(stock_data, sma)
            # NOTE: this data could contain less NaNs because SMA is computed before applying the mask
            return stock_data.loc[mask]
    # download data
    stock_data = yf.download(symbol, start_date, end_date, interval=interval)
    stock_data.index.name = "Time"
    # save data
    _save_stock_data(stock_data, file_path, interval)
    # compute sma
    for sma in sma_periods:
        stock_data = compute_SMA(stock_data, sma)
    return stock_data

#############################################################################################################
# >>> caching behavior of the dataframe will be deprecated (for now !!!!!!!!) in favor of request caching <<<
#############################################################################################################
def get_stock_data_by_symbol_ibkr(
        symbol: str,
        start_date: datetime = None,
        duration: str = "1y",
        interval: str = "15min",
        sma_periods: List[int] | None = None,
) -> pd.DataFrame:
    if sma_periods is None:
        sma_periods = [20, 50, 100]

    # TODO: maybe add caching here

    # get data from ibkr api
    stock_data = WEBAPI.get_historical(
        symbol=symbol,
        start=start_date,
        duration=duration,
        interval=interval,
        outsideRth=True,  # outside regular trading hours
    )
    # compute ema
    stock_data = compute_EMA(stock_data=stock_data, num_periods=20)
    # compute Bollinger Bands
    stock_data = compute_Bollinger_Bands(stock_data)
    # compute MACD (moving average convergence divergence)
    stock_data = compute_MACD(stock_data)
    # compute VWAP
    stock_data = compute_VWAP(stock_data)
    # compute sma
    for sma in sma_periods:
        stock_data = compute_SMA(stock_data, sma)

    return stock_data


get_stock_data_by_symbol = get_stock_data_by_symbol_ibkr


###########################################################################################################################
# Labeling functions
###########################################################################################################################

# now I need x and y, x = [open, adj close, 20sma, 50sma, 100sma, volatility, high, low]
# y: whether next day's close is higher or lower, lower = 0, higher = 1
# training period of 60 days


def create_labels_simple(
    stock_data: pd.DataFrame, training_period: int, max_sma_window_size=100
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    part_of_data = stock_data[
        days_after_start_record: days_after_start_record + training_period
    ]
    # ganz anders vorgehen:
    count = 0
    price_of_last_day = 0
    for i, value in part_of_data.iterrows():
        day_data = value
        x[count] = torch.tensor(
            [
                day_data["Open"],
                day_data["Adj Close"],
                day_data["SMA_20"],
                day_data["SMA_50"],
                day_data["SMA_100"],
                day_data["High"],
                day_data["Low"],
            ]
        )
        # print(stock_data.get(i+1, {}).get('Close', 0), day_data.get('Close', 0))
        volume[count] = torch.tensor([day_data["Volume"]])
        if count != 0:
            y[count - 1] = torch.tensor(
                [1.0] if day_data["Adj Close"] > price_of_last_day else [0.0]
            )
        price_of_last_day = day_data["Adj Close"]
        count += 1

    # set last item randomly to zero, as I do not know how to deal with it
    # I could run the loop one more time, and cut off last label
    y[count - 1] = 0
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)

    # norm all except volume cuz volume kinda big number
    x_tensor = normalize_axis(x_tensor, axis=1)  # NOTE: this is wrong

    # sooo yeah volume is kinda... meh: it needs to be a small, yet "independent" number that carries useful information
    # random idea: ratio that shows whether volume increased or not (there are better ways than this)
    shifted_vol = torch.cat(
        (torch.tensor([[1]]), volume[:-1]), dim=0)  # padding
    volume_ratios = volume / shifted_vol
    # insert volume tensor
    x_tensor = torch.cat(
        (x_tensor[:, :5], volume_ratios, x_tensor[:, -2:]), dim=1)

    return x_tensor, y_tensor


###########################################################################################################################
# Advanced labeling functions
###########################################################################################################################


def _preprocess_data(data: pd.DataFrame, training_period: int) -> pd.DataFrame:
    # automatically discard incomplete rows (instead of using max_sma_window_size to remove the first rows)
    data = data.dropna(axis=0)
    # cut data
    return data[:training_period]


def create_labels_local_min_max(
    stock_data: pd.DataFrame,
    training_period: int,
    extrema_distance: int,
    sma_periods: List[int] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if sma_periods is None:
        sma_periods = [20, 50, 100]

    stock_data = _preprocess_data(stock_data, training_period)

    # find extrema (marked with 1 if local extrema)
    stock_data["Min"] = (
        stock_data.iloc[
            argrelextrema(
                stock_data["High"].values, np.less_equal, order=extrema_distance
            )[0]
        ]["High"]
        > 0
    )
    stock_data["Max"] = (
        stock_data.iloc[
            argrelextrema(
                stock_data["High"].values, np.greater_equal, order=extrema_distance
            )[0]
        ]["High"]
        > 0
    )

    # create x, y
    x = torch.tensor(
        [
            stock_data["Open"],
            stock_data["Adj Close"],
            stock_data["High"],
            stock_data["Low"],
            *[stock_data[f"SMA_{n}"] for n in sma_periods],
            stock_data['VWAP'],
            stock_data['EMA'],
            stock_data['MACD'],
            stock_data['Lower_Bollinger_Band'],
            stock_data['Upper_Bollinger_Band'],
        ]
    )
    y = []
    # TODO: remove later if not triggered
    assert len(stock_data["Min"].shape) == 1
    for is_min, is_max in zip(stock_data["Min"].to_list(), stock_data["Max"].to_list()):
        if is_min:
            y.append(-1)
        elif is_max:
            y.append(1)
        y.append(0)

    y = torch.tensor(y, dtype=torch.int)

    # normalize x
    x = normalize_axis(x, axis=0)  # normalize each row

    # TODO: add volume and F&G data

    return x, y


# tests
if __name__ == "__main__":
    ticker = "GME"
    a_start = "2023-02-01"
    b_start = "2023-01-01"
    a_end = "2023-04-02"
    b_end = "2023-03-01"
    df = get_stock_data_by_symbol(
        ticker,
        b_start,
        b_end,
    )
    print(df.iloc[0], df.iloc[-1])
