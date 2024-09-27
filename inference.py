import torch
from torch import nn
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest, backtesting
from model import SimpleModel, device
from datetime import datetime, timedelta
from dataset import compute_indicators, get_stock_data_by_symbol_ibkr, _preprocess_data, _compute_x

###########################################################################################################################
# Inference
###########################################################################################################################


def _inference(
        model: nn.Module,
        input: torch.Tensor
) -> torch.Tensor:
    model = model.eval().to(device)
    while len(input.shape) < 2:
        input = input.unsqueeze(0)
    output = model(input.to(device)).cpu()
    return output


###########################################################################################################################
# Evaluation (Backtesting)
###########################################################################################################################

# https://kernc.github.io/backtesting.py/doc/examples/Quick%20Start%20User%20Guide.html
# https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html

def precomputed(data): return data


def _find_sma_periods(data: pd.DataFrame):
    sma_periods = []
    for col in data.columns.to_list():
        if "sma" in col:
            sma_periods.append(
                int(col.strip("sma_"))
            )
    return sma_periods

def load_state_dict():
    return 

class _ML_Strat(Strategy):
    takeProfitPerc = 0.15
    """P/L% to sell after. (eg. for `0.15`: if the stock increases by 15% the algo sells)"""
    stopLossPerc = 0.1
    """P/L% to sell after. (eg. for `0.1`: if the stock drops by 10% the algo sells)"""
    maxHoldTime = 30
    """Maximum number of days to hold the stock for. Can be of type `float`."""
    confidenceThreshold = 0.5
    """Minimum confidence in the models prediction for the trade to be made."""
    maxRisk = 0.5
    """Maximum of the current **cash balance** to risk in transaction trade."""

    def __init__(self):
        super().__init__()
        # TODO: automatically get this info / move to model.py
        D_i, D_k, D_o = 8, 4, 1
        # load model
        self.model = SimpleModel(D_i, D_k, D_o)
        self.model.load_state_dict(
            load_state_dict()
        )
        # visible indicators
        for indicator in ["Upper_Bollinger_Band", "Lower_Bollinger_Band", "MACD", "VWAP", "RSI", "EMA"]:
            setattr(self, indicator, self.I(
                precomputed, getattr(self.data, indicator),
                name=indicator.replace("_", " "))
            )
        # perform model inference
        self._complete_inference()

    def _complete_inference(self):
        # new column for predictions
        self.data["pred"] = 0
        stock_data = self.data
        # find sma periods
        sma_periods = _find_sma_periods(stock_data)
        # perform inference for each timestep
        for idx, data in self.data.iterrows():
            x = _compute_x(data, sma_periods)
            # save data to dataframe
            self.data.at[idx, "pred"] = _inference(
                self.model, x
            ).flatten().tolist()[0]

    @property
    def _cash(self):
        return self._broker._cash

    def next(self):
        # NOTE: according to docs are all dataframes np arrays now

        # get prediction and confidence
        pred = float(self.data.pred[-1])
        confidence = abs(pred)
        buy = pred > 0

        # if not confident; abort
        if confidence < self.confidenceThreshold:
            return

        # get position and cash
        pos = self.position
        cash = self._cash
        share_price = float(self.data.Close[-1])

        # case 1: no position & buy signal: open position
        if not pos and buy:
            self.buy(
                # idk what this is supposed to be. default is ~1 and according to docs its the "maximal possible position". so 100% of the account??? or does 1 mean one share??? idk
                size=self.maxRisk,
                # stop loss
                sl=share_price * (1 - self.stopLossPerc),
                # take profit
                tp=share_price * (1 + self.takeProfitPerc),
                # market order
                limit=None,
            )
        # case 2: no position & sell signal: nothing
        if not pos and not buy:
            return
        # case 3: position & buy signal: double down
        if pos and buy:
            # ignore if no money
            try:
                self.buy(
                    # same as above
                    size=self.maxRisk,
                    # stop loss
                    sl=share_price * (1 - self.stopLossPerc),
                    # take profit
                    tp=share_price * (1 + self.takeProfitPerc),
                    # market order
                    limit=None,
                )
            except backtesting._OutOfMoneyError:
                return
        # case 4: position & sell signal: close position
        if pos and not buy:
            self.position.close(
                portion=1  # this is better documentation
            )

        # case 5: position & max hold time reached
        # TODO: implement this case


def backtest(stock_data: pd.DataFrame):
    # drop nans
    stock_data = _preprocess_data(stock_data, None)

    # run backtest
    bt = Backtest(stock_data, _ML_Strat, cash=10000,
                  commission=0.01, trade_on_close=True)
    # display stuff
    stats = bt.run()
    print(stats)
    bt.plot()

if __name__ == "__main__":
    backtest(get_stock_data_by_symbol_ibkr("GOOG")) 
