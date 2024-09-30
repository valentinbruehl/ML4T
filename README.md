# Machine Learning for Trading (wip)
## Description
## Installation
Requires python (preferably `>=3.11`) and git to be installed.
```
git clone https://github.com/Ikeafisch3/ML4T.git
pip install -r requirements.txt
```
For CUDA GPUs use `pip install -r requirements-gpu.txt`

## Installing IBKR Gateway

Refer to [IBKR webpage](https://www.interactivebrokers.com/campus/ibkr-api-page/cpapi-v1/#cpgw).

or

If Java is installed, download the zip file from [here](https://www.interactivebrokers.com/campus/ibkr-api-page/cpapi-v1/#gw-step-one) and extract it to `.\ibkr\gateway`. All files should be in `\ML4T\ibkr\gateway\clientportal.gw\`. 

Start a terminal in `.ibkr\gateway\clientportal.gw\` and run `bin\run.bat root\conf.yaml`. Login with your Paper Trading account at `https://localhost:[port]` (by default the URL is `https://localhost:5000`, however this can be changed in the `root\config.yaml`)

After this `ibkr\api.py` should be functional.

## TODO
- Determine a more fitting model for the task
- Add Paper Trading capability
- Add designated training scripts
- Add the following indicators:
    - Relative Strength index (RSI) --> speed and change of price movements
    - Stochastic Oscillator (compares closing price with severel key historical prices (highs, lows)) --> indicates momentum
    - put / call ratio --> not so relevant for large cap but might include anyway
    - some general economic indicator related to gdp growth, unemployment rate etc. --> Need to find an indicator which updates more regularly than once a month
    - Index Performances: SPX, sector indexes, emerging markets, small cap indexes
    - bid/ask spread --> provides insight into market liquidity and volatility (I think: high spread --> low liquidity)
    - Days until next financial data release or annual shareholder meeting
    - previous candles (?)
- Fixes:
    - properly implement fear and greed data (eg. for different candle sizes)
    - rate limits on IBKR Gateway are not properly implemented (should be different for some requests)
    - `inference.py` needs clean up and better docs


