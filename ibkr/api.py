import requests
import pandas as pd
import json
import yaml
import os
import datetime
from pathlib import Path
from typing import Callable
import logging
from abc import ABC, abstractmethod
import time

# configure logging
logging.basicConfig(level=logging.DEBUG,  # TODO: change to logging.WARNING
                    format='%(asctime)s - %(levelname)s - %(message)s')

# check for ibkr gateway and get baseUrl
GATEWAY_ENABLED = False
baseUrl = None
_GATEWAY_DIR = Path(__file__).resolve().parent / "gateway" / "clientportal.gw"

if os.path.exists(_GATEWAY_DIR):
    # find port
    with open(_GATEWAY_DIR / "root" / "conf.yaml", "r") as F:
        config = yaml.full_load(F)
    # build baseUrl
    baseUrl = f"https://localhost:{config['listenPort']}/v1/api"
    # attempt connection
    try:
        _reponse = requests.get(baseUrl)
        GATEWAY_ENABLED = _reponse.status_code == 401
    except requests.exceptions.ConnectionError:
        pass


# Abstract class for Web and TWS API
class IBKRAPI(ABC):
    @abstractmethod
    def get_historical(symbol: str, start: str, end: str, interval: str | None = None):
        pass

    @abstractmethod
    def get_price(symbol: str):
        pass

    # Order methods
    @abstractmethod
    def place_order():
        pass

    @abstractmethod
    def cancel_order():
        pass

    @abstractmethod
    def get_order_info():
        pass

    @abstractmethod
    def get_all_orders():
        pass

    @abstractmethod
    def get_positions():
        pass

    @abstractmethod
    def close_all():
        pass


###########################################################################################################################
# Web API
# https://www.interactivebrokers.com/campus/ibkr-api-page/webapi-ref
# https://www.interactivebrokers.com/campus/ibkr-api-page/webapi-doc
# https://www.interactivebrokers.com/campus/ibkr-api-page/cpapi-v1
###########################################################################################################################

class _conid_cache():
    def __init__(self, conid_getter: Callable[[str], int]) -> None:
        self._conid_getter = conid_getter
        self.cache_dir = Path(__file__).resolve().parent / "cache"
        self.cache_path = self.cache_dir / "conid.json"
        os.makedirs(self.cache_dir, exist_ok=True)
        # Load file
        self.conid = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as F:
                self.conid = json.load(F)

    def get_conid(self, symbol: str) -> int:
        """get conid for symbol"""
        if not symbol in self.conid:
            self.conid[symbol] = self._conid_getter(symbol)
            self.save()
        return self.conid.get(symbol)

    # alias
    def get(self, symbol: str):
        """get conid for symbol"""
        return self.get_conid(symbol)

    def save(self):
        """save conids to cache"""
        with open(self.cache_path, "w+") as F:
            json.dump(self.conid, F)


class WebAPI():
    DELAY = 1/49  # in seconds

    def __init__(self) -> None:
        self.API_URL = baseUrl
        self.security_type = "STK"  # Stonks TODO: use this attr
        self.conid = _conid_cache(self._get_conid_by_symbol)
        self._last_request_timestamp = 0
        self.return_as_pandas = True

    def _request(self, url: str, accept_list=False, **query_params):
        # for better error handling
        # TODO: implement better error handling
        if self.API_URL not in url:
            url = f"{self.API_URL}/{url.strip('/')}"
        # add params
        # TODO: sanitize keys and value to prevent errors while parsing
        for key, value in query_params.items():
            if "?" not in url:
                url += f"?{key}={str(value).lower()}"
            else:
                url += f"&{key}={str(value).lower()}"
        logging.debug(f"Requesting {url}")
        # ensure delay is met
        current_time = time.time()
        if current_time - self._last_request_timestamp < self.DELAY:
            time.sleep(self.DELAY - (current_time -
                       self._last_request_timestamp))
        # send request and save timestamp
        response = requests.get(url=url, verify=False)
        self._last_request_timestamp = time.time()
        # TODO: handle status_code != 200
        logging.debug(f"Received status code {response.status_code}.")
        if response.status_code >= 400:
            logging.error(
                f"Received status code {response.status_code} while requesting {url}")
        # retrieve content
        def _clean(s: str):
            return s.replace("null", "None").replace("false", "False").replace("true", "True")
        response = eval(_clean(response.content.decode()))
        if isinstance(response, list) and (not accept_list):
            response = response[0]
        logging.debug(f"Response:\n{response}")
        return response

    def _get_conid_by_symbol(self, symbol: str) -> int:
        response = self._request(f"/iserver/secdef/search?symbol={symbol}")
        return response["conid"]

    def get_historical(self, symbol: str, start: datetime.datetime, duration: str = "1y", interval: str = "1h", outsideRth: bool = True):
        """Get historical data for a symbol.

        Args:
            symbol (str): Symbol of stock.
            start (datetime.datetime): Start time of the data.
            duration (str, optional): Available time periods {1-30}min, {1-8}h, {1-1000}d, {1-792}w, {1-182}m, {1-15}y. Defaults to "1y"
            interval (str, optional): 1min, 2min, 3min, 5min, 10min, 15min, 30min, 1h, 2h, 3h, 4h, 8h, 1d, 1w, 1m. Defaults to "1h".
            outsideRth (bool, optional): Whether data should be returned for trades outside regular trading hours. Defaults to True.
        """
        conid = self._get_conid_by_symbol(symbol)
        # time format: 20230821-13:30:00
        start = datetime.datetime.strftime(start, "%Y%m%d-%H:%M:%S")
        request_url = f"{baseUrl}/iserver/marketdata/history?conid={conid}&period={duration}&bar={interval}&startTime={start}"
        response = self._request(request_url, outsideRth=outsideRth)
        if self.return_as_pandas:
            # TODO: implement this
            logging.error("Not yet implemented. Returning as dict.")
        return response


# tests
if __name__ == "__main__":
    api = WebAPI()
    api.return_as_pandas = False
    data = api.get_historical("GME", datetime.datetime(2023, 7, 1, 13, 30, 0))
    print(data)

    # NOTE: delete this later
    """
    >>>
    {'serverId': '1281979', 'symbol': 'GME', 'text': 'GAMESTOP CORP-CLASS A', 'priceFactor': 100, 'startTime': '20230329-22: 00: 00', 'high': '2898/44361.86/100680', 'low': '1806/1532.16/48600', 'timePeriod': '1y', 
    'barLength': 3600, 'mdAvailability': 'S', 'mktDataDelay': 0, 'outsideRth': True, 'volumeFactor': 100, 'priceDisplayRule': 1, 'priceDisplayValue': '2', 'chartPanStartTime': '20230701-13: 30: 00', 'direction': -1, 
    'negativeCapable': False, 'messageVersion': 2, 'data': [
            {'o': 22.51, 'c': 22.5, 'h': 22.51, 'l': 22.45, 'v': 53.02, 't': 1680127200000
            },
            {'o': 22.45, 'c': 22.4, 'h': 22.46, 'l': 22.4, 'v': 26.54, 't': 1680130800000
            },
            {'o': 22.53, 'c': 22.5, 'h': 22.53, 'l': 22.5, 'v': 5.64, 't': 1680163200000
            },
            {'o': 22.5, 'c': 22.46, 'h': 22.5, 'l': 22.42, 'v': 16.01, 't': 1680166800000
            },
            {'o': 22.5, 'c': 22.6, 'h': 22.6, 'l': 22.5, 'v': 6.67, 't': 1680170400000
            },
            {'o': 22.56, 'c': 22.6, 'h': 22.6, 'l': 22.55, 'v': 25.77, 't': 1680174000000
            },
            {'o': 22.55, 'c': 22.6, 'h': 22.74, 'l': 22.5, 'v': 293.08, 't': 1680177600000
            },
            ...
            {'o': 24.45, 'c': 24.4, 'h': 24.45, 'l': 24.4, 'v': 23.76, 't': 1688162400000
            },
            {'o': 24.4, 'c': 24.32, 'h': 24.49, 'l': 24.32, 'v': 19.7, 't': 1688166000000
            }
        ], 'points': 999, 'travelTime': 4785
    }
    serverId: String.
    Internal request identifier.

    symbol: String.
    Returns the ticker symbol of the contract.

    text: String.
    Returns the long name of the ticker symbol.

    priceFactor: String.
    Returns the price increment obtained from the display rules.

    startTime: String.
    Returns the initial time of the historical data request.
    Returned in UTC formatted as YYYYMMDD-HH:mm:ss

    high: String.
    Returns the High values during this time series with format %h/%v/%t.
    %h is the high price (scaled by priceFactor),
    %v is volume (volume factor will always be 100 (reported volume = actual volume/100))
    %t is minutes from start time of the chart

    low: String.
    Returns the low value during this time series with format %l/%v/%t.
    %l is the low price (scaled by priceFactor),
    %v is volume (volume factor will always be 100 (reported volume = actual volume/100))
    %t is minutes from start time of the chart

    timePeriod: String.
    Returns the duration for the historical data request

    barLength: int.
    Returns the number of seconds in a bar.

    mdAvailability: String.
    Returns the Market Data Availability.
    See the Market Data Availability section for more details.

    mktDataDelay: int.
    Returns the amount of delay, in milliseconds, to process the historical data request.

    outsideRth: bool.
    Defines if the market data returned was inside regular trading hours or not.

    volumeFactor: int.
    Returns the factor the volume is multiplied by.

    priceDisplayRule: int.
    Presents the price display rule used.
    For internal use only.

    priceDisplayValue: String.
    Presents the price display rule used.
    For internal use only.

    negativeCapable: bool.
    Returns whether or not the data can return negative values.

    messageVersion: int.
    Internal use only.

    data: Array of objects.
    Returns all historical bars for the requested period.
    [{
    o: float.
    Returns the Open value of the bar.

    c: float.
    Returns the Close value of the bar.

    h: float.
    Returns the High value of the bar.

    l: float.
    Returns the Low value of the bar.

    v: float.
    Returns the Volume of the bar.

    t: int.
    Returns the Operator Timezone Epoch Unix Timestamp of the bar.
    }],

    points: int.
    Returns the total number of data points in the bar.

    travelTime: int.
    Returns the amount of time to return the details.
    """
