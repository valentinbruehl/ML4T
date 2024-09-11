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
    """Caches contract ids"""

    def __init__(self, conid_getter: Callable[[str], int]) -> None:
        # conid_getter: function requesting conid for a symbol from ibkr api
        self._conid_getter = conid_getter
        self.cache_dir = Path(__file__).resolve().parent / "cache"
        self.cache_path = self.cache_dir / "conid.json"
        os.makedirs(self.cache_dir, exist_ok=True)
        # Load cache file
        self.conid = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as F:
                self.conid = json.load(F)

    def get_conid(self, symbol: str) -> int:
        """get conid for symbol"""
        if symbol not in self.conid:
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


class _RequestCache():
    """Caches api requests"""
    categories = ["history"]

    def __init__(self) -> None:
        self.cache_dir = Path(__file__).resolve().parent / "cache"
        self.cache_dict_fp = self.cache_dir / "web_api_request_cache.json"
        self.object_cache_dir = self.cache_dir / "web_api"
        # Create folders
        os.makedirs(self.object_cache_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        for category in self.categories:
            os.makedirs(self.object_cache_dir / category, exist_ok=True)
        os.makedirs(self.object_cache_dir / "misc", exist_ok=True)
        # Load cache dict pointing to each filepath of each request
        if os.path.exists(self.cache_dict_fp):
            with open(self.cache_dict_fp, "r") as F:
                self.cache_dict = json.load(F)
        else:
            self.cache_dict = {category: {} for category in self.categories}
            self.cache_dict["misc"] = {}

    def new_id(self, category: str):
        """create a new file id"""
        return len(os.listdir(self.object_cache_dir / category))

    def read_from_cache(self, request_url: str) -> dict | list | pd.DataFrame:
        """Gets data from cache"""
        # from categories
        for category in self.categories:
            if category in request_url:
                # make entry in category
                return self._read_entry(category, request_url)
        # entries without distinct category
        return self._read_entry("misc", request_url)

    def check_cache(self, request_url: str) -> bool:
        """Checks cache for url"""
        for category in self.categories:
            if category in request_url:
                return request_url in self.cache_dict[category]
        # entries without distinct category
        return request_url in self.cache_dict["misc"]

    def _read_entry(self, category: str, request_url: str):
        """read entry from cache_dir and load file"""
        filename = self.cache_dict[category][request_url]
        fp = self.object_cache_dir / category / filename
        if str(fp).endswith("csv"):
            return pd.read_csv(fp)
        with open(fp, "r") as F:
            return json.load(F)

    def insert_into_cache(self, request_url: str, obj: list | dict | pd.DataFrame):
        """Caches url and data (`obj`)"""
        # organize into categories
        for category in self.categories:
            if category in request_url:
                # make entry in category
                self._write_entry(category, request_url, obj)
                return
        # entries without distinct category
        self._write_entry("misc", request_url, obj)

    def _write_entry(self, category: str, request_url: str, obj: dict | list | pd.DataFrame):
        """save data to cache_dict and disk"""
        # make entry in cache_dict
        filetype = "csv" if isinstance(obj, pd.DataFrame) else "json"
        filename = f"{self.new_id(category)}.{filetype}"
        self.cache_dict[category][request_url] = filename
        # save data
        fp = self.object_cache_dir / category / filename
        with open(fp, "w+") as F:
            if filetype == "json":
                json.dump(obj, F)
            else:
                obj.to_csv(F)
        # save cache_dict
        with open(self.cache_dict_fp, "w+") as F:
            json.dump(self.cache_dict, F)


class WebAPI():
    DELAY = 1/9  # in seconds
    DO_REQUEST_CACHING = True
    """Enables caching of api requests.\n
    **can take up a lot of disk space**"""

    def __init__(self) -> None:
        if not GATEWAY_ENABLED:
            logging.error(
                "Can't connect to IBKR Gateway. WebAPI is therefore inaccessible. Only cached data is available. (Sometimes wrongly triggers, eg. 503, None)")
        self.API_URL = baseUrl
        self.security_type = "STK"  # Stonks
        self.conid = _conid_cache(self._get_conid_by_symbol)
        """Contract id cache"""
        self._last_request_timestamp = 0
        self.return_as_pandas = True
        if self.DO_REQUEST_CACHING:
            self.request_cache = _RequestCache()

    def _request(self, url: str, accept_list=False, **query_params):
        """Sends request to IBKR web API under pacing limitations.
        Parses reponse to python object.

        Args:
            url (str): (partial) url to request.
            accept_list (bool, optional): If the response is a list, do not only return the first item. Defaults to False.
            query_params (dict): Additional parameters added to the url.

        Returns:
            Response: Python object containing the API response.
        """
        # TODO: implement better error handling
        if self.API_URL not in url:
            url = f"{self.API_URL}/{url.strip('/')}"
        # add params
        for key, value in query_params.items():
            if "?" not in url:
                url += f"?{key}={str(value).lower()}"
            else:
                url += f"&{key}={str(value).lower()}"
        logging.debug(f"Requesting {url}")

        # check cache for request
        if self.DO_REQUEST_CACHING:
            if self.request_cache.check_cache(url):
                logging.debug(f"Found url in cache. Loading from there.")
                data = self.request_cache.read_from_cache(url)
                # if the response is a list, select the first item
                if isinstance(data, list) and (not accept_list):
                    data = data[0]
                return data
            else:
                logging.debug(f"Did not find url in cache.")

        # enforce pacing limit
        current_time = time.time()
        if current_time - self._last_request_timestamp < self.DELAY:
            _sleep_time = self.DELAY - \
                (current_time - self._last_request_timestamp)
            logging.debug(f"Pacing. {_sleep_time}")
            time.sleep(_sleep_time)

        # send request and save timestamp
        response = requests.get(url=url, verify=False)
        self._last_request_timestamp = time.time()

        # TODO: handle status_code != 200
        status_code = response.status_code
        logging.debug(f"Received status code {status_code}.")
        if status_code >= 400:
            logging.error(
                f"Received status code {status_code} while requesting {url}")

        # retrieve content and parse to python object (eg. dict, list, ...)
        def _clean(s: str):
            return s.replace("null", "None").replace("false", "False").replace("true", "True")
        response = eval(_clean(response.content.decode()))

        # check cache for request
        if self.DO_REQUEST_CACHING and (status_code < 300):
            logging.debug("Saving request to cache.")
            self.request_cache.insert_into_cache(url, response)

        # if the response is a list, select the first item
        if isinstance(response, list) and (not accept_list):
            response = response[0]

        logging.debug(f"Response:\n{response}")
        return response

    def _get_conid_by_symbol(self, symbol: str) -> int:
        """api request contract id"""
        response = self._request(
            f"/iserver/secdef/search?symbol={symbol}&secType={self.security_type}")
        return response["conid"]

    @staticmethod
    def api_response_to_pandas(data_dict: dict):
        """Convert data dict from api response to DataFrame with descriptive naming."""
        df = pd.DataFrame(data_dict)
        df = df.rename(columns={
            "t": "Time",
            "o": "Open",
            "c": "Close",
            "h": "High",
            "l": "Low",
            "v": "Volume",
        })
        df = df.set_index("Time")
        df.sort_index(inplace=True)
        return df

    def get_historical(self,
                       symbol: str,
                       start: datetime.datetime,
                       duration: str = "1y",
                       interval: str = "1h",
                       outsideRth: bool = True,
                       target_volumeFactor: int = 100,
                       ):
        """Get historical data for a symbol.

        Args:
            symbol (str): Symbol of stock.
            start (datetime.datetime): Start time of the data.
            duration (str, optional): Available time periods {1-30}min, {1-8}h, {1-1000}d, {1-792}w, {1-182}m, {1-15}y. Defaults to "1y"
            interval (str, optional): 1min, 2min, 3min, 5min, 10min, 15min, 30min, 1h, 2h, 3h, 4h, 8h, 1d, 1w, 1m. Defaults to "1h".
            outsideRth (bool, optional): Whether data should be returned for trades outside regular trading hours. Defaults to True.
            target_volumeFactor (int, optional): Desired volumeFactor (factor the volume is multiplied by). Defaults to 100.
        """
        # Get and cache contract id
        conid = self.conid.get(symbol)
        # time format converted to this format: 20230821-13:30:00
        start = datetime.datetime.strftime(start, "%Y%m%d-%H:%M:%S")
        # make request
        request_url = f"{baseUrl}/iserver/marketdata/history?conid={conid}&period={duration}&bar={interval}&startTime={start}"
        response = self._request(request_url, outsideRth=outsideRth)
        # convert to pandas dataframe
        if self.return_as_pandas:
            data = response["data"]
            df = WebAPI.api_response_to_pandas(data)
            # scale volume
            volumeFactor = response["volumeFactor"]
            if volumeFactor != target_volumeFactor:
                df["Volume"] = df["Volume"] * \
                    volumeFactor / target_volumeFactor
            return df
        return response


# tests
if __name__ == "__main__":
    api = WebAPI()
    api.return_as_pandas = True
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
