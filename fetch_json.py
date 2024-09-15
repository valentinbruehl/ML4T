import json
import pandas as pd
from pathlib import Path


def load_data() -> pd.DataFrame:
    # downloads the json file with all relevant data for the fear and greed index
    with open(Path(__file__).resolve() / "data" / "FAGI_since_2023.json") as file:
        data = json.load(file)
    # extracts the relevant data from json file
    data_points = data["fear_and_greed_historical"]["data"]
    # creates a map between dates and their corresponding fear and greed index
    data_index_map = {
        pd.to_datetime(day["x"], unit="ms"): day["y"] for day in data_points
    }
    return data_index_map


def get_fear_and_greed() -> pd.DataFrame:
    return load_data()
