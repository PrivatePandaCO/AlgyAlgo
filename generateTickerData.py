from datetime import datetime, timedelta
from pytz import timezone
import requests
import json


START_TIME = datetime(2022, 1, 1, 0, 0, 0, 0, timezone('UTC'))
END_TIME = datetime(2025, 1, 1, 0, 0, 0, 0, timezone('UTC'))


data = []

while START_TIME < END_TIME:
    d = requests.get("https://data-api.binance.vision/api/v3/klines", params={"symbol": "BTCUSDT", "interval": "4h", "limit": 1000, "startTime": int(START_TIME.timestamp() * 1000)}).json()
    data.extend(d)
    print(START_TIME)
    START_TIME += timedelta(hours=1000 * 4)

data = [[i[0], float(i[1]), float(i[2]), float(i[3]), float(i[4]), float(i[5]), i[8]] for i in data]

json.dump(data, open("data/data.json", "w"), indent=4)