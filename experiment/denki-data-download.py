import os
import requests
from datetime import datetime, timedelta

start = datetime(2022, 1, 1)
end = datetime(2026, 5, 31)

d = start

while d <= end:
    year = d.strftime("%Y")
    month = d.strftime("%m")
    date_str = d.strftime("%Y%m%d")

    # 年/月フォルダ作成
    folder = os.path.join("denki-power-data", year, month)
    os.makedirs(folder, exist_ok=True)

    # ダウンロードURL
    url = f"https://www.okiden.co.jp/denki2/juyo_10_{date_str}.csv"

    response = requests.get(url)

    if response.status_code == 200:
        filename = os.path.join(folder, f"{date_str}.csv")

        with open(filename, "wb") as f:
            f.write(response.content)

        print(f"保存完了: {filename}")

    d += timedelta(days=1)