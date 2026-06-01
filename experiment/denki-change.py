import pandas as pd
import glob

# =========================
# 全CSV取得
# =========================
files = glob.glob("denki-power-data/*.csv")

power_list = []

# =========================
# 各CSV読み込み
# =========================
for file in files:

    # ファイル読み込み
    with open(file, encoding="cp932") as f:
        lines = f.readlines()

    # DATE,TIME の位置を探す
    start_index = None

    for i, line in enumerate(lines):
        if "DATE,TIME" in line:
            start_index = i
            break

    # 1時間データ部分だけ読む
    power = pd.read_csv(
        file,
        encoding="cp932",
        skiprows=start_index,
        nrows=24
    )

    # 必要列
    power = power.iloc[:, [0, 1, 2]]

    power.columns = [
        "date",
        "time",
        "power_demand"
    ]

    # datetime作成
    power["datetime"] = pd.to_datetime(
        power["date"] + " " + power["time"],
        errors="coerce"
    )

    # 数値化
    power["power_demand"] = pd.to_numeric(
        power["power_demand"],
        errors="coerce"
    )

    # 必要列だけ
    power = power[[
        "datetime",
        "power_demand"
    ]]

    power_list.append(power)

# =========================
# 全データ結合
# =========================
all_power = pd.concat(
    power_list,
    ignore_index=True
)

# 時系列順
all_power = all_power.sort_values(
    "datetime"
)

# =========================
# 表示
# =========================
print("\n===== POWER DATA =====")
print(all_power.to_string())

# =========================
# 保存
# =========================
all_power.to_csv(
    "denki-clean.csv",
    index=False
)

print("\ndenki-clean.csv saved.")