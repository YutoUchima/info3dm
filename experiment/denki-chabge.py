import pandas as pd
import glob

# =========================
# 全CSV取得
# =========================
files = glob.glob(
    "denki-power-data/**/*.csv",
    recursive=True
)

power_list = []

# =========================
# 各CSV処理
# =========================
for file in files:

    print(f"reading: {file}")

    # =========================
    # DATE,TIME 行を探す
    # =========================
    with open(file, encoding="cp932") as f:
        lines = f.readlines()

    start_index = None

    for i, line in enumerate(lines):
        if "DATE,TIME" in line:
            start_index = i
            break

    # 見つからなければスキップ
    if start_index is None:
        print("DATE,TIME not found")
        continue

    # =========================
    # 24行だけ取得
    # =========================
    power = pd.read_csv(
        file,
        encoding="cp932",
        skiprows=start_index,
        nrows=24
    )

    # =========================
    # 必要列だけ
    # =========================
    power = power.iloc[:, [0, 1, 2]]

    power.columns = [
        "date",
        "time",
        "power_demand"
    ]

    # =========================
    # datetime作成
    # =========================
    power["datetime"] = pd.to_datetime(
        power["date"].astype(str)
        + " "
        + power["time"].astype(str),
        errors="coerce"
    )

    # =========================
    # 数値化
    # =========================
    power["power_demand"] = pd.to_numeric(
        power["power_demand"],
        errors="coerce"
    )

    # =========================
    # 必要列のみ
    # =========================
    power = power[
        ["datetime", "power_demand"]
    ]

    power_list.append(power)

# =========================
# 全結合
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
print(all_power.to_string())

# =========================
# 保存
# =========================
all_power.to_csv(
    "dennki-clean.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\ndenki-clean.csv saved.")