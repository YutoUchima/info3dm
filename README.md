# info3dm
知能情報総合演習　データマイニング班

テーマ「沖縄の天気と電力需要の関係性分析」

実験再現手順　(ファイルは実行に使う順に説明)


1.データの収集・変換　

-denki_data_download.py 

　　電力消費量のデータをサイトから収集し、denki_power_dataというフォルダに保存。
    
-denki_change.py

　　集めた電気消費量データから1時間ごとの消費量のみを取り出し、denki_clean.csvを作成。

-weather_change.py

　　weather_dataに保存した気象データから、実験に必要な項目のみを取り出し、weather_clean.csvに保存。

-weather_onehot.py

　　weather_clean.csvの天気の項目を、onehotで晴れか曇りか雨など0,1での表記に変換しweather_onehot.csvに保存。

-merge_weather_denki.py

　　denki_clean.csvとweather_onehot.csvを統合し、merge_data.csvにまとめて保存。


2.グラフでの表示

-grath.py

　　電力消費量と、気温・湿度・日射量・風速のそれぞれとの分布図を作成。

-histogram.py

　　2022/1/1/0:00~2026/5/31/23:00の期間での、使用する各特徴量と電力消費量の関係を図に表示。

-senkeikaiki.py

　　線形回帰を行い、各特徴量の影響度などを踏まえて式を作成。

-hisenkeikaiki.py

　　非線形回帰により、各特徴量と電力消費量の非線形な関係をグラフに表示。

  
3.randomForestRegressorを用いた回帰分析

-learn.py

　　集めたデータの80％を学習データ、20%をテストデータとし電力消費量の予測精度を確認。この時点では、データの分割がランダムに行われている。特徴量は気温・湿度・日射量・風速のみ。


4.クラスタリング

5.クラスタリングを用いた回帰分析
