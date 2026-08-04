# info3dm
知能情報総合演習　データマイニング班　G4

テーマ「沖縄の天気と電力需要の関係性分析」

実験再現手順　(ファイルは実行に使う順に説明)


1.データの収集・変換　

-denki_data_download.py 

　　電力消費量のファイルをサイトから収集し、denki_power_dataというフォルダに保存。
    
-denki_change.py

　　集めた電気消費量ファイルから1時間ごとの消費量のみを取り出し、denki_clean.csvを作成。24行を指定しているのは、ファイルの中にある1時間ごとの電力消費量を取得するために0~23時の分の24行である。

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

　　2022/1/1/0:00\~2026/5/31/23:00の期間での、使用する各特徴量と電力消費量の関係を図に表示。

-senkeikaiki.py

　　線形回帰を行い、各特徴量の影響度などを踏まえて式を作成。

-hisenkeikaiki.py

　　非線形回帰により、各特徴量と電力消費量の非線形な関係をグラフに表示。

  
3.回帰分析

-learn.py

　　集めたデータの80％を学習データ、20%をテストデータとし電力消費量の予測精度を確認。この時点では、データの分割がランダムに行われている。特徴量は気温・湿度・日射量・風速のみ。randomForestRegressorを用いた。

-learn2.py

　　learn.pyでランダムに分割していたデータを2022/1/1/0:00\~2024/12/31/23:00を学習データ、2025/1/1/0:00\~2026/5/31/23:00をテストデータとし、時系列順に分割した。「平日or土日祝」、「1時間ごとの時間」、「月」も特徴量に追加。randomForestRegressorを用いた。

  
4.クラスタリング

-clustering_no_denki.py

　　影響度の高い主な特徴量「気温・湿度・日射量・風速」の情報を傾向分けして分析するため、クラスタリングを行う。

-clustering_yes_denki.py

　　clustering_no_denki.pyに加えて、電気消費量の情報も与え、クラスタリングを行った。目的変数を加えることによるクラスタ分けへの影響を確かめた。


5.クラスタリングを用いた回帰分析

-learn3.py

　　クラスタリングによって得られたクラスタをラベルとして特徴量に追加し、randomForestRegressorによる回帰分析を行い精度を確認。

-yosoku.py

　　各特徴量の値を手動で入力すると、未来の電力消費量を予測することができる。

　
