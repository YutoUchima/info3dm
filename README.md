# info3dm
知能情報総合演習　データマイニング班

テーマ「沖縄の天気と電力需要の関係性分析」

実験再現手順

1.データの収集・変換

　　-denki_data_download.py 
  　　電力消費量のデータをサイトから収集し、denki_power_dataというフォルダに保存。
    
   -denki_change.py
     集めた電気消費量データから1時間ごとの消費量を取り出す。

2.グラフでの表示

3.randomForestRegressorを用いた回帰分析

4.クラスタリング

5.クラスタリングを用いた回帰分析
