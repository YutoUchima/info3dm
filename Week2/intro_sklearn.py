from sklearn import datasets #予め用意されてるデータセット読み込み用ライブラリ
iris = datasets.load_iris() #datasets.load[tab]

#sklearnで用意されてるデータセットで、*おおよそ*共通している部分
#print(iris.DESCR) #.DESCR: 概要説明
data = iris.data #.data: データセット(教師データを除外した特徴ベクトルの集合)
target = iris.target #.target: 教師データ
print("###################")
print(iris.target_names) #教師データの種類


#学習器モジュールの読み込み
from sklearn import svm

#学習器の初期化。
#本来は、パラメータ要調整。どんなパラメータがあるかはドキュメント参照したり、help()を利用しよう。
clf = svm.SVC(gamma=0.001, C=100.)

#学習。
#教師あり学習の場合、データセット＋教師データを与えてfitメソッドを実行。
clf.fit(data[:-1], target[:-1])

#予測。
#予測させたいデータのみをpredictメソッドに与えて実行。
#注意: sklearn 0.18? 以降から、「サンプル数1個」に対する予測をさせる際の記載が変更。
#  旧記述: clf.predict(data[-1])
#  新記述: clf.predict(data[-1:])
#旧記述は2017/10/16時点でまだ実行できるが、DeprecationWarningになる。
result = clf.predict(data[-1:])
print("実際の答え={0}, 予測結果={1}".format(target[-1], result))

#スコア関数。
#「いい感じ」にスコア化してくれるが、どうスコア化してくれてるか確認するのを忘れずに。
result = clf.score(data, target)
print(result)



import pickle

#学習器の保存。任意のオブジェクトをファイルに保存可能。
with open('PredictiveModel.pickle', 'wb') as f:
    pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)

#保存したファイルからモデルを復元
with open('PredictiveModel.pickle', 'rb') as f:
    clf2 = pickle.load(f)

#同じ結果になるか確認
clf.predict(data) == clf2.predict(data)