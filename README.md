# kaggle_exercises
Kaggle の練習用レポジトリ

## Competitions

### テーブルコンペ
- 分類
    - [[Getting Started] Titanic: Machine Learning from Disaster](https://github.com/Yagami360/kaggle_exercises/tree/master/titanic)
    - [[Featured] Home Credit Default Risk](https://github.com/Yagami360/kaggle_exercises/tree/master/home-credit-default-risk)

- 回帰
    - [[Getting Started] House Prices: Advanced Regression Techniques](https://github.com/Yagami360/kaggle_exercises/tree/master/house-prices-advanced-regression-techniques)
    - [[Playground] Bike Sharing Demand](https://github.com/Yagami360/kaggle_exercises/tree/master/bike-sharing-demand)

- 未分類
    - [Featured] Restaurant Revenue Prediction
    - [Featured] Mercari Price Suggestion Challenge

### 画像コンペ
- 画像分類
    - [[Getting Started] Digit Recognizer](https://github.com/Yagami360/kaggle_exercises/tree/master/digit-recognizer)
    - [[Playground] Dogs vs. Cats Redux: Kernels Edition](https://github.com/Yagami360/kaggle_exercises/tree/master/dogs-vs-cats-redux-kernels-edition)

- セマンティクス・セグメンテーション
    - [Featured] TGS Salt Identification Challenge
    - [[Research] iMaterialist (Fashion) 2019 at FGVC6](https://github.com/Yagami360/kaggle_exercises/tree/master/imaterialist-fashion-2019-FGVC6)
    - [Research] iMaterialist (Fashion) 2020 at FGVC6

- 物体検出
    - [Featured] Google AI Open Images - Object Detection Track（データ容量が多すぎるので保留）
    - Facial Keypoints Detection

- 動画
    - [Featured] Deepfake Detection Challenge


### 自然言語コンペ
- xxx

## Tips
- EDA と前処理
    1. 目的変数の分布を確認。
        - 偏っているようなら log 変換などで正規分布に近づける
    1. 説明変数の分布を確認。
        - 学習用データとテスト用データでともに情報量ゼロの特徴量をクレンジング
            - 情報量ゼロ：1種類の値しか入ってない特徴量、相関係数が1の特徴量
        - 学習用データには含まれておらず、テスト用データにしか含まれていない特徴量をクレンジング
        - 外れ値がある特徴量は NAN 値で補完以外にも、外れ値存在フラグの特徴量を新たに追加する方法もあり
    1. GDBT での feature_importances を確認
        - XGBoost の `feature_importances_` などで確認可能
        - 相関の低い特徴量は、データセットから除外する
    1. 特徴量の相関確認
        - 各特徴量間や目的変数と説明変数の相関の相関をヒートマップやヒストグラムで確認。
        - 相関が小さいならばそれらの特徴量の独立性は高いので、除外性の低い特徴量となる。

- 特徴抽出
    - 前処理のカテゴリーデータのエンコードを one-hot で行うか否か
    - 時系列データを、年・日・時間・分・曜日・季節・休日などに分解する。
    - 同じ値を持つ特徴量を１つに Group 化するとき（IDなど）には、グループした特徴の count と それに関連した特徴量の min, max, mean, sum などを取る
        - 参考コンペ : home-credit-default-risk のデータセット
    - 外れ値がある特徴量は NAN 値で補完以外にも、外れ値存在フラグの特徴量を新たに追加する方法もあり
    - 目的変数と強い相関を持つ特徴量に対して、多項式特徴量 PolynomialFeatures（目的変数と強い相関を持つ特徴量間で乗算）を追加する方法もあり
        - 参考コンペ : home-credit-default-risk ()
        - 変数EXT_SOURCE_1^2とEXT_SOURCE_2^2の他に、EXT_SOURCE_1×EXT_SOURCE_2、EXT_SOURCE_1×EXT_SOURCE_2^2、EXT_SOURCE_1^2×EXT_SOURCE_2^2などの変数

- ハイパーパラメーターのチューニング
    - ハイパーパラメーターのチューニングは時間がかかるので、一般的に後半で行うほうがよい。
    - ハイパーパラメーターのチューニング時のスコア計算は、計算時間削減のため k > 1 での k-fold CV ではなく k = 1 での k-fold CV で行い、決定したベストモデルでの最終的なスコア計算は、k > 1 の k-fold CV で行う方法もある。
    - GDBT や木構造モデルの `n_estimators` は、NN モデルの epoches のように、学習段階で小さい値から設定値までの値で変化するので、ハイパーパラメーターのチューニングの対象外計算時間の許容できるできるだけ大きい値で固定して他のパラメータのチューニングを行う。

- 評価
    - StratifiedKFold は、テストデータに含まれる各クラスの割合を、学習データに含まれる各クラスの割合とほぼ同じにする CV であり、回帰タスクのように target 値が連続値となるようなケースでは無効。回帰タスクでは通常の k-fold を使用。

- その他
    - 回帰対象の確率分布が正規分布に従っていないとモデルの推論精度が低下するので、対数変換などで正規分布に従うようにする。
        - 参考コンペ : House Prices の SalePrice


## 参考文献
- [Kaggleで勝つデータ分析の技術](https://github.com/ghmagazine/kagglebook)
- [PythonではじめるKaggleスタートブック](https://github.com/upura/python-kaggle-start-book)
- [Kaggleに登録したら次にやること ～ これだけやれば十分闘える！Titanicの先へ行く入門 10 Kernel ～](https://qiita.com/upura/items/3c10ff6fed4e7c3d70f0)
- [【Kaggleのフォルダ構成や管理方法】タイタニック用のGitHubリポジトリを公開しました](https://upura.hatenablog.com/entry/2018/12/28/225234)
- [kaggle_memo](https://github.com/nejumi/kaggle_memo)
- [【随時更新】Kaggleテーブルデータコンペできっと役立つTipsまとめ](https://naotaka1128.hatenadiary.jp/entry/kaggle-compe-tips)