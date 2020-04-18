# kaggle_exercises
Kaggle の練習用レポジトリ

## Competitions

### テーブルコンペ
- 分類
    - [Titanic: Machine Learning from Disaster](https://github.com/Yagami360/kaggle_exercises/tree/master/titanic)
- 回帰
    - [House Prices: Advanced Regression Techniques](https://github.com/Yagami360/kaggle_exercises/tree/master/house-prices-advanced-regression-techniques)
- Restaurant Revenue Prediction
- Home Credit Default Risk

### 画像コンペ
- 画像分類
    - Digit Recognizer
    - [Dogs vs. Cats Redux: Kernels Edition](https://github.com/Yagami360/kaggle_exercises/tree/master/dogs-vs-cats-redux-kernels-edition)

- 物体検出
    - xxx
    - - Facial Keypoints Detection
- Deepfake Detection Challenge


- 自然言語コンペ
    - xxx

## Tips

- 特徴抽出
    - とりあえず、回帰 or 分類対象と強い相関のある特徴を把握することが重要
        - XGBoost の `feature_importances_` などで確認可能
    - 相関の低い特徴量は、データセットから除外する

- ハイパーパラメーターのチューニング
    - ハイパーパラメーターのチューニングは時間がかかるので、一般的に後半で行うほうがよい。
    - ハイパーパラメーターのチューニング時のスコア計算は、計算時間削減のため k > 1 での k-fold CV ではなく k = 1 での k-fold CV で行い、決定したベストモデルでの最終的なスコア計算は、k > 1 の k-fold CV で行う方法もある。

- 評価
    - StratifiedKFold は、テストデータに含まれる各クラスの割合を、学習データに含まれる各クラスの割合とほぼ同じにする CV であり、回帰タスクのように target 値が連続値となるようなケースでは無効。回帰タスクでは通常の k-fold を使用。

- その他
    - 回帰対象の確率分布が正規分布に従っていないとモデルの推論精度が低下するので、対数変換などで正規分布に従うようにする。（例：House Prices の SalePrice）


## 参考文献
- [Kaggleで勝つデータ分析の技術](https://github.com/ghmagazine/kagglebook)
- [PythonではじめるKaggleスタートブック](https://github.com/upura/python-kaggle-start-book)
- [Kaggleに登録したら次にやること ～ これだけやれば十分闘える！Titanicの先へ行く入門 10 Kernel ～](https://qiita.com/upura/items/3c10ff6fed4e7c3d70f0)
- [【Kaggleのフォルダ構成や管理方法】タイタニック用のGitHubリポジトリを公開しました](https://upura.hatenablog.com/entry/2018/12/28/225234)
- [kaggle_memo](https://github.com/nejumi/kaggle_memo)
