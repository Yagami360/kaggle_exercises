# kaggle_exercises
Kaggle の練習用レポジトリ

## Competitions

- テーブルコンペ
    - [Titanic: Machine Learning from Disaster](https://github.com/Yagami360/kaggle_exercises/tree/master/titanic)
    - [House Prices: Advanced Regression Techniques](https://github.com/Yagami360/kaggle_exercises/tree/master/house-prices-advanced-regression-techniques)
    - Restaurant Revenue Prediction
    - Home Credit Default Risk

## Tips


- ハイパーパラメーターのチューニング
    - ハイパーパラメーターのチューニング時のスコア計算は、計算時間削減のため k > 1 値での stratified k-fold CV ではなく k=1 での k-fold CV で行う。決定したベストモデルでの最終的なスコア計算は、stratified k-fold CV で行う方法もある。

- 特徴抽出
    - XGBoost の `feature_importances_` で確認可能

## 参考文献
- [Kaggleで勝つデータ分析の技術](https://github.com/ghmagazine/kagglebook)
- [PythonではじめるKaggleスタートブック](https://github.com/upura/python-kaggle-start-book)
- [Kaggleに登録したら次にやること ～ これだけやれば十分闘える！Titanicの先へ行く入門 10 Kernel ～](https://qiita.com/upura/items/3c10ff6fed4e7c3d70f0)
- [【Kaggleのフォルダ構成や管理方法】タイタニック用のGitHubリポジトリを公開しました](https://upura.hatenablog.com/entry/2018/12/28/225234)
- [kaggle_memo](https://github.com/nejumi/kaggle_memo)
