# 解析ノート

NumPy・pandas・Matplotlib・scikit-learn・SciPyを使えるJupyterカーネルで実行してください。各セルの設定で対象の保存先・rep・時間幅を指定します。図・表はノートブック内に表示し、別の結果フォルダへ保存しません。

- `calibration_liquid.ipynb`: `calibration_uniform`のLiquid E記録を使い、ビン幅別精度、混同行列、発火率、PCA、SVDパターン、ニューロン削減、時間窓別精度を調べます。評価は層別10-fold CVです。同じ100試行での旧Liquidとの対応比較と、200試行／324試行の比較セルもあります。
- `calibration_readout.ipynb`: `calibration_uniform/output_calibration`からLiquid EとReadoutを読み、共通の層別10-fold CVで発火率・PCA・分類を比較します。出力数は保存設定から取得します。
- `liquid.ipynb`: 保存済みLiquid単独結果をrepごとに集計します。ビン幅比較、混同行列の平均割合とrep間SD、発火率と全体平均、時間区間別分類、指定rep・時間窓のPCA、ランダムなニューロン削減を表示します。
- `off.ipynb`: offの保存済み出力記録について、ビン幅別精度、混同行列、素材別発火率と全体平均を表示します。
- `STDP.ipynb`: 同じrep集合の25 msビンの精度から最良の時定数を選び、offと同じ出力層解析・時間窓別分類・PCAを表示します。255 msは選択対象から除外します。選択に同じ評価結果を使うため、最良条件の解析は探索的な比較です。

可視化用PCAは対象試行全体で計算します。分類にPCAを使う場合は各foldの学習側だけで軸と平均を求め、評価側に適用します。標準化は行いません。

校正の保存記録は25 msビンの発火数です。そこから正確に集約でき、500 msを割り切れる25・50・100・125・250・500 msを比較します。通常のLiquid・off・STDPの1 ms記録では、さらに細かいビン幅を指定できます。

現在の実行設定は可塑性学習100試行・テスト200試行／素材です。過去の100試行の記録やノートブック内の計算済み結果が、自動で200試行の結果に変わるわけではありません。混同行列・ビン幅比較のSDは各セルに示したrep間またはfold間のばらつきです。

相互相関の解析入口は次です。

```powershell
python f_run/run_calibration.py correlation --self-test
python f_run/run_calibration.py correlation
```

正lagはLiquid先行です。shift predictor補正は共有する素材応答の成分を比較するためのもので、個別ニューロンの因果性や最適STDP時定数を直接示すものではありません。
