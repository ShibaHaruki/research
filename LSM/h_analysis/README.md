# 解析ノート

NumPy・pandas・Matplotlib・scikit-learn・SciPyを利用できるJupyterカーネルで、上から順に実行してください。入力・結果のファイルは削除しません。

- `calibration_liquid.ipynb`: Liquid Eの発火率、PCA、分類精度、ニューロン×時間の係数・SVDパターン、ニューロン削減、時間幅別の分類を表示します。
- `calibration_output.ipynb`: 校正後のLiquid EとReadoutを同じ分割で比較します。出力数は保存済みの校正設定から読み取り、PCAは最大40成分です。
- `off_output_neuron_reduction.ipynb`: gain 0.2706のoff rep1の保存済み出力記録を使い、80・75・70…5・1個の出力ニューロンで分類精度を比較します。ランダム削減順10通り、固定の層別10分割で評価し、平均・標準偏差と選択ごとの正答率を表示します。

表示用PCAは全試行、分類評価用PCAは各分割の学習側だけで計算します。どちらも標準化はしません。図・表はノート内に表示し、別の結果ファイルへ保存しません。

校正結果の集計は`output_summary.npz`から読みます。過去の`output_summary.json`も読み込み専用で対応します。`test_config_snapshot.json`などのJSONはパラメータ記録です。

相互相関の計算は次の入口を使います。

```powershell
python f_run/run_calibration.py correlation --self-test
python f_run/run_calibration.py correlation
```

既定の保存先は`g_tactile_results/liquid_readout_crosscorrelation_allpairs/`です。正lagはLiquid先行です。shift predictor補正は共有する素材応答の成分を比較するためのもので、単独ニューロンの因果性や最適STDP時定数を直接示すものではありません。
