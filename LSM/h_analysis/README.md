# 解析ノート

NumPy・pandas・Matplotlib・scikit-learn・SciPyを利用できるJupyterカーネルで、上から順に実行してください。入力・結果のファイルは削除しません。

- `calibration_liquid.ipynb`: Liquid Eの発火率、PCA、分類精度、ニューロン×時間の係数・SVDパターン、ニューロン削減、時間幅別の分類を表示します。
- `calibration_output.ipynb`: 校正後のLiquid EとReadoutを同じ分割で比較します。出力数は保存済みの校正設定から読み取り、PCAは最大40成分です。

表示用PCAは全試行、分類評価用PCAは各分割の学習側だけで計算します。どちらも標準化はしません。図・表はノート内に表示し、別の結果ファイルへ保存しません。

校正結果の集計は`output_summary.npz`から読みます。過去の`output_summary.json`も読み込み専用で対応します。`test_config_snapshot.json`などのJSONはパラメータ記録です。

相互相関の計算は次の入口を使います。

```powershell
python f_run/run_calibration.py correlation --self-test
python f_run/run_calibration.py correlation
```

既定の保存先は`g_tactile_results/liquid_readout_crosscorrelation_allpairs/`です。正lagはLiquid先行です。shift predictor補正は共有する素材応答の成分を比較するためのもので、単独ニューロンの因果性や最適STDP時定数を直接示すものではありません。

- `stdp_tau_accuracy.ipynb`：STDPの5〜250 ms・rep1〜5の保存済み分類Excelから、全素材・素材別の正答率を時定数に対して描画します。平均±rep間SD、素材別の重ね描き、8素材の個別図と平均正答率表を表示します。結果はノートブック内だけに表示します。

`stdp_tau_accuracy.ipynb`の末尾では、素材別の最良時定数の出力層から各10ニューロン（計80）を選び、各rep10回のランダム選択・同じ10分割CVで8素材分類を行います。時定数選択に同じ評価データを使う探索的な比較で、保存済み応答だけを組み合わせます。

STDP・230 msのニューロン削減セルでは、80→75→…→5個を各rep10通りの入れ子のランダム選択で評価し、平均正答率・rep間SD・削減曲線を表示します。全80個の再評価は既存の分類Excelとの一致を検証します。

`off_output_neuron_reduction.ipynb`は現在の試行初期化off（rep1〜5）を使い、80→75→…→5個を各rep10通りの入れ子の削減順で評価します。STDP・230 ms側と乱数・CV条件を揃え、全80個の正答率を現在のExcelと照合します。
