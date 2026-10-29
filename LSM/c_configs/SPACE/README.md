# パラメータ候補（SPACE）

通常の実行値は `../FIXED/` です。出力層ごとのSTDP時定数5〜250 msは`FIXED/cfg_network_stdp.py`で定義し、`cfg_learning_rule_models.py`はその候補を参照します。冒頭のseed数・epochは`--sweep`で直接読み込みます。それ以外の候補定義は、置いただけでは通常実行に反映されません。

|ファイル|内容|
|---|---|
|`cfg_base.py`|構造・モデル候補と対応表|
|`cfg_network.py`|結合確率・gain・入力経路の候補|
|`cfg_learning_rule_models.py`|共有LiquidのSTDP sweep設定と、STDP・T-STDP・SRDP・BCPNNの一般的な候補|
|`cfg_neuron_models.py`|LIFパラメータ候補|
|`cfg_synapse_models.py`|シナプス時定数候補|

共有Liquidの`--sweep`では、構造・学習則・層別パラメータ・結合確率・gainを`FIXED/cfg_network_stdp.py`から取り出します。A・重み上下限など層別指定を省略した値は`FIXED/cfg_learning_rule_models.py`、学習・テスト試行数と素材は`a_experiment/experiment_sets.py`の共通値を使用します。

`SPACE_NETWORK`のキーはnetwork辞書へのドット区切りパス、`SPACE_NETWORK_TUPLEPATH`のキーは辞書・リストを順にたどるパスです。タプル内の `(sensor, filter_name)` はIN_ROUTE辞書の1つのキーです。

旧コードのIN_ROUTE_LAYERSを現在のIN_ROUTEへ変更し、I→出力とIへの直接入力は0に固定しています。出力ニューロンは全興奮性のため、r_inh_outの候補はありません。SRDPとT-STDPの抑圧係数は、現在の更新式に合わせて正の候補にしています。

リキッド層の確率・gain・膜時定数を候補から変更した場合、それは校正済み条件とは異なる実験です。`run_calibration.py`の固定条件の校正計算へそのまま渡さず、一般の出力層学習用ネットワークで扱ってください。
