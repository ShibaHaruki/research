"""リキッド層の発火・膜電位を記録するモニタを作成する
ここでは記録対象の選択、記録モニタの作成、E/Iの記録の統合を行う。
作成時は全モニタを無効にし、呼び出し元が必要な記録だけを有効にする。
"""

from __future__ import annotations
import numpy as np
from brian2 import SpikeMonitor, StateMonitor, ms

# 膜電位を記録するニューロンの番号を、重複なしの昇順リストで返す
def select_record_indices(group, n_record: int, rng: np.random.Generator | None = None) -> list[int]:
    
    n_neurons = len(group)
    # 記録数を0〜層全体のニューロン数に制限する。E/Iから合わせて選ぶ。
    n_record = min(max(int(n_record), 0), n_neurons)
    if n_record == 0:
        return []
    if rng is None or n_record == n_neurons:
        # 乱数を使わない場合、または全員を記録する場合は先頭から選ぶ。
        return list(range(n_record))
    # replace=Falseで重複を防ぎ、記録と番号を対応させやすいように並べる。
    return sorted(int(idx) for idx in rng.choice(n_neurons, size=n_record, replace=False))


# 別々のE/I発火モニタを、一つの層全体の発火記録として扱う
class _CombinedSpikeMonitor:
    def __init__(self, exc_monitor, inh_monitor, n_exc: int):
        self.exc = exc_monitor
        self.inh = inh_monitor
        self.n_exc = int(n_exc)  # Iの番号を層全体の番号へ直すためのずらし幅。

    @property
    def active(self):
        """記録の有効・無効を取得する。E/Iは同時に切り替える。"""
        return self.exc.active

    @active.setter
    def active(self, value):
        # 発火記録の有効・無効を、E/Iの両方へ同時に反映する。
        self.exc.active = value
        self.inh.active = value

    def _order(self):
        """E/Iを連結した発火記録を、時刻順に並べるための添字を返す。"""
        # msにそろえて時刻を比較し、同時刻のイベントは元の順を保つ。
        t_exc = np.asarray(self.exc.t / ms)
        t_inh = np.asarray(self.inh.t / ms)
        return np.argsort(np.concatenate([t_exc, t_inh]), kind="stable")

    @property
    def t(self):
        """E/Iを合わせた発火時刻を、Brian2の時間単位付きで返す。"""
        t_exc = np.asarray(self.exc.t / ms)
        t_inh = np.asarray(self.inh.t / ms)
        values = np.concatenate([t_exc, t_inh])[self._order()]
        return values * ms

    @property
    def i(self):
        """発火時刻tと対応する、層全体のニューロン番号を返す。"""
        i_exc = np.asarray(self.exc.i, dtype=int)
        # 例：E80個なら、Iの0〜19番を層全体の80〜99番へ変換する。
        i_inh = self.n_exc + np.asarray(self.inh.i, dtype=int)
        return np.concatenate([i_exc, i_inh])[self._order()]

    def __len__(self):
        """ニューロン数ではなく、E/Iを合わせた発火イベント数を返す。"""
        return len(self.exc.t) + len(self.inh.t)

class _CombinedStateMonitor:
    """E/Iの膜電位記録を、Eを先・Iを後とするニューロン順でまとめる。"""

    def __init__(self, exc_monitor, inh_monitor):
        self.exc = exc_monitor
        self.inh = inh_monitor

    @property
    def active(self):
        return self.exc.active

    @active.setter
    def active(self, value):
        # 膜電位の記録もE/Iを同時に切り替える。
        self.exc.active = value
        self.inh.active = value

    @property
    def t(self):
        # E/Iは同じ間隔・タイミングで記録するため、E側の時刻を使う。
        return self.exc.t

    @property
    def v(self):
        """膜電位を「記録対象ニューロン数×記録時刻数」の配列で返す。"""
        # axis=0はニューロン方向。時間方向には連結しない。
        return np.concatenate(
            [np.asarray(self.exc.v), np.asarray(self.inh.v)], axis=0
        )

    def resize(self, n):
        """E/I両方の記録バッファの時間点数を変更する。0なら記録を空にする。"""
        self.exc.resize(n)
        self.inh.resize(n)

def make_liquid_monitors(G_liq, rng: np.random.Generator, run_cfg: dict) -> dict:
    """各リキッド層のモニタと記録対象番号を辞書で返す。

    G_liq：E/Iグループを持つリキッド層のリスト。
    rng：膜電位を記録するニューロンの選択に使う乱数生成器。
    run_cfg：記録するニューロン数と記録間隔（ms）などの設定。
    返り値の各リストは層順。objectsだけはNetwork.add()用の生モニタ一覧。
    """
    # 各層で詳しく見るニューロンの数と、その膜電位の記録間隔。
    n_voltage = int(run_cfg.get("DEBUG_VOLTAGE_NEURONS", 10))
    voltage_dt_ms = float(run_cfg.get("DEBUG_VOLTAGE_DT_MS", 1.0))

    M_liq_debug = []  # 全ニューロンの発火イベントをE/I統合で扱う。
    V_liq_indices = []  # 一部の膜電位記録に選んだ、層全体での番号。
    V_liq = []  # 選択したニューロンの膜電位モニタ。
    internal_state_dt_ms = float(run_cfg.get("INTERNAL_STATE_DT_MS", 1.0))
    V_liq_internal = []  # 全ニューロンの膜電位モニタ。
    monitor_objects = []  # Brian2のNetworkへ登録する、個々のモニタ。
    for layer_index, layer in enumerate(G_liq):
        # E/Iそれぞれの発火時刻と発火ニューロン番号を記録する。
        m_exc = SpikeMonitor(layer.exc, name=f"M_liq_debug_E_L{layer_index + 1}")
        m_inh = SpikeMonitor(layer.inh, name=f"M_liq_debug_I_L{layer_index + 1}")
        M_liq_debug.append(_CombinedSpikeMonitor(m_exc, m_inh, len(layer.exc)))
        monitor_objects.extend([m_exc, m_inh])

        # 層全体から選んだ番号を、E/I各グループ内の番号へ分ける。
        indices = np.asarray(select_record_indices(layer, n_voltage, rng), dtype=int)
        V_liq_indices.append(indices)
        n_exc = len(layer.exc)
        exc_indices = indices[indices < n_exc]
        inh_indices = indices[indices >= n_exc] - n_exc
        # 選択したニューロンは閾値判定の直前に記録する。
        # order=-1により、発火後のリセットより前の立ち上がりを確認できる。
        v_exc = StateMonitor(
            layer.exc, "v", record=exc_indices, dt=voltage_dt_ms * ms,
            when="thresholds", order=-1,
            name=f"V_liq_E_L{layer_index + 1}",
        )
        v_inh = StateMonitor(
            layer.inh, "v", record=inh_indices, dt=voltage_dt_ms * ms,
            when="thresholds", order=-1,
            name=f"V_liq_I_L{layer_index + 1}",
        )
        V_liq.append(_CombinedStateMonitor(v_exc, v_inh))
        monitor_objects.extend([v_exc, v_inh])

        # record=Trueで各グループの全員を記録する。
        # 全ニューロンの膜電位は、既定のstart段階で記録する。
        v_internal_exc = StateMonitor(
            layer.exc, "v", record=True, dt=internal_state_dt_ms * ms,
            name=f"V_liq_internal_E_L{layer_index + 1}",
        )
        v_internal_inh = StateMonitor(
            layer.inh, "v", record=True, dt=internal_state_dt_ms * ms,
            name=f"V_liq_internal_I_L{layer_index + 1}",
        )
        V_liq_internal.append(_CombinedStateMonitor(v_internal_exc, v_internal_inh))
        monitor_objects.extend([v_internal_exc, v_internal_inh])
    # 作成だけでは記録しない。呼び出し元が必要なモニタを有効にする。
    for monitor in M_liq_debug + V_liq + V_liq_internal:
        monitor.active = False

    return {
        "M_liq_debug": M_liq_debug,
        "V_liq": V_liq,
        "V_liq_indices": V_liq_indices,
        "V_liq_internal": V_liq_internal,
        # 統合用クラスではなく、実際のBrian2モニタをNetworkへ登録する。
        "objects": monitor_objects,
    }
