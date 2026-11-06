"""出力群を一括計算し、保存・評価では従来の層別インターフェースを維持する。"""

from collections import defaultdict
import re

import numpy as np
from brian2 import SpikeMonitor, StateMonitor, Synapses, ms


class _SynapseArray:
    """一括シナプス配列の連続区間を、従来の結合番号で読み書きする。"""

    def __init__(self, view, name):
        self.view, self.name = view, name

    def __getitem__(self, item):
        view = self.view
        values = getattr(view.parent, self.name)[view.start:view.stop]
        if self.name == "j":
            values = values - view.post_offset
        return values[item]

    def __setitem__(self, item, values):
        if self.name in ("i", "j"):
            raise TypeError("Connectivity is read-only")
        # 部分更新でもコピーだけを書き換えず、一括計算側の実配列へ反映する。
        block = self[:].copy()
        block[item] = values
        getattr(self.view.parent, self.name)[self.view.start:self.view.stop] = block

    def __array__(self, dtype=None, copy=None):
        values = np.asarray(self[:], dtype=dtype)
        return values.copy() if copy else values


class PackedSynapseView:
    """1候補の実結合。名前・i/j・w・接続先は従来の保存形式と同じにする。"""

    def __init__(self, parent, plan, start, stop, post_offset):
        self.parent = parent
        self.name = plan["name"]
        self.source, self.target = plan["source"], plan["target"]
        self.start, self.stop, self.post_offset = start, stop, post_offset
        self.variables, self.clock = parent.variables, parent.clock

    def __len__(self):
        return self.stop - self.start

    def __getattr__(self, name):
        if name in self.parent.variables:
            return _SynapseArray(self, name)
        raise AttributeError(name)

    def __setattr__(self, name, values):
        if name == "w" and "parent" in self.__dict__:
            self.parent.w[self.start:self.stop] = values
        else:
            object.__setattr__(self, name, values)


class _SpikeView:
    """共通発火モニタのうち、指定出力群だけを相対ニューロン番号で返す。"""

    def __init__(self, monitor, start, stop):
        self.monitor, self.start, self.stop = monitor, start, stop

    @property
    def active(self):
        return self.monitor.active

    @active.setter
    def active(self, value):
        self.monitor.active = value

    def _mask(self):
        ids = np.asarray(self.monitor.i)
        return (ids >= self.start) & (ids < self.stop)

    @property
    def t(self):
        return self.monitor.t[self._mask()]

    @property
    def i(self):
        return np.asarray(self.monitor.i)[self._mask()] - self.start

    @property
    def count(self):
        return self.monitor.count[self.start:self.stop]


class _VoltageView:
    """共通膜電位モニタの記録行だけを候補別に取り出す。"""

    def __init__(self, monitor, start, stop):
        self.monitor, self.start, self.stop = monitor, start, stop

    @property
    def active(self):
        return self.monitor.active

    @active.setter
    def active(self, value):
        self.monitor.active = value

    @property
    def t(self):
        return self.monitor.t

    @property
    def v(self):
        return self.monitor.v[self.start:self.stop]

    def resize(self, n):
        self.monitor.resize(n)


def make_packed_output_groups(cfg, rng):
    """1つのNeuronGroupを80個等の区画へ分け、候補間の結合は追加しない。"""
    from .build_network import _create_spiking_group, _init_group_state, _layer_sizes, _synapse_post_equations
    from .models.neuron_models import NEURON_MODELS

    sizes = _layer_sizes(cfg, "N_out")
    if any(size < 1 for size in sizes):
        raise ValueError("Each output group must have at least one neuron")
    model = NEURON_MODELS["LIF_E"]
    equations = model["eqs"] + "\nI_in = 0 : 1\n" + _synapse_post_equations(cfg)
    parent = _create_spiking_group(sum(sizes), equations, model, "G_out_packed")
    # 位置の乱数は元から生成していないため、全員の初期化で乱数順序は変わらない。
    _init_group_state(parent, cfg, rng, model, typ_value=1)
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    return parent, [parent[int(a):int(b)] for a, b in zip(offsets[:-1], offsets[1:])]


def make_packed_output_synapses(liquid, groups, parent, rng, cfg):
    """同じLiquid・学習則の結合をまとめ、学習係数を各実結合へ保持する。"""
    from .build_network import output_connection_plans
    from .models.learning_rule_models import LEARNING_RULES
    from .models.synapse_models import SYNAPSE_MODELS

    plans = list(output_connection_plans(liquid, groups, rng, cfg))
    buckets = defaultdict(list)
    for index, plan in enumerate(plans):
        buckets[(plan["liquid_index"], plan["pre_offset"], plan["learning_rule"])].append((index, plan))
    views, metadata = [None] * len(plans), [None] * len(plans)
    parents = []
    syn_model = SYNAPSE_MODELS[cfg["synapse_model"]]["synapse"]
    for bucket_index, entries in enumerate(buckets.values()):
        rule = LEARNING_RULES[entries[0][1]["learning_rule"]]
        learnable = rule.get("learnable", False)
        equations = syn_model["eqs"] + (rule["eqs"] if learnable else "")
        # 時定数等はnamespaceの共通スカラーではなく、実結合ごとの定数にする。
        parameters = rule["ns_vars"] if learnable else []
        # Brian2では末尾_pre/_postが予約されているため、SRDP等の定数名だけ置換する。
        parameter_names = {
            key: f"packed_{key}_value" if key.endswith(("_pre", "_post")) else key
            for key in parameters
        }
        pre_code = syn_model["on_pre"] + (rule["on_pre"] if learnable else "")
        post_code = rule["on_post"] if learnable else ""
        for key, variable in parameter_names.items():
            equations = re.sub(r"\b" + re.escape(key) + r"\b", variable, equations)
            pre_code = re.sub(r"\b" + re.escape(key) + r"\b", variable, pre_code)
            post_code = re.sub(r"\b" + re.escape(key) + r"\b", variable, post_code)
        # 従来のスカラー時定数ではBrian2が逆数を先に計算する。
        # 結合ごとの定数でも同じ丸め順を保つため、逆数を初期化時に保持する。
        inverse_names = {key: f"packed_inverse_{key}_value"
                         for key in parameters if key.startswith("tau")}
        for key, inverse in inverse_names.items():
            equations = re.sub(r"/\s*" + re.escape(parameter_names[key]) + r"\b",
                               "* " + inverse, equations)
        equations += "\n" + "\n".join(
            f"{parameter_names[key]} : {'second' if key.startswith('tau') else '1'} (constant)"
            for key in parameters
        )
        equations += "\n" + "\n".join(f"{name} : Hz (constant)" for name in inverse_names.values())
        kwargs = dict(
            model=equations, method="euler", name=f"S_out_packed_{bucket_index}",
            namespace=syn_model.get("namespace", {}),
            on_pre=pre_code,
        )
        if learnable and rule["on_post"]:
            kwargs["on_post"] = post_code
        syn = Synapses(entries[0][1]["source"], parent, **kwargs)
        pres = np.concatenate([plan["i"] for _, plan in entries])
        posts = np.concatenate([plan["j"] + plan["target"].start for _, plan in entries])
        syn.connect(i=pres, j=posts)
        syn.w = np.concatenate([plan["weights"] for _, plan in entries])
        syn.delay = 0 * ms
        offset = 0
        for index, plan in entries:
            end = offset + len(plan["i"])
            for key in parameters:
                getattr(syn, parameter_names[key])[offset:end] = plan["namespace"][key]
                if key in inverse_names:
                    getattr(syn, inverse_names[key])[offset:end] = 1.0 / plan["namespace"][key]
            views[index] = PackedSynapseView(syn, plan, offset, end, plan["target"].start)
            metadata[index] = {key: plan[key] for key in
                               ("liquid_index", "output_index", "learning_rule", "pre_offset")}
            metadata[index]["synapses"] = views[index]
            offset = end
        parents.append(syn)
    return views, metadata, parents


def make_packed_output_monitors(groups, parent, rng, cfg):
    """候補別の記録対象を維持し、実際の発火・膜電位モニタは各1個にする。"""
    from .monitor import select_record_indices

    indices = [np.asarray(select_record_indices(g, cfg.get("DEBUG_VOLTAGE_NEURONS", 10), rng), int)
               for g in groups]
    recorded = np.concatenate([ids + group.start for ids, group in zip(indices, groups)])
    spikes = SpikeMonitor(parent, name="M_out_packed")
    voltages = StateMonitor(parent, "v", record=recorded, clock=parent.clock,
                            when="thresholds", order=-1, name="V_out_packed")
    spikes.active = voltages.active = False
    rows = np.concatenate(([0], np.cumsum([len(ids) for ids in indices])))
    return {
        "M_out": [_SpikeView(spikes, group.start, group.stop) for group in groups],
        "V_out": [_VoltageView(voltages, int(a), int(b)) for a, b in zip(rows[:-1], rows[1:])],
        "V_out_indices": indices, "objects": [spikes, voltages],
    }
