"""matplotlib を安全に読み込み、GUI 不要の保存処理でも使えるようにする補助。"""

from __future__ import annotations


def try_import_pyplot(*, force_agg: bool = True):
    """Return matplotlib.pyplot when available, otherwise None.

    Most analysis scripts only need plotting as an optional save step. Keeping
    this helper in one place avoids repeating backend setup and keeps scripts
    usable in environments without matplotlib.
    """

    try:
        if force_agg:
            import matplotlib

            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except (ImportError, ModuleNotFoundError):
        return None
    return plt
