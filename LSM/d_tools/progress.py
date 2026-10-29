"""学習則に依存しない進捗表示と、ログへ転送された進捗のターミナル表示。"""

import itertools
import json
import os
import subprocess
from pathlib import Path

from tqdm import tqdm

# プロセス内で各進捗区間を区別する。実験の乱数は消費しない。
_SEQUENCE = itertools.count()
_PREFIX = "[LSM_PROGRESS] "


def progress_items(iterable, *, total, description, rep=None, unit="trial", label=None):
    """項目の処理完了後に進捗を増やす。共有Liquidの子では表示情報をログへ送る。"""
    rep = int(rep if rep is not None else os.environ.get("LSM_PROGRESS_REP", 1))
    token = next(_SEQUENCE)
    completed = 0
    event_mode = os.environ.get("LSM_PROGRESS_EVENTS") == "1"
    bar = (
        None
        if event_mode
        else tqdm(
            total=total,
            desc=f"rep{rep} | {description}",
            unit=unit,
            dynamic_ncols=True,
            position=max(rep - 1, 0),
            leave=False,
        )
    )

    def report(detail="", status="running"):
        """表示だけを更新し、試行データ・提示順・ネットワーク状態には触れない。"""
        if event_mode:
            print(
                _PREFIX
                + json.dumps(
                    {
                        "token": token,
                        "rep": rep,
                        "description": description,
                        "completed": completed,
                        "total": total,
                        "unit": unit,
                        "detail": detail,
                        "status": status,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        elif bar is not None:
            bar.set_postfix_str(detail, refresh=False)
            bar.update(completed - bar.n)

    try:
        report()
        for item in iterable:
            detail = str(label(item)) if label is not None else ""
            report(detail)
            if bar is not None:
                # 計算終了を待たず、これから提示する素材・試行番号を表示する。
                bar.refresh()
            yield item
            completed += 1
            report(detail)
        report(status="complete")
    except BaseException:
        report(status="interrupted")
        raise
    finally:
        if bar is not None:
            bar.close()


def wait_with_progress(process, log_path, *, start_offset=0, position=0, rep=1):
    """子のログを読みながら待機し、進捗イベントを親のtqdmへ表示する。"""
    token = None
    last_detail = None
    with Path(log_path).open("r", encoding="utf-8", errors="replace") as stream, tqdm(
        total=None,
        desc=f"rep{rep} | Preparing",
        position=position,
        dynamic_ncols=True,
        unit="trial",
        leave=False,
    ) as bar:
        stream.seek(start_offset)
        pending = ""
        while True:
            chunk = stream.read()
            pending += chunk
            lines = pending.split("\n")
            pending = lines.pop()
            for line in lines:
                if not line.startswith(_PREFIX):
                    continue
                try:
                    event = json.loads(line[len(_PREFIX) :])
                    if event["token"] != token:
                        token = event["token"]
                        bar.reset(total=event["total"])
                        bar.unit = event["unit"]
                        bar.set_description_str(f"rep{rep} | {event['description']}", refresh=False)
                        last_detail = None
                    bar.set_postfix_str(event["detail"], refresh=False)
                    bar.update(event["completed"] - bar.n)
                    if event["detail"] != last_detail:
                        bar.refresh()
                        last_detail = event["detail"]
                except (ValueError, KeyError, TypeError):
                    # 通常のエラーログはそのまま残し、表示の破損で実験を中断しない。
                    continue
            if process.poll() is not None:
                # 終了後にも読み取りを繰り返し、末尾まで処理してから戻る。
                if not chunk:
                    return process.returncode
            else:
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass


def initialize_progress_lock(lock):
    """rep並列処理でターミナルへの書き込みを同じロックで保護する。"""
    tqdm.set_lock(lock)
