"""生成キャッシュや一時ファイルを確認し、必要なら削除する補助スクリプト。"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]


def _inside_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT.resolve())
        return True
    except ValueError:
        return False


def _size_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += int(child.stat().st_size)
        except OSError:
            continue
    return total


def collect_generated_paths(*, include_runtime_cache: bool = True) -> list[Path]:
    # 削除候補を集めるだけ。実際に消すかどうかは cleanup_generated の apply で決める。
    paths: list[Path] = []
    paths.extend(PROJECT_ROOT.rglob("__pycache__"))
    paths.extend(PROJECT_ROOT.rglob(".ipynb_checkpoints"))
    paths.extend(PROJECT_ROOT.rglob("*.pyc"))
    paths.extend(PROJECT_ROOT.rglob("*.pyo"))
    if include_runtime_cache:
        runtime_cache = PROJECT_ROOT / "g_tactile_results" / "_runtime_cache"
        if runtime_cache.exists():
            paths.append(runtime_cache)

    unique = {}
    for path in paths:
        path = path.resolve()
        if _inside_project(path) and path.exists():
            unique[str(path)] = path
    return sorted(unique.values(), key=lambda item: (len(item.parts), str(item)))


def cleanup_generated(*, apply: bool, include_runtime_cache: bool = True) -> dict:
    # apply=False なら dry-run として候補とサイズだけ返し、ファイルは消さない。
    paths = collect_generated_paths(include_runtime_cache=include_runtime_cache)
    total_size = sum(_size_bytes(path) for path in paths)
    removed = []
    failed = []

    def on_rm_error(function, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
            function(path)
        except OSError as exc:
            failed.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    for path in paths:
        if not apply:
            continue
        if not _inside_project(path):
            raise RuntimeError(f"Refusing to remove outside project: {path}")
        try:
            if path.is_dir():
                shutil.rmtree(path, onerror=on_rm_error)
            elif path.exists():
                path.unlink()
            if not path.exists():
                removed.append(str(path))
            else:
                failed.append({"path": str(path), "error": "path still exists after removal"})
        except OSError as exc:
            failed.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    return {
        "project_root": str(PROJECT_ROOT),
        "apply": bool(apply),
        "count": len(paths),
        "size_mb": round(total_size / (1024 * 1024), 2),
        "paths": [str(path) for path in paths],
        "removed": removed,
        "failed": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove generated Python caches and local runtime cache from the LSM project."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove files. Without this flag, only show what would be removed.",
    )
    parser.add_argument(
        "--keep-runtime-cache",
        action="store_true",
        help="Keep g_tactile_results/_runtime_cache.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = cleanup_generated(
        apply=bool(args.apply),
        include_runtime_cache=not bool(args.keep_runtime_cache),
    )
    action = "removed" if result["apply"] else "would remove"
    print(f"[cleanup] {action} {result['count']} generated paths ({result['size_mb']} MB)")
    for path in result["paths"]:
        print(path)
    if result["failed"]:
        print(f"[cleanup] failed paths: {len(result['failed'])}")
        for item in result["failed"]:
            print(f"[failed] {item['path']} :: {item['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
