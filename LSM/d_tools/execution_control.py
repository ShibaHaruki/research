"""同じ実験の重複起動を防ぎ、明示的な中断時には計算用の子孫プロセスも終了する。"""

from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path
import subprocess


@contextmanager
def execution_lock(path):
    """PIDファイルの有無ではなくOSのロックで判定し、異常終了後も再起動可能にする。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                f"This experiment is already running. Wait for it to finish; do not start a duplicate: {path}"
            ) from error
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def exclusive_execution(path_for_arguments):
    """実行関数の全体をロックし、同一条件への並列書き込みを防ぐ。"""

    def decorate(function):
        """元の引数と関数名を保ったままロックを適用する。"""

        @wraps(function)
        def run(*args, **kwargs):
            """ロックを取得できた場合だけ実験を開始する。"""
            with execution_lock(path_for_arguments(*args, **kwargs)):
                return function(*args, **kwargs)

        return run

    return decorate


def assert_recording_available(path):
    """Windowsで既存記録が使用中か、切り詰め・削除をせずに事前確認する。"""
    path = Path(path)
    if os.name != "nt" or not path.is_file():
        return
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    # OPEN_EXISTING + GENERIC_READ + FileShareなし。書き込みは一切行わない。
    handle = create_file(str(path.resolve()), 0x80000000, 0, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        error = ctypes.get_last_error()
        if error in (32, 33):
            raise RuntimeError(
                f"A previous test is still using this recording. Let it finish before --resume: {path}"
            )
        raise ctypes.WinError(error)
    kernel.CloseHandle(handle)


def stop_process_tree(process):
    """中断された実験のPopenだけを対象に、Windowsのvenvランチャーの子も終了する。"""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
