"""Python-side driver for the Blender stages: run estate/blender/<script>.py in blender.exe and return its JSON.

Each job runs in its own background Blender (``-b --factory-startup --python-exit-code 1``). Arguments travel in
a temp JSON file (no Windows command-line quoting), and the job writes its result to a second JSON file; the
result is also printed after ``_boot.MARKER`` so the log alone is enough. stdout/stderr go to a log file under
build/logs/blender/ (never a pipe, so a chatty Blender cannot dead-lock on a full buffer). Failures raise
BlenderError with the log tail. ``run_many`` runs jobs on a thread pool; each thread just waits on its process.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from estate import env

SCRIPTS = Path(__file__).resolve().parent
LOGS = env.BUILD / "logs" / "blender"
JOBS = env.BUILD / "blender_jobs"
MARKER = "@@ESTATE_RESULT@@"
DROP_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONNOUSERSITE", "VIRTUAL_ENV")
_ACTIVE: set = set()
_LOCK = threading.Lock()


class BlenderError(RuntimeError):
    def __init__(self, msg, result=None, log=None):
        super().__init__(msg)
        self.result = result or {}
        self.log = log


def _tag(script, args) -> str:
    stem = ""
    for k in ("blend", "ifc", "site", "out"):
        if args.get(k):
            stem = Path(str(args[k])).stem
            break
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{Path(script).stem}-{stem}" if stem else Path(script).stem)


def _claim_log(log_dir: Path, tag: str) -> Path:
    """A log path no other running job of this process uses (parallel jobs on one IFC share a tag)."""
    with _LOCK:
        n, name = 1, tag
        while name in _ACTIVE:
            n += 1
            name = f"{tag}-{n}"
        _ACTIVE.add(name)
    return log_dir / f"{name}.log"


def _release_log(path: Path) -> None:
    with _LOCK:
        _ACTIVE.discard(path.stem)


def blender_cmd(script: Path, args_file: Path, background=True, factory=True) -> list[str]:
    cmd = [str(env.BLENDER_EXE)]
    if background:
        cmd.append("-b")
    if factory:
        cmd.append("--factory-startup")
    cmd += ["--python-exit-code", "1", "--python", str(script), "--", "--json", str(args_file)]
    return cmd


def child_env() -> dict:
    e = {k: v for k, v in os.environ.items() if k not in DROP_ENV}
    e.setdefault("PYTHONUTF8", "1")
    return e


def tail(path, n=60) -> str:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def parse_marker(text: str) -> dict | None:
    """The JSON line after the last marker line in a log."""
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == MARKER and i + 1 < len(lines):
            try:
                return json.loads(lines[i + 1])
            except json.JSONDecodeError:
                return None
    return None


def run_blender(script_name: str, args: dict | None = None, timeout: float = 1800, log_dir: Path | None = None,
                background: bool = True, check: bool = True) -> dict:
    """Run estate/blender/<script_name> in blender.exe; returns the job's result dict (raises BlenderError)."""
    script = Path(script_name)
    if not script.is_absolute():
        script = SCRIPTS / script.name
    if script.suffix != ".py":
        script = script.with_suffix(".py")
    assert script.exists(), f"no Blender script {script}"
    args = dict(args or {})
    JOBS.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir) if log_dir else LOGS
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = _tag(script, args)
    fd, args_path = tempfile.mkstemp(prefix=f"{tag}-", suffix=".json", dir=JOBS)
    os.close(fd)
    args_file = Path(args_path)
    result_file = args_file.with_name(args_file.stem + ".result.json")
    args["result"] = str(result_file)
    args_file.write_text(json.dumps(args, indent=1, default=str), encoding="utf-8")
    log_path = _claim_log(log_dir, tag)
    t = time.time()
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            log.write(" ".join(blender_cmd(script, args_file, background)) + "\n")
            log.flush()
            proc = subprocess.Popen(blender_cmd(script, args_file, background), stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=child_env(), cwd=str(env.ROOT))
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise BlenderError(f"{script.name} timed out after {timeout:.0f} s\n{tail(log_path)}", log=log_path)
    finally:
        _release_log(log_path)
    result = None
    if result_file.exists():
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = None
    if result is None:
        result = parse_marker(Path(log_path).read_text(encoding="utf-8", errors="replace")) or {}
    result.setdefault("ok", code == 0)
    result["exit_code"] = code
    result["wall_seconds"] = round(time.time() - t, 2)
    result["log"] = str(log_path)
    result_file.unlink(missing_ok=True)
    if code == 0 and result.get("ok"):
        args_file.unlink(missing_ok=True)
    else:
        result["args_file"] = str(args_file)      # kept on failure, to rerun the job by hand
    if check and (code != 0 or not result.get("ok")):
        raise BlenderError(f"{script.name} failed (exit {code}): {result.get('error', '')}\n--- log tail "
                           f"({env.rel(log_path)}) ---\n{tail(log_path)}", result, log_path)
    return result


def run_many(jobs, max_workers: int = 4, timeout: float = 1800) -> list:
    """Run [(script_name, args), ...] in parallel; returns results in job order (a BlenderError in place of failures)."""
    def one(job):
        script, a = job
        try:
            return run_blender(script, a, timeout=timeout)
        except BlenderError as e:
            return e

    jobs = list(jobs)
    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(jobs)))) as ex:
        return list(ex.map(one, jobs))


def launch_gui(script_name: str, args: dict) -> subprocess.Popen:
    """Start an interactive Blender (user preferences, Bonsai enabled by the script) and return immediately."""
    script = SCRIPTS / Path(script_name).with_suffix(".py").name
    JOBS.mkdir(parents=True, exist_ok=True)
    fd, args_path = tempfile.mkstemp(prefix="gui-", suffix=".json", dir=JOBS)
    os.close(fd)
    Path(args_path).write_text(json.dumps(args, indent=1, default=str), encoding="utf-8")
    return subprocess.Popen(blender_cmd(script, Path(args_path), background=False, factory=False), env=child_env(),
                            cwd=str(env.ROOT))
