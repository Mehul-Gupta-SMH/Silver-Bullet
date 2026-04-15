"""Sequential training + eval chainer for SilverBullet.

Runs as a background daemon. Waits for a running training process to finish,
then chains: eval → train next mode → eval → … → update readme flag.

Usage:
    python -m backend.chain_train --wait-for rvg

Sequence when started with --wait-for rvg:
    wait train_rvg.log      → done
    eval RVG                → test_rvg_v55.log
    kill stray procs        (cleanup)
    train MVM               → train_mvm_v55.log
    eval MVM                → test_mvm_v55.log
    kill stray procs        (cleanup)
    eval CVG                → test_cvg_v55.log
    write chain_done.flag   (signals completion to any watcher)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
from datetime import datetime

# Force UTF-8 stdout so log() never crashes on arrow/emoji chars on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".sbvenv" / "Scripts" / "python.exe"
ENV    = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}

LOG_TAG = "[Chainer]"

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{LOG_TAG} {ts}  {msg}", flush=True)


# ── process management ──────────────────────────────────────────────────────

def kill_stray_trainers() -> None:
    """Kill any python.exe processes running backend.train (except ourselves)."""
    import subprocess as sp
    own_pid = os.getpid()
    try:
        out = sp.check_output(
            ["wmic", "process", "where", "name='python.exe'", "get",
             "processid,commandline"],
            text=True, errors="replace"
        )
        for line in out.splitlines():
            if "backend.train" in line or "backend.test" in line:
                parts = line.strip().split()
                pid = parts[-1] if parts else None
                if pid and pid.isdigit() and int(pid) != own_pid:
                    sp.call(["taskkill", "/PID", pid, "/F"],
                            stdout=sp.DEVNULL, stderr=sp.DEVNULL)
                    log(f"killed stray PID {pid}")
    except Exception as exc:
        log(f"warning: could not enumerate stray processes ({exc})")


# ── log polling ─────────────────────────────────────────────────────────────

def wait_for_log(log_path: Path, success: str = "Training complete.",
                 error_tokens: tuple[str, ...] = ("Traceback", "OOM", "alloc_cpu"),
                 poll_secs: int = 30,
                 timeout_secs: int = 86400) -> str:
    """
    Poll log_path every poll_secs seconds.
    Only scans content written AFTER this function was called — avoids false
    positives from error text in previous runs appended to the same log.
    Returns 'done', 'error', or 'timeout'.
    """
    log(f"waiting for '{success}' in {log_path.name} …")
    deadline = time.time() + timeout_secs

    # Record baseline size so we only inspect new content
    baseline = 0
    if log_path.exists():
        try:
            baseline = log_path.stat().st_size
        except OSError:
            baseline = 0
    log(f"  baseline offset: {baseline} bytes (ignoring prior content)")

    last_size = baseline

    while time.time() < deadline:
        time.sleep(poll_secs)
        if not log_path.exists():
            continue
        try:
            size = log_path.stat().st_size
        except OSError:
            continue
        if size <= baseline:
            continue  # no new content yet

        try:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                fh.seek(baseline)
                new_text = fh.read()
        except OSError:
            continue

        if success in new_text:
            log(f"{log_path.name}: SUCCESS — '{success}' found")
            return "done"

        # Only flag errors in genuinely new content AND only when the tail of
        # the file shows the error (guards against old crash text from prior
        # runs being scanned when the process restarts and appends to the log).
        tail = new_text[-3000:]
        if any(tok in tail for tok in error_tokens):
            # Require the error to appear in the final 500 bytes — if training
            # is actively progressing the tail will be tqdm lines, not errors.
            recent = new_text[-500:]
            if any(tok in recent for tok in error_tokens):
                log(f"{log_path.name}: ERROR — crash detected in recent tail")
                log(f"  snippet: {recent.strip()[-200:]}")
                return "error"

        # Log progress if file grew
        if size != last_size:
            last_size = size
            for line in reversed(new_text.splitlines()):
                line = line.strip()
                if line and not line.startswith("it/s") and not line.startswith("[A"):
                    safe = line[:120].encode("ascii", errors="replace").decode("ascii")
                    log(f"  tail: {safe}")
                    break

    log(f"{log_path.name}: TIMEOUT after {timeout_secs}s")
    return "timeout"


def wait_for_process(log_path: Path, success: str = "done\n",
                     poll_secs: int = 30, timeout_secs: int = 3600) -> str:
    """Same as wait_for_log but for test/eval runs (shorter timeout)."""
    return wait_for_log(log_path, success=success,
                        error_tokens=("Traceback", "Error"),
                        poll_secs=poll_secs, timeout_secs=timeout_secs)


# ── step runner ─────────────────────────────────────────────────────────────

def run_step(cmd: list[str], log_path: Path,
             wait_token: str = "done\n",
             timeout: int = 7200) -> bool:
    """
    Spawn subprocess writing to log_path, wait for completion.
    Returns True on success.
    """
    log(f"starting: {' '.join(str(c) for c in cmd)}")
    log(f"  → log: {log_path.name}")
    pid_file = log_path.with_suffix(".pid")
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [str(PYTHON)] + cmd[1:],
            stdout=fh, stderr=fh,
            cwd=str(ROOT), env=ENV,
        )
    pid_file.write_text(str(proc.pid), encoding="utf-8")
    log(f"  PID {proc.pid} → {pid_file.name}")
    result = wait_for_process(log_path, success=wait_token, timeout_secs=timeout)
    if proc.poll() is None:
        proc.wait(timeout=60)
    rc = proc.returncode
    pid_file.unlink(missing_ok=True)
    if result == "done" or rc == 0:
        log(f"  ✓ step complete (rc={rc})")
        return True
    log(f"  ✗ step failed (rc={rc}, result={result})")
    return False


# ── eval summary helper ─────────────────────────────────────────────────────

def extract_auc(log_path: Path) -> str:
    """Pull ROC-AUC from test log."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if "ROC-AUC" in line or "roc_auc" in line.lower():
                return line.strip()
    except Exception:
        pass
    return "AUC unknown"


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for", choices=["rvg", "none"], default="rvg",
                        help="Which in-flight training to wait for before chaining")
    args = parser.parse_args()

    log("=== SilverBullet sequential training chainer started ===")
    log(f"ROOT={ROOT}  PYTHON={PYTHON}")

    results: dict[str, str] = {}

    # ── Step 0: wait for in-flight run ──────────────────────────────────────
    if args.wait_for == "rvg":
        log("Step 0: waiting for in-flight RVG training …")
        rvg_log = ROOT / "train_rvg.log"
        outcome = wait_for_log(rvg_log, success="Training complete.",
                               timeout_secs=43200)  # 12h max
        if outcome != "done":
            log(f"RVG training did not complete cleanly ({outcome}). Aborting chain.")
            sys.exit(1)
        results["rvg_train"] = "done"
        log("Step 0 complete.")

    # ── Step 1: eval RVG ────────────────────────────────────────────────────
    log("Step 1: evaluating RVG …")
    kill_stray_trainers()
    ok = run_step(
        [str(PYTHON), "-m", "backend.test", "--mode", "reference-vs-generated"],
        ROOT / "test_rvg_v55.log",
        wait_token="Test complete",
        timeout=3600,
    )
    results["rvg_eval"] = extract_auc(ROOT / "test_rvg_v55.log") if ok else "FAILED"
    if not ok:
        log("RVG eval failed — continuing to MVM anyway")

    # ── Step 2: train MVM ───────────────────────────────────────────────────
    log("Step 2: training MVM …")
    kill_stray_trainers()
    mvm_log = ROOT / "train_mvm_v55.log"
    with open(mvm_log, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [str(PYTHON), "-m", "backend.train", "--mode", "model-vs-model"],
            stdout=fh, stderr=fh, cwd=str(ROOT), env=ENV,
        )
    outcome = wait_for_log(mvm_log, success="Training complete.", timeout_secs=43200)
    if proc.poll() is None:
        proc.wait(timeout=120)
    results["mvm_train"] = "done" if outcome == "done" else f"FAILED ({outcome})"
    log(f"MVM train: {results['mvm_train']}")

    # ── Step 3: eval MVM ────────────────────────────────────────────────────
    log("Step 3: evaluating MVM …")
    kill_stray_trainers()
    ok = run_step(
        [str(PYTHON), "-m", "backend.test", "--mode", "model-vs-model"],
        ROOT / "test_mvm_v55.log",
        wait_token="Test complete",
        timeout=3600,
    )
    results["mvm_eval"] = extract_auc(ROOT / "test_mvm_v55.log") if ok else "FAILED"

    # ── Step 4: eval CVG (already trained) ──────────────────────────────────
    log("Step 4: evaluating CVG …")
    kill_stray_trainers()
    ok = run_step(
        [str(PYTHON), "-m", "backend.test", "--mode", "context-vs-generated"],
        ROOT / "test_cvg_v55.log",
        wait_token="Test complete",
        timeout=3600,
    )
    results["cvg_eval"] = extract_auc(ROOT / "test_cvg_v55.log") if ok else "FAILED"

    # ── Done ─────────────────────────────────────────────────────────────────
    log("=== ALL STEPS COMPLETE ===")
    for k, v in results.items():
        log(f"  {k:20s}: {v}")

    # Write completion flag for watchers
    flag = ROOT / "chain_done.flag"
    flag.write_text(
        "\n".join(f"{k}: {v}" for k, v in results.items()) + "\n",
        encoding="utf-8",
    )
    log(f"Results written to {flag.name}")


if __name__ == "__main__":
    main()
