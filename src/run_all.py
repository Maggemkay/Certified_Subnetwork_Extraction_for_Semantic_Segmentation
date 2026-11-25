"""run_all · Runs the pipeline stages one after another in a single job, with config.py frozen at the start.

STAGES lists the stage scripts in run order; comment a line out to skip that stage.
  - Each stage runs as its own Python process, exactly as the job script would run it (same log tag),
    so all of its GPU and CPU memory is freed before the next stage starts.
  - config.py is read ONCE, when this script starts, and copied to a temporary file. Every stage loads
    that copy as its `config` module, so edits to config.py during the job do not reach later stages.
    The copy keeps __file__ = src/config.py, so PROJECT_ROOT and every path built from it are unchanged.
    Only config.py is frozen: the stage scripts themselves are read when each stage starts.
  - The job stops at the first stage that fails (exit code != 0) and names it in the log. A stop signal
    (SIGTERM, e.g. a SLURM time limit or scancel; Ctrl+C) is logged with the stage that was running.
  - All stage output goes to this script's stdout/stderr, i.e. into the same log file.

A stage run by hand (or through LOCAL_JOB.sh) still reads the live config.py.
"""

import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from socket import gethostname

# ---- Stages, in run order (comment out to skip) ------------------------------------------
STAGES = [
    # "setup_check_and_baseline.py",       # data/model checks, not part of the pipeline
    "s01_discovery_samples_dataset.py",
    "s02_score_cache.py",
    "s03_sample_circuits.py",
    "s04_certify.py",
    "s04b_logit_statistics.py",          # needed when LOGIT_STANDARDIZATION_VARIANTS (s05, s06) or
                                         # PREDICT_LOGIT_STANDARDIZATION_VARIANTS (s07) is non-empty, or with
                                         # PREDICT_CONFIDENCE_MAPS (s07)
    "s05_evaluate.py",
    # "s05b_evaluate_merged_circuits.py",  # optional: merged class-pair circuits; long: up to ≈ 19× s05's
                                         # passes per eval set. s06 draws its figure when its output exists
    "s06_plots.py",
    "s07_predictions.py",                # prediction examples (figures) of the circuits
    # s05c and s06c come last: they need COCO, ADE20K and Cityscapes, so without those datasets the job stops at
    # s05c only after every other stage has run.
    "s05c_evaluate_label_remapped_datasets.py",    # long: accuracy of s04's circuits on COCO val2017, ADE20K and
                                                   # Cityscapes (GT remapped to VOC); needs s04 (+ s04b with
                                                   # LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS), not s05
    "s06c_plots_label_remapped_datasets.py",       # s05c's figures; its VOC val columns need s05's voc_val counts
                                                   # (the folder of the current MAX_EVAL_IMAGES)
]

SRC = Path(__file__).resolve().parent
LOG_TAG = sys.argv[1] if len(sys.argv) > 1 else "manual"

# Runs one stage in the child process: load the frozen copy as the module `config` (with __file__ of the
# real src/config.py), then run the stage file unchanged as __main__ with argv = [stage, log tag].
LAUNCHER = """
import os, runpy, sys, types
frozen, real_config, stage = sys.argv[1:4]
config = types.ModuleType("config")
config.__file__ = real_config
with open(frozen) as f:
    exec(compile(f.read(), frozen, "exec"), config.__dict__)
sys.modules["config"] = config
sys.path.insert(0, os.path.dirname(real_config))  # src/, for `from common import ...`
sys.argv = [stage] + sys.argv[4:]
runpy.run_path(stage, run_name="__main__")
"""

LINE, HASHES = "=" * 100, "#" * 100


class Stopped(Exception):
    """Raised by the SIGTERM handler, so the stop can be logged like Ctrl+C."""


def on_sigterm(signum, frame):
    raise Stopped("SIGTERM (e.g. a SLURM time limit or scancel)")


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def duration(seconds):
    return str(timedelta(seconds=round(seconds)))


def explain(code):
    """What a stage's exit code means."""
    if code == 0:
        return "finished normally"
    if code == 1:
        return ("the stage raised an exception (traceback above; a CUDA out-of-memory error also ends up here) "
                "or stopped itself with a message (e.g. \"[missing input]\", printed above)")
    if code < 0:
        name = signal.Signals(-code).name
        reasons = {
            "SIGKILL": "often the out-of-memory killer (RAM), or a scheduler killing the job",
            "SIGTERM": "e.g. a SLURM time limit or scancel",
            "SIGINT": "Ctrl+C",
        }
        return f"killed by {name}" + (f": {reasons[name]}" if name in reasons else "")
    return f"exited with code {code}"


def say(text=""):
    print(text, flush=True)


def summary(status, start):
    """Per-stage outcome table and the overall result."""
    say(f"\n{LINE}\nrun_all summary | job {LOG_TAG} | finished {now()} | total {duration(time.time() - start)}\n{LINE}")
    width = max(len(s) for s in STAGES)
    say(f"  {'stage':<{width}}  {'outcome':<36s} {'time':>9s}")
    for stage in STAGES:
        outcome, seconds = status.get(stage, ("not run", None))
        say(f"  {stage:<{width}}  {outcome:<36s} {'' if seconds is None else duration(seconds):>9s}")
    failed = [s for s in STAGES if s in status and status[s][0] != "OK"]
    if failed:
        say(f"\nRESULT: STOPPED at {failed[0]}. The stages after it were NOT run; their output folders hold "
            f"either nothing or the results of an EARLIER job.")
    else:
        say(f"\nRESULT: all {len(STAGES)} stages finished OK.")
    say(LINE)


def main():
    start = time.time()
    signal.signal(signal.SIGTERM, on_sigterm)

    missing = [s for s in STAGES if not (SRC / s).is_file()]
    if missing:
        say(f"run_all: stage scripts not found in {SRC}: {missing}. Nothing was run.")
        sys.exit(1)

    config_text = (SRC / "config.py").read_text()
    with tempfile.TemporaryDirectory(prefix="run_all_") as tmp:
        frozen = Path(tmp) / "config.py"
        frozen.write_text(config_text)

        say(LINE)
        say(f"run_all | job {LOG_TAG} | {now()}")
        say(f"host: {gethostname()} | python: {sys.executable}")
        say("-" * 100)
        say("Stages, in this order. Each runs as its own process; the job stops at the first one that fails:")
        for n, stage in enumerate(STAGES, start=1):
            say(f"  {n}. {stage}")
        say(f"\nconfig.py is FROZEN now: copied to {frozen}.\n"
            f"Every stage below reads this copy, so edits to {SRC / 'config.py'} after {now()} do NOT affect "
            f"this job.\nIts full content follows; each stage also prints the values it uses.")
        say(f"{'-' * 40} begin frozen config.py {'-' * 36}")
        say(config_text.rstrip("\n"))
        say(f"{'-' * 40} end frozen config.py {'-' * 38}")
        say(LINE)

        status = {}
        for n, stage in enumerate(STAGES, start=1):
            label = f"run_all · stage {n}/{len(STAGES)}: {stage}"
            say(f"\n{HASHES}\n{label} | started {now()}\n{HASHES}")
            stage_start = time.time()
            process = subprocess.Popen([sys.executable, "-u", "-c", LAUNCHER,
                                        str(frozen), str(SRC / "config.py"), str(SRC / stage), LOG_TAG])
            try:
                code = process.wait()
            except (Stopped, KeyboardInterrupt) as stop:
                reason = stop.args[0] if isinstance(stop, Stopped) else "SIGINT (Ctrl+C)"
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                status[stage] = (f"STOPPED by {reason.split(' ')[0]}", time.time() - stage_start)
                say(f"\n{HASHES}\n{label} STOPPED after {duration(time.time() - stage_start)}: run_all received "
                    f"{reason}.\nThe stage process was terminated; its output above may be incomplete.\n{HASHES}")
                summary(status, start)
                sys.exit(1)

            seconds = time.time() - stage_start
            if code == 0:
                status[stage] = ("OK", seconds)
                say(f"\n{label} | finished OK in {duration(seconds)} (exit code 0)")
                continue
            status[stage] = (f"FAILED (exit code {code})", seconds)
            say(f"\n{HASHES}\n{label} FAILED after {duration(seconds)}\n"
                f"exit code {code}: {explain(code)}.\n{HASHES}")
            summary(status, start)
            sys.exit(1)

        summary(status, start)


if __name__ == "__main__":
    main()
