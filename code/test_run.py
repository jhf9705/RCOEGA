import argparse
import subprocess
import sys
import time
from pathlib import Path
from code import build_process_gif

# Batch list: edit this list directly to control which schedule JSON files run.
BATCH_SCHEDULE_FILES = [
    #"final_schedule_150_111_时间最短.json",
    "final_schedule_150_111_二期类似.json"
    #"final_schedule_150_211_时间最短_3260.json",
    #"final_schedule_150_211_二期类似_4494.json"
    # "final_schedule_200_111_4092.json",
    # "final_schedule_200_211_4094.json"
    # "final_schedule_250_111_4874.json",
    # "final_schedule_250_211_4878.json",
    # Edit this list to control which schedule files to run.
    # If left empty the script will try to auto-discover schedule JSON files
    # in the same folder (matching '*schedule*.json').
]

# Baseline layout JSON (由 docx_to_json.py 从 baseline_2.docx 生成)
# 修改此变量即可切换 baseline 布局文件
BASELINE_LAYOUT = "baseline_from_docx.json"

BASELINE_JSON = BASELINE_LAYOUT  # 兼容旧引用


def run(algo, schedule, extra_args=None):
    base = Path(__file__).resolve().parent
    cmd = [sys.executable, "code.py", "--algo", algo]
    if schedule:
        cmd.extend(["--schedule", schedule])
    if BASELINE_LAYOUT:
        cmd.extend(["--baseline", BASELINE_LAYOUT])
    if extra_args:
        cmd.extend(extra_args)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=base, check=True)


def run_baseline_only(schedule):
    """Stage_MODE=0: 仅评估 baseline，生成 baseline_result.json 和 baseline_layout.png。"""
    base = Path(__file__).resolve().parent
    cmd = [
        sys.executable, "code.py",
        "--schedule", schedule,
        "--baseline", BASELINE_JSON,
        "--stage", "0",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=base, check=True)


def redraw_pareto_iter_only(schedule):
    base = Path(__file__).resolve().parent
    cmd = [sys.executable, "code.py", "--redraw-pareto-iter"]
    if schedule:
        cmd.extend(["--schedule", schedule])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=base, check=True)


def _get_stage_mode():
    """读取 code.py 中的 Stage_MODE 值。"""
    import code as C
    return C.Stage_MODE


if __name__ == "__main__":
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["nsga2"], default="nsga2")
    parser.add_argument("--schedule", help="Schedule json passed to code.py (optional)")
    parser.add_argument("--run-all-json", action="store_true", help="Run all matched schedule json files")
    parser.add_argument("--quick", action="store_true", help="Pass --quick to code.py")
    parser.add_argument("--redraw-pareto-iter", action="store_true", help="Only redraw pareto_iter.gif")
    parser.add_argument("--baseline-only", action="store_true", help="Stage_MODE=0: only evaluate baseline and save result/png")

    args = parser.parse_args()

    # 若未显式指定 --baseline-only，自动根据 code.py 中的 Stage_MODE 判断
    if not args.baseline_only and _get_stage_mode() == 0:
        args.baseline_only = True

    base = Path(__file__).resolve().parent
    try:
        # One-click behavior: if no --schedule is provided, run all schedules in the batch list.
        schedules = [args.schedule] if args.schedule else []
        # Auto-discover when user requested run-all or BATCH_SCHEDULE_FILES is empty
        if args.run_all_json or not schedules:
            schedules = BATCH_SCHEDULE_FILES[:]
            if not schedules:
                # Try to discover schedule json files in this folder
                cand = sorted([p.name for p in base.glob("*schedule*.json")])
                if not cand:
                    # Fallback to final_schedule.json (common default)
                    cand = ["final_schedule.json"]
                schedules = cand

        total = len(schedules)
        for idx, schedule in enumerate(schedules, start=1):
            print(f"\n=== [{idx}/{total}] schedule: {schedule} ===")
            if args.redraw_pareto_iter:
                redraw_pareto_iter_only(schedule)
            elif args.baseline_only:
                run_baseline_only(schedule)
            else:
                extra = ["--quick"] if args.quick else []
                run(args.algo, schedule, extra_args=extra)
                if schedule:
                    schedule_path = Path(schedule)
                    if not schedule_path.is_absolute():
                        schedule_path = (base / schedule_path).resolve()
                    output_root = schedule_path.parent / schedule_path.stem
                    build_process_gif(output_root / "image", output_root / "process.gif", fps=2)
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"Program runtime: {elapsed:.2f} s")
