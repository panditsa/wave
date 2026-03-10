#!/usr/bin/env python3
"""
Runs rocprofv3 ATT tracing across CUs until a valid trace is captured.
Uses a single output directory, wiping and reusing it between attempts.
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys


def has_valid_trace(trace_dir: str) -> bool:
    """Check if a trace directory has non-zero hitcount entries in its stats CSV."""
    stats_files = glob.glob(os.path.join(trace_dir, "stats_*.csv"))
    if not stats_files:
        return False
    for stats_file in stats_files:
        try:
            with open(stats_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row.get("Hitcount", 0)) > 0:
                        return True
        except (ValueError, KeyError, FileNotFoundError):
            continue
    return False


def clear_dir(path: str):
    """Remove all contents inside a directory, but keep the directory itself."""
    if os.path.isdir(path):
        shutil.rmtree(path)


def run_trace(
    cu: int,
    output_dir: str,
    app_cmd: list[str],
    kernel_regex: str = "gemm",
    buffer_size: str = "0x20000000",
    se_mask: str = "0xFFFFFFFF",
    perfcounter_ctrl: int = 10,
    perfcounters: str = "SQ_LDS_BANK_CONFLICT",
    decoder_dir: str | None = None,
) -> bool:
    """Run rocprofv3 ATT for a given CU. Returns True if trace has valid data."""
    cmd = [
        "rocprofv3",
        "--kernel-trace",
        "--att",
        "--att-target-cu", str(cu),
        "--att-buffer-size", buffer_size,
        "--att-shader-engine-mask", se_mask,
        "--kernel-include-regex", kernel_regex,
        "-d", output_dir,
        "--stats", "TRUE",
        "--output-format", "csv",
        "--att-perfcounter-ctrl", str(perfcounter_ctrl),
        "--att-perfcounters", perfcounters,
    ]
    if decoder_dir:
        cmd.extend(["--att-library-path", decoder_dir])
    cmd.append("--")
    cmd.extend(app_cmd)

    env = os.environ.copy()
    env["WAVE_CACHE_ON"] = "0"

    print(f"\n{'='*60}")
    print(f"  Running ATT trace on CU {cu}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"  rocprofv3 exited with code {result.returncode}")
        return False

    valid = has_valid_trace(output_dir)
    if valid:
        print(f"\n  >>> CU {cu}: Valid trace captured!")
    else:
        print(f"\n  >>> CU {cu}: No trace data (waves didn't land on this CU)")
    return valid


def main():
    parser = argparse.ArgumentParser(
        description="Run rocprofv3 ATT tracing across CUs until a valid trace is found."
    )
    parser.add_argument(
        "--cu-range", type=str, default="0-10",
        help="CU range to sweep, e.g. '0-10' or '0-15' (default: 0-10)",
    )
    parser.add_argument(
        "-d", "--output-dir", type=str, default="trace_att",
        help="Trace output directory (default: trace_att)",
    )
    parser.add_argument(
        "--kernel-regex", type=str, default="gemm",
        help="Kernel name regex filter (default: gemm)",
    )
    parser.add_argument(
        "--buffer-size", type=str, default="0x20000000",
        help="ATT buffer size (default: 0x20000000)",
    )
    parser.add_argument(
        "--se-mask", type=str, default="0xFFFFFFFF",
        help="Shader engine mask (default: 0xFFFFFFFF)",
    )
    parser.add_argument(
        "--perfcounter-ctrl", type=int, default=10,
        help="Perfcounter control period (default: 10)",
    )
    parser.add_argument(
        "--perfcounters", type=str, default="SQ_LDS_BANK_CONFLICT",
        help="SQ perfcounters to collect (default: SQ_LDS_BANK_CONFLICT)",
    )
    parser.add_argument(
        "--decoder-dir", type=str, default=os.environ.get("DECODER_DIR"),
        help="Path to ATT decoder library (default: $DECODER_DIR)",
    )
    parser.add_argument(
        "app_cmd", nargs=argparse.REMAINDER,
        help="Application command to profile, e.g.: -- python 7.1_schedule.py --test test_name",
    )

    args = parser.parse_args()

    # Parse CU range
    try:
        lo, hi = args.cu_range.split("-")
        cu_range = range(int(lo), int(hi) + 1)
    except ValueError:
        print(f"Error: invalid --cu-range '{args.cu_range}', expected format like '0-10'")
        sys.exit(1)

    # Parse app command (strip leading --)
    app_cmd = args.app_cmd
    if app_cmd and app_cmd[0] == "--":
        app_cmd = app_cmd[1:]
    if not app_cmd:
        print("Error: no application command provided. Use: -- python your_script.py --test test_name")
        sys.exit(1)

    output_dir = args.output_dir

    print(f"Sweeping CUs {cu_range.start}-{cu_range.stop - 1}")
    print(f"Output directory: {output_dir}")
    print(f"App command: {' '.join(app_cmd)}")

    success_cu = None

    for cu in cu_range:
        # Wipe the directory before each attempt so rocprofv3 starts fresh
        clear_dir(output_dir)

        success = run_trace(
            cu=cu,
            output_dir=output_dir,
            app_cmd=app_cmd,
            kernel_regex=args.kernel_regex,
            buffer_size=args.buffer_size,
            se_mask=args.se_mask,
            perfcounter_ctrl=args.perfcounter_ctrl,
            perfcounters=args.perfcounters,
            decoder_dir=args.decoder_dir,
        )
        if success:
            success_cu = cu
            break

    # If no CU succeeded, clean up the empty directory
    if success_cu is None:
        clear_dir(output_dir)

    # Summary
    print(f"\n{'='*60}")
    if success_cu is not None:
        print(f"  SUCCESS: Valid trace captured on CU {success_cu}")
        print(f"  Trace directory: {output_dir}")
    else:
        print(f"  FAILED: No valid trace found across CUs {cu_range.start}-{cu_range.stop - 1}")
        print(f"  Try a wider CU range or check kernel launch dimensions.")
    print(f"{'='*60}")

    sys.exit(0 if success_cu is not None else 1)


if __name__ == "__main__":
    main()
