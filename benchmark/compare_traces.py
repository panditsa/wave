"""Compare ATT traces: our 3-phase kernel vs aiter reference."""

import json
from collections import Counter


def analyze_kernel(path, name):
    with open(path) as f:
        code = json.load(f)
    insts = code["code"]
    mfma_locs = [i for i, inst in enumerate(insts) if "v_mfma_scale" in inst[0]]
    barrier_locs = [i for i, inst in enumerate(insts) if "s_barrier" in inst[0] and "cbranch" not in inst[0]]
    waitcnt_locs = [(i, insts[i][0]) for i in range(len(insts)) if "s_waitcnt" in insts[i][0]]

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {name}")
    print(f"{sep}")
    print(f"Total instructions: {len(insts)}")
    print(f"Total MFMAs: {len(mfma_locs)}")
    print(f"s_barriers: {len(barrier_locs)}")

    mfma_start = min(mfma_locs)
    mfma_end = max(mfma_locs)

    # Count consecutive MFMA runs
    runs = []
    current_run = 0
    for i in range(mfma_start, mfma_end + 1):
        if "v_mfma_scale" in insts[i][0]:
            current_run += 1
        elif current_run > 0:
            runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)

    run_counts = Counter(runs)
    total_runs = len(runs)
    avg_run = sum(runs) / total_runs if total_runs > 0 else 0
    print(f"\nMFMA run lengths (consecutive MFMAs):")
    print(f"  Total runs: {total_runs}")
    print(f"  Average run: {avg_run:.2f}")
    print(f"  Distribution: {dict(sorted(run_counts.items()))}")
    print(f"  Max run: {max(runs)}")

    # Count barriers/waits in MFMA region
    region_barriers = [b for b in barrier_locs if mfma_start - 10 <= b <= mfma_end + 10]
    print(f"\nBarriers in MFMA region: {len(region_barriers)}")

    lgkmcnt5 = sum(1 for i, asm in waitcnt_locs if "lgkmcnt(5)" in asm and mfma_start <= i <= mfma_end)
    lgkmcnt0 = sum(1 for i, asm in waitcnt_locs if "lgkmcnt(0)" in asm and mfma_start <= i <= mfma_end)
    vmcnt_waits = sum(1 for i, asm in waitcnt_locs if "vmcnt" in asm and mfma_start <= i <= mfma_end)
    print(f"  lgkmcnt(5): {lgkmcnt5}")
    print(f"  lgkmcnt(0): {lgkmcnt0}")
    print(f"  vmcnt waits: {vmcnt_waits}")

    # Phase structure: count ops between barriers
    print(f"\nPhase structure (between barriers in MFMA region):")
    all_markers = sorted(region_barriers)
    sections = [mfma_start - 1] + all_markers + [mfma_end + 1]
    for idx in range(len(sections) - 1):
        s = sections[idx] + 1
        e = sections[idx + 1]
        mfma = g2s = ds_rd = vgpr = 0
        for j in range(s, e):
            asm = insts[j][0]
            if "v_mfma_scale" in asm:
                mfma += 1
            elif "buffer_load" in asm and "lds" in asm:
                g2s += 1
            elif "ds_read" in asm:
                ds_rd += 1
            elif "buffer_load" in asm and "lds" not in asm:
                vgpr += 1
        marker_name = insts[sections[idx]][0][:35] if sections[idx] >= 0 else "start"
        print(f"  [{sections[idx]}] {marker_name:35s} -> MFMA={mfma:3d} G2S={g2s:2d} ds_rd={ds_rd:2d} vgpr={vgpr:2d}")


# Find our kernel trace dynamically
import glob
our_code = glob.glob("/workspace/wave_ref/benchmark/trace_att/ui_output_*/code.json")
if not our_code:
    print("No trace found!")
    exit(1)

# Analyze both kernels
analyze_kernel(our_code[0], "Our 3-phase kernel")

analyze_kernel(
    "/workspace/wave_ref/aiter_3072x8192x184576/aiter_3072x8192x184576/ui_output_agent_10739_dispatch_4/code.json",
    "AITER reference kernel",
)
