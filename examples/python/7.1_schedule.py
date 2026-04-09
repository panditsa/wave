"""
MXFP4 Scaled GEMM Scheduling for GFX950 (MI350)

Double-buffered MXFP4 GEMM with 4-wave and 8-wave configurations.
Uses get_tagged_mxfp4_gemm (templates) + get_mxfp4_dbuf_schedule (schedules).

Usage:
    python 7.1_schedule.py --test test_dbuf_4wave_mxfp_gemm
    python 7.1_schedule.py --test test_dbuf_8wave_mxfp_gemm
    python 7.1_schedule.py --test test_dbuf_8wave_mxfp_gemm --debug
    python 7.1_schedule.py --list_tests
"""

import os
import torch
import wave_lang.kernel.lang as tkl

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.templates import (
    get_tagged_mxfp4_gemm,
    get_tagged_mxfp4_gemm_preshuffle_a,
    get_tagged_mxfp4_gemm_preshuffle_b,
    get_tagged_mxfp4_gemm_preshuffle_scales,
    get_tagged_mxfp4_gemm_preshuffle_scales_and_B,
)
from wave_lang.kernel.wave.schedules import (
    get_mxfp4_dbuf_schedule,
    get_mxfp4_dbuf_pingpong_schedule,
    get_mxfp4_dbuf_mixed_pingpong_schedule,
    get_mxfp4_asymmetric_schedule,
    get_mxfp4_asymmetric_schedule_mirrored,
    get_mxfp4_asymmetric_schedule_mirrored_3phase_experimental,
    get_mxfp4_dbuf_mixed_pingpong_shuffle_schedule,
    get_mxfp4_dbuf_pingpong_schedule_Bshuffled,
    get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds,
)
from wave_lang.kernel.wave.utils.mxfp_utils import (
    generate_gemm_afp4wfp4_inputs,
    torchScaledGemmMXFP4,
    b_preshuffle,
    e8m0_shuffle,
)
from wave_lang.kernel.lang.global_symbols import (
    GLOBAL_ADDRESS_SPACE,
    SHARED_ADDRESS_SPACE,
)
from utils import parse_args, list_tests, run_test


def _run_mxfp_gemm(gemm, shape):
    """Run compiled GEMM kernel and verify against reference."""
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    x, w = x.cuda(), w.cuda()
    x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
    out = torch.zeros(x.shape[0], w.shape[1], dtype=torch.float32).cuda()

    gemm(x, x_scales, w.T.contiguous(), w_scales, out)
    torch.testing.assert_close(
        torch_out, out.cpu(), check_dtype=False, check_device=False
    )


def _run_mxfp_gemm_preshuffle(
    gemm, shape, all=False, only_scale=False, only_b=False,
    output_dtype=torch.float32,
):
    """Run compiled GEMM kernel with preshuffled operands, verify against reference.

    Shuffling is applied based on the flags:
      all        - shuffle a_scale (x_scales), b_scale (w_scales), and b (w_t)
      only_scale - shuffle a_scale (x_scales) and b_scale (w_scales) only
      only_b     - shuffle b_scale (w_scales) only
    """
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    w_t = w.T.contiguous()

    # Apply b (w_t) preshuffle only when all=True
    w_t_ps = b_preshuffle(w_t) if all else w_t

    # Apply a_scale shuffle when all=True or only_scale=True
    x_scales_ps = e8m0_shuffle(x_scales) if (all or only_scale) else x_scales

    # Apply b_scale shuffle when all=True, only_scale=True, or only_b=True
    w_scales_ps = e8m0_shuffle(w_scales) if (all or only_scale or only_b) else w_scales

    x, w_t_ps = x.cuda(), w_t_ps.cuda()
    x_scales_ps, w_scales_ps = x_scales_ps.cuda(), w_scales_ps.cuda()
    out = torch.zeros(x.shape[0], w_t_ps.shape[0], dtype=output_dtype).cuda()

    gemm(x, x_scales_ps, w_t_ps, w_scales_ps, out)

    torch.testing.assert_close(
        torch_out, out.cpu(), check_dtype=False, check_device=False
    )


def _get_8wave_shape_from_block(block):
    """Choose an 8-wave shape (4x2 or 2x4) from block M/N dims.

    If either block M or N is 32, force that corresponding wave dimension to 2.
    """
    m_blk, n_blk = block[0], block[1]
    if m_blk == 32 and n_blk == 32:
        raise ValueError(
            "Cannot satisfy both M and N=32 with an 8-wave shape constrained to (4, 2) or (2, 4)."
        )
    if m_blk == 32:
        return (2, 4)
    if n_blk == 32:
        return (4, 2)
    return (4, 2)


def test_dbuf_4wave_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Double-buffered MXFP4 GEMM, 4 waves, no stagger."""
    gemm, options = get_tagged_mxfp4_gemm(shape, block, wave_shape=(2, 2))
    schedule = get_mxfp4_dbuf_schedule(use_stagger=False)

    options.print_ir_after = "all" if is_debug else []
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave.mlir"
    options.print_mlir = True
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM double-buffer 4-wave test passed!")


def test_dbuf_8wave_pingpong_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256), dynamic=False
):
    """Double-buffered MXFP4 GEMM, 8 waves, ping-pong with stagger.
    A&B scales are preshuffled and read from global memory directly to VGPRs.
    A and B are read from global memory directly to LDS.

    Note: for dynamic mode, keep block MxN at or below 128x256 or 256x128
    to avoid exceeding shared-memory limits.
    """
    wave_shape = _get_8wave_shape_from_block(block)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
        shape, block, wave_shape=wave_shape
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True

    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]

    schedule = get_mxfp4_dbuf_pingpong_schedule(use_stagger=True, shape=shape)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, only_scale=True)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM double-buffer 8-wave ping pong with scale shuffling ({mode}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm_Bshuffle(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256), dynamic=False
):
    """Double-buffered MXFP4 GEMM, 8 waves, ping-pong with stagger.
    A&B scales are preshuffled and read from global memory directly to VGPRs.
    Same for B data. However, prefetching shuffled B directly to VGPR consumes too many VGPRs and causes spilling.
    A is read from global memory directly to LDS.
    """
    wave_shape = _get_8wave_shape_from_block(block)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales_and_B(
        shape, block, wave_shape=wave_shape
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True

    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]
    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled(use_stagger=True, shape=shape)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM double-buffer 8-wave ping pong with scale and B shuffling and B->VGPR ({mode}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm_Bshuffle_lds(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256), dynamic=False
):
    """Double-buffered MXFP4 GEMM, 8 waves, ping-pong with stagger.
    A&B scales are preshuffled and read from global memory directly to VGPRs.
    B data is preshuffled and loaded to LDS (shared memory), not directly to VGPRs.
    A data is read from global memory directly to LDS.
    """
    wave_shape = _get_8wave_shape_from_block(block)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales_and_B(
        shape,
        block,
        wave_shape=wave_shape,
        b_address_space=SHARED_ADDRESS_SPACE,
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = False
    options.linearize_shared_access = True

    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]
    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds(
        use_stagger=True, shape=shape
    )

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM double-buffer 8-wave ping pong with scales and B shuffling and B->LDS ({mode}) test passed!"
    )


def test_dbuf_8wave_mixed_pingpong_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Double-buffered MXFP4 GEMM, 8 waves, with stagger.

    A variant of the ping-pong schedule that hides the latency of the extra
    WorkgroupBarrier required for large shapes. With staggering, the two
    clusters of waves write to LDS at different times.
    When the bus becomes congested, memory operations loaded by the later cluster may not arrive
    in LDS before the other cluster attempts to read from it. In this case,
    we add a second workgroup barrier to fix the timing and prevent incorrect output results.

    This schedule overlaps that barrier with useful work by splitting LDS loads:
      - "Safe" loads: rows this wave wrote itself — readable immediately after
        memory_counter_wait, before the global WorkgroupBarrier.
      - "Dependent" loads: rows written by other waves — deferred until after
        the global WorkgroupBarrier.

    This lets the MFMAs on the safe operands start firing as soon as the
    barrier releases, effectively hiding the second barrier's latency behind
    the early loads and compute.
    """
    gemm, options = get_tagged_mxfp4_gemm(shape, block, wave_shape=(4, 2))
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    schedule = get_mxfp4_dbuf_mixed_pingpong_schedule(use_stagger=True)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM double-buffer 8-wave mixed ping pong test passed!")


def test_dbuf_8wave_mixed_pingpong_shuffle_mxfp_gemm(
    is_debug=False, shape=(16384, 16384, 16384), block=(256, 256, 256)
):
    """Like :func:`test_dbuf_8wave_mixed_pingpong_mxfp_gemm` but with A_scale & B_scale
    preshuffled and prefetched to VGPRs.

    Note: preshuffling B and loading it directly to VGPRs combined with prefetching
    consumes too many VGPRs and causes spilling.
    """

    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
        shape, block, wave_shape=(4, 2)
    )

    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    schedule = get_mxfp4_dbuf_mixed_pingpong_shuffle_schedule(use_stagger=True)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, only_scale=True)
    print("MXFP GEMM double-buffer 8-wave mixed ping pong with shuffling test passed!")


def test_dbuf_4wave_mxfp_asymmetric_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Asymmetric-prefetch MXFP4 GEMM: A through LDS (2x prefetch), B direct from global."""
    gemm, options = get_tagged_mxfp4_gemm(
        shape, block, wave_shape=(1, 4), b_address_space=GLOBAL_ADDRESS_SPACE
    )
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave_asymmetric.mlir"
    options.print_mlir = True
    options.dump_binaries = "build/binaries"
    options.dump_intermediates = "build/intermediates"
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.use_water_backend = True
    schedule = get_mxfp4_asymmetric_schedule()

    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM asymmetric-prefetch 4-wave test passed!")


def test_dbuf_4wave_mxfp_preshuffle_b_gemm(
    is_debug=False,
    shape=(1024, 1024, 8192),
    block=(128, 256, 256),
    eliminate_epilogue=True,
):
    """Asymmetric MXFP4 GEMM with preshuffled B data and B scales."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(shape, block, wave_shape=(1, 4))
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.eliminate_epilogue = eliminate_epilogue
    options.dump_intermediates = "build/intermediates/"
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )

    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    print("MXFP GEMM preshuffle-B 4-wave test passed!")


def test_dbuf_4wave_mxfp_asymmetric_gemm_cpp(
    is_debug=False, shape=(1024, 1024, 8192), block=(128, 256, 256)
):
    """Asymmetric MXFP4 GEMM using C++ WaveASM backend (no preshuffle)."""
    gemm, options = get_tagged_mxfp4_gemm(
        shape, block, wave_shape=(1, 4), b_address_space=GLOBAL_ADDRESS_SPACE
    )
    options.backend = "asm"
    options.wave_runtime = True
    options.dump_intermediates = "build/intermediates"
    schedule = get_mxfp4_asymmetric_schedule()
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM asymmetric 4-wave (WaveASM backend) test passed!")


def test_dbuf_4wave_mxfp_preshuffle_b_gemm_cpp(
    is_debug=False,
    shape=(512, 1024, 8192),  # 4*T0, 4*T1, 8192
    block=(128, 256, 256),
    eliminate_epilogue=True,
):
    """Preshuffle-B MXFP4 GEMM using C++ WaveASM backend."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape, block, wave_shape=(2, 2), reorder_workgroups=True
    )
    options.backend = "asm"
    options.use_buffer_ops = True
    options.wave_runtime = True
    options.use_wave_asm_backend = True
    options.dump_intermediates = "build/intermediates"
    options.eliminate_epilogue = eliminate_epilogue
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    print(
        f"MXFP GEMM preshuffle-B 4-wave (WaveASM) epilogue elimination={eliminate_epilogue} PASSED"
    )


def test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm(
    is_debug=False,
    shape=(1024, 1024, 8192),
    block=(128, 256, 256),
    eliminate_epilogue=True,
):
    """Preshuffle-B MXFP4 GEMM with dynamic M, N, K."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(shape, block, wave_shape=(1, 4))
    # Make M, N, K dynamic so the compiler does not specialize on problem size.
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "llvm"
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    options.dump_intermediates = "build/intermediates/"
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    print("MXFP GEMM preshuffle-B 4-wave dynamic M, N, K (LLVM backend) test passed!")


def test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm_asm(
    is_debug=False,
    shape=(1024, 1024, 8192),
    block=(256, 192, 256),
    eliminate_epilogue=True,
):
    """Preshuffle-B MXFP4 GEMM with dynamic M, N, K."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape, block, wave_shape=(2, 2), reorder_workgroups=True, output_dtype=tkl.bf16
    )
    # Make M, N, K dynamic so the compiler does not specialize on problem size.
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "asm"
    options.use_wave_asm_backend = True
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    options.dump_intermediates = "build/intermediates/"
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave_asymmetric.mlir"
    options.print_mlir = True
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True, output_dtype=torch.bfloat16)
    print(
        "MXFP GEMM preshuffle-B 4-wave dynamic M, N, K (WaveASM backend) test passed!"
    )


def test_mirrored(
    is_debug=False,
    shape=(8192, 3072, 8192),
    block=(256, 192, 256),
    eliminate_epilogue=True,
):
    """Mirrored asymmetric MXFP4 GEMM: preshuffle-A (direct), B through LDS."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_a(shape, block, wave_shape=(4, 1), reorder_workgroups=True)
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "asm"
    options.use_wave_asm_backend = True
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    options.dump_intermediates = "build/intermediatesmirrored/"
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave_asymmetric_mirrored.mlir"
    options.print_mlir = True
    schedule = get_mxfp4_asymmetric_schedule_mirrored(
        eliminate_epilogue=eliminate_epilogue, is_ascale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    print("Mirrored asymmetric MXFP4 GEMM (preshuffle-A, B via LDS) test passed!")


def test_mirrored_3phase(
    is_debug=False,
    shape=(8192, 3072, 8192),
    block=(256, 192, 256),
    eliminate_epilogue=True,
):
    """Mirrored AITER-exact bringup schedule; assembly first, correctness later."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_a(
        shape, block, wave_shape=(4, 1), reorder_workgroups=True
    )
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.wave_runtime = True
    use_llvm = os.environ.get("USE_LLVM_BACKEND", "0") == "1"
    if use_llvm:
        options.backend = "llvm"
        options.use_wave_asm_backend = False
    else:
        options.backend = "asm"
        options.use_wave_asm_backend = True
    options.eliminate_epilogue = eliminate_epilogue
    options.dump_intermediates = "build/intermediates_3phase/"
    options.print_mlir_file = (
        "gemm_mxfp4_dbuf_4wave_asymmetric_mirrored_3phase.mlir"
    )
    options.print_mlir = True
    schedule = get_mxfp4_asymmetric_schedule_mirrored_3phase_experimental(
        eliminate_epilogue=eliminate_epilogue, is_ascale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    # options.waveasm_print_ir_after = "all"
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)
    w_t = w.T.contiguous()
    w_t_ps = b_preshuffle(w_t)
    x_scales_ps = e8m0_shuffle(x_scales)
    w_scales_ps = e8m0_shuffle(w_scales)
    x, w_t_ps = x.cuda(), w_t_ps.cuda()
    x_scales_ps, w_scales_ps = x_scales_ps.cuda(), w_scales_ps.cuda()
    out = torch.zeros(x.shape[0], w_t_ps.shape[0], dtype=torch.float32).cuda()
    gemm(x, x_scales_ps, w_t_ps, w_scales_ps, out)
    out_cpu = out.cpu()
    ref = torch_out

    ref = ref.cpu() if ref.is_cuda else ref
    diff = (out_cpu - ref).abs()
    ref_abs = ref.abs().clamp(min=1e-6)
    rel = diff / ref_abs

    print(f"Max abs diff: {diff.max():.8f}, Max rel err: {rel.max():.8f}")
    print(f"Mean abs diff: {diff.mean():.8f}, Mean rel err: {rel.mean():.8f}")
    try:
        torch.testing.assert_close(ref, out_cpu, check_dtype=False, check_device=False)
        print("assert_close PASSED (default tol)")
    except AssertionError as e:
        msg = str(e)
        print(f"assert_close FAILED: {msg[:500]}")
    mismatch = rel > 0.01
    n_bad = mismatch.sum().item()
    n_total = mismatch.numel()
    print(f"Shape: {out_cpu.shape}, bad elements: {n_bad}/{n_total} ({100*n_bad/n_total:.2f}%)")

    if n_bad > 0:
        bad_rows = mismatch.any(dim=1).nonzero(as_tuple=True)[0]
        bad_cols = mismatch.any(dim=0).nonzero(as_tuple=True)[0]
        print(f"Bad rows: {bad_rows.numel()}/{out_cpu.shape[0]}")
        print(f"Bad cols: {bad_cols.numel()}/{out_cpu.shape[1]}")

        M, N = out_cpu.shape
        BLOCK_M, BLOCK_N = block[0], block[1]
        n_blocks_m = (M + BLOCK_M - 1) // BLOCK_M
        n_blocks_n = (N + BLOCK_N - 1) // BLOCK_N
        print(f"\nBlock layout: {n_blocks_m} x {n_blocks_n} blocks ({BLOCK_M}x{BLOCK_N})")

        print("\nPer-block-row bad column distribution:")
        for bm in range(min(n_blocks_m, 8)):
            r0, r1 = bm * BLOCK_M, min((bm + 1) * BLOCK_M, M)
            block_mismatch = mismatch[r0:r1, :]
            bad_in_block_row = block_mismatch.sum().item()
            if bad_in_block_row > 0:
                bad_cols_here = block_mismatch.any(dim=0).nonzero(as_tuple=True)[0]
                print(f"  block_row {bm} (rows {r0}-{r1-1}): {bad_in_block_row} bad, cols: {bad_cols_here[:20].tolist()}{'...' if len(bad_cols_here)>20 else ''}")

        print("\nPer-block-col bad row distribution:")
        for bn in range(min(n_blocks_n, 8)):
            c0, c1 = bn * BLOCK_N, min((bn + 1) * BLOCK_N, N)
            block_mismatch = mismatch[:, c0:c1]
            bad_in_block_col = block_mismatch.sum().item()
            if bad_in_block_col > 0:
                bad_rows_here = block_mismatch.any(dim=1).nonzero(as_tuple=True)[0]
                print(f"  block_col {bn} (cols {c0}-{c1-1}): {bad_in_block_col} bad, rows: {bad_rows_here[:20].tolist()}{'...' if len(bad_rows_here)>20 else ''}")

        print("\nFirst 10 bad elements (row, col, got, expected, rel_err):")
        bad_idx = mismatch.nonzero()
        for i in range(min(10, len(bad_idx))):
            r, c = bad_idx[i][0].item(), bad_idx[i][1].item()
            print(f"  ({r}, {c}): got={out_cpu[r,c]:.6f} ref={ref[r,c]:.6f} rel={rel[r,c]:.4f}")

        print(f"\nBad rows mod {BLOCK_M}: {sorted(set((r.item() % BLOCK_M) for r in bad_rows[:100]))}")
        print(f"Bad cols mod {BLOCK_N}: {sorted(set((c.item() % BLOCK_N) for c in bad_cols[:100]))}")

        print(f"\nBad rows mod 16: {sorted(set((r.item() % 16) for r in bad_rows[:200]))}")
        print(f"Bad cols mod 16: {sorted(set((c.item() % 16) for c in bad_cols[:200]))}")

        print(f"\nBad rows mod 64: {sorted(set((r.item() % 64) for r in bad_rows[:200]))}")
        print(f"Bad cols mod 64: {sorted(set((c.item() % 64) for c in bad_cols[:200]))}")

        # Map mismatch to thread/workgroup structure
        print("\n=== Thread/Workgroup Analysis ===")
        # Within block (256x192): wave_shape=(4,1), MFMA 16x16x128
        # M-row within block: wave_id*64 + lane_group*4 + row_in_group
        #   wave_id = thread_id // 64 (0-3)
        #   lane_group = (thread_id % 64) // 16 (0-3)
        #   row_in_group = 0-3 (from accumulator index)
        #   M-row offsets per wave: 0-3, 16-19, 32-35, 48-51
        # N-col within block: (thread_id % 16) + n_tile*16
        #   n_tile = 0-11 (12 tiles of 16 cols each = 192)

        # Analyze first block (0,0) in detail
        blk_mm = mismatch[:BLOCK_M, :BLOCK_N]
        if blk_mm.sum().item() > 0:
            print(f"\nBlock (0,0) detail: {blk_mm.sum().item()} bad elements")
            bad_local = blk_mm.nonzero()
            # Count by intra-block row
            row_counts = {}
            for idx in range(len(bad_local)):
                lr = bad_local[idx][0].item()
                row_counts[lr] = row_counts.get(lr, 0) + 1
            print(f"  Bad intra-block rows ({len(row_counts)}): {sorted(row_counts.keys())[:30]}{'...' if len(row_counts)>30 else ''}")

            # Map rows to wave_id
            wave_bad = {}
            for lr in sorted(row_counts.keys()):
                wave_id = lr // 64
                wave_bad[wave_id] = wave_bad.get(wave_id, 0) + row_counts[lr]
            print(f"  Per-wave bad counts: {wave_bad}")

            # Map to lane_group within wave
            lg_bad = {}
            for lr in sorted(row_counts.keys()):
                wave_id = lr // 64
                local_row = lr % 64
                # Rows 0-3,16-19,32-35,48-51 map to lane_groups 0-3
                row_group = local_row // 16
                row_in_group = local_row % 16
                if row_in_group < 4:
                    lane_group = 0
                elif 4 <= row_in_group < 8:
                    lane_group = 1
                elif 8 <= row_in_group < 12:
                    lane_group = 2
                else:
                    lane_group = 3
                key = (wave_id, row_group, lane_group, row_in_group)
                lg_bad[key] = lg_bad.get(key, 0) + row_counts[lr]
            print(f"  (wave, row_group, lane_group, row_in_grp) -> count:")
            for k in sorted(lg_bad.keys())[:20]:
                print(f"    {k}: {lg_bad[k]} bad cols")

            # N-col pattern within block
            col_counts = {}
            for idx in range(len(bad_local)):
                lc = bad_local[idx][1].item()
                col_counts[lc] = col_counts.get(lc, 0) + 1
            n_tile_bad = {}
            lane_in_tile_bad = {}
            for lc in sorted(col_counts.keys()):
                n_tile = lc // 16
                lane_in_tile = lc % 16
                n_tile_bad[n_tile] = n_tile_bad.get(n_tile, 0) + col_counts[lc]
                lane_in_tile_bad[lane_in_tile] = lane_in_tile_bad.get(lane_in_tile, 0) + col_counts[lc]
            print(f"\n  Per N-tile (0-11) bad counts: {n_tile_bad}")
            print(f"  Per lane-in-tile (0-15) bad counts: {lane_in_tile_bad}")

        # Check if pattern is identical across blocks
        print("\n=== Cross-block pattern consistency ===")
        ref_pattern = mismatch[:BLOCK_M, :BLOCK_N]
        same_count = 0
        diff_count = 0
        for bm in range(min(4, (M + BLOCK_M - 1) // BLOCK_M)):
            for bn in range(min(4, (N + BLOCK_N - 1) // BLOCK_N)):
                if bm == 0 and bn == 0:
                    continue
                r0, r1 = bm * BLOCK_M, min((bm+1)*BLOCK_M, M)
                c0, c1 = bn * BLOCK_N, min((bn+1)*BLOCK_N, N)
                blk = mismatch[r0:r1, c0:c1]
                if blk.shape == ref_pattern.shape and (blk == ref_pattern).all():
                    same_count += 1
                else:
                    diff_count += 1
        print(f"  Blocks matching (0,0) pattern: {same_count}, different: {diff_count}")

        n_zero_out = (out_cpu.abs() < 1e-10).sum().item()
        n_nan_out = out_cpu.isnan().sum().item()
        n_inf_out = out_cpu.isinf().sum().item()
        print(f"\nOutput stats: zeros={n_zero_out}, nans={n_nan_out}, infs={n_inf_out}")
        print(f"Output range: [{out_cpu.min():.4f}, {out_cpu.max():.4f}]")
        print(f"Ref range: [{ref.min():.4f}, {ref.max():.4f}]")

        print(f"\nMax rel error: {rel.max():.6f}")
        print(f"Mean rel error (bad only): {rel[mismatch].mean():.6f}")

        print("\nMismatch heatmap (8x16 blocks):")
        for bm in range(min(n_blocks_m, 8)):
            r0, r1 = bm * BLOCK_M, min((bm + 1) * BLOCK_M, M)
            row_str = f"  bm{bm:2d}: "
            for bn in range(min(n_blocks_n, 16)):
                c0, c1 = bn * BLOCK_N, min((bn + 1) * BLOCK_N, N)
                pct = 100 * mismatch[r0:r1, c0:c1].float().mean().item()
                if pct == 0:
                    row_str += " . "
                elif pct < 10:
                    row_str += f" {pct:.0f} "
                else:
                    row_str += f"{pct:3.0f}"
            print(row_str)


if __name__ == "__main__":
    args = parse_args()

    if args.list_tests:
        list_tests(globals())
        exit(0)

    if not args.test:
        print("Error: --test argument is required")
        print("Use --list_tests to see available tests")
        exit(1)

    success = run_test(
        args.test,
        globals(),
        args.debug,
        args.repeat,
        args.shape,
        args.block,
        args.eliminate_epilogue,
    )
    exit(0 if success else 1)
