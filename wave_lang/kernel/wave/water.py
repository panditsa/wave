# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import tempfile
import json
import re
from collections import defaultdict
from pathlib import Path
import linecache
import os
import subprocess
import sys
import math
from typing import Any, Sequence

from wave_lang.kernel.wave.compile_options import WaveCompileOptions
from wave_lang.support.detect_water import get_water_mlir_pkg_path, get_water_opt
from wave_lang.support.ir_imports import (
    Attribute,
    BlockArgument,
    FunctionType,
    InsertionPoint,
    IntegerType,
    MemRefType,
    Module,
    Operation,
    TypeAttr,
    WalkResult,
    gpu_d,
    llvm_d,
    memref_d,
    stream_d,
)


def _find_single_nested(name: str, parent: Operation) -> Operation:
    """Find a single operation with the specified name in a single-block parent operation.

    Raises a RuntimeError if there is no such operation or if there are multiple.
    Raises a ValueError if the parent operation is not a single-block one.
    """

    if len(parent.regions) != 1 or len(parent.regions[0].blocks) != 1:
        raise ValueError("Expected a single-block operation.")
    captured = None
    for op in parent.regions[0].blocks[0].operations:
        # Dynamic typing is hard: must to op.operation.name in case some specific class has .name that has a different meaning.
        if op.operation.name == name:
            if captured:
                raise RuntimeError(f"More than one '{name}' operation found.")
            captured = op
    if not captured:
        raise RuntimeError(f"No {name} operation found.")
    return captured


def _get_numeric_memory_space(memory_space: Attribute) -> int | None:
    """Return the numeric address space used in LLVM pointers that matches the given GPU dialect address space."""

    # TODO: expose construction of these attributes in upstream bindings.
    if memory_space == Attribute.parse("#gpu.address_space<workgroup>"):
        return int(gpu_d.AddressSpace.Workgroup)
    if memory_space == Attribute.parse("#gpu.address_space<global>"):
        return int(gpu_d.AddressSpace.Global)
    if memory_space == Attribute.parse("#gpu.address_space<private>"):
        return int(gpu_d.AddressSpace.Private)
    return None


def _deiree(module: Module) -> str:
    """Return a copy of the module without IREE-specific operations, suitable for MLIR processing."""
    # Uglily clone the module by printing and parsing back.
    module = Module.parse(module.operation.get_asm(), context=module.context)

    executable = _find_single_nested("stream.executable", module.operation)
    local_module = _find_single_nested("builtin.module", executable)
    func = _find_single_nested("func.func", local_module)

    # TODO: add launch bounds

    to_delete = []  # type: list[Operation]
    subspans = []  # type: list[stream_d.BindingSubspanOp]

    def replace_ops_and_collect_subspans(op: Operation) -> WalkResult:
        """Callback for the function walk dispatching based on the operation kind."""

        # Replace IREE workgroup IDs with GPU dialect block IDs.
        if isinstance(op.opview, stream_d.DispatchWorkgroupIDOp):
            dispatch = op.opview  # type: stream_d.DispatchWorkgroupIDOp
            match dispatch.dimension.value:
                case 0:
                    dimension = gpu_d.Dimension.x
                case 1:
                    dimension = gpu_d.Dimension.y
                case 2:
                    dimension = gpu_d.Dimension.z
            with InsertionPoint(op), op.location:
                block_id = gpu_d.BlockIdOp(dimension)
            op.result.replace_all_uses_with(block_id.result)
            to_delete.append(op)
            return WalkResult.ADVANCE

        # Record IREE subspans so they can be replaced with function arguments.
        if isinstance(op.opview, stream_d.BindingSubspanOp):
            subspan = op.opview  # type: stream_d.BindingSubspanOp
            subspans.append(subspan)
            return WalkResult.ADVANCE

        # Replace allocations with poison values, at this point we don't support indirect loads/stores.
        # TODO: when adding support for indirect loads/stores, implement a check for not using data from local allocations in control flow or address computation; potentially with a fallback to allocating in another memory space.
        if isinstance(op.opview, memref_d.AllocOp):
            with op.context, InsertionPoint(op), op.location:
                original_memref_type = MemRefType(op.opview.memref.type)
                llvm_ptr_type = llvm_d.PointerType.get(
                    address_space=_get_numeric_memory_space(
                        original_memref_type.memory_space
                    )
                )
                llvm_descriptor_type = llvm_d.StructType.get_literal(
                    [llvm_ptr_type, llvm_ptr_type, IntegerType.get_signless(64)]
                )
                poison = llvm_d.PoisonOp(llvm_descriptor_type)
                base_memref_type = MemRefType.get(
                    [],
                    element_type=original_memref_type.element_type,
                    memory_space=original_memref_type.memory_space,
                )
                cast = Operation.create(
                    "builtin.unrealized_conversion_cast",
                    results=[base_memref_type],
                    operands=[poison.results[0]],
                )
                strides, _ = original_memref_type.get_strides_and_offset()
                assert (
                    MemRefType.get_dynamic_stride_or_offset() not in strides
                ), "Allocation is not expected to have dynamic strides."
                cast2 = memref_d.ReinterpretCastOp(
                    original_memref_type,
                    cast.results[0],
                    [],
                    op.opview.dynamicSizes,
                    [],
                    static_offsets=[0] * len(original_memref_type.shape),
                    static_sizes=original_memref_type.shape,
                    static_strides=strides,
                )
                op.opview.memref.replace_all_uses_with(cast2.result)
                to_delete.append(op)
                return WalkResult.ADVANCE

        return WalkResult.ADVANCE

    func.walk(replace_ops_and_collect_subspans)
    old_func_type = func.attributes["function_type"].value
    func_input_types = old_func_type.inputs
    for subspan in subspans:
        subspan.binding.set_type(subspan.result.type)
        arg_number = BlockArgument(subspan.binding).arg_number
        func_input_types[arg_number] = subspan.result.type
        subspan.result.replace_all_uses_except(subspan.binding, subspan.operation)
        to_delete.append(subspan)

    with old_func_type.context:
        func.attributes["function_type"] = TypeAttr.get(
            FunctionType.get(func_input_types, old_func_type.results)
        )

    for op in to_delete:
        op.erase()

    if int(os.environ.get("WAVE_WATER_DUMP_MLIR_BEFORE", "0")) != 0:
        print(local_module, file=sys.stderr)
    return local_module.get_asm(binary=False, print_generic_op_form=True)


def make_linear_pass_pipeline(
    pipeline: Sequence[
        tuple[str, dict[str, Any]] | tuple[str, dict[str, Any], str] | str
    ],
) -> str:
    """
    Construct a pass pipeline string for mlir-opt style tool.

    Args:
        pipeline: A sequence of pass names and arguments.
            - For the pass with no arguments/all default arguments, pass just the name as a string.
            - For the pass with arguments, pass a tuple with the name and a dictionary of arguments.
            - For the pass with a root op, pass a tuple with the name, a dictionary of arguments, and the root op name.
              Arguments dict can be empty.
    Returns:
        A string representing the pass pipeline command line argument.
    """

    def make_pass_arguments(
        name: str, args: dict[str, Any], root_op: str | None = None
    ) -> str:
        ret = (
            name
            + "{"
            + " ".join("=".join((key, str(value))) for (key, value) in args.items())
            + "}"
        )
        if root_op:
            ret = root_op + "(" + ret + ")"
        return ret

    return (
        "--pass-pipeline=builtin.module("
        + ",".join(
            entry if isinstance(entry, str) else make_pass_arguments(*entry)
            for entry in pipeline
        )
        + ")"
    )


def water_leak_in_bounds_check(module: Module, override_ir: str = ""):
    binary = get_water_opt()
    generic_mlir = _deiree(module) if override_ir == "" else override_ir
    pipeline = [
        (
            "water-assert-in-bounds",
            {"include-vector-load-store": 1, "create-speculative-funcs": 1},
        ),
        "lower-affine",
        "canonicalize",
        "cse",
        "loop-invariant-code-motion",
        "int-range-optimizations",
        "canonicalize",
        "water-check-static-assertions",
    ]

    def get_code_context(
        filename: str, start_line: int, end_line: int, context: int = 2
    ) -> str:
        """
        Retrieves a line and a few lines of context around it.

        Args:
            filename (str): The path to the file.
            line_number (int): The central line number to retrieve.
            context (int): The number of lines to show before and after.

        Returns:
            A string with the code + context.
        """
        start = max(1, start_line - context)
        end = end_line + context + 1

        num_characters = int(math.ceil(math.log10(end)))
        format_string = "{0:" + str(num_characters) + "d}"

        lines = []
        for i in range(start, end + 1):
            line = linecache.getline(filename, i)
            if not line:
                break
            lines.append(
                format_string.format(i)
                + f"{'*' if start_line <= i <= end_line else ' '}| {line.rstrip()}"
            )

        return "\n".join(lines)

    def diagnostic_from_json(
        json_obj: dict[str, Any], *, include_context: bool = False
    ) -> str:
        if "unknown" in json_obj:
            return "<unknown location>"
        if "name" in json_obj:
            child = diagnostic_from_json(
                json_obj["loc"], include_context=include_context
            )
            return '"' + json_obj["name"] + '" at' + child
        if "fused" in json_obj:
            result = "fused<["
            for d in json_obj["fused"]:
                result += "\n  " + diagnostic_from_json(
                    d, include_context=include_context
                )
            result += "]>"
            return result
        if "start_line" in json_obj:
            start_line, end_line, start_col, end_col = tuple(
                int(json_obj[key])
                for key in ("start_line", "end_line", "start_column", "end_column")
            )
            result = json_obj["file"] + ":" + str(start_line)
            zero_column = start_col == 0
            if not zero_column:
                result += ":" + str(start_col)
            same_line = start_line == end_line
            same_column = start_col == end_col
            if not same_line or (not same_column and not zero_column):
                result += " to "
                if not same_line:
                    result += str(end_line)
                if not same_column and not zero_column:
                    result += ":" + str(end_col)
            if include_context:
                result += "\n" + get_code_context(
                    json_obj["file"], start_line, end_line
                )
            return result
        if "callstack" in json_obj:
            # TODO: consider capturing Python stack frame objects in a
            # dictionary with unique names, using those names in named locations
            # in MLIR and then reconstructing the traceback programmatically.
            return "\ncalled from ".join(
                diagnostic_from_json(d, include_context=(i == 0 and include_context))
                for i, d in enumerate(json_obj["callstack"])
            )
        raise ValueError(f"Unhandled diagnostic: {json_obj}")

    exceptions = []

    with tempfile.TemporaryDirectory() as temp_dir:
        diagnosticsFile = Path(temp_dir) / "diagnostics.txt"
        result = subprocess.run(
            [
                binary,
                "--diagnostics-file",
                diagnosticsFile,
                "--allow-unregistered-dialect",
                make_linear_pass_pipeline(pipeline),
            ],
            input=generic_mlir,
            capture_output=True,
            text=True,
        )
        if diagnosticsFile.is_file():
            with open(diagnosticsFile, "r") as file:
                for line in file:
                    diag = json.loads(line.rstrip())
                    msg = (
                        diag["severity"]
                        + ": "
                        + diag["message"]
                        + "\nAt "
                        + diagnostic_from_json(diag, include_context=True)
                    )
                    exception = RuntimeError(f"{msg}")
                    exceptions.append(exception)

    if len(result.stderr) != 0:
        exceptions.append(RuntimeError("Water MLIR error (stderr): " + result.stderr))

    if int(os.environ.get("WAVE_WATER_DUMP_MLIR_AFTER", "0")) != 0:
        print(result.stdout, file=sys.stderr)

    if len(exceptions) > 0:
        # Exception groups are only available for Python >= 3.11
        assert sys.version_info.major == 3, "Unexpected Python version"
        if sys.version_info.minor >= 11:
            raise ExceptionGroup("Water errors: ", exceptions)

        if len(exceptions) == 1:
            raise exceptions[0]
        e = exceptions[0]
        e.add_note(f"{len(exceptions) - 1} other exceptions were generated")
        raise e

    if "cf.assert" in result.stdout:
        print(
            "[warning] Couldn't statically determine the absence of out-of-bounds accesses."
        )
    else:
        print("[info] No out-of-bounds accesses detected.")


def coalesce_scale_loads(mlir_asm: str) -> str:
    """Coalesce pairs of vector.load<4xi8> from scale memrefs into vector.load<8xi8>.

    Scale tensors in LDS use memref<Nx8xi8, workgroup> layout (no padding).
    The compiler emits two vector.load<4xi8> per row at columns 0 and 4
    (one per WMMA in the K-unroll). The LLVM backend fuses these into
    ds_load_2addr_b32 (dual-address 32-bit loads).

    This function replaces each pair with a single vector.load<8xi8> from
    column 0, plus two vector.extract_strided_slice ops to split the result.
    After memref decomposition, the wide load becomes a single llvm.load i64
    which maps to ds_load_b64 (consecutive 64-bit load) -- avoiding the
    scattered ds_load_2addr_b32 instruction.
    """
    lines = mlir_asm.split("\n")

    # Match: %NAME = vector.load %MEM[%ROW, %COL] : memref<NxMxi8, ...workgroup...>, vector<4xi8>
    # where M >= 8 (covers both padded stride-24 and unpadded stride-8 layouts).
    load_re = re.compile(
        r"^(\s+)"  # 1: indent
        r"(%\S+) = vector\.load "  # 2: result SSA
        r"(%\S+)"  # 3: memref SSA
        r"\[(%\S+), (%\S+)\]"  # 4: row index, 5: col index
        r" : (memref<\d+x\d+xi8, #gpu\.address_space<workgroup>>)"  # 6: type
        r", vector<4xi8>"
        r"\s*$"
    )

    # Collect all matching loads: (line_idx, indent, result, memref, row, col, memref_type)
    load_info = []
    for i, line in enumerate(lines):
        m = load_re.match(line)
        if m:
            load_info.append(
                (i, m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6))
            )

    # Group by (memref, row_index) to find pairs at columns %c0 and %c4
    groups = defaultdict(list)
    for info in load_info:
        key = (info[3], info[4])  # (memref SSA, row SSA)
        groups[key].append(info)

    replacements = {}  # line_idx -> list of replacement lines
    coalesce_id = 0

    for _key, loads in groups.items():
        col0_loads = [l for l in loads if l[5] == "%c0"]
        col4_loads = [l for l in loads if l[5] == "%c4"]

        if not col0_loads or not col4_loads:
            continue

        for c0, c4 in zip(col0_loads, col4_loads):
            # The wide load must be defined before the second extract uses it.
            # Ensure col-0 load (where we place the wide load) comes first.
            if c0[0] > c4[0]:
                continue  # Skip if ordering is unexpected

            wide_name = f"%_coalesced_scale_{coalesce_id}"
            coalesce_id += 1
            indent = c0[1]

            # Replace the col-0 load with a wide load + extract of low half
            replacements[c0[0]] = [
                f"{indent}{wide_name} = vector.load {c0[3]}[{c0[4]}, {c0[5]}]"
                f" : {c0[6]}, vector<8xi8>",
                f"{indent}{c0[2]} = vector.extract_strided_slice {wide_name}"
                f" {{offsets = [0], sizes = [4], strides = [1]}}"
                f" : vector<8xi8> to vector<4xi8>",
            ]

            # Replace the col-4 load with an extract of high half
            replacements[c4[0]] = [
                f"{indent}{c4[2]} = vector.extract_strided_slice {wide_name}"
                f" {{offsets = [4], sizes = [4], strides = [1]}}"
                f" : vector<8xi8> to vector<4xi8>",
            ]

    if not replacements:
        return mlir_asm

    new_lines = []
    for i, line in enumerate(lines):
        if i in replacements:
            new_lines.extend(replacements[i])
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def water_lowering_pipeline(module: Module, options: WaveCompileOptions) -> Module:
    binary = get_water_opt()
    mlir_asm = module.operation.get_asm()

    # Coalesce adjacent scale loads (vector.load<4xi8> pairs at columns 0,4)
    # into single vector.load<8xi8> + extract ops to get ds_load_b64.
    mlir_asm = coalesce_scale_loads(mlir_asm)

    target_chip = options.target

    def add_opt(pipeline):
        if options.optimization_level:
            return [pipeline]

        return []

    def add_transform(transform: str, entry_point: str) -> tuple[str, dict[str, Any]]:
        nonlocal mlir_asm
        # Erase the last occurrence of '}' from mlir_asm which closes the module operation
        last_close = mlir_asm.rfind("}")
        if last_close != -1:
            mlir_asm = mlir_asm[:last_close]
        mlir_asm += transform
        mlir_asm += "}\n"
        return ("transform-interpreter", {"entry-point": entry_point})

    # TODO: this transform refuses to work.
    alloc_to_alloca = """
  transform.named_sequence @__transform_alloc_to_alloca(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["gpu.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.alloc_to_alloca
    } : !transform.any_op
    transform.yield
  }
"""

    alloca_to_global = """
  transform.named_sequence @__transform_alloca_to_global(%arg0: !transform.any_op {transform.readonly}) {
    %alloca = transform.structured.match ops{["memref.alloca"]} in %arg0
        : (!transform.any_op) -> !transform.op<"memref.alloca">
    %get_global, %global = transform.memref.alloca_to_global %alloca
          : (!transform.op<"memref.alloca">)
            -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
"""

    canonicalize_cse = "composite-fixed-point-pass", {
        "name": "canonicalize_cse",
        "pipeline": "any(canonicalize,cse)",
    }

    int_range_optimizations = "composite-fixed-point-pass", {
        "name": "int-range-optimizations",
        "pipeline": 'any(int-range-optimizations,arith-int-range-narrowing{int-bitwidths-supported="32"},canonicalize,cse)',
    }

    llvm_opt_level = 3 if options.optimization_level else 0
    dump_intermediates = options.dump_intermediates or ""
    toolkit_path = get_water_mlir_pkg_path()

    pipeline = [
        "water-memref-decomposition",
        "water-fuse-scale-loads",  # operates on llvm.load ops so should ge after memref decomposition
        *add_opt(canonicalize_cse),
        "lower-affine",
        *add_opt(int_range_optimizations),
        *add_opt("loop-invariant-code-motion"),
        ("water-alloc-to-alloca", {}, "gpu.module"),
        # add_transform(alloc_to_alloca, "__transform_alloc_to_alloca"),
        add_transform(alloca_to_global, "__transform_alloca_to_global"),
        "convert-scf-to-cf",
        ("convert-amdgpu-to-rocdl", {"chipset": target_chip}),
        ("convert-gpu-to-rocdl", {"use-bare-ptr-memref-call-conv": "1"}, "gpu.module"),
        ("rocdl-attach-target", {"chip": target_chip, "O": llvm_opt_level}),
        ("gpu-to-llvm", {"use-bare-pointers-for-kernels": "1"}),
        "convert-vector-to-llvm",
        "reconcile-unrealized-casts",
        *add_opt(canonicalize_cse),
        (
            "water-gpu-module-to-binary",
            {"dump-intermediates": dump_intermediates, "toolkit": toolkit_path},
        ),
        "water-gpu-to-gpu-runtime",
        "water-drop-transform-ops",
        "symbol-dce",
        *add_opt(canonicalize_cse),
    ]

    args = [binary, make_linear_pass_pipeline(pipeline)]
    if options.mlir_print_ir_after_all:
        args.append("--mlir-print-ir-after-all")

    try:
        result = subprocess.check_output(
            args,
            input=mlir_asm,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        error_msg = f"Subprocess failed with return code {e.returncode}."
        raise RuntimeError(error_msg) from e

    with module.context:
        return Module.parse(result)


def apply_water_middle_end_passes(mlir_text: str) -> str:
    """Apply Water middle-end pipeline using subprocess water-opt.

    This function applies the following passes:
    - water-wave-detect-normal-forms
    - (nested in normalform.module)
      - water-wave-propagate-elements-per-thread
      - water-wave-resolve-distributed-allocations
      - water-wave-detect-normal-forms
      - lower-wave-to-mlir
    - canonicalize
    - cse

    Args:
        mlir_text: Input Wave dialect MLIR as string

    Returns:
        Optimized MLIR as string after applying the passes

    Raises:
        RuntimeError: If water-opt is not available or passes fail
    """
    binary = get_water_opt()

    # Define the pass pipeline for Wave lowering
    # Note: water-wave-detect-normal-forms wraps contents in normalform.module,
    # so subsequent passes that operate on normalform::ModuleOp must be nested together
    # in the same normalform.module() pass manager to avoid duplicate normal form attributes.
    pass_pipeline = (
        "--pass-pipeline=builtin.module("
        "water-wave-detect-normal-forms,"
        "normalform.module("
        "water-wave-propagate-elements-per-thread,"
        "water-wave-resolve-distributed-allocations,"
        "water-wave-detect-normal-forms,"
        "lower-wave-to-mlir"
        "),"
        "lower-normalform-module,"
        "canonicalize,"
        "cse"
        ")"
    )

    try:
        result = subprocess.check_output(
            [
                binary,
                "--allow-unregistered-dialect",
                pass_pipeline,
            ],
            input=mlir_text,
            text=True,
        )

        return result

    except subprocess.CalledProcessError as e:
        error_msg = f"water-opt subprocess failed with return code {e.returncode}."
        if e.stderr:
            error_msg += f" Error: {e.stderr}"
        raise RuntimeError(error_msg) from e
