import torch
from typing import Sequence

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    torch_dtype_to_wave,
)

def get_vector_add_kernel(
    shape: Tuple[int, int],
    dtype: torch.dtype = torch.float32,
):
    # Some symbol representing the problem size   
    M = tkl.sym.M
    N = tkl.sym.N

    # Some symbol representing tile size each wave will work on
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N

    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    dtype_wave = torch_dtype_to_wave(dtype)

    # Number of workgroups
    # Assuming M and N are divisible by BLOCK_M and BLOCK_N respectively
    # Two things are happening here:
    # 1. We are saying that M maps to Y dimension and N maps to X dimension (because of 1 and 0 argument)
    # 2. We are saying that number of blocks in Y dimension is M // BLOCK_M and in X dimension is N // BLOCK_N
    # each block may contain multiple waves (?)
    constraints : list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    
    # Each wave will work on a tile size of blockM/2 and blockN/2 meaning, we will need 4 waves
    # to cover the whole tile of BLOCK_M x BLOCK_N
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    
    constraints += [tkw.HardwareConstraint(threads_per_wave=64)]

    @tkw.wave(constraints)
    def vector_add_kernel(
        a: tkl.Memory[M, N, ADDRESS_SPACE, dtype_wave],
        b: tkl.Memory[M, N, ADDRESS_SPACE, dtype_wave],
        c: tkl.Memory[M, N, ADDRESS_SPACE, dtype_wave],
    ):

        # each wave will work on a tile of (BLOCK_M/2) x (BLOCK_N/2)
        # each workgroup will have 4 waves to cover the workgroup constraint of BLOCK_M x BLOCK_N
        lhs = tkw.read(a)
        rhs = tkw.read(b)
        result = lhs + rhs
        tkw.write(c, result)