// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s 2>&1 | FileCheck %s
//
// Test: Full assembly output for a pipelined triple-buffer loop.
// Verifies the SGPR rotation pattern for 3-way LDS buffer rotation.
//
// The loop carries three memref iter_args (LDS buffers A, B, C) and rotates
// them each iteration: yield (B, C, A).  The assembly must contain:
// 1. s_mov_b32 initialization of three SGPR offsets (0, 4096, 8192)
// 2. A loop label
// 3. ds_read using one of the SGPR offsets
// 4. s_mov_b32 rotation (4-instruction cycle: tmp=A, A=B, B=C, C=tmp)
// 5. s_cbranch_scc1 back to the loop label

module {
  gpu.module @test_pipelined_triple_buffer {

    // CHECK-LABEL: pipelined_triple_buffer:
    gpu.func @pipelined_triple_buffer(%tidx: index {llvm.mlir.workitem_id_x}) kernel {
      %c0 = arith.constant 0 : index
      %c15 = arith.constant 15 : index
      %c1 = arith.constant 1 : index

      %lds = memref.alloc() : memref<12288xi8, 3>
      %c0_byte = arith.constant 0 : index
      %c4096_byte = arith.constant 4096 : index
      %c8192_byte = arith.constant 8192 : index
      %viewA = memref.view %lds[%c0_byte][] : memref<12288xi8, 3> to memref<64x16xf16, 3>
      %viewB = memref.view %lds[%c4096_byte][] : memref<12288xi8, 3> to memref<64x16xf16, 3>
      %viewC = memref.view %lds[%c8192_byte][] : memref<12288xi8, 3> to memref<64x16xf16, 3>

      %acc_init = arith.constant dense<0.0> : vector<4xf32>

      // Init: s_mov_b32 for three LDS offsets
      // CHECK-DAG:  s_mov_b32 s{{[0-9]+}}, 0
      // CHECK-DAG:  s_mov_b32 s{{[0-9]+}}, 4096
      // CHECK-DAG:  s_mov_b32 s{{[0-9]+}}, 8192
      //
      // Loop with ds_read using SGPR offset
      // CHECK:      L_loop_0:
      // CHECK:      ds_read_b64
      //
      // SGPR rotation: 4-instruction cycle pattern (3-way rotation)
      // tmp = A, A = B, B = C, C = tmp
      // CHECK:      s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK-NEXT: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK-NEXT: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK-NEXT: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK-NEXT: s_cbranch_scc1 L_loop_0
      %result:4 = scf.for %i = %c0 to %c15 step %c1
          iter_args(%acc = %acc_init, %curA = %viewA, %curB = %viewB, %curC = %viewC)
          -> (vector<4xf32>, memref<64x16xf16, 3>, memref<64x16xf16, 3>, memref<64x16xf16, 3>) {
        %data = vector.load %curA[%tidx, %c0] : memref<64x16xf16, 3>, vector<4xf16>
        scf.yield %acc, %curB, %curC, %curA : vector<4xf32>, memref<64x16xf16, 3>, memref<64x16xf16, 3>, memref<64x16xf16, 3>
      }

      // CHECK: s_endpgm
      gpu.return
    }
  }
}
