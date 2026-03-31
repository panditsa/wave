// RUN: waveasm-translate %s | FileCheck %s
//
// Test: scf.if where the then branch yields a computed VGPR value and the
// else branch yields a constant (which translates to an immediate).
//
// In split-K kernels the prologue is wrapped in an scf.if that guards
// against out-of-bounds splits.  The then branch executes computation
// (producing registers) while the else branch yields zero-initialized
// values (immediates).  The backend must coerce the else-yield immediates
// into register types so that both branches yield type-compatible values.
//
// Note: the translator currently produces a vreg condition for waveasm.if
// (instead of scc), so the output is dumped in generic form after
// verification. The CHECK patterns below match generic form.

module {
  gpu.module @test_if_else_coercion {

    // CHECK-LABEL: sym_name = "if_else_coercion"
    gpu.func @if_else_coercion() kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cond_i1 = arith.cmpi slt, %c0, %c1 : index

      %zero_i32 = arith.constant 0 : i32
      %one_i32 = arith.constant 1 : i32

      // Then branch: compute a VGPR value (addi -> v_add_u32)
      // Else branch: yield a constant zero (-> immediate coerced to vreg)
      //
      // CHECK:      "waveasm.if"
      // CHECK:        "waveasm.v_add_u32"
      // CHECK:        "waveasm.yield"
      // CHECK:      }, {
      // CHECK:        "waveasm.v_mov_b32"
      // CHECK:        "waveasm.yield"
      %result = scf.if %cond_i1 -> i32 {
        %val = arith.addi %zero_i32, %one_i32 : i32
        scf.yield %val : i32
      } else {
        scf.yield %zero_i32 : i32
      }

      // CHECK: "waveasm.s_endpgm"
      gpu.return
    }
  }
}
