#map = affine_map<()[s0, s1] -> ((s1 * 16 + s0 floordiv 16) mod 64)>
#map1 = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 16) * 32)>
#map2 = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
#map3 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>
#map4 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 16)>
#map5 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
#map6 = affine_map<()[s0, s1] -> (s0 * 32 + s1 * 2 - (s1 floordiv 16) * 32)>
#map7 = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4)>
#map8 = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 1)>
#map9 = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 2)>
#map10 = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 3)>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 4, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @gemm {
    stream.executable.export public @gemm workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4608 = arith.constant 4608 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %thread_id_x = gpu.thread_id  x upper_bound 256
        %thread_id_y = gpu.thread_id  y upper_bound 4
        %alloc = memref.alloc() : memref<9216xi8, #gpu.address_space<workgroup>>
        %view = memref.view %alloc[%c0][] : memref<9216xi8, #gpu.address_space<workgroup>> to memref<64x36xf16, #gpu.address_space<workgroup>>
        %view_0 = memref.view %alloc[%c4608][] : memref<9216xi8, #gpu.address_space<workgroup>> to memref<64x36xf16, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<128x64xf16, strided<[64, 1], offset: ?>>
        %1 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<128x64xf16, strided<[64, 1], offset: ?>>
        %2 = affine.apply #map()[%thread_id_x, %thread_id_y]
        %3 = affine.apply #map1()[%thread_id_x]
        %4 = affine.apply #map2()[%thread_id_x, %thread_id_y]
        %5 = affine.apply #map3()[%thread_id_x]
        %6 = affine.apply #map4()[%thread_id_x]
        %7 = affine.apply #map5()[%thread_id_x]
        %8 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %cst) -> (vector<4xf32>) {
          %19 = affine.apply #map6()[%arg3, %thread_id_x]
          %20 = vector.load %1[%2, %19] : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<2xf16>
          amdgpu.lds_barrier
          vector.store %20, %view_0[%2, %3] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<2xf16>
          %21 = vector.load %0[%2, %19] : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<2xf16>
          vector.store %21, %view[%2, %3] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<2xf16>
          amdgpu.lds_barrier
          %22 = vector.load %view[%4, %5] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %23 = vector.load %view[%4, %6] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %24 = vector.load %view_0[%7, %5] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %25 = vector.load %view_0[%7, %6] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %26 = amdgpu.mfma %24 * %22 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %27 = amdgpu.mfma %25 * %23 + %26 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          scf.yield %27 : vector<4xf32>
        }
        %9 = vector.extract_strided_slice %8 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %10 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<128x128xf32, strided<[128, 1], offset: ?>>
        %11 = affine.apply #map7()[%thread_id_x]
        %12 = affine.apply #map2()[%thread_id_x, %thread_id_y]
        vector.store %9, %10[%11, %12] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %13 = vector.extract_strided_slice %8 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %14 = affine.apply #map8()[%thread_id_x]
        vector.store %13, %10[%14, %12] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %15 = vector.extract_strided_slice %8 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %16 = affine.apply #map9()[%thread_id_x]
        vector.store %15, %10[%16, %12] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %17 = vector.extract_strided_slice %8 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %18 = affine.apply #map10()[%thread_id_x]
        vector.store %17, %10[%18, %12] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
    %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<128x64xf16>
    %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<128x64xf16>
    %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<128x128xf32>
    %3 = flow.dispatch @gemm::@gemm(%0, %1, %2) : (tensor<128x64xf16>, tensor<128x64xf16>, tensor<128x128xf32>) -> %2
    %4 = hal.tensor.barrier join(%3 : tensor<128x128xf32>) => %arg4 : !hal.fence
    %5 = hal.tensor.export %4 : tensor<128x128xf32> -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}