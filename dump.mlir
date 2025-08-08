#map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 128 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 128) * 128)>
#map1 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 8) * 64)>
#map2 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 128 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + 64)>
#map3 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 256) * 256)>
#map4 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
#map5 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
#map6 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 256 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
#map7 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 128)>
#map8 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 128) * 128 + 64)>
#map9 = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 8) mod 256)>
#map10 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 64) floordiv 256) * 256 + 64)>
#map11 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 128) floordiv 256) * 256 + 128)>
#map12 = affine_map<()[s0, s1] -> (s1 * 32 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8 + 192) floordiv 256) * 256 + 192)>
#map13 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
#map14 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 32)>
#map15 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>
#map16 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 16)>
#map17 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 32 + 16)>
#map18 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16)>
#map19 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 16)>
#map20 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 32)>
#map21 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 48)>
#map22 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 64)>
#map23 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 80)>
#map24 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 96)>
#map25 = affine_map<()[s0, s1] -> (s0 + s1 * 128 - (s0 floordiv 16) * 16 + 112)>
#map26 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 32)>
#map27 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 48)>
#map28 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8 - (s1 floordiv 8) * 64 + 64)>
#map29 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 16) * 4)>
#map30 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 128 - (s0 floordiv 16) * 16)>
#map31 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 16) * 4 + 1)>
#map32 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 16) * 4 + 2)>
#map33 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 16) * 4 + 3)>
#map34 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 128 - (s0 floordiv 16) * 16 + 16)>
#map35 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 128 - (s0 floordiv 16) * 16 + 32)>
#map36 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 128 - (s0 floordiv 16) * 16 + 48)>
#map37 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 128 - (s0 floordiv 16) * 16 + 64)>
#map38 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 128 - (s0 floordiv 16) * 16 + 80)>
#map39 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 128 - (s0 floordiv 16) * 16 + 96)>
#map40 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 128 - (s0 floordiv 16) * 16 + 112)>
#map41 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 16) * 4 + 16)>
#map42 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 16) * 4 + 17)>
#map43 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 16) * 4 + 18)>
#map44 = affine_map<()[s0, s1] -> (s0 * 128 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 16) * 4 + 19)>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @gemm {
    stream.executable.export public @gemm workgroups() -> (index, index, index) {
      %c32 = arith.constant 32 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      stream.return %c32, %c16, %c1 : index, index, index
    }
    builtin.module {
      func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
        %c4_i32 = arith.constant 4 : i32
        %c0_i32 = arith.constant 0 : i32
        %c63 = arith.constant 63 : index
        %c1 = arith.constant 1 : index
        %c34816 = arith.constant 34816 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %block_id_x = gpu.block_id  x upper_bound 32
        %block_id_y = gpu.block_id  y upper_bound 16
        %thread_id_x = gpu.thread_id  x upper_bound 256
        %thread_id_y = gpu.thread_id  y upper_bound 2
        %alloc = memref.alloc() : memref<52224xi8, #gpu.address_space<workgroup>>
        %view = memref.view %alloc[%c0][] : memref<52224xi8, #gpu.address_space<workgroup>> to memref<256x68xf16, #gpu.address_space<workgroup>>
        %view_0 = memref.view %alloc[%c34816][] : memref<52224xi8, #gpu.address_space<workgroup>> to memref<128x68xf16, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<4096x4096xf16, strided<[4096, 1], offset: ?>>
        %1 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
        %2 = affine.apply #map1()[%thread_id_x]
        %3 = vector.load %0[%1, %2] : memref<4096x4096xf16, strided<[4096, 1], offset: ?>>, vector<8xf16>
        %4 = affine.apply #map2()[%thread_id_x, %thread_id_y, %block_id_x]
        %5 = vector.load %0[%4, %2] : memref<4096x4096xf16, strided<[4096, 1], offset: ?>>, vector<8xf16>
        %6 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<4096x4096xf16, strided<[4096, 1], offset: ?>>
        %7 = affine.apply #map3()[%thread_id_x, %thread_id_y, %block_id_y]
        %8 = vector.load %6[%7, %2] : memref<4096x4096xf16, strided<[4096, 1], offset: ?>>, vector<8xf16>
        %9 = affine.apply #map4()[%thread_id_x, %thread_id_y, %block_id_y]
        %10 = vector.load %6[%9, %2] : memref<4096x4096xf16, strided<[4096, 1], offset: ?>>, vector<8xf16>
        %11 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_y]
        %12 = vector.load %6[%11, %2] : memref<4096x4096xf16, strided<[4096, 1], offset: ?>>, vector<8xf16>
        %13 = affine.apply #map6()[%thread_id_x, %thread_id_y, %block_id_y]
        %14 = vector.load %6[%13, %2] : memref<4096x4096xf16, strided<[4096, 1], offset: ?>>, vector<8xf16>
        %15 = affine.apply #map7()[%thread_id_x, %thread_id_y]
        vector.store %3, %view_0[%15, %2] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %16 = affine.apply #map8()[%thread_id_x, %thread_id_y]
        vector.store %5, %view_0[%16, %2] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %17 = affine.apply #map9()[%thread_id_x, %thread_id_y]
        vector.store %8, %view[%17, %2] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %18 = affine.apply #map10()[%thread_id_x, %thread_id_y]
        vector.store %10, %view[%18, %2] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %19 = affine.apply #map11()[%thread_id_x, %thread_id_y]
        vector.store %12, %view[%19, %2] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %20 = affine.apply #map12()[%thread_id_x, %thread_id_y]
        vector.store %14, %view[%20, %2] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        amdgpu.lds_barrier
        %21 = affine.apply #map13()[%thread_id_x, %thread_id_y]
        %22 = arith.index_cast %21 : index to i32
        %23 = arith.cmpi sge, %22, %c4_i32 : i32
        %24 = arith.cmpi slt, %22, %c4_i32 : i32
        %25 = vector.broadcast %23 : i1 to vector<i1>
        %26 = vector.extractelement %25[] : vector<i1>
        scf.if %26 {
          rocdl.s.barrier
        }
        %27 = affine.apply #map14()[%thread_id_x]
        %28 = affine.apply #map15()[%thread_id_x]
        %29 = affine.apply #map16()[%thread_id_x]
        %30 = affine.apply #map17()[%thread_id_x]
        %31 = affine.apply #map18()[%thread_id_x, %thread_id_y]
        %32 = affine.apply #map19()[%thread_id_x, %thread_id_y]
        %33 = affine.apply #map20()[%thread_id_x, %thread_id_y]
        %34 = affine.apply #map21()[%thread_id_x, %thread_id_y]
        %35 = affine.apply #map22()[%thread_id_x, %thread_id_y]
        %36 = affine.apply #map23()[%thread_id_x, %thread_id_y]
        %37 = affine.apply #map24()[%thread_id_x, %thread_id_y]
        %38 = affine.apply #map25()[%thread_id_x, %thread_id_y]
        %39 = affine.apply #map26()[%thread_id_x]
        %40 = affine.apply #map27()[%thread_id_x]
        %41:16 = scf.for %arg3 = %c0 to %c63 step %c1 iter_args(%arg4 = %cst, %arg5 = %cst, %arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %243 = vector.load %view_0[%27, %28] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %244 = vector.load %view_0[%27, %29] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %245 = vector.load %view_0[%30, %28] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %246 = vector.load %view_0[%30, %29] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %247 = vector.load %view[%31, %28] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %248 = vector.load %view[%31, %29] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %249 = vector.load %view[%32, %28] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %250 = vector.load %view[%32, %29] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %251 = vector.load %view[%33, %28] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %252 = vector.load %view[%33, %29] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %253 = vector.load %view[%34, %28] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %254 = vector.load %view[%34, %29] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %255 = vector.load %view[%35, %28] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %256 = vector.load %view[%35, %29] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %257 = vector.load %view[%36, %28] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %258 = vector.load %view[%36, %29] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %259 = vector.load %view[%37, %28] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %260 = vector.load %view[%37, %29] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %261 = vector.load %view[%38, %28] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %262 = vector.load %view[%38, %29] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
          %263 = affine.apply #map28()[%arg3, %thread_id_x]
          %264 = vector.load %0[%4, %263] : memref<4096x4096xf16, strided<[4096, 1], offset: ?>>, vector<8xf16>
          %265 = vector.load %0[%1, %263] : memref<4096x4096xf16, strided<[4096, 1], offset: ?>>, vector<8xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
          %266 = vector.load %view_0[%27, %39] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %267 = vector.load %view_0[%27, %40] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %268 = vector.load %view_0[%30, %39] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %269 = vector.load %view_0[%30, %40] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %270 = vector.load %view[%31, %39] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %271 = vector.load %view[%31, %40] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %272 = vector.load %view[%32, %39] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %273 = vector.load %view[%32, %40] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %274 = vector.load %view[%33, %39] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %275 = vector.load %view[%33, %40] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %276 = vector.load %view[%34, %39] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %277 = vector.load %view[%34, %40] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %278 = vector.load %view[%35, %39] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %279 = vector.load %view[%35, %40] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %280 = vector.load %view[%36, %39] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %281 = vector.load %view[%36, %40] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %282 = vector.load %view[%37, %39] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %283 = vector.load %view[%37, %40] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %284 = vector.load %view[%38, %39] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %285 = vector.load %view[%38, %40] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
          %286 = vector.load %6[%11, %263] : memref<4096x4096xf16, strided<[4096, 1], offset: ?>>, vector<8xf16>
          %287 = vector.load %6[%7, %263] : memref<4096x4096xf16, strided<[4096, 1], offset: ?>>, vector<8xf16>
          %288 = vector.load %6[%13, %263] : memref<4096x4096xf16, strided<[4096, 1], offset: ?>>, vector<8xf16>
          %289 = vector.load %6[%9, %263] : memref<4096x4096xf16, strided<[4096, 1], offset: ?>>, vector<8xf16>
          rocdl.s.barrier
          llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
          rocdl.s.setprio 1
          %290 = amdgpu.mfma %243 * %247 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %291 = amdgpu.mfma %244 * %248 + %290 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %292 = amdgpu.mfma %243 * %249 + %arg5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %293 = amdgpu.mfma %244 * %250 + %292 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %294 = amdgpu.mfma %243 * %251 + %arg6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %295 = amdgpu.mfma %244 * %252 + %294 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %296 = amdgpu.mfma %243 * %253 + %arg7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %297 = amdgpu.mfma %244 * %254 + %296 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %298 = amdgpu.mfma %243 * %255 + %arg8 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %299 = amdgpu.mfma %244 * %256 + %298 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %300 = amdgpu.mfma %243 * %257 + %arg9 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %301 = amdgpu.mfma %244 * %258 + %300 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %302 = amdgpu.mfma %243 * %259 + %arg10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %303 = amdgpu.mfma %244 * %260 + %302 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %304 = amdgpu.mfma %243 * %261 + %arg11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %305 = amdgpu.mfma %244 * %262 + %304 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %306 = amdgpu.mfma %245 * %247 + %arg12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %307 = amdgpu.mfma %246 * %248 + %306 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %308 = amdgpu.mfma %245 * %249 + %arg13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %309 = amdgpu.mfma %246 * %250 + %308 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %310 = amdgpu.mfma %245 * %251 + %arg14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %311 = amdgpu.mfma %246 * %252 + %310 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %312 = amdgpu.mfma %245 * %253 + %arg15 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %313 = amdgpu.mfma %246 * %254 + %312 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %314 = amdgpu.mfma %245 * %255 + %arg16 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %315 = amdgpu.mfma %246 * %256 + %314 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %316 = amdgpu.mfma %245 * %257 + %arg17 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %317 = amdgpu.mfma %246 * %258 + %316 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %318 = amdgpu.mfma %245 * %259 + %arg18 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %319 = amdgpu.mfma %246 * %260 + %318 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %320 = amdgpu.mfma %245 * %261 + %arg19 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %321 = amdgpu.mfma %246 * %262 + %320 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          rocdl.s.setprio 0
          amdgpu.lds_barrier
          llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
          vector.store %265, %view_0[%15, %2] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %264, %view_0[%16, %2] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %287, %view[%17, %2] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %288, %view[%20, %2] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %289, %view[%18, %2] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %286, %view[%19, %2] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          rocdl.s.barrier
          llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
          rocdl.s.setprio 1
          %322 = amdgpu.mfma %266 * %270 + %291 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %323 = amdgpu.mfma %267 * %271 + %322 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %324 = amdgpu.mfma %266 * %272 + %293 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %325 = amdgpu.mfma %267 * %273 + %324 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %326 = amdgpu.mfma %266 * %274 + %295 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %327 = amdgpu.mfma %267 * %275 + %326 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %328 = amdgpu.mfma %266 * %276 + %297 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %329 = amdgpu.mfma %267 * %277 + %328 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %330 = amdgpu.mfma %266 * %278 + %299 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %331 = amdgpu.mfma %267 * %279 + %330 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %332 = amdgpu.mfma %266 * %280 + %301 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %333 = amdgpu.mfma %267 * %281 + %332 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %334 = amdgpu.mfma %266 * %282 + %303 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %335 = amdgpu.mfma %267 * %283 + %334 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %336 = amdgpu.mfma %266 * %284 + %305 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %337 = amdgpu.mfma %267 * %285 + %336 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %338 = amdgpu.mfma %268 * %270 + %307 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %339 = amdgpu.mfma %269 * %271 + %338 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %340 = amdgpu.mfma %268 * %272 + %309 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %341 = amdgpu.mfma %269 * %273 + %340 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %342 = amdgpu.mfma %268 * %274 + %311 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %343 = amdgpu.mfma %269 * %275 + %342 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %344 = amdgpu.mfma %268 * %276 + %313 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %345 = amdgpu.mfma %269 * %277 + %344 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %346 = amdgpu.mfma %268 * %278 + %315 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %347 = amdgpu.mfma %269 * %279 + %346 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %348 = amdgpu.mfma %268 * %280 + %317 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %349 = amdgpu.mfma %269 * %281 + %348 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %350 = amdgpu.mfma %268 * %282 + %319 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %351 = amdgpu.mfma %269 * %283 + %350 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %352 = amdgpu.mfma %268 * %284 + %321 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %353 = amdgpu.mfma %269 * %285 + %352 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          rocdl.s.setprio 0
          llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
          amdgpu.lds_barrier
          scf.yield %323, %325, %327, %329, %331, %333, %335, %337, %339, %341, %343, %345, %347, %349, %351, %353 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %42 = vector.broadcast %24 : i1 to vector<i1>
        %43 = vector.extractelement %42[] : vector<i1>
        scf.if %43 {
          rocdl.s.barrier
        }
        %44 = affine.apply #map18()[%thread_id_x, %thread_id_y]
        %45 = affine.apply #map15()[%thread_id_x]
        %46 = vector.load %view[%44, %45] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %47 = affine.apply #map16()[%thread_id_x]
        %48 = vector.load %view[%44, %47] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %49 = affine.apply #map26()[%thread_id_x]
        %50 = vector.load %view[%44, %49] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %51 = affine.apply #map27()[%thread_id_x]
        %52 = vector.load %view[%44, %51] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %53 = affine.apply #map19()[%thread_id_x, %thread_id_y]
        %54 = vector.load %view[%53, %45] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %55 = vector.load %view[%53, %47] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %56 = vector.load %view[%53, %49] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %57 = vector.load %view[%53, %51] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %58 = affine.apply #map20()[%thread_id_x, %thread_id_y]
        %59 = vector.load %view[%58, %45] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %60 = vector.load %view[%58, %47] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %61 = vector.load %view[%58, %49] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %62 = vector.load %view[%58, %51] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %63 = affine.apply #map21()[%thread_id_x, %thread_id_y]
        %64 = vector.load %view[%63, %45] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %65 = vector.load %view[%63, %47] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %66 = vector.load %view[%63, %49] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %67 = vector.load %view[%63, %51] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %68 = affine.apply #map22()[%thread_id_x, %thread_id_y]
        %69 = vector.load %view[%68, %45] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %70 = vector.load %view[%68, %47] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %71 = vector.load %view[%68, %49] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %72 = vector.load %view[%68, %51] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %73 = affine.apply #map23()[%thread_id_x, %thread_id_y]
        %74 = vector.load %view[%73, %45] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %75 = vector.load %view[%73, %47] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %76 = vector.load %view[%73, %49] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %77 = vector.load %view[%73, %51] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %78 = affine.apply #map24()[%thread_id_x, %thread_id_y]
        %79 = vector.load %view[%78, %45] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %80 = vector.load %view[%78, %47] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %81 = vector.load %view[%78, %49] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %82 = vector.load %view[%78, %51] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %83 = affine.apply #map25()[%thread_id_x, %thread_id_y]
        %84 = vector.load %view[%83, %45] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %85 = vector.load %view[%83, %47] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %86 = vector.load %view[%83, %49] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %87 = vector.load %view[%83, %51] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %88 = affine.apply #map14()[%thread_id_x]
        %89 = vector.load %view_0[%88, %45] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %90 = vector.load %view_0[%88, %47] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %91 = vector.load %view_0[%88, %49] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %92 = vector.load %view_0[%88, %51] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %93 = affine.apply #map17()[%thread_id_x]
        %94 = vector.load %view_0[%93, %45] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %95 = vector.load %view_0[%93, %47] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %96 = vector.load %view_0[%93, %49] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %97 = vector.load %view_0[%93, %51] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %98 = amdgpu.mfma %89 * %46 + %41#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %99 = amdgpu.mfma %90 * %48 + %98 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %100 = amdgpu.mfma %91 * %50 + %99 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %101 = amdgpu.mfma %92 * %52 + %100 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %102 = amdgpu.mfma %89 * %54 + %41#1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %103 = amdgpu.mfma %90 * %55 + %102 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %104 = amdgpu.mfma %91 * %56 + %103 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %105 = amdgpu.mfma %92 * %57 + %104 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %106 = amdgpu.mfma %89 * %59 + %41#2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %107 = amdgpu.mfma %90 * %60 + %106 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %108 = amdgpu.mfma %91 * %61 + %107 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %109 = amdgpu.mfma %92 * %62 + %108 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %110 = amdgpu.mfma %89 * %64 + %41#3 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %111 = amdgpu.mfma %90 * %65 + %110 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %112 = amdgpu.mfma %91 * %66 + %111 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %113 = amdgpu.mfma %92 * %67 + %112 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %114 = amdgpu.mfma %89 * %69 + %41#4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %115 = amdgpu.mfma %90 * %70 + %114 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %116 = amdgpu.mfma %91 * %71 + %115 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %117 = amdgpu.mfma %92 * %72 + %116 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %118 = amdgpu.mfma %89 * %74 + %41#5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %119 = amdgpu.mfma %90 * %75 + %118 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %120 = amdgpu.mfma %91 * %76 + %119 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %121 = amdgpu.mfma %92 * %77 + %120 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %122 = amdgpu.mfma %89 * %79 + %41#6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %123 = amdgpu.mfma %90 * %80 + %122 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %124 = amdgpu.mfma %91 * %81 + %123 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %125 = amdgpu.mfma %92 * %82 + %124 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %126 = amdgpu.mfma %89 * %84 + %41#7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %127 = amdgpu.mfma %90 * %85 + %126 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %128 = amdgpu.mfma %91 * %86 + %127 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %129 = amdgpu.mfma %92 * %87 + %128 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %130 = amdgpu.mfma %94 * %46 + %41#8 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %131 = amdgpu.mfma %95 * %48 + %130 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %132 = amdgpu.mfma %96 * %50 + %131 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %133 = amdgpu.mfma %97 * %52 + %132 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %134 = amdgpu.mfma %94 * %54 + %41#9 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %135 = amdgpu.mfma %95 * %55 + %134 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %136 = amdgpu.mfma %96 * %56 + %135 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %137 = amdgpu.mfma %97 * %57 + %136 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %138 = amdgpu.mfma %94 * %59 + %41#10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %139 = amdgpu.mfma %95 * %60 + %138 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %140 = amdgpu.mfma %96 * %61 + %139 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %141 = amdgpu.mfma %97 * %62 + %140 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %142 = amdgpu.mfma %94 * %64 + %41#11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %143 = amdgpu.mfma %95 * %65 + %142 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %144 = amdgpu.mfma %96 * %66 + %143 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %145 = amdgpu.mfma %97 * %67 + %144 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %146 = amdgpu.mfma %94 * %69 + %41#12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %147 = amdgpu.mfma %95 * %70 + %146 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %148 = amdgpu.mfma %96 * %71 + %147 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %149 = amdgpu.mfma %97 * %72 + %148 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %150 = amdgpu.mfma %94 * %74 + %41#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %151 = amdgpu.mfma %95 * %75 + %150 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %152 = amdgpu.mfma %96 * %76 + %151 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %153 = amdgpu.mfma %97 * %77 + %152 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %154 = amdgpu.mfma %94 * %79 + %41#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %155 = amdgpu.mfma %95 * %80 + %154 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %156 = amdgpu.mfma %96 * %81 + %155 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %157 = amdgpu.mfma %97 * %82 + %156 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %158 = amdgpu.mfma %94 * %84 + %41#15 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %159 = amdgpu.mfma %95 * %85 + %158 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %160 = amdgpu.mfma %96 * %86 + %159 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %161 = amdgpu.mfma %97 * %87 + %160 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %162 = vector.extract_strided_slice %101 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %163 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<4096x4096xf32, strided<[4096, 1], offset: ?>>
        %164 = affine.apply #map29()[%block_id_x, %thread_id_x]
        %165 = affine.apply #map30()[%thread_id_x, %block_id_y, %thread_id_y]
        vector.store %162, %163[%164, %165] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %166 = vector.extract_strided_slice %101 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %167 = affine.apply #map31()[%block_id_x, %thread_id_x]
        vector.store %166, %163[%167, %165] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %168 = vector.extract_strided_slice %101 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %169 = affine.apply #map32()[%block_id_x, %thread_id_x]
        vector.store %168, %163[%169, %165] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %170 = vector.extract_strided_slice %101 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %171 = affine.apply #map33()[%block_id_x, %thread_id_x]
        vector.store %170, %163[%171, %165] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %172 = vector.extract_strided_slice %105 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %173 = affine.apply #map34()[%thread_id_x, %block_id_y, %thread_id_y]
        vector.store %172, %163[%164, %173] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %174 = vector.extract_strided_slice %105 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %174, %163[%167, %173] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %175 = vector.extract_strided_slice %105 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %175, %163[%169, %173] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %176 = vector.extract_strided_slice %105 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %176, %163[%171, %173] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %177 = vector.extract_strided_slice %109 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %178 = affine.apply #map35()[%thread_id_x, %block_id_y, %thread_id_y]
        vector.store %177, %163[%164, %178] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %179 = vector.extract_strided_slice %109 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %179, %163[%167, %178] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %180 = vector.extract_strided_slice %109 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %180, %163[%169, %178] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %181 = vector.extract_strided_slice %109 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %181, %163[%171, %178] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %182 = vector.extract_strided_slice %113 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %183 = affine.apply #map36()[%thread_id_x, %block_id_y, %thread_id_y]
        vector.store %182, %163[%164, %183] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %184 = vector.extract_strided_slice %113 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %184, %163[%167, %183] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %185 = vector.extract_strided_slice %113 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %185, %163[%169, %183] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %186 = vector.extract_strided_slice %113 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %186, %163[%171, %183] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %187 = vector.extract_strided_slice %117 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %188 = affine.apply #map37()[%thread_id_x, %block_id_y, %thread_id_y]
        vector.store %187, %163[%164, %188] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %189 = vector.extract_strided_slice %117 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %189, %163[%167, %188] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %190 = vector.extract_strided_slice %117 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %190, %163[%169, %188] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %191 = vector.extract_strided_slice %117 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %191, %163[%171, %188] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %192 = vector.extract_strided_slice %121 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %193 = affine.apply #map38()[%thread_id_x, %block_id_y, %thread_id_y]
        vector.store %192, %163[%164, %193] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %194 = vector.extract_strided_slice %121 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %194, %163[%167, %193] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %195 = vector.extract_strided_slice %121 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %195, %163[%169, %193] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %196 = vector.extract_strided_slice %121 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %196, %163[%171, %193] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %197 = vector.extract_strided_slice %125 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %198 = affine.apply #map39()[%thread_id_x, %block_id_y, %thread_id_y]
        vector.store %197, %163[%164, %198] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %199 = vector.extract_strided_slice %125 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %199, %163[%167, %198] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %200 = vector.extract_strided_slice %125 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %200, %163[%169, %198] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %201 = vector.extract_strided_slice %125 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %201, %163[%171, %198] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %202 = vector.extract_strided_slice %129 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %203 = affine.apply #map40()[%thread_id_x, %block_id_y, %thread_id_y]
        vector.store %202, %163[%164, %203] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %204 = vector.extract_strided_slice %129 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %204, %163[%167, %203] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %205 = vector.extract_strided_slice %129 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %205, %163[%169, %203] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %206 = vector.extract_strided_slice %129 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %206, %163[%171, %203] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %207 = vector.extract_strided_slice %133 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %208 = affine.apply #map41()[%block_id_x, %thread_id_x]
        vector.store %207, %163[%208, %165] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %209 = vector.extract_strided_slice %133 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %210 = affine.apply #map42()[%block_id_x, %thread_id_x]
        vector.store %209, %163[%210, %165] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %211 = vector.extract_strided_slice %133 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %212 = affine.apply #map43()[%block_id_x, %thread_id_x]
        vector.store %211, %163[%212, %165] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %213 = vector.extract_strided_slice %133 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %214 = affine.apply #map44()[%block_id_x, %thread_id_x]
        vector.store %213, %163[%214, %165] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %215 = vector.extract_strided_slice %137 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %215, %163[%208, %173] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %216 = vector.extract_strided_slice %137 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %216, %163[%210, %173] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %217 = vector.extract_strided_slice %137 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %217, %163[%212, %173] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %218 = vector.extract_strided_slice %137 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %218, %163[%214, %173] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %219 = vector.extract_strided_slice %141 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %219, %163[%208, %178] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %220 = vector.extract_strided_slice %141 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %220, %163[%210, %178] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %221 = vector.extract_strided_slice %141 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %221, %163[%212, %178] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %222 = vector.extract_strided_slice %141 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %222, %163[%214, %178] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %223 = vector.extract_strided_slice %145 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %223, %163[%208, %183] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %224 = vector.extract_strided_slice %145 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %224, %163[%210, %183] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %225 = vector.extract_strided_slice %145 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %225, %163[%212, %183] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %226 = vector.extract_strided_slice %145 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %226, %163[%214, %183] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %227 = vector.extract_strided_slice %149 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %227, %163[%208, %188] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %228 = vector.extract_strided_slice %149 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %228, %163[%210, %188] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %229 = vector.extract_strided_slice %149 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %229, %163[%212, %188] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %230 = vector.extract_strided_slice %149 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %230, %163[%214, %188] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %231 = vector.extract_strided_slice %153 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %231, %163[%208, %193] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %232 = vector.extract_strided_slice %153 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %232, %163[%210, %193] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %233 = vector.extract_strided_slice %153 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %233, %163[%212, %193] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %234 = vector.extract_strided_slice %153 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %234, %163[%214, %193] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %235 = vector.extract_strided_slice %157 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %235, %163[%208, %198] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %236 = vector.extract_strided_slice %157 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %236, %163[%210, %198] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %237 = vector.extract_strided_slice %157 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %237, %163[%212, %198] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %238 = vector.extract_strided_slice %157 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %238, %163[%214, %198] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %239 = vector.extract_strided_slice %161 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %239, %163[%208, %203] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %240 = vector.extract_strided_slice %161 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %240, %163[%210, %203] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %241 = vector.extract_strided_slice %161 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %241, %163[%212, %203] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        %242 = vector.extract_strided_slice %161 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %242, %163[%214, %203] : memref<4096x4096xf32, strided<[4096, 1], offset: ?>>, vector<1xf32>
        return
      }
    }
  }
  func.func @main(
      %arg0: tensor<8192x4096xf16>,
      %arg1: tensor<4096x4096xf16>,
      %arg2: tensor<8192x4096xf32>
  ) -> (tensor<4096x4096xf32>, tensor<4096x4096xf32>) {
    %c4096 = arith.constant 4096 : index
    %c0 = arith.constant 0 : index
    %3 = flow.tensor.slice %arg0[%c0, %c0 for %c4096, %c4096] : tensor<8192x4096xf16> -> tensor<4096x4096xf16>
    %4 = flow.tensor.slice %arg0[%c4096, %c0 for %c4096, %c4096] : tensor<8192x4096xf16> -> tensor<4096x4096xf16>
    %9 = flow.tensor.slice %arg2[%c0, %c0 for %c4096, %c4096] : tensor<8192x4096xf32> -> tensor<4096x4096xf32>
    %10 = flow.tensor.slice %arg2[%c4096, %c0 for %c4096, %c4096] : tensor<8192x4096xf32> -> tensor<4096x4096xf32>

    // Dispatch to device 0
    %13 = flow.dispatch @gemm::@gemm(%3, %arg1, %9) {
      stream.affinity = #hal.device.affinity<@__device_0>
    } : (tensor<4096x4096xf16>, tensor<4096x4096xf16>, tensor<4096x4096xf32>) -> %9

    // Dispatch to device 1
    %14 = flow.dispatch @gemm::@gemm(%4, %arg1, %10) {
      stream.affinity = #hal.device.affinity<@__device_1>
    } : (tensor<4096x4096xf16>, tensor<4096x4096xf16>, tensor<4096x4096xf32>) -> %10

    return %13, %14 : tensor<4096x4096xf32>, tensor<4096x4096xf32>
  }
}
