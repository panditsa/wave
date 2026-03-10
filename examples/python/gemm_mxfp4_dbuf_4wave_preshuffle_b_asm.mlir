module {
func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 4, 1] subgroup_size = 64>} {
  %c1024_i14 = arith.constant 1024 : i14
  %c256_i14 = arith.constant 256 : i14
  %c4096_i14 = arith.constant 4096 : i14
  %c2147483643_i64 = arith.constant 2147483643 : i64
  %c30 = arith.constant 30 : index
  %c1024 = arith.constant 1024 : index
  %c2 = arith.constant 2 : index
  %c256 = arith.constant 256 : index
  %c3 = arith.constant 3 : index
  %c2147483646_i64 = arith.constant 2147483646 : i64
  %c4096 = arith.constant 4096 : index
  %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
  %c0 = arith.constant 0 : index
  %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<i8>
  %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<i8>
  %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<i8>
  %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<i8>
  %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<f32>
  %block_id_x = gpu.block_id  x upper_bound 8
  %block_id_y = gpu.block_id  y upper_bound 4
  %thread_id_x = gpu.thread_id  x upper_bound 64
  %thread_id_y = gpu.thread_id  y upper_bound 4
  %alloc = memref.alloc() : memref<1024x1xi8, #gpu.address_space<workgroup>>
  %alloc_0 = memref.alloc() : memref<1024x1xi8, #gpu.address_space<workgroup>>
  %alloc_1 = memref.alloc() : memref<128x128xi8, #gpu.address_space<workgroup>>
  %alloc_2 = memref.alloc() : memref<128x128xi8, #gpu.address_space<workgroup>>
  %5 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 128 + s0 floordiv 8 - ((s1 * 8 + s0 floordiv 8) floordiv 128) * 128)>()[%thread_id_x, %thread_id_y, %block_id_x]
  %6 = affine.apply affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>()[%thread_id_x]
  %7 = affine.apply affine_map<()[s0] -> (s0 mod 8)>()[%thread_id_x]
  %8 = arith.xori %7, %6 : index
  %9 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%8]
  %10 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 16) * 128)>()[%thread_id_y]
  %11 = gpu.subgroup_broadcast %10,  first_active_lane : index
  %12 = gpu.subgroup_broadcast %c0,  first_active_lane : index
  %13 = arith.muli %5, %c4096 overflow<nsw> : index
  %14 = arith.addi %13, %9 overflow<nsw> : index
  %reinterpret_cast = memref.reinterpret_cast %0 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
  %cast = memref.cast %reinterpret_cast : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
  %15 = amdgpu.fat_raw_buffer_cast %cast validBytes(%c2147483646_i64) cacheSwizzleStride(%c4096_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
  amdgpu.gather_to_lds %15[%14], %alloc_2[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
  %16 = affine.apply affine_map<()[s0] -> (s0 * 16 + 128)>()[%8]
  %17 = arith.addi %13, %16 overflow<nsw> : index
  amdgpu.gather_to_lds %15[%17], %alloc_1[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
  %18 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 128 + s0 floordiv 8 - ((s1 * 8 + s0 floordiv 8 + 32) floordiv 128) * 128 + 32)>()[%thread_id_x, %thread_id_y, %block_id_x]
  %19 = affine.apply affine_map<()[s0] -> (s0 * 8 - ((s0 + 4) floordiv 16) * 128 + 32)>()[%thread_id_y]
  %20 = gpu.subgroup_broadcast %19,  first_active_lane : index
  %21 = arith.muli %18, %c4096 overflow<nsw> : index
  %22 = arith.addi %21, %9 overflow<nsw> : index
  amdgpu.gather_to_lds %15[%22], %alloc_2[%20, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
  %23 = arith.addi %21, %16 overflow<nsw> : index
  amdgpu.gather_to_lds %15[%23], %alloc_1[%20, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
  %24 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 128 + s0 floordiv 8 - ((s1 * 8 + s0 floordiv 8 + 64) floordiv 128) * 128 + 64)>()[%thread_id_x, %thread_id_y, %block_id_x]
  %25 = affine.apply affine_map<()[s0] -> (s0 * 8 - ((s0 + 8) floordiv 16) * 128 + 64)>()[%thread_id_y]
  %26 = gpu.subgroup_broadcast %25,  first_active_lane : index
  %27 = arith.muli %24, %c4096 overflow<nsw> : index
  %28 = arith.addi %27, %9 overflow<nsw> : index
  amdgpu.gather_to_lds %15[%28], %alloc_2[%26, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
  %29 = arith.addi %27, %16 overflow<nsw> : index
  amdgpu.gather_to_lds %15[%29], %alloc_1[%26, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
  %30 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 128 + s0 floordiv 8 - ((s1 * 8 + s0 floordiv 8 + 96) floordiv 128) * 128 + 96)>()[%thread_id_x, %thread_id_y, %block_id_x]
  %31 = affine.apply affine_map<()[s0] -> (s0 * 8 - ((s0 + 12) floordiv 16) * 128 + 96)>()[%thread_id_y]
  %32 = gpu.subgroup_broadcast %31,  first_active_lane : index
  %33 = arith.muli %30, %c4096 overflow<nsw> : index
  %34 = arith.addi %33, %9 overflow<nsw> : index
  amdgpu.gather_to_lds %15[%34], %alloc_2[%32, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
  %35 = arith.addi %33, %16 overflow<nsw> : index
  amdgpu.gather_to_lds %15[%35], %alloc_1[%32, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
  %36 = affine.apply affine_map<()[s0, s1] -> (s1 + s0 floordiv 64)>()[%thread_id_x, %thread_id_y]
  %37 = arith.minsi %36, %c3 : index
  %38 = affine.apply affine_map<()[s0, s1] -> (s0 * 128 + s1 * 32)>()[%block_id_x, %37]
  %39 = affine.apply affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 64) * 256)>()[%thread_id_x]
  %40 = arith.minsi %thread_id_y, %c3 : index
  %41 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%40]
  %42 = gpu.subgroup_broadcast %41,  first_active_lane : index
  %43 = arith.muli %38, %c256 overflow<nsw> : index
  %44 = arith.addi %43, %39 overflow<nsw> : index
  %reinterpret_cast_3 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
  %cast_4 = memref.cast %reinterpret_cast_3 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
  %45 = amdgpu.fat_raw_buffer_cast %cast_4 validBytes(%c2147483646_i64) cacheSwizzleStride(%c256_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
  amdgpu.gather_to_lds %45[%44], %alloc_0[%42, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<1024x1xi8, #gpu.address_space<workgroup>>
  %46 = affine.apply affine_map<()[s0, s1] -> (s0 * 128 + s1 * 32 + 1)>()[%block_id_x, %37]
  %47 = arith.muli %46, %c256 overflow<nsw> : index
  %48 = arith.addi %47, %39 overflow<nsw> : index
  amdgpu.gather_to_lds %45[%48], %alloc[%42, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<1024x1xi8, #gpu.address_space<workgroup>>
  %49 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%block_id_y]
  %50 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256)>()[%thread_id_x, %thread_id_y]
  %51 = affine.apply affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256) * 4096)>()[%thread_id_x]
  %52 = arith.muli %49, %c4096 overflow<nsw> : index
  %53 = arith.muli %50, %c4096 overflow<nsw> : index
  %54 = arith.addi %53, %51 overflow<nsw> : index
  %reinterpret_cast_5 = memref.reinterpret_cast %2 to offset: [%52], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1], offset: ?>>
  %cast_6 = memref.cast %reinterpret_cast_5 : memref<2147483646xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
  %55 = amdgpu.fat_raw_buffer_cast %cast_6 validBytes(%c2147483646_i64) cacheSwizzleStride(%c4096_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
  %56 = vector.load %55[%54] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %57 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256)>()[%thread_id_x, %thread_id_y]
  %58 = affine.apply affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256) * 4096 + 1024)>()[%thread_id_x]
  %59 = arith.muli %57, %c4096 overflow<nsw> : index
  %60 = arith.addi %59, %58 overflow<nsw> : index
  %61 = vector.load %55[%60] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %62 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 16)>()[%thread_id_x, %thread_id_y]
  %63 = arith.muli %62, %c4096 overflow<nsw> : index
  %64 = arith.addi %63, %51 overflow<nsw> : index
  %65 = vector.load %55[%64] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %66 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256 + 16)>()[%thread_id_x, %thread_id_y]
  %67 = arith.muli %66, %c4096 overflow<nsw> : index
  %68 = arith.addi %67, %58 overflow<nsw> : index
  %69 = vector.load %55[%68] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %70 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 32)>()[%thread_id_x, %thread_id_y]
  %71 = arith.muli %70, %c4096 overflow<nsw> : index
  %72 = arith.addi %71, %51 overflow<nsw> : index
  %73 = vector.load %55[%72] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %74 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256 + 32)>()[%thread_id_x, %thread_id_y]
  %75 = arith.muli %74, %c4096 overflow<nsw> : index
  %76 = arith.addi %75, %58 overflow<nsw> : index
  %77 = vector.load %55[%76] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %78 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 48)>()[%thread_id_x, %thread_id_y]
  %79 = arith.muli %78, %c4096 overflow<nsw> : index
  %80 = arith.addi %79, %51 overflow<nsw> : index
  %81 = vector.load %55[%80] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %82 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256 + 48)>()[%thread_id_x, %thread_id_y]
  %83 = arith.muli %82, %c4096 overflow<nsw> : index
  %84 = arith.addi %83, %58 overflow<nsw> : index
  %85 = vector.load %55[%84] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  rocdl.sched.barrier 0
  %86 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64)>()[%thread_id_x, %thread_id_y]
  %87 = affine.apply affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 16) * 64 + ((s0 mod 64) floordiv 16) * 64 - ((s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64) * 256)>()[%thread_id_x]
  %88 = arith.muli %49, %c256 overflow<nsw> : index
  %89 = arith.muli %86, %c256 overflow<nsw> : index
  %90 = arith.addi %89, %87 overflow<nsw> : index
  %reinterpret_cast_7 = memref.reinterpret_cast %3 to offset: [%88], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1], offset: ?>>
  %cast_8 = memref.cast %reinterpret_cast_7 : memref<2147483646xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
  %91 = amdgpu.fat_raw_buffer_cast %cast_8 validBytes(%c2147483646_i64) cacheSwizzleStride(%c256_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
  %92 = vector.load %91[%90] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
  %93 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64 + 32)>()[%thread_id_x, %thread_id_y]
  %94 = arith.muli %93, %c256 overflow<nsw> : index
  %95 = arith.addi %94, %87 overflow<nsw> : index
  %96 = vector.load %91[%95] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
  rocdl.sched.barrier 0
  amdgpu.memory_counter_wait load(0)
  rocdl.s.barrier
  rocdl.sched.barrier 0
  %97 = affine.apply affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 128)>()[%thread_id_x]
  %98 = affine.apply affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>()[%thread_id_x]
  %99 = arith.xori %98, %7 : index
  %100 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%99]
  %101 = vector.load %alloc_2[%97, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %102 = affine.apply affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>()[%thread_id_x]
  %103 = arith.xori %102, %7 : index
  %104 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%103]
  %105 = vector.load %alloc_2[%97, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %106 = affine.apply affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 128 + 16)>()[%thread_id_x]
  %107 = vector.load %alloc_2[%106, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %108 = vector.load %alloc_2[%106, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %109 = affine.apply affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 128 + 32)>()[%thread_id_x]
  %110 = vector.load %alloc_2[%109, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %111 = vector.load %alloc_2[%109, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %112 = affine.apply affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 128 + 48)>()[%thread_id_x]
  %113 = vector.load %alloc_2[%112, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %114 = vector.load %alloc_2[%112, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %115 = affine.apply affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 768)>()[%thread_id_x]
  %116 = vector.load %alloc_0[%c0, %115] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
  %117 = affine.apply affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 768 + 256)>()[%thread_id_x]
  %118 = vector.load %alloc_0[%c0, %117] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
  %119 = affine.apply affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 128 + 64)>()[%thread_id_x]
  %120 = affine.apply affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 128 + 80)>()[%thread_id_x]
  %121 = affine.apply affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 128 + 96)>()[%thread_id_x]
  %122 = affine.apply affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 128 + 112)>()[%thread_id_x]
  %123 = affine.apply affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 768 + 512)>()[%thread_id_x]
  %124 = affine.apply affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 768 + 768)>()[%thread_id_x]
  %125:52 = scf.for %arg5 = %c0 to %c30 step %c2 iter_args(%arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst, %arg37 = %cst, %arg38 = %116, %arg39 = %118, %arg40 = %101, %arg41 = %105, %arg42 = %107, %arg43 = %108, %arg44 = %110, %arg45 = %111, %arg46 = %113, %arg47 = %114, %arg48 = %56, %arg49 = %61, %arg50 = %65, %arg51 = %69, %arg52 = %73, %arg53 = %77, %arg54 = %81, %arg55 = %85, %arg56 = %92, %arg57 = %96) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<4xi8>, vector<4xi8>) {
    %714 = vector.bitcast %arg39 : vector<4xi8> to vector<4xf8E8M0FNU>
    %715 = vector.bitcast %arg57 : vector<4xi8> to vector<4xf8E8M0FNU>
    %716 = vector.bitcast %arg56 : vector<4xi8> to vector<4xf8E8M0FNU>
    %717 = vector.bitcast %arg38 : vector<4xi8> to vector<4xf8E8M0FNU>
    amdgpu.memory_counter_wait load(5) ds(0)
    rocdl.s.barrier
    %718 = vector.bitcast %arg40 : vector<16xi8> to vector<32xf4E2M1FN>
    %719 = vector.bitcast %arg41 : vector<16xi8> to vector<32xf4E2M1FN>
    %720 = vector.bitcast %arg42 : vector<16xi8> to vector<32xf4E2M1FN>
    %721 = vector.bitcast %arg43 : vector<16xi8> to vector<32xf4E2M1FN>
    %722 = vector.bitcast %arg44 : vector<16xi8> to vector<32xf4E2M1FN>
    %723 = vector.bitcast %arg45 : vector<16xi8> to vector<32xf4E2M1FN>
    %724 = vector.bitcast %arg46 : vector<16xi8> to vector<32xf4E2M1FN>
    %725 = vector.bitcast %arg47 : vector<16xi8> to vector<32xf4E2M1FN>
    %726 = vector.bitcast %arg48 : vector<16xi8> to vector<32xf4E2M1FN>
    %727 = vector.bitcast %arg49 : vector<16xi8> to vector<32xf4E2M1FN>
    %728 = vector.bitcast %arg50 : vector<16xi8> to vector<32xf4E2M1FN>
    %729 = vector.bitcast %arg51 : vector<16xi8> to vector<32xf4E2M1FN>
    %730 = vector.bitcast %arg52 : vector<16xi8> to vector<32xf4E2M1FN>
    %731 = vector.bitcast %arg53 : vector<16xi8> to vector<32xf4E2M1FN>
    %732 = vector.bitcast %arg54 : vector<16xi8> to vector<32xf4E2M1FN>
    %733 = vector.bitcast %arg55 : vector<16xi8> to vector<32xf4E2M1FN>
    rocdl.sched.barrier 0
    %734 = amdgpu.scaled_mfma 16x16x128 (%717[0] * %718) * (%716[0] * %726) + %arg6 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %735 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 128) floordiv 256)>()[%thread_id_x, %arg5, %thread_id_y]
    %736 = affine.apply affine_map<()[s0, s1] -> (s0 * 16 + s1 * 2048 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 128) floordiv 256) * 4096 + 2048)>()[%thread_id_x, %arg5]
    %737 = arith.muli %735, %c4096 overflow<nsw> : index
    %738 = arith.addi %737, %736 overflow<nsw> : index
    %739 = vector.load %55[%738] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %740 = amdgpu.scaled_mfma 16x16x128 (%717[2] * %719) * (%716[2] * %727) + %734 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %741 = amdgpu.scaled_mfma 16x16x128 (%717[0] * %718) * (%716[1] * %728) + %arg7 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %742 = amdgpu.scaled_mfma 16x16x128 (%717[2] * %719) * (%716[3] * %729) + %741 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %743 = vector.load %alloc_2[%119, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %744 = amdgpu.scaled_mfma 16x16x128 (%717[0] * %718) * (%715[0] * %730) + %arg8 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %745 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 192) floordiv 256)>()[%thread_id_x, %arg5, %thread_id_y]
    %746 = affine.apply affine_map<()[s0, s1] -> (s0 * 16 + s1 * 2048 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 192) floordiv 256) * 4096 + 3072)>()[%thread_id_x, %arg5]
    %747 = arith.muli %745, %c4096 overflow<nsw> : index
    %748 = arith.addi %747, %746 overflow<nsw> : index
    %749 = vector.load %55[%748] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %750 = amdgpu.scaled_mfma 16x16x128 (%717[2] * %719) * (%715[2] * %731) + %744 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %751 = amdgpu.scaled_mfma 16x16x128 (%717[0] * %718) * (%715[1] * %732) + %arg9 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %752 = amdgpu.scaled_mfma 16x16x128 (%717[2] * %719) * (%715[3] * %733) + %751 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %753 = vector.load %alloc_2[%119, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %754 = amdgpu.scaled_mfma 16x16x128 (%717[1] * %720) * (%716[0] * %726) + %arg10 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %755 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 128) floordiv 256 + 16)>()[%thread_id_x, %arg5, %thread_id_y]
    %756 = arith.muli %755, %c4096 overflow<nsw> : index
    %757 = arith.addi %756, %736 overflow<nsw> : index
    %758 = vector.load %55[%757] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %759 = amdgpu.scaled_mfma 16x16x128 (%717[3] * %721) * (%716[2] * %727) + %754 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %760 = amdgpu.scaled_mfma 16x16x128 (%717[1] * %720) * (%716[1] * %728) + %arg11 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %761 = amdgpu.scaled_mfma 16x16x128 (%717[3] * %721) * (%716[3] * %729) + %760 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %762 = vector.load %alloc_2[%120, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %763 = amdgpu.scaled_mfma 16x16x128 (%717[1] * %720) * (%715[0] * %730) + %arg12 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %764 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 192) floordiv 256 + 16)>()[%thread_id_x, %arg5, %thread_id_y]
    %765 = arith.muli %764, %c4096 overflow<nsw> : index
    %766 = arith.addi %765, %746 overflow<nsw> : index
    %767 = vector.load %55[%766] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %768 = amdgpu.scaled_mfma 16x16x128 (%717[3] * %721) * (%715[2] * %731) + %763 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %769 = amdgpu.scaled_mfma 16x16x128 (%717[1] * %720) * (%715[1] * %732) + %arg13 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %770 = amdgpu.scaled_mfma 16x16x128 (%717[3] * %721) * (%715[3] * %733) + %769 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %771 = vector.load %alloc_2[%120, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %772 = amdgpu.scaled_mfma 16x16x128 (%714[0] * %722) * (%716[0] * %726) + %arg14 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %773 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 128) floordiv 256 + 32)>()[%thread_id_x, %arg5, %thread_id_y]
    %774 = arith.muli %773, %c4096 overflow<nsw> : index
    %775 = arith.addi %774, %736 overflow<nsw> : index
    %776 = vector.load %55[%775] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %777 = amdgpu.scaled_mfma 16x16x128 (%714[2] * %723) * (%716[2] * %727) + %772 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %778 = amdgpu.scaled_mfma 16x16x128 (%714[0] * %722) * (%716[1] * %728) + %arg15 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %779 = amdgpu.scaled_mfma 16x16x128 (%714[2] * %723) * (%716[3] * %729) + %778 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %780 = vector.load %alloc_2[%121, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %781 = amdgpu.scaled_mfma 16x16x128 (%714[0] * %722) * (%715[0] * %730) + %arg16 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %782 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 192) floordiv 256 + 32)>()[%thread_id_x, %arg5, %thread_id_y]
    %783 = arith.muli %782, %c4096 overflow<nsw> : index
    %784 = arith.addi %783, %746 overflow<nsw> : index
    %785 = vector.load %55[%784] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %786 = amdgpu.scaled_mfma 16x16x128 (%714[2] * %723) * (%715[2] * %731) + %781 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %787 = amdgpu.scaled_mfma 16x16x128 (%714[0] * %722) * (%715[1] * %732) + %arg17 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %788 = amdgpu.scaled_mfma 16x16x128 (%714[2] * %723) * (%715[3] * %733) + %787 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %789 = vector.load %alloc_2[%121, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %790 = amdgpu.scaled_mfma 16x16x128 (%714[1] * %724) * (%716[0] * %726) + %arg18 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %791 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 128) floordiv 256 + 48)>()[%thread_id_x, %arg5, %thread_id_y]
    %792 = arith.muli %791, %c4096 overflow<nsw> : index
    %793 = arith.addi %792, %736 overflow<nsw> : index
    %794 = vector.load %55[%793] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %795 = amdgpu.scaled_mfma 16x16x128 (%714[3] * %725) * (%716[2] * %727) + %790 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %796 = amdgpu.scaled_mfma 16x16x128 (%714[1] * %724) * (%716[1] * %728) + %arg19 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %797 = amdgpu.scaled_mfma 16x16x128 (%714[3] * %725) * (%716[3] * %729) + %796 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %798 = vector.load %alloc_2[%122, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %799 = amdgpu.scaled_mfma 16x16x128 (%714[1] * %724) * (%715[0] * %730) + %arg20 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %800 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 192) floordiv 256 + 48)>()[%thread_id_x, %arg5, %thread_id_y]
    %801 = arith.muli %800, %c4096 overflow<nsw> : index
    %802 = arith.addi %801, %746 overflow<nsw> : index
    %803 = vector.load %55[%802] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %804 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 64 + s2 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64 + 1)>()[%thread_id_x, %thread_id_y, %arg5]
    %805 = arith.muli %804, %c256 overflow<nsw> : index
    %806 = arith.addi %805, %87 overflow<nsw> : index
    %807 = vector.load %91[%806] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
    %808 = vector.bitcast %807 : vector<4xi8> to vector<4xf8E8M0FNU>
    %809 = amdgpu.scaled_mfma 16x16x128 (%714[3] * %725) * (%715[2] * %731) + %799 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %810 = amdgpu.scaled_mfma 16x16x128 (%714[1] * %724) * (%715[1] * %732) + %arg21 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %811 = amdgpu.scaled_mfma 16x16x128 (%714[3] * %725) * (%715[3] * %733) + %810 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %812 = vector.load %alloc_2[%122, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %813 = vector.load %alloc_0[%c0, %123] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    %814 = vector.bitcast %813 : vector<4xi8> to vector<4xf8E8M0FNU>
    %815 = vector.load %alloc_0[%c0, %124] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    %816 = vector.bitcast %815 : vector<4xi8> to vector<4xf8E8M0FNU>
    %817 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 64 + s2 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64 + 33)>()[%thread_id_x, %thread_id_y, %arg5]
    %818 = arith.muli %817, %c256 overflow<nsw> : index
    %819 = arith.addi %818, %87 overflow<nsw> : index
    %820 = vector.load %91[%819] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
    %821 = vector.bitcast %820 : vector<4xi8> to vector<4xf8E8M0FNU>
    rocdl.sched.barrier 0
    amdgpu.memory_counter_wait load(10) ds(0)
    rocdl.s.barrier
    rocdl.sched.barrier 0
    %822 = vector.bitcast %743 : vector<16xi8> to vector<32xf4E2M1FN>
    %823 = vector.bitcast %753 : vector<16xi8> to vector<32xf4E2M1FN>
    %824 = vector.bitcast %762 : vector<16xi8> to vector<32xf4E2M1FN>
    %825 = vector.bitcast %771 : vector<16xi8> to vector<32xf4E2M1FN>
    %826 = vector.bitcast %780 : vector<16xi8> to vector<32xf4E2M1FN>
    %827 = vector.bitcast %789 : vector<16xi8> to vector<32xf4E2M1FN>
    %828 = vector.bitcast %798 : vector<16xi8> to vector<32xf4E2M1FN>
    %829 = vector.bitcast %812 : vector<16xi8> to vector<32xf4E2M1FN>
    rocdl.sched.barrier 0
    %830 = amdgpu.scaled_mfma 16x16x128 (%814[0] * %822) * (%716[0] * %726) + %arg22 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %831 = affine.apply affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + 256)>()[%arg5, %8]
    %832 = arith.addi %13, %831 overflow<nsw> : index
    amdgpu.gather_to_lds %15[%832], %alloc_2[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
    %833 = amdgpu.scaled_mfma 16x16x128 (%814[2] * %823) * (%716[2] * %727) + %830 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %834 = amdgpu.scaled_mfma 16x16x128 (%814[0] * %822) * (%716[1] * %728) + %arg23 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %835 = amdgpu.scaled_mfma 16x16x128 (%814[2] * %823) * (%716[3] * %729) + %834 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %836 = vector.load %alloc_1[%97, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %837 = amdgpu.scaled_mfma 16x16x128 (%814[0] * %822) * (%715[0] * %730) + %arg24 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %838 = arith.addi %21, %831 overflow<nsw> : index
    amdgpu.gather_to_lds %15[%838], %alloc_2[%20, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
    %839 = amdgpu.scaled_mfma 16x16x128 (%814[2] * %823) * (%715[2] * %731) + %837 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %840 = amdgpu.scaled_mfma 16x16x128 (%814[0] * %822) * (%715[1] * %732) + %arg25 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %841 = amdgpu.scaled_mfma 16x16x128 (%814[2] * %823) * (%715[3] * %733) + %840 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %842 = vector.load %alloc_1[%97, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %843 = amdgpu.scaled_mfma 16x16x128 (%814[1] * %824) * (%716[0] * %726) + %arg26 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %844 = arith.addi %27, %831 overflow<nsw> : index
    amdgpu.gather_to_lds %15[%844], %alloc_2[%26, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
    %845 = amdgpu.scaled_mfma 16x16x128 (%814[3] * %825) * (%716[2] * %727) + %843 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %846 = amdgpu.scaled_mfma 16x16x128 (%814[1] * %824) * (%716[1] * %728) + %arg27 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %847 = amdgpu.scaled_mfma 16x16x128 (%814[3] * %825) * (%716[3] * %729) + %846 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %848 = vector.load %alloc_1[%106, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %849 = amdgpu.scaled_mfma 16x16x128 (%814[1] * %824) * (%715[0] * %730) + %arg28 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %850 = arith.addi %33, %831 overflow<nsw> : index
    amdgpu.gather_to_lds %15[%850], %alloc_2[%32, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
    %851 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * 128 + s1 * 32 + s2 + 2)>()[%block_id_x, %37, %arg5]
    %852 = arith.muli %851, %c256 overflow<nsw> : index
    %853 = arith.addi %852, %39 overflow<nsw> : index
    amdgpu.gather_to_lds %45[%853], %alloc_0[%42, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<1024x1xi8, #gpu.address_space<workgroup>>
    %854 = amdgpu.scaled_mfma 16x16x128 (%814[3] * %825) * (%715[2] * %731) + %849 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %855 = amdgpu.scaled_mfma 16x16x128 (%814[1] * %824) * (%715[1] * %732) + %arg29 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %856 = amdgpu.scaled_mfma 16x16x128 (%814[3] * %825) * (%715[3] * %733) + %855 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %857 = vector.load %alloc_1[%106, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %858 = amdgpu.scaled_mfma 16x16x128 (%816[0] * %826) * (%716[0] * %726) + %arg30 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %859 = amdgpu.scaled_mfma 16x16x128 (%816[2] * %827) * (%716[2] * %727) + %858 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %860 = amdgpu.scaled_mfma 16x16x128 (%816[0] * %826) * (%716[1] * %728) + %arg31 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %861 = amdgpu.scaled_mfma 16x16x128 (%816[2] * %827) * (%716[3] * %729) + %860 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %862 = vector.load %alloc_1[%109, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %863 = amdgpu.scaled_mfma 16x16x128 (%816[0] * %826) * (%715[0] * %730) + %arg32 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %864 = amdgpu.scaled_mfma 16x16x128 (%816[2] * %827) * (%715[2] * %731) + %863 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %865 = amdgpu.scaled_mfma 16x16x128 (%816[0] * %826) * (%715[1] * %732) + %arg33 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %866 = amdgpu.scaled_mfma 16x16x128 (%816[2] * %827) * (%715[3] * %733) + %865 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %867 = vector.load %alloc_1[%109, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %868 = amdgpu.scaled_mfma 16x16x128 (%816[1] * %828) * (%716[0] * %726) + %arg34 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %869 = amdgpu.scaled_mfma 16x16x128 (%816[3] * %829) * (%716[2] * %727) + %868 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %870 = amdgpu.scaled_mfma 16x16x128 (%816[1] * %828) * (%716[1] * %728) + %arg35 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %871 = amdgpu.scaled_mfma 16x16x128 (%816[3] * %829) * (%716[3] * %729) + %870 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %872 = vector.load %alloc_1[%112, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %873 = amdgpu.scaled_mfma 16x16x128 (%816[1] * %828) * (%715[0] * %730) + %arg36 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %874 = amdgpu.scaled_mfma 16x16x128 (%816[3] * %829) * (%715[2] * %731) + %873 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %875 = amdgpu.scaled_mfma 16x16x128 (%816[1] * %828) * (%715[1] * %732) + %arg37 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %876 = amdgpu.scaled_mfma 16x16x128 (%816[3] * %829) * (%715[3] * %733) + %875 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %877 = vector.load %alloc_1[%112, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %878 = vector.load %alloc[%c0, %115] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    %879 = vector.bitcast %878 : vector<4xi8> to vector<4xf8E8M0FNU>
    %880 = vector.load %alloc[%c0, %117] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    %881 = vector.bitcast %880 : vector<4xi8> to vector<4xf8E8M0FNU>
    rocdl.sched.barrier 0
    amdgpu.memory_counter_wait load(5) ds(0)
    rocdl.s.barrier
    rocdl.sched.barrier 0
    %882 = vector.bitcast %836 : vector<16xi8> to vector<32xf4E2M1FN>
    %883 = vector.bitcast %842 : vector<16xi8> to vector<32xf4E2M1FN>
    %884 = vector.bitcast %848 : vector<16xi8> to vector<32xf4E2M1FN>
    %885 = vector.bitcast %857 : vector<16xi8> to vector<32xf4E2M1FN>
    %886 = vector.bitcast %862 : vector<16xi8> to vector<32xf4E2M1FN>
    %887 = vector.bitcast %867 : vector<16xi8> to vector<32xf4E2M1FN>
    %888 = vector.bitcast %872 : vector<16xi8> to vector<32xf4E2M1FN>
    %889 = vector.bitcast %877 : vector<16xi8> to vector<32xf4E2M1FN>
    %890 = vector.bitcast %739 : vector<16xi8> to vector<32xf4E2M1FN>
    %891 = vector.bitcast %749 : vector<16xi8> to vector<32xf4E2M1FN>
    %892 = vector.bitcast %758 : vector<16xi8> to vector<32xf4E2M1FN>
    %893 = vector.bitcast %767 : vector<16xi8> to vector<32xf4E2M1FN>
    %894 = vector.bitcast %776 : vector<16xi8> to vector<32xf4E2M1FN>
    %895 = vector.bitcast %785 : vector<16xi8> to vector<32xf4E2M1FN>
    %896 = vector.bitcast %794 : vector<16xi8> to vector<32xf4E2M1FN>
    %897 = vector.bitcast %803 : vector<16xi8> to vector<32xf4E2M1FN>
    rocdl.sched.barrier 0
    %898 = amdgpu.scaled_mfma 16x16x128 (%879[0] * %882) * (%808[0] * %890) + %740 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %899 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 1)>()[%thread_id_x, %arg5, %thread_id_y]
    %900 = affine.apply affine_map<()[s0, s1] -> (s0 * 16 + s1 * 2048 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256) * 4096)>()[%thread_id_x, %arg5]
    %901 = arith.muli %899, %c4096 overflow<nsw> : index
    %902 = arith.addi %901, %900 overflow<nsw> : index
    %903 = vector.load %55[%902] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %904 = amdgpu.scaled_mfma 16x16x128 (%879[2] * %883) * (%808[2] * %891) + %898 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %905 = amdgpu.scaled_mfma 16x16x128 (%879[0] * %882) * (%808[1] * %892) + %742 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %906 = amdgpu.scaled_mfma 16x16x128 (%879[2] * %883) * (%808[3] * %893) + %905 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %907 = vector.load %alloc_1[%119, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %908 = amdgpu.scaled_mfma 16x16x128 (%879[0] * %882) * (%821[0] * %894) + %750 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %909 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 320) floordiv 256)>()[%thread_id_x, %arg5, %thread_id_y]
    %910 = affine.apply affine_map<()[s0, s1] -> (s0 * 16 + s1 * 2048 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256) * 4096 + 1024)>()[%thread_id_x, %arg5]
    %911 = arith.muli %909, %c4096 overflow<nsw> : index
    %912 = arith.addi %911, %910 overflow<nsw> : index
    %913 = vector.load %55[%912] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %914 = amdgpu.scaled_mfma 16x16x128 (%879[2] * %883) * (%821[2] * %895) + %908 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %915 = amdgpu.scaled_mfma 16x16x128 (%879[0] * %882) * (%821[1] * %896) + %752 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %916 = amdgpu.scaled_mfma 16x16x128 (%879[2] * %883) * (%821[3] * %897) + %915 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %917 = vector.load %alloc_1[%119, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %918 = amdgpu.scaled_mfma 16x16x128 (%879[1] * %884) * (%808[0] * %890) + %759 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %919 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 17)>()[%thread_id_x, %arg5, %thread_id_y]
    %920 = arith.muli %919, %c4096 overflow<nsw> : index
    %921 = arith.addi %920, %900 overflow<nsw> : index
    %922 = vector.load %55[%921] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %923 = amdgpu.scaled_mfma 16x16x128 (%879[3] * %885) * (%808[2] * %891) + %918 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %924 = amdgpu.scaled_mfma 16x16x128 (%879[1] * %884) * (%808[1] * %892) + %761 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %925 = amdgpu.scaled_mfma 16x16x128 (%879[3] * %885) * (%808[3] * %893) + %924 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %926 = vector.load %alloc_1[%120, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %927 = amdgpu.scaled_mfma 16x16x128 (%879[1] * %884) * (%821[0] * %894) + %768 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %928 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 320) floordiv 256 + 16)>()[%thread_id_x, %arg5, %thread_id_y]
    %929 = arith.muli %928, %c4096 overflow<nsw> : index
    %930 = arith.addi %929, %910 overflow<nsw> : index
    %931 = vector.load %55[%930] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %932 = amdgpu.scaled_mfma 16x16x128 (%879[3] * %885) * (%821[2] * %895) + %927 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %933 = amdgpu.scaled_mfma 16x16x128 (%879[1] * %884) * (%821[1] * %896) + %770 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %934 = amdgpu.scaled_mfma 16x16x128 (%879[3] * %885) * (%821[3] * %897) + %933 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %935 = vector.load %alloc_1[%120, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %936 = amdgpu.scaled_mfma 16x16x128 (%881[0] * %886) * (%808[0] * %890) + %777 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %937 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 33)>()[%thread_id_x, %arg5, %thread_id_y]
    %938 = arith.muli %937, %c4096 overflow<nsw> : index
    %939 = arith.addi %938, %900 overflow<nsw> : index
    %940 = vector.load %55[%939] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %941 = amdgpu.scaled_mfma 16x16x128 (%881[2] * %887) * (%808[2] * %891) + %936 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %942 = amdgpu.scaled_mfma 16x16x128 (%881[0] * %886) * (%808[1] * %892) + %779 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %943 = amdgpu.scaled_mfma 16x16x128 (%881[2] * %887) * (%808[3] * %893) + %942 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %944 = vector.load %alloc_1[%121, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %945 = amdgpu.scaled_mfma 16x16x128 (%881[0] * %886) * (%821[0] * %894) + %786 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %946 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 320) floordiv 256 + 32)>()[%thread_id_x, %arg5, %thread_id_y]
    %947 = arith.muli %946, %c4096 overflow<nsw> : index
    %948 = arith.addi %947, %910 overflow<nsw> : index
    %949 = vector.load %55[%948] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %950 = amdgpu.scaled_mfma 16x16x128 (%881[2] * %887) * (%821[2] * %895) + %945 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %951 = amdgpu.scaled_mfma 16x16x128 (%881[0] * %886) * (%821[1] * %896) + %788 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %952 = amdgpu.scaled_mfma 16x16x128 (%881[2] * %887) * (%821[3] * %897) + %951 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %953 = vector.load %alloc_1[%121, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %954 = amdgpu.scaled_mfma 16x16x128 (%881[1] * %888) * (%808[0] * %890) + %795 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %955 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 49)>()[%thread_id_x, %arg5, %thread_id_y]
    %956 = arith.muli %955, %c4096 overflow<nsw> : index
    %957 = arith.addi %956, %900 overflow<nsw> : index
    %958 = vector.load %55[%957] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %959 = amdgpu.scaled_mfma 16x16x128 (%881[3] * %889) * (%808[2] * %891) + %954 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %960 = amdgpu.scaled_mfma 16x16x128 (%881[1] * %888) * (%808[1] * %892) + %797 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %961 = amdgpu.scaled_mfma 16x16x128 (%881[3] * %889) * (%808[3] * %893) + %960 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %962 = vector.load %alloc_1[%122, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %963 = amdgpu.scaled_mfma 16x16x128 (%881[1] * %888) * (%821[0] * %894) + %809 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %964 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 320) floordiv 256 + 48)>()[%thread_id_x, %arg5, %thread_id_y]
    %965 = arith.muli %964, %c4096 overflow<nsw> : index
    %966 = arith.addi %965, %910 overflow<nsw> : index
    %967 = vector.load %55[%966] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %968 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 64 + s2 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64 + 2)>()[%thread_id_x, %thread_id_y, %arg5]
    %969 = arith.muli %968, %c256 overflow<nsw> : index
    %970 = arith.addi %969, %87 overflow<nsw> : index
    %971 = vector.load %91[%970] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
    %972 = amdgpu.scaled_mfma 16x16x128 (%881[3] * %889) * (%821[2] * %895) + %963 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %973 = amdgpu.scaled_mfma 16x16x128 (%881[1] * %888) * (%821[1] * %896) + %811 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %974 = amdgpu.scaled_mfma 16x16x128 (%881[3] * %889) * (%821[3] * %897) + %973 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %975 = vector.load %alloc_1[%122, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %976 = vector.load %alloc[%c0, %123] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    %977 = vector.bitcast %976 : vector<4xi8> to vector<4xf8E8M0FNU>
    %978 = vector.load %alloc[%c0, %124] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    %979 = vector.bitcast %978 : vector<4xi8> to vector<4xf8E8M0FNU>
    %980 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 64 + s2 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64 + 34)>()[%thread_id_x, %thread_id_y, %arg5]
    %981 = arith.muli %980, %c256 overflow<nsw> : index
    %982 = arith.addi %981, %87 overflow<nsw> : index
    %983 = vector.load %91[%982] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
    rocdl.sched.barrier 0
    amdgpu.memory_counter_wait load(10) ds(0)
    rocdl.s.barrier
    rocdl.sched.barrier 0
    %984 = vector.bitcast %907 : vector<16xi8> to vector<32xf4E2M1FN>
    %985 = vector.bitcast %917 : vector<16xi8> to vector<32xf4E2M1FN>
    %986 = vector.bitcast %926 : vector<16xi8> to vector<32xf4E2M1FN>
    %987 = vector.bitcast %935 : vector<16xi8> to vector<32xf4E2M1FN>
    %988 = vector.bitcast %944 : vector<16xi8> to vector<32xf4E2M1FN>
    %989 = vector.bitcast %953 : vector<16xi8> to vector<32xf4E2M1FN>
    %990 = vector.bitcast %962 : vector<16xi8> to vector<32xf4E2M1FN>
    %991 = vector.bitcast %975 : vector<16xi8> to vector<32xf4E2M1FN>
    rocdl.sched.barrier 0
    %992 = amdgpu.scaled_mfma 16x16x128 (%977[0] * %984) * (%808[0] * %890) + %833 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %993 = affine.apply affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + 384)>()[%arg5, %8]
    %994 = arith.addi %13, %993 overflow<nsw> : index
    amdgpu.gather_to_lds %15[%994], %alloc_1[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
    %995 = amdgpu.scaled_mfma 16x16x128 (%977[2] * %985) * (%808[2] * %891) + %992 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %996 = amdgpu.scaled_mfma 16x16x128 (%977[0] * %984) * (%808[1] * %892) + %835 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %997 = amdgpu.scaled_mfma 16x16x128 (%977[2] * %985) * (%808[3] * %893) + %996 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %998 = vector.load %alloc_2[%97, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %999 = amdgpu.scaled_mfma 16x16x128 (%977[0] * %984) * (%821[0] * %894) + %839 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1000 = arith.addi %21, %993 overflow<nsw> : index
    amdgpu.gather_to_lds %15[%1000], %alloc_1[%20, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
    %1001 = amdgpu.scaled_mfma 16x16x128 (%977[2] * %985) * (%821[2] * %895) + %999 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1002 = amdgpu.scaled_mfma 16x16x128 (%977[0] * %984) * (%821[1] * %896) + %841 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1003 = amdgpu.scaled_mfma 16x16x128 (%977[2] * %985) * (%821[3] * %897) + %1002 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1004 = vector.load %alloc_2[%97, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1005 = amdgpu.scaled_mfma 16x16x128 (%977[1] * %986) * (%808[0] * %890) + %845 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1006 = arith.addi %27, %993 overflow<nsw> : index
    amdgpu.gather_to_lds %15[%1006], %alloc_1[%26, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
    %1007 = amdgpu.scaled_mfma 16x16x128 (%977[3] * %987) * (%808[2] * %891) + %1005 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1008 = amdgpu.scaled_mfma 16x16x128 (%977[1] * %986) * (%808[1] * %892) + %847 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1009 = amdgpu.scaled_mfma 16x16x128 (%977[3] * %987) * (%808[3] * %893) + %1008 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1010 = vector.load %alloc_2[%106, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1011 = amdgpu.scaled_mfma 16x16x128 (%977[1] * %986) * (%821[0] * %894) + %854 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1012 = arith.addi %33, %993 overflow<nsw> : index
    amdgpu.gather_to_lds %15[%1012], %alloc_1[%32, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
    %1013 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * 128 + s1 * 32 + s2 + 3)>()[%block_id_x, %37, %arg5]
    %1014 = arith.muli %1013, %c256 overflow<nsw> : index
    %1015 = arith.addi %1014, %39 overflow<nsw> : index
    amdgpu.gather_to_lds %45[%1015], %alloc[%42, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<1024x1xi8, #gpu.address_space<workgroup>>
    %1016 = amdgpu.scaled_mfma 16x16x128 (%977[3] * %987) * (%821[2] * %895) + %1011 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1017 = amdgpu.scaled_mfma 16x16x128 (%977[1] * %986) * (%821[1] * %896) + %856 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1018 = amdgpu.scaled_mfma 16x16x128 (%977[3] * %987) * (%821[3] * %897) + %1017 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1019 = vector.load %alloc_2[%106, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1020 = amdgpu.scaled_mfma 16x16x128 (%979[0] * %988) * (%808[0] * %890) + %859 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1021 = amdgpu.scaled_mfma 16x16x128 (%979[2] * %989) * (%808[2] * %891) + %1020 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1022 = amdgpu.scaled_mfma 16x16x128 (%979[0] * %988) * (%808[1] * %892) + %861 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1023 = amdgpu.scaled_mfma 16x16x128 (%979[2] * %989) * (%808[3] * %893) + %1022 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1024 = vector.load %alloc_2[%109, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1025 = amdgpu.scaled_mfma 16x16x128 (%979[0] * %988) * (%821[0] * %894) + %864 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1026 = amdgpu.scaled_mfma 16x16x128 (%979[2] * %989) * (%821[2] * %895) + %1025 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1027 = amdgpu.scaled_mfma 16x16x128 (%979[0] * %988) * (%821[1] * %896) + %866 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1028 = amdgpu.scaled_mfma 16x16x128 (%979[2] * %989) * (%821[3] * %897) + %1027 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1029 = vector.load %alloc_2[%109, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1030 = amdgpu.scaled_mfma 16x16x128 (%979[1] * %990) * (%808[0] * %890) + %869 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1031 = amdgpu.scaled_mfma 16x16x128 (%979[3] * %991) * (%808[2] * %891) + %1030 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1032 = amdgpu.scaled_mfma 16x16x128 (%979[1] * %990) * (%808[1] * %892) + %871 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1033 = amdgpu.scaled_mfma 16x16x128 (%979[3] * %991) * (%808[3] * %893) + %1032 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1034 = vector.load %alloc_2[%112, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1035 = amdgpu.scaled_mfma 16x16x128 (%979[1] * %990) * (%821[0] * %894) + %874 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1036 = amdgpu.scaled_mfma 16x16x128 (%979[3] * %991) * (%821[2] * %895) + %1035 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1037 = amdgpu.scaled_mfma 16x16x128 (%979[1] * %990) * (%821[1] * %896) + %876 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1038 = amdgpu.scaled_mfma 16x16x128 (%979[3] * %991) * (%821[3] * %897) + %1037 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1039 = vector.load %alloc_2[%112, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1040 = vector.load %alloc_0[%c0, %115] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    %1041 = vector.load %alloc_0[%c0, %117] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    rocdl.sched.barrier 0
    amdgpu.memory_counter_wait load(5) ds(0)
    rocdl.s.barrier
    rocdl.sched.barrier 0
    scf.yield %904, %906, %914, %916, %923, %925, %932, %934, %941, %943, %950, %952, %959, %961, %972, %974, %995, %997, %1001, %1003, %1007, %1009, %1016, %1018, %1021, %1023, %1026, %1028, %1031, %1033, %1036, %1038, %1040, %1041, %998, %1004, %1010, %1019, %1024, %1029, %1034, %1039, %903, %913, %922, %931, %940, %949, %958, %967, %971, %983 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<4xi8>, vector<4xi8>
  }
  %126 = vector.bitcast %125#33 : vector<4xi8> to vector<4xf8E8M0FNU>
  %127 = vector.bitcast %125#51 : vector<4xi8> to vector<4xf8E8M0FNU>
  %128 = vector.bitcast %125#50 : vector<4xi8> to vector<4xf8E8M0FNU>
  %129 = vector.bitcast %125#32 : vector<4xi8> to vector<4xf8E8M0FNU>
  amdgpu.memory_counter_wait load(0) ds(0)
  rocdl.s.barrier
  %130 = vector.load %alloc_0[%c0, %123] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
  %131 = vector.bitcast %130 : vector<4xi8> to vector<4xf8E8M0FNU>
  %132 = vector.load %alloc_0[%c0, %124] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
  %133 = vector.bitcast %132 : vector<4xi8> to vector<4xf8E8M0FNU>
  %134 = vector.load %alloc_2[%119, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %135 = vector.load %alloc_2[%119, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %136 = vector.load %alloc_2[%120, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %137 = vector.load %alloc_2[%120, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %138 = vector.load %alloc_2[%121, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %139 = vector.load %alloc_2[%121, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %140 = vector.load %alloc_2[%122, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %141 = vector.load %alloc_2[%122, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %142 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 3968) floordiv 256)>()[%thread_id_x, %thread_id_y]
  %143 = affine.apply affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 128) floordiv 256) * 4096 + 2048)>()[%thread_id_x]
  %144 = arith.muli %142, %c4096 overflow<nsw> : index
  %145 = arith.addi %144, %143 overflow<nsw> : index
  %146 = vector.load %55[%145] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %147 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 4032) floordiv 256)>()[%thread_id_x, %thread_id_y]
  %148 = affine.apply affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 192) floordiv 256) * 4096 + 3072)>()[%thread_id_x]
  %149 = arith.muli %147, %c4096 overflow<nsw> : index
  %150 = arith.addi %149, %148 overflow<nsw> : index
  %151 = vector.load %55[%150] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %152 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 3968) floordiv 256 + 16)>()[%thread_id_x, %thread_id_y]
  %153 = arith.muli %152, %c4096 overflow<nsw> : index
  %154 = arith.addi %153, %143 overflow<nsw> : index
  %155 = vector.load %55[%154] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %156 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 4032) floordiv 256 + 16)>()[%thread_id_x, %thread_id_y]
  %157 = arith.muli %156, %c4096 overflow<nsw> : index
  %158 = arith.addi %157, %148 overflow<nsw> : index
  %159 = vector.load %55[%158] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %160 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 3968) floordiv 256 + 32)>()[%thread_id_x, %thread_id_y]
  %161 = arith.muli %160, %c4096 overflow<nsw> : index
  %162 = arith.addi %161, %143 overflow<nsw> : index
  %163 = vector.load %55[%162] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %164 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 4032) floordiv 256 + 32)>()[%thread_id_x, %thread_id_y]
  %165 = arith.muli %164, %c4096 overflow<nsw> : index
  %166 = arith.addi %165, %148 overflow<nsw> : index
  %167 = vector.load %55[%166] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %168 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 3968) floordiv 256 + 48)>()[%thread_id_x, %thread_id_y]
  %169 = arith.muli %168, %c4096 overflow<nsw> : index
  %170 = arith.addi %169, %143 overflow<nsw> : index
  %171 = vector.load %55[%170] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %172 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 4032) floordiv 256 + 48)>()[%thread_id_x, %thread_id_y]
  %173 = arith.muli %172, %c4096 overflow<nsw> : index
  %174 = arith.addi %173, %148 overflow<nsw> : index
  %175 = vector.load %55[%174] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %176 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64 + 31)>()[%thread_id_x, %thread_id_y]
  %177 = arith.muli %176, %c256 overflow<nsw> : index
  %178 = arith.addi %177, %87 overflow<nsw> : index
  %179 = vector.load %91[%178] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
  %180 = vector.bitcast %179 : vector<4xi8> to vector<4xf8E8M0FNU>
  %181 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64 + 63)>()[%thread_id_x, %thread_id_y]
  %182 = arith.muli %181, %c256 overflow<nsw> : index
  %183 = arith.addi %182, %87 overflow<nsw> : index
  %184 = vector.load %91[%183] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
  %185 = vector.bitcast %184 : vector<4xi8> to vector<4xf8E8M0FNU>
  amdgpu.memory_counter_wait load(0) ds(0)
  rocdl.s.barrier
  %186 = vector.bitcast %125#34 : vector<16xi8> to vector<32xf4E2M1FN>
  %187 = vector.bitcast %125#35 : vector<16xi8> to vector<32xf4E2M1FN>
  %188 = vector.bitcast %125#36 : vector<16xi8> to vector<32xf4E2M1FN>
  %189 = vector.bitcast %125#37 : vector<16xi8> to vector<32xf4E2M1FN>
  %190 = vector.bitcast %125#38 : vector<16xi8> to vector<32xf4E2M1FN>
  %191 = vector.bitcast %125#39 : vector<16xi8> to vector<32xf4E2M1FN>
  %192 = vector.bitcast %125#40 : vector<16xi8> to vector<32xf4E2M1FN>
  %193 = vector.bitcast %125#41 : vector<16xi8> to vector<32xf4E2M1FN>
  %194 = vector.bitcast %134 : vector<16xi8> to vector<32xf4E2M1FN>
  %195 = vector.bitcast %135 : vector<16xi8> to vector<32xf4E2M1FN>
  %196 = vector.bitcast %136 : vector<16xi8> to vector<32xf4E2M1FN>
  %197 = vector.bitcast %137 : vector<16xi8> to vector<32xf4E2M1FN>
  %198 = vector.bitcast %138 : vector<16xi8> to vector<32xf4E2M1FN>
  %199 = vector.bitcast %139 : vector<16xi8> to vector<32xf4E2M1FN>
  %200 = vector.bitcast %140 : vector<16xi8> to vector<32xf4E2M1FN>
  %201 = vector.bitcast %141 : vector<16xi8> to vector<32xf4E2M1FN>
  %202 = vector.bitcast %125#42 : vector<16xi8> to vector<32xf4E2M1FN>
  %203 = vector.bitcast %125#43 : vector<16xi8> to vector<32xf4E2M1FN>
  %204 = vector.bitcast %125#44 : vector<16xi8> to vector<32xf4E2M1FN>
  %205 = vector.bitcast %125#45 : vector<16xi8> to vector<32xf4E2M1FN>
  %206 = vector.bitcast %125#46 : vector<16xi8> to vector<32xf4E2M1FN>
  %207 = vector.bitcast %125#47 : vector<16xi8> to vector<32xf4E2M1FN>
  %208 = vector.bitcast %125#48 : vector<16xi8> to vector<32xf4E2M1FN>
  %209 = vector.bitcast %125#49 : vector<16xi8> to vector<32xf4E2M1FN>
  rocdl.sched.barrier 0
  %210 = vector.load %alloc[%c0, %115] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
  %211 = vector.bitcast %210 : vector<4xi8> to vector<4xf8E8M0FNU>
  %212 = vector.load %alloc[%c0, %117] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
  %213 = vector.bitcast %212 : vector<4xi8> to vector<4xf8E8M0FNU>
  %214 = vector.load %alloc_1[%97, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %215 = vector.load %alloc_1[%97, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %216 = vector.load %alloc_1[%106, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %217 = vector.load %alloc_1[%106, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %218 = vector.load %alloc_1[%109, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %219 = vector.load %alloc_1[%109, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %220 = vector.load %alloc_1[%112, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %221 = vector.load %alloc_1[%112, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %222 = amdgpu.scaled_mfma 16x16x128 (%129[0] * %186) * (%128[0] * %202) + %125#0 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %223 = amdgpu.scaled_mfma 16x16x128 (%129[2] * %187) * (%128[2] * %203) + %222 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %224 = amdgpu.scaled_mfma 16x16x128 (%129[0] * %186) * (%128[1] * %204) + %125#1 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %225 = amdgpu.scaled_mfma 16x16x128 (%129[2] * %187) * (%128[3] * %205) + %224 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %226 = amdgpu.scaled_mfma 16x16x128 (%129[0] * %186) * (%127[0] * %206) + %125#2 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %227 = amdgpu.scaled_mfma 16x16x128 (%129[2] * %187) * (%127[2] * %207) + %226 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %228 = amdgpu.scaled_mfma 16x16x128 (%129[0] * %186) * (%127[1] * %208) + %125#3 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %229 = amdgpu.scaled_mfma 16x16x128 (%129[2] * %187) * (%127[3] * %209) + %228 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %230 = amdgpu.scaled_mfma 16x16x128 (%129[1] * %188) * (%128[0] * %202) + %125#4 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %231 = amdgpu.scaled_mfma 16x16x128 (%129[3] * %189) * (%128[2] * %203) + %230 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %232 = amdgpu.scaled_mfma 16x16x128 (%129[1] * %188) * (%128[1] * %204) + %125#5 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %233 = amdgpu.scaled_mfma 16x16x128 (%129[3] * %189) * (%128[3] * %205) + %232 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %234 = amdgpu.scaled_mfma 16x16x128 (%129[1] * %188) * (%127[0] * %206) + %125#6 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %235 = amdgpu.scaled_mfma 16x16x128 (%129[3] * %189) * (%127[2] * %207) + %234 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %236 = amdgpu.scaled_mfma 16x16x128 (%129[1] * %188) * (%127[1] * %208) + %125#7 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %237 = amdgpu.scaled_mfma 16x16x128 (%129[3] * %189) * (%127[3] * %209) + %236 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %238 = amdgpu.scaled_mfma 16x16x128 (%126[0] * %190) * (%128[0] * %202) + %125#8 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %239 = amdgpu.scaled_mfma 16x16x128 (%126[2] * %191) * (%128[2] * %203) + %238 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %240 = amdgpu.scaled_mfma 16x16x128 (%126[0] * %190) * (%128[1] * %204) + %125#9 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %241 = amdgpu.scaled_mfma 16x16x128 (%126[2] * %191) * (%128[3] * %205) + %240 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %242 = amdgpu.scaled_mfma 16x16x128 (%126[0] * %190) * (%127[0] * %206) + %125#10 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %243 = amdgpu.scaled_mfma 16x16x128 (%126[2] * %191) * (%127[2] * %207) + %242 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %244 = amdgpu.scaled_mfma 16x16x128 (%126[0] * %190) * (%127[1] * %208) + %125#11 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %245 = amdgpu.scaled_mfma 16x16x128 (%126[2] * %191) * (%127[3] * %209) + %244 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %246 = amdgpu.scaled_mfma 16x16x128 (%126[1] * %192) * (%128[0] * %202) + %125#12 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %247 = amdgpu.scaled_mfma 16x16x128 (%126[3] * %193) * (%128[2] * %203) + %246 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %248 = amdgpu.scaled_mfma 16x16x128 (%126[1] * %192) * (%128[1] * %204) + %125#13 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %249 = amdgpu.scaled_mfma 16x16x128 (%126[3] * %193) * (%128[3] * %205) + %248 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %250 = amdgpu.scaled_mfma 16x16x128 (%126[1] * %192) * (%127[0] * %206) + %125#14 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %251 = amdgpu.scaled_mfma 16x16x128 (%126[3] * %193) * (%127[2] * %207) + %250 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %252 = amdgpu.scaled_mfma 16x16x128 (%126[1] * %192) * (%127[1] * %208) + %125#15 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %253 = amdgpu.scaled_mfma 16x16x128 (%126[3] * %193) * (%127[3] * %209) + %252 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %254 = amdgpu.scaled_mfma 16x16x128 (%131[0] * %194) * (%128[0] * %202) + %125#16 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %255 = amdgpu.scaled_mfma 16x16x128 (%131[2] * %195) * (%128[2] * %203) + %254 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %256 = amdgpu.scaled_mfma 16x16x128 (%131[0] * %194) * (%128[1] * %204) + %125#17 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %257 = amdgpu.scaled_mfma 16x16x128 (%131[2] * %195) * (%128[3] * %205) + %256 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %258 = amdgpu.scaled_mfma 16x16x128 (%131[0] * %194) * (%127[0] * %206) + %125#18 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %259 = amdgpu.scaled_mfma 16x16x128 (%131[2] * %195) * (%127[2] * %207) + %258 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %260 = amdgpu.scaled_mfma 16x16x128 (%131[0] * %194) * (%127[1] * %208) + %125#19 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %261 = amdgpu.scaled_mfma 16x16x128 (%131[2] * %195) * (%127[3] * %209) + %260 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %262 = amdgpu.scaled_mfma 16x16x128 (%131[1] * %196) * (%128[0] * %202) + %125#20 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %263 = amdgpu.scaled_mfma 16x16x128 (%131[3] * %197) * (%128[2] * %203) + %262 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %264 = amdgpu.scaled_mfma 16x16x128 (%131[1] * %196) * (%128[1] * %204) + %125#21 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %265 = amdgpu.scaled_mfma 16x16x128 (%131[3] * %197) * (%128[3] * %205) + %264 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %266 = amdgpu.scaled_mfma 16x16x128 (%131[1] * %196) * (%127[0] * %206) + %125#22 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %267 = amdgpu.scaled_mfma 16x16x128 (%131[3] * %197) * (%127[2] * %207) + %266 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %268 = amdgpu.scaled_mfma 16x16x128 (%131[1] * %196) * (%127[1] * %208) + %125#23 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %269 = amdgpu.scaled_mfma 16x16x128 (%131[3] * %197) * (%127[3] * %209) + %268 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %270 = amdgpu.scaled_mfma 16x16x128 (%133[0] * %198) * (%128[0] * %202) + %125#24 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %271 = amdgpu.scaled_mfma 16x16x128 (%133[2] * %199) * (%128[2] * %203) + %270 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %272 = amdgpu.scaled_mfma 16x16x128 (%133[0] * %198) * (%128[1] * %204) + %125#25 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %273 = amdgpu.scaled_mfma 16x16x128 (%133[2] * %199) * (%128[3] * %205) + %272 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %274 = amdgpu.scaled_mfma 16x16x128 (%133[0] * %198) * (%127[0] * %206) + %125#26 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %275 = amdgpu.scaled_mfma 16x16x128 (%133[2] * %199) * (%127[2] * %207) + %274 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %276 = amdgpu.scaled_mfma 16x16x128 (%133[0] * %198) * (%127[1] * %208) + %125#27 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %277 = amdgpu.scaled_mfma 16x16x128 (%133[2] * %199) * (%127[3] * %209) + %276 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %278 = amdgpu.scaled_mfma 16x16x128 (%133[1] * %200) * (%128[0] * %202) + %125#28 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %279 = amdgpu.scaled_mfma 16x16x128 (%133[3] * %201) * (%128[2] * %203) + %278 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %280 = amdgpu.scaled_mfma 16x16x128 (%133[1] * %200) * (%128[1] * %204) + %125#29 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %281 = amdgpu.scaled_mfma 16x16x128 (%133[3] * %201) * (%128[3] * %205) + %280 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %282 = amdgpu.scaled_mfma 16x16x128 (%133[1] * %200) * (%127[0] * %206) + %125#30 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %283 = amdgpu.scaled_mfma 16x16x128 (%133[3] * %201) * (%127[2] * %207) + %282 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %284 = amdgpu.scaled_mfma 16x16x128 (%133[1] * %200) * (%127[1] * %208) + %125#31 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %285 = amdgpu.scaled_mfma 16x16x128 (%133[3] * %201) * (%127[3] * %209) + %284 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  amdgpu.memory_counter_wait load(0) ds(0)
  rocdl.s.barrier
  %286 = vector.load %alloc[%c0, %123] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
  %287 = vector.bitcast %286 : vector<4xi8> to vector<4xf8E8M0FNU>
  %288 = vector.load %alloc[%c0, %124] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
  %289 = vector.bitcast %288 : vector<4xi8> to vector<4xf8E8M0FNU>
  %290 = vector.load %alloc_1[%119, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %291 = vector.load %alloc_1[%119, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %292 = vector.load %alloc_1[%120, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %293 = vector.load %alloc_1[%120, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %294 = vector.load %alloc_1[%121, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %295 = vector.load %alloc_1[%121, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %296 = vector.load %alloc_1[%122, %100] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %297 = vector.load %alloc_1[%122, %104] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  amdgpu.memory_counter_wait load(0) ds(0)
  rocdl.s.barrier
  %298 = vector.bitcast %214 : vector<16xi8> to vector<32xf4E2M1FN>
  %299 = vector.bitcast %215 : vector<16xi8> to vector<32xf4E2M1FN>
  %300 = vector.bitcast %216 : vector<16xi8> to vector<32xf4E2M1FN>
  %301 = vector.bitcast %217 : vector<16xi8> to vector<32xf4E2M1FN>
  %302 = vector.bitcast %218 : vector<16xi8> to vector<32xf4E2M1FN>
  %303 = vector.bitcast %219 : vector<16xi8> to vector<32xf4E2M1FN>
  %304 = vector.bitcast %220 : vector<16xi8> to vector<32xf4E2M1FN>
  %305 = vector.bitcast %221 : vector<16xi8> to vector<32xf4E2M1FN>
  %306 = vector.bitcast %290 : vector<16xi8> to vector<32xf4E2M1FN>
  %307 = vector.bitcast %291 : vector<16xi8> to vector<32xf4E2M1FN>
  %308 = vector.bitcast %292 : vector<16xi8> to vector<32xf4E2M1FN>
  %309 = vector.bitcast %293 : vector<16xi8> to vector<32xf4E2M1FN>
  %310 = vector.bitcast %294 : vector<16xi8> to vector<32xf4E2M1FN>
  %311 = vector.bitcast %295 : vector<16xi8> to vector<32xf4E2M1FN>
  %312 = vector.bitcast %296 : vector<16xi8> to vector<32xf4E2M1FN>
  %313 = vector.bitcast %297 : vector<16xi8> to vector<32xf4E2M1FN>
  %314 = vector.bitcast %146 : vector<16xi8> to vector<32xf4E2M1FN>
  %315 = vector.bitcast %151 : vector<16xi8> to vector<32xf4E2M1FN>
  %316 = vector.bitcast %155 : vector<16xi8> to vector<32xf4E2M1FN>
  %317 = vector.bitcast %159 : vector<16xi8> to vector<32xf4E2M1FN>
  %318 = vector.bitcast %163 : vector<16xi8> to vector<32xf4E2M1FN>
  %319 = vector.bitcast %167 : vector<16xi8> to vector<32xf4E2M1FN>
  %320 = vector.bitcast %171 : vector<16xi8> to vector<32xf4E2M1FN>
  %321 = vector.bitcast %175 : vector<16xi8> to vector<32xf4E2M1FN>
  rocdl.sched.barrier 0
  %322 = amdgpu.scaled_mfma 16x16x128 (%211[0] * %298) * (%180[0] * %314) + %223 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %323 = amdgpu.scaled_mfma 16x16x128 (%211[2] * %299) * (%180[2] * %315) + %322 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %324 = amdgpu.scaled_mfma 16x16x128 (%211[0] * %298) * (%180[1] * %316) + %225 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %325 = amdgpu.scaled_mfma 16x16x128 (%211[2] * %299) * (%180[3] * %317) + %324 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %326 = amdgpu.scaled_mfma 16x16x128 (%211[0] * %298) * (%185[0] * %318) + %227 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %327 = amdgpu.scaled_mfma 16x16x128 (%211[2] * %299) * (%185[2] * %319) + %326 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %328 = amdgpu.scaled_mfma 16x16x128 (%211[0] * %298) * (%185[1] * %320) + %229 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %329 = amdgpu.scaled_mfma 16x16x128 (%211[2] * %299) * (%185[3] * %321) + %328 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %330 = amdgpu.scaled_mfma 16x16x128 (%211[1] * %300) * (%180[0] * %314) + %231 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %331 = amdgpu.scaled_mfma 16x16x128 (%211[3] * %301) * (%180[2] * %315) + %330 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %332 = amdgpu.scaled_mfma 16x16x128 (%211[1] * %300) * (%180[1] * %316) + %233 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %333 = amdgpu.scaled_mfma 16x16x128 (%211[3] * %301) * (%180[3] * %317) + %332 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %334 = amdgpu.scaled_mfma 16x16x128 (%211[1] * %300) * (%185[0] * %318) + %235 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %335 = amdgpu.scaled_mfma 16x16x128 (%211[3] * %301) * (%185[2] * %319) + %334 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %336 = amdgpu.scaled_mfma 16x16x128 (%211[1] * %300) * (%185[1] * %320) + %237 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %337 = amdgpu.scaled_mfma 16x16x128 (%211[3] * %301) * (%185[3] * %321) + %336 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %338 = amdgpu.scaled_mfma 16x16x128 (%213[0] * %302) * (%180[0] * %314) + %239 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %339 = amdgpu.scaled_mfma 16x16x128 (%213[2] * %303) * (%180[2] * %315) + %338 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %340 = amdgpu.scaled_mfma 16x16x128 (%213[0] * %302) * (%180[1] * %316) + %241 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %341 = amdgpu.scaled_mfma 16x16x128 (%213[2] * %303) * (%180[3] * %317) + %340 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %342 = amdgpu.scaled_mfma 16x16x128 (%213[0] * %302) * (%185[0] * %318) + %243 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %343 = amdgpu.scaled_mfma 16x16x128 (%213[2] * %303) * (%185[2] * %319) + %342 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %344 = amdgpu.scaled_mfma 16x16x128 (%213[0] * %302) * (%185[1] * %320) + %245 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %345 = amdgpu.scaled_mfma 16x16x128 (%213[2] * %303) * (%185[3] * %321) + %344 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %346 = amdgpu.scaled_mfma 16x16x128 (%213[1] * %304) * (%180[0] * %314) + %247 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %347 = amdgpu.scaled_mfma 16x16x128 (%213[3] * %305) * (%180[2] * %315) + %346 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %348 = amdgpu.scaled_mfma 16x16x128 (%213[1] * %304) * (%180[1] * %316) + %249 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %349 = amdgpu.scaled_mfma 16x16x128 (%213[3] * %305) * (%180[3] * %317) + %348 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %350 = amdgpu.scaled_mfma 16x16x128 (%213[1] * %304) * (%185[0] * %318) + %251 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %351 = amdgpu.scaled_mfma 16x16x128 (%213[3] * %305) * (%185[2] * %319) + %350 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %352 = amdgpu.scaled_mfma 16x16x128 (%213[1] * %304) * (%185[1] * %320) + %253 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %353 = amdgpu.scaled_mfma 16x16x128 (%213[3] * %305) * (%185[3] * %321) + %352 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %354 = amdgpu.scaled_mfma 16x16x128 (%287[0] * %306) * (%180[0] * %314) + %255 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %355 = amdgpu.scaled_mfma 16x16x128 (%287[2] * %307) * (%180[2] * %315) + %354 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %356 = amdgpu.scaled_mfma 16x16x128 (%287[0] * %306) * (%180[1] * %316) + %257 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %357 = amdgpu.scaled_mfma 16x16x128 (%287[2] * %307) * (%180[3] * %317) + %356 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %358 = amdgpu.scaled_mfma 16x16x128 (%287[0] * %306) * (%185[0] * %318) + %259 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %359 = amdgpu.scaled_mfma 16x16x128 (%287[2] * %307) * (%185[2] * %319) + %358 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %360 = amdgpu.scaled_mfma 16x16x128 (%287[0] * %306) * (%185[1] * %320) + %261 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %361 = amdgpu.scaled_mfma 16x16x128 (%287[2] * %307) * (%185[3] * %321) + %360 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %362 = amdgpu.scaled_mfma 16x16x128 (%287[1] * %308) * (%180[0] * %314) + %263 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %363 = amdgpu.scaled_mfma 16x16x128 (%287[3] * %309) * (%180[2] * %315) + %362 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %364 = amdgpu.scaled_mfma 16x16x128 (%287[1] * %308) * (%180[1] * %316) + %265 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %365 = amdgpu.scaled_mfma 16x16x128 (%287[3] * %309) * (%180[3] * %317) + %364 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %366 = amdgpu.scaled_mfma 16x16x128 (%287[1] * %308) * (%185[0] * %318) + %267 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %367 = amdgpu.scaled_mfma 16x16x128 (%287[3] * %309) * (%185[2] * %319) + %366 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %368 = amdgpu.scaled_mfma 16x16x128 (%287[1] * %308) * (%185[1] * %320) + %269 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %369 = amdgpu.scaled_mfma 16x16x128 (%287[3] * %309) * (%185[3] * %321) + %368 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %370 = amdgpu.scaled_mfma 16x16x128 (%289[0] * %310) * (%180[0] * %314) + %271 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %371 = amdgpu.scaled_mfma 16x16x128 (%289[2] * %311) * (%180[2] * %315) + %370 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %372 = amdgpu.scaled_mfma 16x16x128 (%289[0] * %310) * (%180[1] * %316) + %273 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %373 = amdgpu.scaled_mfma 16x16x128 (%289[2] * %311) * (%180[3] * %317) + %372 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %374 = amdgpu.scaled_mfma 16x16x128 (%289[0] * %310) * (%185[0] * %318) + %275 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %375 = amdgpu.scaled_mfma 16x16x128 (%289[2] * %311) * (%185[2] * %319) + %374 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %376 = amdgpu.scaled_mfma 16x16x128 (%289[0] * %310) * (%185[1] * %320) + %277 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %377 = amdgpu.scaled_mfma 16x16x128 (%289[2] * %311) * (%185[3] * %321) + %376 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %378 = amdgpu.scaled_mfma 16x16x128 (%289[1] * %312) * (%180[0] * %314) + %279 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %379 = amdgpu.scaled_mfma 16x16x128 (%289[3] * %313) * (%180[2] * %315) + %378 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %380 = amdgpu.scaled_mfma 16x16x128 (%289[1] * %312) * (%180[1] * %316) + %281 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %381 = amdgpu.scaled_mfma 16x16x128 (%289[3] * %313) * (%180[3] * %317) + %380 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %382 = amdgpu.scaled_mfma 16x16x128 (%289[1] * %312) * (%185[0] * %318) + %283 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %383 = amdgpu.scaled_mfma 16x16x128 (%289[3] * %313) * (%185[2] * %319) + %382 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %384 = amdgpu.scaled_mfma 16x16x128 (%289[1] * %312) * (%185[1] * %320) + %285 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %385 = amdgpu.scaled_mfma 16x16x128 (%289[3] * %313) * (%185[3] * %321) + %384 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %386 = vector.extract_strided_slice %323 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %387 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%block_id_x]
  %388 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4)>()[%thread_id_x]
  %389 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16)>()[%thread_id_x, %thread_id_y]
  %390 = arith.muli %387, %c1024 overflow<nsw> : index
  %391 = arith.muli %388, %c1024 overflow<nsw> : index
  %392 = arith.addi %390, %49 overflow<nsw> : index
  %393 = arith.addi %391, %389 overflow<nsw> : index
  %reinterpret_cast_9 = memref.reinterpret_cast %4 to offset: [%392], sizes: [536870910], strides: [1] : memref<f32> to memref<536870910xf32, strided<[1], offset: ?>>
  %cast_10 = memref.cast %reinterpret_cast_9 : memref<536870910xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  %394 = amdgpu.fat_raw_buffer_cast %cast_10 validBytes(%c2147483643_i64) cacheSwizzleStride(%c1024_i14) resetOffset : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
  vector.store %386, %394[%393] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %395 = vector.extract_strided_slice %323 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %396 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1)>()[%thread_id_x]
  %397 = arith.muli %396, %c1024 overflow<nsw> : index
  %398 = arith.addi %397, %389 overflow<nsw> : index
  vector.store %395, %394[%398] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %399 = vector.extract_strided_slice %323 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %400 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 2)>()[%thread_id_x]
  %401 = arith.muli %400, %c1024 overflow<nsw> : index
  %402 = arith.addi %401, %389 overflow<nsw> : index
  vector.store %399, %394[%402] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %403 = vector.extract_strided_slice %323 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %404 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3)>()[%thread_id_x]
  %405 = arith.muli %404, %c1024 overflow<nsw> : index
  %406 = arith.addi %405, %389 overflow<nsw> : index
  vector.store %403, %394[%406] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %407 = vector.extract_strided_slice %325 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %408 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 16)>()[%thread_id_x, %thread_id_y]
  %409 = arith.addi %391, %408 overflow<nsw> : index
  vector.store %407, %394[%409] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %410 = vector.extract_strided_slice %325 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %411 = arith.addi %397, %408 overflow<nsw> : index
  vector.store %410, %394[%411] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %412 = vector.extract_strided_slice %325 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %413 = arith.addi %401, %408 overflow<nsw> : index
  vector.store %412, %394[%413] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %414 = vector.extract_strided_slice %325 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %415 = arith.addi %405, %408 overflow<nsw> : index
  vector.store %414, %394[%415] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %416 = vector.extract_strided_slice %327 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %417 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 32)>()[%thread_id_x, %thread_id_y]
  %418 = arith.addi %391, %417 overflow<nsw> : index
  vector.store %416, %394[%418] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %419 = vector.extract_strided_slice %327 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %420 = arith.addi %397, %417 overflow<nsw> : index
  vector.store %419, %394[%420] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %421 = vector.extract_strided_slice %327 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %422 = arith.addi %401, %417 overflow<nsw> : index
  vector.store %421, %394[%422] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %423 = vector.extract_strided_slice %327 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %424 = arith.addi %405, %417 overflow<nsw> : index
  vector.store %423, %394[%424] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %425 = vector.extract_strided_slice %329 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %426 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 48)>()[%thread_id_x, %thread_id_y]
  %427 = arith.addi %391, %426 overflow<nsw> : index
  vector.store %425, %394[%427] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %428 = vector.extract_strided_slice %329 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %429 = arith.addi %397, %426 overflow<nsw> : index
  vector.store %428, %394[%429] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %430 = vector.extract_strided_slice %329 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %431 = arith.addi %401, %426 overflow<nsw> : index
  vector.store %430, %394[%431] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %432 = vector.extract_strided_slice %329 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %433 = arith.addi %405, %426 overflow<nsw> : index
  vector.store %432, %394[%433] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %434 = vector.extract_strided_slice %331 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %435 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 16)>()[%thread_id_x]
  %436 = arith.muli %435, %c1024 overflow<nsw> : index
  %437 = arith.addi %436, %389 overflow<nsw> : index
  vector.store %434, %394[%437] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %438 = vector.extract_strided_slice %331 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %439 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 17)>()[%thread_id_x]
  %440 = arith.muli %439, %c1024 overflow<nsw> : index
  %441 = arith.addi %440, %389 overflow<nsw> : index
  vector.store %438, %394[%441] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %442 = vector.extract_strided_slice %331 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %443 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 18)>()[%thread_id_x]
  %444 = arith.muli %443, %c1024 overflow<nsw> : index
  %445 = arith.addi %444, %389 overflow<nsw> : index
  vector.store %442, %394[%445] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %446 = vector.extract_strided_slice %331 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %447 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 19)>()[%thread_id_x]
  %448 = arith.muli %447, %c1024 overflow<nsw> : index
  %449 = arith.addi %448, %389 overflow<nsw> : index
  vector.store %446, %394[%449] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %450 = vector.extract_strided_slice %333 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %451 = arith.addi %436, %408 overflow<nsw> : index
  vector.store %450, %394[%451] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %452 = vector.extract_strided_slice %333 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %453 = arith.addi %440, %408 overflow<nsw> : index
  vector.store %452, %394[%453] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %454 = vector.extract_strided_slice %333 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %455 = arith.addi %444, %408 overflow<nsw> : index
  vector.store %454, %394[%455] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %456 = vector.extract_strided_slice %333 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %457 = arith.addi %448, %408 overflow<nsw> : index
  vector.store %456, %394[%457] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %458 = vector.extract_strided_slice %335 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %459 = arith.addi %436, %417 overflow<nsw> : index
  vector.store %458, %394[%459] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %460 = vector.extract_strided_slice %335 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %461 = arith.addi %440, %417 overflow<nsw> : index
  vector.store %460, %394[%461] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %462 = vector.extract_strided_slice %335 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %463 = arith.addi %444, %417 overflow<nsw> : index
  vector.store %462, %394[%463] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %464 = vector.extract_strided_slice %335 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %465 = arith.addi %448, %417 overflow<nsw> : index
  vector.store %464, %394[%465] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %466 = vector.extract_strided_slice %337 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %467 = arith.addi %436, %426 overflow<nsw> : index
  vector.store %466, %394[%467] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %468 = vector.extract_strided_slice %337 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %469 = arith.addi %440, %426 overflow<nsw> : index
  vector.store %468, %394[%469] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %470 = vector.extract_strided_slice %337 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %471 = arith.addi %444, %426 overflow<nsw> : index
  vector.store %470, %394[%471] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %472 = vector.extract_strided_slice %337 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %473 = arith.addi %448, %426 overflow<nsw> : index
  vector.store %472, %394[%473] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %474 = vector.extract_strided_slice %339 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %475 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 32)>()[%thread_id_x]
  %476 = arith.muli %475, %c1024 overflow<nsw> : index
  %477 = arith.addi %476, %389 overflow<nsw> : index
  vector.store %474, %394[%477] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %478 = vector.extract_strided_slice %339 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %479 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 33)>()[%thread_id_x]
  %480 = arith.muli %479, %c1024 overflow<nsw> : index
  %481 = arith.addi %480, %389 overflow<nsw> : index
  vector.store %478, %394[%481] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %482 = vector.extract_strided_slice %339 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %483 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 34)>()[%thread_id_x]
  %484 = arith.muli %483, %c1024 overflow<nsw> : index
  %485 = arith.addi %484, %389 overflow<nsw> : index
  vector.store %482, %394[%485] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %486 = vector.extract_strided_slice %339 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %487 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 35)>()[%thread_id_x]
  %488 = arith.muli %487, %c1024 overflow<nsw> : index
  %489 = arith.addi %488, %389 overflow<nsw> : index
  vector.store %486, %394[%489] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %490 = vector.extract_strided_slice %341 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %491 = arith.addi %476, %408 overflow<nsw> : index
  vector.store %490, %394[%491] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %492 = vector.extract_strided_slice %341 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %493 = arith.addi %480, %408 overflow<nsw> : index
  vector.store %492, %394[%493] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %494 = vector.extract_strided_slice %341 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %495 = arith.addi %484, %408 overflow<nsw> : index
  vector.store %494, %394[%495] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %496 = vector.extract_strided_slice %341 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %497 = arith.addi %488, %408 overflow<nsw> : index
  vector.store %496, %394[%497] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %498 = vector.extract_strided_slice %343 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %499 = arith.addi %476, %417 overflow<nsw> : index
  vector.store %498, %394[%499] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %500 = vector.extract_strided_slice %343 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %501 = arith.addi %480, %417 overflow<nsw> : index
  vector.store %500, %394[%501] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %502 = vector.extract_strided_slice %343 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %503 = arith.addi %484, %417 overflow<nsw> : index
  vector.store %502, %394[%503] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %504 = vector.extract_strided_slice %343 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %505 = arith.addi %488, %417 overflow<nsw> : index
  vector.store %504, %394[%505] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %506 = vector.extract_strided_slice %345 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %507 = arith.addi %476, %426 overflow<nsw> : index
  vector.store %506, %394[%507] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %508 = vector.extract_strided_slice %345 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %509 = arith.addi %480, %426 overflow<nsw> : index
  vector.store %508, %394[%509] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %510 = vector.extract_strided_slice %345 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %511 = arith.addi %484, %426 overflow<nsw> : index
  vector.store %510, %394[%511] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %512 = vector.extract_strided_slice %345 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %513 = arith.addi %488, %426 overflow<nsw> : index
  vector.store %512, %394[%513] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %514 = vector.extract_strided_slice %347 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %515 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 48)>()[%thread_id_x]
  %516 = arith.muli %515, %c1024 overflow<nsw> : index
  %517 = arith.addi %516, %389 overflow<nsw> : index
  vector.store %514, %394[%517] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %518 = vector.extract_strided_slice %347 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %519 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 49)>()[%thread_id_x]
  %520 = arith.muli %519, %c1024 overflow<nsw> : index
  %521 = arith.addi %520, %389 overflow<nsw> : index
  vector.store %518, %394[%521] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %522 = vector.extract_strided_slice %347 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %523 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 50)>()[%thread_id_x]
  %524 = arith.muli %523, %c1024 overflow<nsw> : index
  %525 = arith.addi %524, %389 overflow<nsw> : index
  vector.store %522, %394[%525] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %526 = vector.extract_strided_slice %347 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %527 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 51)>()[%thread_id_x]
  %528 = arith.muli %527, %c1024 overflow<nsw> : index
  %529 = arith.addi %528, %389 overflow<nsw> : index
  vector.store %526, %394[%529] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %530 = vector.extract_strided_slice %349 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %531 = arith.addi %516, %408 overflow<nsw> : index
  vector.store %530, %394[%531] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %532 = vector.extract_strided_slice %349 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %533 = arith.addi %520, %408 overflow<nsw> : index
  vector.store %532, %394[%533] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %534 = vector.extract_strided_slice %349 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %535 = arith.addi %524, %408 overflow<nsw> : index
  vector.store %534, %394[%535] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %536 = vector.extract_strided_slice %349 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %537 = arith.addi %528, %408 overflow<nsw> : index
  vector.store %536, %394[%537] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %538 = vector.extract_strided_slice %351 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %539 = arith.addi %516, %417 overflow<nsw> : index
  vector.store %538, %394[%539] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %540 = vector.extract_strided_slice %351 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %541 = arith.addi %520, %417 overflow<nsw> : index
  vector.store %540, %394[%541] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %542 = vector.extract_strided_slice %351 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %543 = arith.addi %524, %417 overflow<nsw> : index
  vector.store %542, %394[%543] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %544 = vector.extract_strided_slice %351 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %545 = arith.addi %528, %417 overflow<nsw> : index
  vector.store %544, %394[%545] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %546 = vector.extract_strided_slice %353 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %547 = arith.addi %516, %426 overflow<nsw> : index
  vector.store %546, %394[%547] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %548 = vector.extract_strided_slice %353 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %549 = arith.addi %520, %426 overflow<nsw> : index
  vector.store %548, %394[%549] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %550 = vector.extract_strided_slice %353 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %551 = arith.addi %524, %426 overflow<nsw> : index
  vector.store %550, %394[%551] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %552 = vector.extract_strided_slice %353 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %553 = arith.addi %528, %426 overflow<nsw> : index
  vector.store %552, %394[%553] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %554 = vector.extract_strided_slice %355 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %555 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 64)>()[%thread_id_x]
  %556 = arith.muli %555, %c1024 overflow<nsw> : index
  %557 = arith.addi %556, %389 overflow<nsw> : index
  vector.store %554, %394[%557] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %558 = vector.extract_strided_slice %355 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %559 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 65)>()[%thread_id_x]
  %560 = arith.muli %559, %c1024 overflow<nsw> : index
  %561 = arith.addi %560, %389 overflow<nsw> : index
  vector.store %558, %394[%561] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %562 = vector.extract_strided_slice %355 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %563 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 66)>()[%thread_id_x]
  %564 = arith.muli %563, %c1024 overflow<nsw> : index
  %565 = arith.addi %564, %389 overflow<nsw> : index
  vector.store %562, %394[%565] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %566 = vector.extract_strided_slice %355 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %567 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 67)>()[%thread_id_x]
  %568 = arith.muli %567, %c1024 overflow<nsw> : index
  %569 = arith.addi %568, %389 overflow<nsw> : index
  vector.store %566, %394[%569] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %570 = vector.extract_strided_slice %357 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %571 = arith.addi %556, %408 overflow<nsw> : index
  vector.store %570, %394[%571] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %572 = vector.extract_strided_slice %357 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %573 = arith.addi %560, %408 overflow<nsw> : index
  vector.store %572, %394[%573] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %574 = vector.extract_strided_slice %357 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %575 = arith.addi %564, %408 overflow<nsw> : index
  vector.store %574, %394[%575] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %576 = vector.extract_strided_slice %357 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %577 = arith.addi %568, %408 overflow<nsw> : index
  vector.store %576, %394[%577] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %578 = vector.extract_strided_slice %359 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %579 = arith.addi %556, %417 overflow<nsw> : index
  vector.store %578, %394[%579] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %580 = vector.extract_strided_slice %359 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %581 = arith.addi %560, %417 overflow<nsw> : index
  vector.store %580, %394[%581] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %582 = vector.extract_strided_slice %359 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %583 = arith.addi %564, %417 overflow<nsw> : index
  vector.store %582, %394[%583] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %584 = vector.extract_strided_slice %359 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %585 = arith.addi %568, %417 overflow<nsw> : index
  vector.store %584, %394[%585] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %586 = vector.extract_strided_slice %361 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %587 = arith.addi %556, %426 overflow<nsw> : index
  vector.store %586, %394[%587] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %588 = vector.extract_strided_slice %361 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %589 = arith.addi %560, %426 overflow<nsw> : index
  vector.store %588, %394[%589] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %590 = vector.extract_strided_slice %361 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %591 = arith.addi %564, %426 overflow<nsw> : index
  vector.store %590, %394[%591] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %592 = vector.extract_strided_slice %361 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %593 = arith.addi %568, %426 overflow<nsw> : index
  vector.store %592, %394[%593] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %594 = vector.extract_strided_slice %363 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %595 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 80)>()[%thread_id_x]
  %596 = arith.muli %595, %c1024 overflow<nsw> : index
  %597 = arith.addi %596, %389 overflow<nsw> : index
  vector.store %594, %394[%597] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %598 = vector.extract_strided_slice %363 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %599 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 81)>()[%thread_id_x]
  %600 = arith.muli %599, %c1024 overflow<nsw> : index
  %601 = arith.addi %600, %389 overflow<nsw> : index
  vector.store %598, %394[%601] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %602 = vector.extract_strided_slice %363 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %603 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 82)>()[%thread_id_x]
  %604 = arith.muli %603, %c1024 overflow<nsw> : index
  %605 = arith.addi %604, %389 overflow<nsw> : index
  vector.store %602, %394[%605] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %606 = vector.extract_strided_slice %363 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %607 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 83)>()[%thread_id_x]
  %608 = arith.muli %607, %c1024 overflow<nsw> : index
  %609 = arith.addi %608, %389 overflow<nsw> : index
  vector.store %606, %394[%609] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %610 = vector.extract_strided_slice %365 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %611 = arith.addi %596, %408 overflow<nsw> : index
  vector.store %610, %394[%611] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %612 = vector.extract_strided_slice %365 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %613 = arith.addi %600, %408 overflow<nsw> : index
  vector.store %612, %394[%613] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %614 = vector.extract_strided_slice %365 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %615 = arith.addi %604, %408 overflow<nsw> : index
  vector.store %614, %394[%615] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %616 = vector.extract_strided_slice %365 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %617 = arith.addi %608, %408 overflow<nsw> : index
  vector.store %616, %394[%617] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %618 = vector.extract_strided_slice %367 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %619 = arith.addi %596, %417 overflow<nsw> : index
  vector.store %618, %394[%619] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %620 = vector.extract_strided_slice %367 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %621 = arith.addi %600, %417 overflow<nsw> : index
  vector.store %620, %394[%621] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %622 = vector.extract_strided_slice %367 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %623 = arith.addi %604, %417 overflow<nsw> : index
  vector.store %622, %394[%623] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %624 = vector.extract_strided_slice %367 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %625 = arith.addi %608, %417 overflow<nsw> : index
  vector.store %624, %394[%625] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %626 = vector.extract_strided_slice %369 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %627 = arith.addi %596, %426 overflow<nsw> : index
  vector.store %626, %394[%627] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %628 = vector.extract_strided_slice %369 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %629 = arith.addi %600, %426 overflow<nsw> : index
  vector.store %628, %394[%629] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %630 = vector.extract_strided_slice %369 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %631 = arith.addi %604, %426 overflow<nsw> : index
  vector.store %630, %394[%631] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %632 = vector.extract_strided_slice %369 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %633 = arith.addi %608, %426 overflow<nsw> : index
  vector.store %632, %394[%633] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %634 = vector.extract_strided_slice %371 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %635 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 96)>()[%thread_id_x]
  %636 = arith.muli %635, %c1024 overflow<nsw> : index
  %637 = arith.addi %636, %389 overflow<nsw> : index
  vector.store %634, %394[%637] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %638 = vector.extract_strided_slice %371 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %639 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 97)>()[%thread_id_x]
  %640 = arith.muli %639, %c1024 overflow<nsw> : index
  %641 = arith.addi %640, %389 overflow<nsw> : index
  vector.store %638, %394[%641] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %642 = vector.extract_strided_slice %371 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %643 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 98)>()[%thread_id_x]
  %644 = arith.muli %643, %c1024 overflow<nsw> : index
  %645 = arith.addi %644, %389 overflow<nsw> : index
  vector.store %642, %394[%645] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %646 = vector.extract_strided_slice %371 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %647 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 99)>()[%thread_id_x]
  %648 = arith.muli %647, %c1024 overflow<nsw> : index
  %649 = arith.addi %648, %389 overflow<nsw> : index
  vector.store %646, %394[%649] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %650 = vector.extract_strided_slice %373 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %651 = arith.addi %636, %408 overflow<nsw> : index
  vector.store %650, %394[%651] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %652 = vector.extract_strided_slice %373 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %653 = arith.addi %640, %408 overflow<nsw> : index
  vector.store %652, %394[%653] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %654 = vector.extract_strided_slice %373 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %655 = arith.addi %644, %408 overflow<nsw> : index
  vector.store %654, %394[%655] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %656 = vector.extract_strided_slice %373 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %657 = arith.addi %648, %408 overflow<nsw> : index
  vector.store %656, %394[%657] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %658 = vector.extract_strided_slice %375 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %659 = arith.addi %636, %417 overflow<nsw> : index
  vector.store %658, %394[%659] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %660 = vector.extract_strided_slice %375 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %661 = arith.addi %640, %417 overflow<nsw> : index
  vector.store %660, %394[%661] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %662 = vector.extract_strided_slice %375 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %663 = arith.addi %644, %417 overflow<nsw> : index
  vector.store %662, %394[%663] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %664 = vector.extract_strided_slice %375 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %665 = arith.addi %648, %417 overflow<nsw> : index
  vector.store %664, %394[%665] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %666 = vector.extract_strided_slice %377 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %667 = arith.addi %636, %426 overflow<nsw> : index
  vector.store %666, %394[%667] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %668 = vector.extract_strided_slice %377 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %669 = arith.addi %640, %426 overflow<nsw> : index
  vector.store %668, %394[%669] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %670 = vector.extract_strided_slice %377 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %671 = arith.addi %644, %426 overflow<nsw> : index
  vector.store %670, %394[%671] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %672 = vector.extract_strided_slice %377 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %673 = arith.addi %648, %426 overflow<nsw> : index
  vector.store %672, %394[%673] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %674 = vector.extract_strided_slice %379 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %675 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 112)>()[%thread_id_x]
  %676 = arith.muli %675, %c1024 overflow<nsw> : index
  %677 = arith.addi %676, %389 overflow<nsw> : index
  vector.store %674, %394[%677] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %678 = vector.extract_strided_slice %379 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %679 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 113)>()[%thread_id_x]
  %680 = arith.muli %679, %c1024 overflow<nsw> : index
  %681 = arith.addi %680, %389 overflow<nsw> : index
  vector.store %678, %394[%681] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %682 = vector.extract_strided_slice %379 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %683 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 114)>()[%thread_id_x]
  %684 = arith.muli %683, %c1024 overflow<nsw> : index
  %685 = arith.addi %684, %389 overflow<nsw> : index
  vector.store %682, %394[%685] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %686 = vector.extract_strided_slice %379 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %687 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 115)>()[%thread_id_x]
  %688 = arith.muli %687, %c1024 overflow<nsw> : index
  %689 = arith.addi %688, %389 overflow<nsw> : index
  vector.store %686, %394[%689] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %690 = vector.extract_strided_slice %381 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %691 = arith.addi %676, %408 overflow<nsw> : index
  vector.store %690, %394[%691] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %692 = vector.extract_strided_slice %381 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %693 = arith.addi %680, %408 overflow<nsw> : index
  vector.store %692, %394[%693] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %694 = vector.extract_strided_slice %381 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %695 = arith.addi %684, %408 overflow<nsw> : index
  vector.store %694, %394[%695] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %696 = vector.extract_strided_slice %381 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %697 = arith.addi %688, %408 overflow<nsw> : index
  vector.store %696, %394[%697] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %698 = vector.extract_strided_slice %383 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %699 = arith.addi %676, %417 overflow<nsw> : index
  vector.store %698, %394[%699] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %700 = vector.extract_strided_slice %383 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %701 = arith.addi %680, %417 overflow<nsw> : index
  vector.store %700, %394[%701] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %702 = vector.extract_strided_slice %383 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %703 = arith.addi %684, %417 overflow<nsw> : index
  vector.store %702, %394[%703] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %704 = vector.extract_strided_slice %383 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %705 = arith.addi %688, %417 overflow<nsw> : index
  vector.store %704, %394[%705] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %706 = vector.extract_strided_slice %385 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %707 = arith.addi %676, %426 overflow<nsw> : index
  vector.store %706, %394[%707] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %708 = vector.extract_strided_slice %385 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %709 = arith.addi %680, %426 overflow<nsw> : index
  vector.store %708, %394[%709] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %710 = vector.extract_strided_slice %385 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %711 = arith.addi %684, %426 overflow<nsw> : index
  vector.store %710, %394[%711] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %712 = vector.extract_strided_slice %385 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %713 = arith.addi %688, %426 overflow<nsw> : index
  vector.store %712, %394[%713] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  return
}
}
