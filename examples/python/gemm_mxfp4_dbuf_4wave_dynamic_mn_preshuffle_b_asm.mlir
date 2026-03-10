module {
func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding, %arg5: index, %arg6: index) attributes {translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 4, 1] subgroup_size = 64>} {
  %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
  %cst_0 = arith.constant dense<2147483647> : vector<16xindex>
  %c256_i14 = arith.constant 256 : i14
  %c4096_i14 = arith.constant 4096 : i14
  %c536870911 = arith.constant 536870911 : index
  %c2147483643_i64 = arith.constant 2147483643 : i64
  %c30 = arith.constant 30 : index
  %c2 = arith.constant 2 : index
  %c256 = arith.constant 256 : index
  %c3 = arith.constant 3 : index
  %c2147483647 = arith.constant 2147483647 : index
  %c2147483646_i64 = arith.constant 2147483646 : i64
  %c4096 = arith.constant 4096 : index
  %cst_1 = arith.constant dense<0.000000e+00> : vector<4xf32>
  %c0 = arith.constant 0 : index
  %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<i8>
  %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<i8>
  %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<i8>
  %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<i8>
  %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<f32>
  %block_id_x = gpu.block_id  x upper_bound 2147483647
  %block_id_y = gpu.block_id  y upper_bound 2147483647
  %thread_id_x = gpu.thread_id  x upper_bound 64
  %thread_id_y = gpu.thread_id  y upper_bound 4
  %alloc = memref.alloc() : memref<1024x1xi8, #gpu.address_space<workgroup>>
  %alloc_2 = memref.alloc() : memref<1024x1xi8, #gpu.address_space<workgroup>>
  %alloc_3 = memref.alloc() : memref<128x128xi8, #gpu.address_space<workgroup>>
  %alloc_4 = memref.alloc() : memref<128x128xi8, #gpu.address_space<workgroup>>
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
  %16 = arith.cmpi slt, %5, %arg5 : index
  %17 = arith.select %16, %14, %c2147483647 : index
  amdgpu.gather_to_lds %15[%17], %alloc_4[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
  %18 = affine.apply affine_map<()[s0] -> (s0 * 16 + 128)>()[%8]
  %19 = arith.addi %13, %18 overflow<nsw> : index
  %20 = arith.select %16, %19, %c2147483647 : index
  amdgpu.gather_to_lds %15[%20], %alloc_3[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
  %21 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 128 + s0 floordiv 8 - ((s1 * 8 + s0 floordiv 8 + 32) floordiv 128) * 128 + 32)>()[%thread_id_x, %thread_id_y, %block_id_x]
  %22 = affine.apply affine_map<()[s0] -> (s0 * 8 - ((s0 + 4) floordiv 16) * 128 + 32)>()[%thread_id_y]
  %23 = gpu.subgroup_broadcast %22,  first_active_lane : index
  %24 = arith.muli %21, %c4096 overflow<nsw> : index
  %25 = arith.addi %24, %9 overflow<nsw> : index
  %26 = arith.cmpi slt, %21, %arg5 : index
  %27 = arith.select %26, %25, %c2147483647 : index
  amdgpu.gather_to_lds %15[%27], %alloc_4[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
  %28 = arith.addi %24, %18 overflow<nsw> : index
  %29 = arith.select %26, %28, %c2147483647 : index
  amdgpu.gather_to_lds %15[%29], %alloc_3[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
  %30 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 128 + s0 floordiv 8 - ((s1 * 8 + s0 floordiv 8 + 64) floordiv 128) * 128 + 64)>()[%thread_id_x, %thread_id_y, %block_id_x]
  %31 = affine.apply affine_map<()[s0] -> (s0 * 8 - ((s0 + 8) floordiv 16) * 128 + 64)>()[%thread_id_y]
  %32 = gpu.subgroup_broadcast %31,  first_active_lane : index
  %33 = arith.muli %30, %c4096 overflow<nsw> : index
  %34 = arith.addi %33, %9 overflow<nsw> : index
  %35 = arith.cmpi slt, %30, %arg5 : index
  %36 = arith.select %35, %34, %c2147483647 : index
  amdgpu.gather_to_lds %15[%36], %alloc_4[%32, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
  %37 = arith.addi %33, %18 overflow<nsw> : index
  %38 = arith.select %35, %37, %c2147483647 : index
  amdgpu.gather_to_lds %15[%38], %alloc_3[%32, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
  %39 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 128 + s0 floordiv 8 - ((s1 * 8 + s0 floordiv 8 + 96) floordiv 128) * 128 + 96)>()[%thread_id_x, %thread_id_y, %block_id_x]
  %40 = affine.apply affine_map<()[s0] -> (s0 * 8 - ((s0 + 12) floordiv 16) * 128 + 96)>()[%thread_id_y]
  %41 = gpu.subgroup_broadcast %40,  first_active_lane : index
  %42 = arith.muli %39, %c4096 overflow<nsw> : index
  %43 = arith.addi %42, %9 overflow<nsw> : index
  %44 = arith.cmpi slt, %39, %arg5 : index
  %45 = arith.select %44, %43, %c2147483647 : index
  amdgpu.gather_to_lds %15[%45], %alloc_4[%41, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
  %46 = arith.addi %42, %18 overflow<nsw> : index
  %47 = arith.select %44, %46, %c2147483647 : index
  amdgpu.gather_to_lds %15[%47], %alloc_3[%41, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
  %48 = affine.apply affine_map<()[s0, s1] -> (s1 + s0 floordiv 64)>()[%thread_id_x, %thread_id_y]
  %49 = arith.minsi %48, %c3 : index
  %50 = affine.apply affine_map<()[s0, s1] -> (s0 * 128 + s1 * 32)>()[%block_id_x, %49]
  %51 = affine.apply affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 64) * 256)>()[%thread_id_x]
  %52 = arith.minsi %thread_id_y, %c3 : index
  %53 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%52]
  %54 = gpu.subgroup_broadcast %53,  first_active_lane : index
  %55 = arith.muli %50, %c256 overflow<nsw> : index
  %56 = arith.addi %55, %51 overflow<nsw> : index
  %reinterpret_cast_5 = memref.reinterpret_cast %1 to offset: [0], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1]>>
  %cast_6 = memref.cast %reinterpret_cast_5 : memref<2147483646xi8, strided<[1]>> to memref<?xi8, strided<[1], offset: ?>>
  %57 = amdgpu.fat_raw_buffer_cast %cast_6 validBytes(%c2147483646_i64) cacheSwizzleStride(%c256_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
  amdgpu.gather_to_lds %57[%56], %alloc_2[%54, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<1024x1xi8, #gpu.address_space<workgroup>>
  %58 = affine.apply affine_map<()[s0, s1] -> (s0 * 128 + s1 * 32 + 1)>()[%block_id_x, %49]
  %59 = arith.muli %58, %c256 overflow<nsw> : index
  %60 = arith.addi %59, %51 overflow<nsw> : index
  amdgpu.gather_to_lds %57[%60], %alloc[%54, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<1024x1xi8, #gpu.address_space<workgroup>>
  %61 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 64 - (s0 floordiv 16) * 16)>()[%thread_id_x, %block_id_y, %thread_id_y]
  %62 = arith.cmpi slt, %61, %arg6 : index
  %63 = vector.broadcast %62 : i1 to vector<16xi1>
  %64 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%block_id_y]
  %65 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256)>()[%thread_id_x, %thread_id_y]
  %66 = affine.apply affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256) * 4096)>()[%thread_id_x]
  %67 = arith.muli %64, %c4096 overflow<nsw> : index
  %68 = arith.muli %65, %c4096 overflow<nsw> : index
  %69 = arith.addi %68, %66 overflow<nsw> : index
  %reinterpret_cast_7 = memref.reinterpret_cast %2 to offset: [%67], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1], offset: ?>>
  %cast_8 = memref.cast %reinterpret_cast_7 : memref<2147483646xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
  %70 = amdgpu.fat_raw_buffer_cast %cast_8 validBytes(%c2147483646_i64) cacheSwizzleStride(%c4096_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
  %71 = arith.index_cast %69 : index to i32
  %72 = vector.broadcast %71 : i32 to vector<16xi32>
  %73 = arith.addi %72, %cst : vector<16xi32>
  %74 = arith.index_cast %73 : vector<16xi32> to vector<16xindex>
  %75 = arith.select %63, %74, %cst_0 : vector<16xi1>, vector<16xindex>
  %76 = vector.extract %75[0] : index from vector<16xindex>
  %77 = vector.load %70[%76] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %78 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256)>()[%thread_id_x, %thread_id_y]
  %79 = affine.apply affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256) * 4096 + 1024)>()[%thread_id_x]
  %80 = arith.muli %78, %c4096 overflow<nsw> : index
  %81 = arith.addi %80, %79 overflow<nsw> : index
  %82 = arith.index_cast %81 : index to i32
  %83 = vector.broadcast %82 : i32 to vector<16xi32>
  %84 = arith.addi %83, %cst : vector<16xi32>
  %85 = arith.index_cast %84 : vector<16xi32> to vector<16xindex>
  %86 = arith.select %63, %85, %cst_0 : vector<16xi1>, vector<16xindex>
  %87 = vector.extract %86[0] : index from vector<16xindex>
  %88 = vector.load %70[%87] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %89 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 64 - (s0 floordiv 16) * 16 + 16)>()[%thread_id_x, %block_id_y, %thread_id_y]
  %90 = arith.cmpi slt, %89, %arg6 : index
  %91 = vector.broadcast %90 : i1 to vector<16xi1>
  %92 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 16)>()[%thread_id_x, %thread_id_y]
  %93 = arith.muli %92, %c4096 overflow<nsw> : index
  %94 = arith.addi %93, %66 overflow<nsw> : index
  %95 = arith.index_cast %94 : index to i32
  %96 = vector.broadcast %95 : i32 to vector<16xi32>
  %97 = arith.addi %96, %cst : vector<16xi32>
  %98 = arith.index_cast %97 : vector<16xi32> to vector<16xindex>
  %99 = arith.select %91, %98, %cst_0 : vector<16xi1>, vector<16xindex>
  %100 = vector.extract %99[0] : index from vector<16xindex>
  %101 = vector.load %70[%100] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %102 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256 + 16)>()[%thread_id_x, %thread_id_y]
  %103 = arith.muli %102, %c4096 overflow<nsw> : index
  %104 = arith.addi %103, %79 overflow<nsw> : index
  %105 = arith.index_cast %104 : index to i32
  %106 = vector.broadcast %105 : i32 to vector<16xi32>
  %107 = arith.addi %106, %cst : vector<16xi32>
  %108 = arith.index_cast %107 : vector<16xi32> to vector<16xindex>
  %109 = arith.select %91, %108, %cst_0 : vector<16xi1>, vector<16xindex>
  %110 = vector.extract %109[0] : index from vector<16xindex>
  %111 = vector.load %70[%110] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %112 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 64 - (s0 floordiv 16) * 16 + 32)>()[%thread_id_x, %block_id_y, %thread_id_y]
  %113 = arith.cmpi slt, %112, %arg6 : index
  %114 = vector.broadcast %113 : i1 to vector<16xi1>
  %115 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 32)>()[%thread_id_x, %thread_id_y]
  %116 = arith.muli %115, %c4096 overflow<nsw> : index
  %117 = arith.addi %116, %66 overflow<nsw> : index
  %118 = arith.index_cast %117 : index to i32
  %119 = vector.broadcast %118 : i32 to vector<16xi32>
  %120 = arith.addi %119, %cst : vector<16xi32>
  %121 = arith.index_cast %120 : vector<16xi32> to vector<16xindex>
  %122 = arith.select %114, %121, %cst_0 : vector<16xi1>, vector<16xindex>
  %123 = vector.extract %122[0] : index from vector<16xindex>
  %124 = vector.load %70[%123] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %125 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256 + 32)>()[%thread_id_x, %thread_id_y]
  %126 = arith.muli %125, %c4096 overflow<nsw> : index
  %127 = arith.addi %126, %79 overflow<nsw> : index
  %128 = arith.index_cast %127 : index to i32
  %129 = vector.broadcast %128 : i32 to vector<16xi32>
  %130 = arith.addi %129, %cst : vector<16xi32>
  %131 = arith.index_cast %130 : vector<16xi32> to vector<16xindex>
  %132 = arith.select %114, %131, %cst_0 : vector<16xi1>, vector<16xindex>
  %133 = vector.extract %132[0] : index from vector<16xindex>
  %134 = vector.load %70[%133] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %135 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * 256 + s2 * 64 - (s0 floordiv 16) * 16 + 48)>()[%thread_id_x, %block_id_y, %thread_id_y]
  %136 = arith.cmpi slt, %135, %arg6 : index
  %137 = vector.broadcast %136 : i1 to vector<16xi1>
  %138 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 48)>()[%thread_id_x, %thread_id_y]
  %139 = arith.muli %138, %c4096 overflow<nsw> : index
  %140 = arith.addi %139, %66 overflow<nsw> : index
  %141 = arith.index_cast %140 : index to i32
  %142 = vector.broadcast %141 : i32 to vector<16xi32>
  %143 = arith.addi %142, %cst : vector<16xi32>
  %144 = arith.index_cast %143 : vector<16xi32> to vector<16xindex>
  %145 = arith.select %137, %144, %cst_0 : vector<16xi1>, vector<16xindex>
  %146 = vector.extract %145[0] : index from vector<16xindex>
  %147 = vector.load %70[%146] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %148 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256 + 48)>()[%thread_id_x, %thread_id_y]
  %149 = arith.muli %148, %c4096 overflow<nsw> : index
  %150 = arith.addi %149, %79 overflow<nsw> : index
  %151 = arith.index_cast %150 : index to i32
  %152 = vector.broadcast %151 : i32 to vector<16xi32>
  %153 = arith.addi %152, %cst : vector<16xi32>
  %154 = arith.index_cast %153 : vector<16xi32> to vector<16xindex>
  %155 = arith.select %137, %154, %cst_0 : vector<16xi1>, vector<16xindex>
  %156 = vector.extract %155[0] : index from vector<16xindex>
  %157 = vector.load %70[%156] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  rocdl.sched.barrier 0
  %158 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64)>()[%thread_id_x, %thread_id_y]
  %159 = affine.apply affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 16) * 64 + ((s0 mod 64) floordiv 16) * 64 - ((s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64) * 256)>()[%thread_id_x]
  %160 = arith.muli %64, %c256 overflow<nsw> : index
  %161 = arith.muli %158, %c256 overflow<nsw> : index
  %162 = arith.addi %161, %159 overflow<nsw> : index
  %reinterpret_cast_9 = memref.reinterpret_cast %3 to offset: [%160], sizes: [2147483646], strides: [1] : memref<i8> to memref<2147483646xi8, strided<[1], offset: ?>>
  %cast_10 = memref.cast %reinterpret_cast_9 : memref<2147483646xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
  %163 = amdgpu.fat_raw_buffer_cast %cast_10 validBytes(%c2147483646_i64) cacheSwizzleStride(%c256_i14) resetOffset : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>
  %164 = vector.load %163[%162] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
  %165 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64 + 32)>()[%thread_id_x, %thread_id_y]
  %166 = arith.muli %165, %c256 overflow<nsw> : index
  %167 = arith.addi %166, %159 overflow<nsw> : index
  %168 = vector.load %163[%167] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
  rocdl.sched.barrier 0
  amdgpu.memory_counter_wait load(0)
  rocdl.s.barrier
  rocdl.sched.barrier 0
  %169 = affine.apply affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 128)>()[%thread_id_x]
  %170 = affine.apply affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>()[%thread_id_x]
  %171 = arith.xori %170, %7 : index
  %172 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%171]
  %173 = vector.load %alloc_4[%169, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %174 = affine.apply affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>()[%thread_id_x]
  %175 = arith.xori %174, %7 : index
  %176 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%175]
  %177 = vector.load %alloc_4[%169, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %178 = affine.apply affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 128 + 16)>()[%thread_id_x]
  %179 = vector.load %alloc_4[%178, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %180 = vector.load %alloc_4[%178, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %181 = affine.apply affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 128 + 32)>()[%thread_id_x]
  %182 = vector.load %alloc_4[%181, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %183 = vector.load %alloc_4[%181, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %184 = affine.apply affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 128 + 48)>()[%thread_id_x]
  %185 = vector.load %alloc_4[%184, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %186 = vector.load %alloc_4[%184, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %187 = affine.apply affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 768)>()[%thread_id_x]
  %188 = vector.load %alloc_2[%c0, %187] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
  %189 = affine.apply affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 768 + 256)>()[%thread_id_x]
  %190 = vector.load %alloc_2[%c0, %189] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
  %191 = affine.apply affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 128 + 64)>()[%thread_id_x]
  %192 = affine.apply affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 128 + 80)>()[%thread_id_x]
  %193 = affine.apply affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 128 + 96)>()[%thread_id_x]
  %194 = affine.apply affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 128 + 112)>()[%thread_id_x]
  %195 = affine.apply affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 768 + 512)>()[%thread_id_x]
  %196 = affine.apply affine_map<()[s0] -> (s0 * 4 + (s0 floordiv 64) * 768 + 768)>()[%thread_id_x]
  %197:52 = scf.for %arg7 = %c0 to %c30 step %c2 iter_args(%arg8 = %cst_1, %arg9 = %cst_1, %arg10 = %cst_1, %arg11 = %cst_1, %arg12 = %cst_1, %arg13 = %cst_1, %arg14 = %cst_1, %arg15 = %cst_1, %arg16 = %cst_1, %arg17 = %cst_1, %arg18 = %cst_1, %arg19 = %cst_1, %arg20 = %cst_1, %arg21 = %cst_1, %arg22 = %cst_1, %arg23 = %cst_1, %arg24 = %cst_1, %arg25 = %cst_1, %arg26 = %cst_1, %arg27 = %cst_1, %arg28 = %cst_1, %arg29 = %cst_1, %arg30 = %cst_1, %arg31 = %cst_1, %arg32 = %cst_1, %arg33 = %cst_1, %arg34 = %cst_1, %arg35 = %cst_1, %arg36 = %cst_1, %arg37 = %cst_1, %arg38 = %cst_1, %arg39 = %cst_1, %arg40 = %188, %arg41 = %190, %arg42 = %173, %arg43 = %177, %arg44 = %179, %arg45 = %180, %arg46 = %182, %arg47 = %183, %arg48 = %185, %arg49 = %186, %arg50 = %77, %arg51 = %88, %arg52 = %101, %arg53 = %111, %arg54 = %124, %arg55 = %134, %arg56 = %147, %arg57 = %157, %arg58 = %164, %arg59 = %168) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<4xi8>, vector<4xi8>) {
    %1154 = vector.bitcast %arg41 : vector<4xi8> to vector<4xf8E8M0FNU>
    %1155 = vector.bitcast %arg59 : vector<4xi8> to vector<4xf8E8M0FNU>
    %1156 = vector.bitcast %arg58 : vector<4xi8> to vector<4xf8E8M0FNU>
    %1157 = vector.bitcast %arg40 : vector<4xi8> to vector<4xf8E8M0FNU>
    amdgpu.memory_counter_wait load(5) ds(0)
    rocdl.s.barrier
    %1158 = vector.bitcast %arg42 : vector<16xi8> to vector<32xf4E2M1FN>
    %1159 = vector.bitcast %arg43 : vector<16xi8> to vector<32xf4E2M1FN>
    %1160 = vector.bitcast %arg44 : vector<16xi8> to vector<32xf4E2M1FN>
    %1161 = vector.bitcast %arg45 : vector<16xi8> to vector<32xf4E2M1FN>
    %1162 = vector.bitcast %arg46 : vector<16xi8> to vector<32xf4E2M1FN>
    %1163 = vector.bitcast %arg47 : vector<16xi8> to vector<32xf4E2M1FN>
    %1164 = vector.bitcast %arg48 : vector<16xi8> to vector<32xf4E2M1FN>
    %1165 = vector.bitcast %arg49 : vector<16xi8> to vector<32xf4E2M1FN>
    %1166 = vector.bitcast %arg50 : vector<16xi8> to vector<32xf4E2M1FN>
    %1167 = vector.bitcast %arg51 : vector<16xi8> to vector<32xf4E2M1FN>
    %1168 = vector.bitcast %arg52 : vector<16xi8> to vector<32xf4E2M1FN>
    %1169 = vector.bitcast %arg53 : vector<16xi8> to vector<32xf4E2M1FN>
    %1170 = vector.bitcast %arg54 : vector<16xi8> to vector<32xf4E2M1FN>
    %1171 = vector.bitcast %arg55 : vector<16xi8> to vector<32xf4E2M1FN>
    %1172 = vector.bitcast %arg56 : vector<16xi8> to vector<32xf4E2M1FN>
    %1173 = vector.bitcast %arg57 : vector<16xi8> to vector<32xf4E2M1FN>
    rocdl.sched.barrier 0
    %1174 = amdgpu.scaled_mfma 16x16x128 (%1157[0] * %1158) * (%1156[0] * %1166) + %arg8 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1175 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 128) floordiv 256)>()[%thread_id_x, %arg7, %thread_id_y]
    %1176 = affine.apply affine_map<()[s0, s1] -> (s0 * 16 + s1 * 2048 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 128) floordiv 256) * 4096 + 2048)>()[%thread_id_x, %arg7]
    %1177 = arith.muli %1175, %c4096 overflow<nsw> : index
    %1178 = arith.addi %1177, %1176 overflow<nsw> : index
    %1179 = arith.index_cast %1178 : index to i32
    %1180 = vector.broadcast %1179 : i32 to vector<16xi32>
    %1181 = arith.addi %1180, %cst : vector<16xi32>
    %1182 = arith.index_cast %1181 : vector<16xi32> to vector<16xindex>
    %1183 = arith.select %63, %1182, %cst_0 : vector<16xi1>, vector<16xindex>
    %1184 = vector.extract %1183[0] : index from vector<16xindex>
    %1185 = vector.load %70[%1184] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %1186 = amdgpu.scaled_mfma 16x16x128 (%1157[2] * %1159) * (%1156[2] * %1167) + %1174 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1187 = amdgpu.scaled_mfma 16x16x128 (%1157[0] * %1158) * (%1156[1] * %1168) + %arg9 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1188 = amdgpu.scaled_mfma 16x16x128 (%1157[2] * %1159) * (%1156[3] * %1169) + %1187 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1189 = vector.load %alloc_4[%191, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1190 = amdgpu.scaled_mfma 16x16x128 (%1157[0] * %1158) * (%1155[0] * %1170) + %arg10 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1191 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 192) floordiv 256)>()[%thread_id_x, %arg7, %thread_id_y]
    %1192 = affine.apply affine_map<()[s0, s1] -> (s0 * 16 + s1 * 2048 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 192) floordiv 256) * 4096 + 3072)>()[%thread_id_x, %arg7]
    %1193 = arith.muli %1191, %c4096 overflow<nsw> : index
    %1194 = arith.addi %1193, %1192 overflow<nsw> : index
    %1195 = arith.index_cast %1194 : index to i32
    %1196 = vector.broadcast %1195 : i32 to vector<16xi32>
    %1197 = arith.addi %1196, %cst : vector<16xi32>
    %1198 = arith.index_cast %1197 : vector<16xi32> to vector<16xindex>
    %1199 = arith.select %63, %1198, %cst_0 : vector<16xi1>, vector<16xindex>
    %1200 = vector.extract %1199[0] : index from vector<16xindex>
    %1201 = vector.load %70[%1200] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %1202 = amdgpu.scaled_mfma 16x16x128 (%1157[2] * %1159) * (%1155[2] * %1171) + %1190 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1203 = amdgpu.scaled_mfma 16x16x128 (%1157[0] * %1158) * (%1155[1] * %1172) + %arg11 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1204 = amdgpu.scaled_mfma 16x16x128 (%1157[2] * %1159) * (%1155[3] * %1173) + %1203 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1205 = vector.load %alloc_4[%191, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1206 = amdgpu.scaled_mfma 16x16x128 (%1157[1] * %1160) * (%1156[0] * %1166) + %arg12 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1207 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 128) floordiv 256 + 16)>()[%thread_id_x, %arg7, %thread_id_y]
    %1208 = arith.muli %1207, %c4096 overflow<nsw> : index
    %1209 = arith.addi %1208, %1176 overflow<nsw> : index
    %1210 = arith.index_cast %1209 : index to i32
    %1211 = vector.broadcast %1210 : i32 to vector<16xi32>
    %1212 = arith.addi %1211, %cst : vector<16xi32>
    %1213 = arith.index_cast %1212 : vector<16xi32> to vector<16xindex>
    %1214 = arith.select %91, %1213, %cst_0 : vector<16xi1>, vector<16xindex>
    %1215 = vector.extract %1214[0] : index from vector<16xindex>
    %1216 = vector.load %70[%1215] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %1217 = amdgpu.scaled_mfma 16x16x128 (%1157[3] * %1161) * (%1156[2] * %1167) + %1206 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1218 = amdgpu.scaled_mfma 16x16x128 (%1157[1] * %1160) * (%1156[1] * %1168) + %arg13 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1219 = amdgpu.scaled_mfma 16x16x128 (%1157[3] * %1161) * (%1156[3] * %1169) + %1218 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1220 = vector.load %alloc_4[%192, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1221 = amdgpu.scaled_mfma 16x16x128 (%1157[1] * %1160) * (%1155[0] * %1170) + %arg14 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1222 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 192) floordiv 256 + 16)>()[%thread_id_x, %arg7, %thread_id_y]
    %1223 = arith.muli %1222, %c4096 overflow<nsw> : index
    %1224 = arith.addi %1223, %1192 overflow<nsw> : index
    %1225 = arith.index_cast %1224 : index to i32
    %1226 = vector.broadcast %1225 : i32 to vector<16xi32>
    %1227 = arith.addi %1226, %cst : vector<16xi32>
    %1228 = arith.index_cast %1227 : vector<16xi32> to vector<16xindex>
    %1229 = arith.select %91, %1228, %cst_0 : vector<16xi1>, vector<16xindex>
    %1230 = vector.extract %1229[0] : index from vector<16xindex>
    %1231 = vector.load %70[%1230] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %1232 = amdgpu.scaled_mfma 16x16x128 (%1157[3] * %1161) * (%1155[2] * %1171) + %1221 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1233 = amdgpu.scaled_mfma 16x16x128 (%1157[1] * %1160) * (%1155[1] * %1172) + %arg15 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1234 = amdgpu.scaled_mfma 16x16x128 (%1157[3] * %1161) * (%1155[3] * %1173) + %1233 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1235 = vector.load %alloc_4[%192, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1236 = amdgpu.scaled_mfma 16x16x128 (%1154[0] * %1162) * (%1156[0] * %1166) + %arg16 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1237 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 128) floordiv 256 + 32)>()[%thread_id_x, %arg7, %thread_id_y]
    %1238 = arith.muli %1237, %c4096 overflow<nsw> : index
    %1239 = arith.addi %1238, %1176 overflow<nsw> : index
    %1240 = arith.index_cast %1239 : index to i32
    %1241 = vector.broadcast %1240 : i32 to vector<16xi32>
    %1242 = arith.addi %1241, %cst : vector<16xi32>
    %1243 = arith.index_cast %1242 : vector<16xi32> to vector<16xindex>
    %1244 = arith.select %114, %1243, %cst_0 : vector<16xi1>, vector<16xindex>
    %1245 = vector.extract %1244[0] : index from vector<16xindex>
    %1246 = vector.load %70[%1245] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %1247 = amdgpu.scaled_mfma 16x16x128 (%1154[2] * %1163) * (%1156[2] * %1167) + %1236 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1248 = amdgpu.scaled_mfma 16x16x128 (%1154[0] * %1162) * (%1156[1] * %1168) + %arg17 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1249 = amdgpu.scaled_mfma 16x16x128 (%1154[2] * %1163) * (%1156[3] * %1169) + %1248 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1250 = vector.load %alloc_4[%193, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1251 = amdgpu.scaled_mfma 16x16x128 (%1154[0] * %1162) * (%1155[0] * %1170) + %arg18 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1252 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 192) floordiv 256 + 32)>()[%thread_id_x, %arg7, %thread_id_y]
    %1253 = arith.muli %1252, %c4096 overflow<nsw> : index
    %1254 = arith.addi %1253, %1192 overflow<nsw> : index
    %1255 = arith.index_cast %1254 : index to i32
    %1256 = vector.broadcast %1255 : i32 to vector<16xi32>
    %1257 = arith.addi %1256, %cst : vector<16xi32>
    %1258 = arith.index_cast %1257 : vector<16xi32> to vector<16xindex>
    %1259 = arith.select %114, %1258, %cst_0 : vector<16xi1>, vector<16xindex>
    %1260 = vector.extract %1259[0] : index from vector<16xindex>
    %1261 = vector.load %70[%1260] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %1262 = amdgpu.scaled_mfma 16x16x128 (%1154[2] * %1163) * (%1155[2] * %1171) + %1251 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1263 = amdgpu.scaled_mfma 16x16x128 (%1154[0] * %1162) * (%1155[1] * %1172) + %arg19 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1264 = amdgpu.scaled_mfma 16x16x128 (%1154[2] * %1163) * (%1155[3] * %1173) + %1263 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1265 = vector.load %alloc_4[%193, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1266 = amdgpu.scaled_mfma 16x16x128 (%1154[1] * %1164) * (%1156[0] * %1166) + %arg20 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1267 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 128) floordiv 256 + 48)>()[%thread_id_x, %arg7, %thread_id_y]
    %1268 = arith.muli %1267, %c4096 overflow<nsw> : index
    %1269 = arith.addi %1268, %1176 overflow<nsw> : index
    %1270 = arith.index_cast %1269 : index to i32
    %1271 = vector.broadcast %1270 : i32 to vector<16xi32>
    %1272 = arith.addi %1271, %cst : vector<16xi32>
    %1273 = arith.index_cast %1272 : vector<16xi32> to vector<16xindex>
    %1274 = arith.select %137, %1273, %cst_0 : vector<16xi1>, vector<16xindex>
    %1275 = vector.extract %1274[0] : index from vector<16xindex>
    %1276 = vector.load %70[%1275] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %1277 = amdgpu.scaled_mfma 16x16x128 (%1154[3] * %1165) * (%1156[2] * %1167) + %1266 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1278 = amdgpu.scaled_mfma 16x16x128 (%1154[1] * %1164) * (%1156[1] * %1168) + %arg21 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1279 = amdgpu.scaled_mfma 16x16x128 (%1154[3] * %1165) * (%1156[3] * %1169) + %1278 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1280 = vector.load %alloc_4[%194, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1281 = amdgpu.scaled_mfma 16x16x128 (%1154[1] * %1164) * (%1155[0] * %1170) + %arg22 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1282 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 192) floordiv 256 + 48)>()[%thread_id_x, %arg7, %thread_id_y]
    %1283 = arith.muli %1282, %c4096 overflow<nsw> : index
    %1284 = arith.addi %1283, %1192 overflow<nsw> : index
    %1285 = arith.index_cast %1284 : index to i32
    %1286 = vector.broadcast %1285 : i32 to vector<16xi32>
    %1287 = arith.addi %1286, %cst : vector<16xi32>
    %1288 = arith.index_cast %1287 : vector<16xi32> to vector<16xindex>
    %1289 = arith.select %137, %1288, %cst_0 : vector<16xi1>, vector<16xindex>
    %1290 = vector.extract %1289[0] : index from vector<16xindex>
    %1291 = vector.load %70[%1290] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %1292 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 64 + s2 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64 + 1)>()[%thread_id_x, %thread_id_y, %arg7]
    %1293 = arith.muli %1292, %c256 overflow<nsw> : index
    %1294 = arith.addi %1293, %159 overflow<nsw> : index
    %1295 = vector.load %163[%1294] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
    %1296 = vector.bitcast %1295 : vector<4xi8> to vector<4xf8E8M0FNU>
    %1297 = amdgpu.scaled_mfma 16x16x128 (%1154[3] * %1165) * (%1155[2] * %1171) + %1281 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1298 = amdgpu.scaled_mfma 16x16x128 (%1154[1] * %1164) * (%1155[1] * %1172) + %arg23 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1299 = amdgpu.scaled_mfma 16x16x128 (%1154[3] * %1165) * (%1155[3] * %1173) + %1298 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1300 = vector.load %alloc_4[%194, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1301 = vector.load %alloc_2[%c0, %195] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    %1302 = vector.bitcast %1301 : vector<4xi8> to vector<4xf8E8M0FNU>
    %1303 = vector.load %alloc_2[%c0, %196] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    %1304 = vector.bitcast %1303 : vector<4xi8> to vector<4xf8E8M0FNU>
    %1305 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 64 + s2 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64 + 33)>()[%thread_id_x, %thread_id_y, %arg7]
    %1306 = arith.muli %1305, %c256 overflow<nsw> : index
    %1307 = arith.addi %1306, %159 overflow<nsw> : index
    %1308 = vector.load %163[%1307] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
    %1309 = vector.bitcast %1308 : vector<4xi8> to vector<4xf8E8M0FNU>
    rocdl.sched.barrier 0
    amdgpu.memory_counter_wait load(10) ds(0)
    rocdl.s.barrier
    rocdl.sched.barrier 0
    %1310 = vector.bitcast %1189 : vector<16xi8> to vector<32xf4E2M1FN>
    %1311 = vector.bitcast %1205 : vector<16xi8> to vector<32xf4E2M1FN>
    %1312 = vector.bitcast %1220 : vector<16xi8> to vector<32xf4E2M1FN>
    %1313 = vector.bitcast %1235 : vector<16xi8> to vector<32xf4E2M1FN>
    %1314 = vector.bitcast %1250 : vector<16xi8> to vector<32xf4E2M1FN>
    %1315 = vector.bitcast %1265 : vector<16xi8> to vector<32xf4E2M1FN>
    %1316 = vector.bitcast %1280 : vector<16xi8> to vector<32xf4E2M1FN>
    %1317 = vector.bitcast %1300 : vector<16xi8> to vector<32xf4E2M1FN>
    rocdl.sched.barrier 0
    %1318 = amdgpu.scaled_mfma 16x16x128 (%1302[0] * %1310) * (%1156[0] * %1166) + %arg24 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1319 = affine.apply affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + 256)>()[%arg7, %8]
    %1320 = arith.addi %13, %1319 overflow<nsw> : index
    %1321 = arith.select %16, %1320, %c2147483647 : index
    amdgpu.gather_to_lds %15[%1321], %alloc_4[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
    %1322 = amdgpu.scaled_mfma 16x16x128 (%1302[2] * %1311) * (%1156[2] * %1167) + %1318 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1323 = amdgpu.scaled_mfma 16x16x128 (%1302[0] * %1310) * (%1156[1] * %1168) + %arg25 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1324 = amdgpu.scaled_mfma 16x16x128 (%1302[2] * %1311) * (%1156[3] * %1169) + %1323 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1325 = vector.load %alloc_3[%169, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1326 = amdgpu.scaled_mfma 16x16x128 (%1302[0] * %1310) * (%1155[0] * %1170) + %arg26 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1327 = arith.addi %24, %1319 overflow<nsw> : index
    %1328 = arith.select %26, %1327, %c2147483647 : index
    amdgpu.gather_to_lds %15[%1328], %alloc_4[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
    %1329 = amdgpu.scaled_mfma 16x16x128 (%1302[2] * %1311) * (%1155[2] * %1171) + %1326 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1330 = amdgpu.scaled_mfma 16x16x128 (%1302[0] * %1310) * (%1155[1] * %1172) + %arg27 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1331 = amdgpu.scaled_mfma 16x16x128 (%1302[2] * %1311) * (%1155[3] * %1173) + %1330 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1332 = vector.load %alloc_3[%169, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1333 = amdgpu.scaled_mfma 16x16x128 (%1302[1] * %1312) * (%1156[0] * %1166) + %arg28 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1334 = arith.addi %33, %1319 overflow<nsw> : index
    %1335 = arith.select %35, %1334, %c2147483647 : index
    amdgpu.gather_to_lds %15[%1335], %alloc_4[%32, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
    %1336 = amdgpu.scaled_mfma 16x16x128 (%1302[3] * %1313) * (%1156[2] * %1167) + %1333 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1337 = amdgpu.scaled_mfma 16x16x128 (%1302[1] * %1312) * (%1156[1] * %1168) + %arg29 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1338 = amdgpu.scaled_mfma 16x16x128 (%1302[3] * %1313) * (%1156[3] * %1169) + %1337 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1339 = vector.load %alloc_3[%178, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1340 = amdgpu.scaled_mfma 16x16x128 (%1302[1] * %1312) * (%1155[0] * %1170) + %arg30 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1341 = arith.addi %42, %1319 overflow<nsw> : index
    %1342 = arith.select %44, %1341, %c2147483647 : index
    amdgpu.gather_to_lds %15[%1342], %alloc_4[%41, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
    %1343 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * 128 + s1 * 32 + s2 + 2)>()[%block_id_x, %49, %arg7]
    %1344 = arith.muli %1343, %c256 overflow<nsw> : index
    %1345 = arith.addi %1344, %51 overflow<nsw> : index
    amdgpu.gather_to_lds %57[%1345], %alloc_2[%54, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<1024x1xi8, #gpu.address_space<workgroup>>
    %1346 = amdgpu.scaled_mfma 16x16x128 (%1302[3] * %1313) * (%1155[2] * %1171) + %1340 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1347 = amdgpu.scaled_mfma 16x16x128 (%1302[1] * %1312) * (%1155[1] * %1172) + %arg31 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1348 = amdgpu.scaled_mfma 16x16x128 (%1302[3] * %1313) * (%1155[3] * %1173) + %1347 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1349 = vector.load %alloc_3[%178, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1350 = amdgpu.scaled_mfma 16x16x128 (%1304[0] * %1314) * (%1156[0] * %1166) + %arg32 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1351 = amdgpu.scaled_mfma 16x16x128 (%1304[2] * %1315) * (%1156[2] * %1167) + %1350 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1352 = amdgpu.scaled_mfma 16x16x128 (%1304[0] * %1314) * (%1156[1] * %1168) + %arg33 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1353 = amdgpu.scaled_mfma 16x16x128 (%1304[2] * %1315) * (%1156[3] * %1169) + %1352 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1354 = vector.load %alloc_3[%181, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1355 = amdgpu.scaled_mfma 16x16x128 (%1304[0] * %1314) * (%1155[0] * %1170) + %arg34 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1356 = amdgpu.scaled_mfma 16x16x128 (%1304[2] * %1315) * (%1155[2] * %1171) + %1355 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1357 = amdgpu.scaled_mfma 16x16x128 (%1304[0] * %1314) * (%1155[1] * %1172) + %arg35 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1358 = amdgpu.scaled_mfma 16x16x128 (%1304[2] * %1315) * (%1155[3] * %1173) + %1357 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1359 = vector.load %alloc_3[%181, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1360 = amdgpu.scaled_mfma 16x16x128 (%1304[1] * %1316) * (%1156[0] * %1166) + %arg36 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1361 = amdgpu.scaled_mfma 16x16x128 (%1304[3] * %1317) * (%1156[2] * %1167) + %1360 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1362 = amdgpu.scaled_mfma 16x16x128 (%1304[1] * %1316) * (%1156[1] * %1168) + %arg37 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1363 = amdgpu.scaled_mfma 16x16x128 (%1304[3] * %1317) * (%1156[3] * %1169) + %1362 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1364 = vector.load %alloc_3[%184, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1365 = amdgpu.scaled_mfma 16x16x128 (%1304[1] * %1316) * (%1155[0] * %1170) + %arg38 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1366 = amdgpu.scaled_mfma 16x16x128 (%1304[3] * %1317) * (%1155[2] * %1171) + %1365 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1367 = amdgpu.scaled_mfma 16x16x128 (%1304[1] * %1316) * (%1155[1] * %1172) + %arg39 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1368 = amdgpu.scaled_mfma 16x16x128 (%1304[3] * %1317) * (%1155[3] * %1173) + %1367 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1369 = vector.load %alloc_3[%184, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1370 = vector.load %alloc[%c0, %187] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    %1371 = vector.bitcast %1370 : vector<4xi8> to vector<4xf8E8M0FNU>
    %1372 = vector.load %alloc[%c0, %189] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    %1373 = vector.bitcast %1372 : vector<4xi8> to vector<4xf8E8M0FNU>
    rocdl.sched.barrier 0
    amdgpu.memory_counter_wait load(5) ds(0)
    rocdl.s.barrier
    rocdl.sched.barrier 0
    %1374 = vector.bitcast %1325 : vector<16xi8> to vector<32xf4E2M1FN>
    %1375 = vector.bitcast %1332 : vector<16xi8> to vector<32xf4E2M1FN>
    %1376 = vector.bitcast %1339 : vector<16xi8> to vector<32xf4E2M1FN>
    %1377 = vector.bitcast %1349 : vector<16xi8> to vector<32xf4E2M1FN>
    %1378 = vector.bitcast %1354 : vector<16xi8> to vector<32xf4E2M1FN>
    %1379 = vector.bitcast %1359 : vector<16xi8> to vector<32xf4E2M1FN>
    %1380 = vector.bitcast %1364 : vector<16xi8> to vector<32xf4E2M1FN>
    %1381 = vector.bitcast %1369 : vector<16xi8> to vector<32xf4E2M1FN>
    %1382 = vector.bitcast %1185 : vector<16xi8> to vector<32xf4E2M1FN>
    %1383 = vector.bitcast %1201 : vector<16xi8> to vector<32xf4E2M1FN>
    %1384 = vector.bitcast %1216 : vector<16xi8> to vector<32xf4E2M1FN>
    %1385 = vector.bitcast %1231 : vector<16xi8> to vector<32xf4E2M1FN>
    %1386 = vector.bitcast %1246 : vector<16xi8> to vector<32xf4E2M1FN>
    %1387 = vector.bitcast %1261 : vector<16xi8> to vector<32xf4E2M1FN>
    %1388 = vector.bitcast %1276 : vector<16xi8> to vector<32xf4E2M1FN>
    %1389 = vector.bitcast %1291 : vector<16xi8> to vector<32xf4E2M1FN>
    rocdl.sched.barrier 0
    %1390 = amdgpu.scaled_mfma 16x16x128 (%1371[0] * %1374) * (%1296[0] * %1382) + %1186 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1391 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 1)>()[%thread_id_x, %arg7, %thread_id_y]
    %1392 = affine.apply affine_map<()[s0, s1] -> (s0 * 16 + s1 * 2048 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256) * 4096)>()[%thread_id_x, %arg7]
    %1393 = arith.muli %1391, %c4096 overflow<nsw> : index
    %1394 = arith.addi %1393, %1392 overflow<nsw> : index
    %1395 = arith.index_cast %1394 : index to i32
    %1396 = vector.broadcast %1395 : i32 to vector<16xi32>
    %1397 = arith.addi %1396, %cst : vector<16xi32>
    %1398 = arith.index_cast %1397 : vector<16xi32> to vector<16xindex>
    %1399 = arith.select %63, %1398, %cst_0 : vector<16xi1>, vector<16xindex>
    %1400 = vector.extract %1399[0] : index from vector<16xindex>
    %1401 = vector.load %70[%1400] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %1402 = amdgpu.scaled_mfma 16x16x128 (%1371[2] * %1375) * (%1296[2] * %1383) + %1390 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1403 = amdgpu.scaled_mfma 16x16x128 (%1371[0] * %1374) * (%1296[1] * %1384) + %1188 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1404 = amdgpu.scaled_mfma 16x16x128 (%1371[2] * %1375) * (%1296[3] * %1385) + %1403 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1405 = vector.load %alloc_3[%191, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1406 = amdgpu.scaled_mfma 16x16x128 (%1371[0] * %1374) * (%1309[0] * %1386) + %1202 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1407 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 320) floordiv 256)>()[%thread_id_x, %arg7, %thread_id_y]
    %1408 = affine.apply affine_map<()[s0, s1] -> (s0 * 16 + s1 * 2048 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 64) floordiv 256) * 4096 + 1024)>()[%thread_id_x, %arg7]
    %1409 = arith.muli %1407, %c4096 overflow<nsw> : index
    %1410 = arith.addi %1409, %1408 overflow<nsw> : index
    %1411 = arith.index_cast %1410 : index to i32
    %1412 = vector.broadcast %1411 : i32 to vector<16xi32>
    %1413 = arith.addi %1412, %cst : vector<16xi32>
    %1414 = arith.index_cast %1413 : vector<16xi32> to vector<16xindex>
    %1415 = arith.select %63, %1414, %cst_0 : vector<16xi1>, vector<16xindex>
    %1416 = vector.extract %1415[0] : index from vector<16xindex>
    %1417 = vector.load %70[%1416] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %1418 = amdgpu.scaled_mfma 16x16x128 (%1371[2] * %1375) * (%1309[2] * %1387) + %1406 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1419 = amdgpu.scaled_mfma 16x16x128 (%1371[0] * %1374) * (%1309[1] * %1388) + %1204 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1420 = amdgpu.scaled_mfma 16x16x128 (%1371[2] * %1375) * (%1309[3] * %1389) + %1419 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1421 = vector.load %alloc_3[%191, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1422 = amdgpu.scaled_mfma 16x16x128 (%1371[1] * %1376) * (%1296[0] * %1382) + %1217 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1423 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 17)>()[%thread_id_x, %arg7, %thread_id_y]
    %1424 = arith.muli %1423, %c4096 overflow<nsw> : index
    %1425 = arith.addi %1424, %1392 overflow<nsw> : index
    %1426 = arith.index_cast %1425 : index to i32
    %1427 = vector.broadcast %1426 : i32 to vector<16xi32>
    %1428 = arith.addi %1427, %cst : vector<16xi32>
    %1429 = arith.index_cast %1428 : vector<16xi32> to vector<16xindex>
    %1430 = arith.select %91, %1429, %cst_0 : vector<16xi1>, vector<16xindex>
    %1431 = vector.extract %1430[0] : index from vector<16xindex>
    %1432 = vector.load %70[%1431] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %1433 = amdgpu.scaled_mfma 16x16x128 (%1371[3] * %1377) * (%1296[2] * %1383) + %1422 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1434 = amdgpu.scaled_mfma 16x16x128 (%1371[1] * %1376) * (%1296[1] * %1384) + %1219 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1435 = amdgpu.scaled_mfma 16x16x128 (%1371[3] * %1377) * (%1296[3] * %1385) + %1434 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1436 = vector.load %alloc_3[%192, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1437 = amdgpu.scaled_mfma 16x16x128 (%1371[1] * %1376) * (%1309[0] * %1386) + %1232 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1438 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 320) floordiv 256 + 16)>()[%thread_id_x, %arg7, %thread_id_y]
    %1439 = arith.muli %1438, %c4096 overflow<nsw> : index
    %1440 = arith.addi %1439, %1408 overflow<nsw> : index
    %1441 = arith.index_cast %1440 : index to i32
    %1442 = vector.broadcast %1441 : i32 to vector<16xi32>
    %1443 = arith.addi %1442, %cst : vector<16xi32>
    %1444 = arith.index_cast %1443 : vector<16xi32> to vector<16xindex>
    %1445 = arith.select %91, %1444, %cst_0 : vector<16xi1>, vector<16xindex>
    %1446 = vector.extract %1445[0] : index from vector<16xindex>
    %1447 = vector.load %70[%1446] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %1448 = amdgpu.scaled_mfma 16x16x128 (%1371[3] * %1377) * (%1309[2] * %1387) + %1437 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1449 = amdgpu.scaled_mfma 16x16x128 (%1371[1] * %1376) * (%1309[1] * %1388) + %1234 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1450 = amdgpu.scaled_mfma 16x16x128 (%1371[3] * %1377) * (%1309[3] * %1389) + %1449 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1451 = vector.load %alloc_3[%192, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1452 = amdgpu.scaled_mfma 16x16x128 (%1373[0] * %1378) * (%1296[0] * %1382) + %1247 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1453 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 33)>()[%thread_id_x, %arg7, %thread_id_y]
    %1454 = arith.muli %1453, %c4096 overflow<nsw> : index
    %1455 = arith.addi %1454, %1392 overflow<nsw> : index
    %1456 = arith.index_cast %1455 : index to i32
    %1457 = vector.broadcast %1456 : i32 to vector<16xi32>
    %1458 = arith.addi %1457, %cst : vector<16xi32>
    %1459 = arith.index_cast %1458 : vector<16xi32> to vector<16xindex>
    %1460 = arith.select %114, %1459, %cst_0 : vector<16xi1>, vector<16xindex>
    %1461 = vector.extract %1460[0] : index from vector<16xindex>
    %1462 = vector.load %70[%1461] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %1463 = amdgpu.scaled_mfma 16x16x128 (%1373[2] * %1379) * (%1296[2] * %1383) + %1452 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1464 = amdgpu.scaled_mfma 16x16x128 (%1373[0] * %1378) * (%1296[1] * %1384) + %1249 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1465 = amdgpu.scaled_mfma 16x16x128 (%1373[2] * %1379) * (%1296[3] * %1385) + %1464 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1466 = vector.load %alloc_3[%193, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1467 = amdgpu.scaled_mfma 16x16x128 (%1373[0] * %1378) * (%1309[0] * %1386) + %1262 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1468 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 320) floordiv 256 + 32)>()[%thread_id_x, %arg7, %thread_id_y]
    %1469 = arith.muli %1468, %c4096 overflow<nsw> : index
    %1470 = arith.addi %1469, %1408 overflow<nsw> : index
    %1471 = arith.index_cast %1470 : index to i32
    %1472 = vector.broadcast %1471 : i32 to vector<16xi32>
    %1473 = arith.addi %1472, %cst : vector<16xi32>
    %1474 = arith.index_cast %1473 : vector<16xi32> to vector<16xindex>
    %1475 = arith.select %114, %1474, %cst_0 : vector<16xi1>, vector<16xindex>
    %1476 = vector.extract %1475[0] : index from vector<16xindex>
    %1477 = vector.load %70[%1476] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %1478 = amdgpu.scaled_mfma 16x16x128 (%1373[2] * %1379) * (%1309[2] * %1387) + %1467 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1479 = amdgpu.scaled_mfma 16x16x128 (%1373[0] * %1378) * (%1309[1] * %1388) + %1264 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1480 = amdgpu.scaled_mfma 16x16x128 (%1373[2] * %1379) * (%1309[3] * %1389) + %1479 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1481 = vector.load %alloc_3[%193, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1482 = amdgpu.scaled_mfma 16x16x128 (%1373[1] * %1380) * (%1296[0] * %1382) + %1277 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1483 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 256 + 49)>()[%thread_id_x, %arg7, %thread_id_y]
    %1484 = arith.muli %1483, %c4096 overflow<nsw> : index
    %1485 = arith.addi %1484, %1392 overflow<nsw> : index
    %1486 = arith.index_cast %1485 : index to i32
    %1487 = vector.broadcast %1486 : i32 to vector<16xi32>
    %1488 = arith.addi %1487, %cst : vector<16xi32>
    %1489 = arith.index_cast %1488 : vector<16xi32> to vector<16xindex>
    %1490 = arith.select %137, %1489, %cst_0 : vector<16xi1>, vector<16xindex>
    %1491 = vector.extract %1490[0] : index from vector<16xindex>
    %1492 = vector.load %70[%1491] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %1493 = amdgpu.scaled_mfma 16x16x128 (%1373[3] * %1381) * (%1296[2] * %1383) + %1482 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1494 = amdgpu.scaled_mfma 16x16x128 (%1373[1] * %1380) * (%1296[1] * %1384) + %1279 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1495 = amdgpu.scaled_mfma 16x16x128 (%1373[3] * %1381) * (%1296[3] * %1385) + %1494 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1496 = vector.load %alloc_3[%194, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1497 = amdgpu.scaled_mfma 16x16x128 (%1373[1] * %1380) * (%1309[0] * %1386) + %1297 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1498 = affine.apply affine_map<()[s0, s1, s2] -> (s2 * 64 + (s0 + s1 * 128 - (s0 floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 16 + 320) floordiv 256 + 48)>()[%thread_id_x, %arg7, %thread_id_y]
    %1499 = arith.muli %1498, %c4096 overflow<nsw> : index
    %1500 = arith.addi %1499, %1408 overflow<nsw> : index
    %1501 = arith.index_cast %1500 : index to i32
    %1502 = vector.broadcast %1501 : i32 to vector<16xi32>
    %1503 = arith.addi %1502, %cst : vector<16xi32>
    %1504 = arith.index_cast %1503 : vector<16xi32> to vector<16xindex>
    %1505 = arith.select %137, %1504, %cst_0 : vector<16xi1>, vector<16xindex>
    %1506 = vector.extract %1505[0] : index from vector<16xindex>
    %1507 = vector.load %70[%1506] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
    %1508 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 64 + s2 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64 + 2)>()[%thread_id_x, %thread_id_y, %arg7]
    %1509 = arith.muli %1508, %c256 overflow<nsw> : index
    %1510 = arith.addi %1509, %159 overflow<nsw> : index
    %1511 = vector.load %163[%1510] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
    %1512 = amdgpu.scaled_mfma 16x16x128 (%1373[3] * %1381) * (%1309[2] * %1387) + %1497 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1513 = amdgpu.scaled_mfma 16x16x128 (%1373[1] * %1380) * (%1309[1] * %1388) + %1299 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1514 = amdgpu.scaled_mfma 16x16x128 (%1373[3] * %1381) * (%1309[3] * %1389) + %1513 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1515 = vector.load %alloc_3[%194, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1516 = vector.load %alloc[%c0, %195] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    %1517 = vector.bitcast %1516 : vector<4xi8> to vector<4xf8E8M0FNU>
    %1518 = vector.load %alloc[%c0, %196] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    %1519 = vector.bitcast %1518 : vector<4xi8> to vector<4xf8E8M0FNU>
    %1520 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 64 + s2 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64 + 34)>()[%thread_id_x, %thread_id_y, %arg7]
    %1521 = arith.muli %1520, %c256 overflow<nsw> : index
    %1522 = arith.addi %1521, %159 overflow<nsw> : index
    %1523 = vector.load %163[%1522] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
    rocdl.sched.barrier 0
    amdgpu.memory_counter_wait load(10) ds(0)
    rocdl.s.barrier
    rocdl.sched.barrier 0
    %1524 = vector.bitcast %1405 : vector<16xi8> to vector<32xf4E2M1FN>
    %1525 = vector.bitcast %1421 : vector<16xi8> to vector<32xf4E2M1FN>
    %1526 = vector.bitcast %1436 : vector<16xi8> to vector<32xf4E2M1FN>
    %1527 = vector.bitcast %1451 : vector<16xi8> to vector<32xf4E2M1FN>
    %1528 = vector.bitcast %1466 : vector<16xi8> to vector<32xf4E2M1FN>
    %1529 = vector.bitcast %1481 : vector<16xi8> to vector<32xf4E2M1FN>
    %1530 = vector.bitcast %1496 : vector<16xi8> to vector<32xf4E2M1FN>
    %1531 = vector.bitcast %1515 : vector<16xi8> to vector<32xf4E2M1FN>
    rocdl.sched.barrier 0
    %1532 = amdgpu.scaled_mfma 16x16x128 (%1517[0] * %1524) * (%1296[0] * %1382) + %1322 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1533 = affine.apply affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16 + 384)>()[%arg7, %8]
    %1534 = arith.addi %13, %1533 overflow<nsw> : index
    %1535 = arith.select %16, %1534, %c2147483647 : index
    amdgpu.gather_to_lds %15[%1535], %alloc_3[%11, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
    %1536 = amdgpu.scaled_mfma 16x16x128 (%1517[2] * %1525) * (%1296[2] * %1383) + %1532 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1537 = amdgpu.scaled_mfma 16x16x128 (%1517[0] * %1524) * (%1296[1] * %1384) + %1324 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1538 = amdgpu.scaled_mfma 16x16x128 (%1517[2] * %1525) * (%1296[3] * %1385) + %1537 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1539 = vector.load %alloc_4[%169, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1540 = amdgpu.scaled_mfma 16x16x128 (%1517[0] * %1524) * (%1309[0] * %1386) + %1329 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1541 = arith.addi %24, %1533 overflow<nsw> : index
    %1542 = arith.select %26, %1541, %c2147483647 : index
    amdgpu.gather_to_lds %15[%1542], %alloc_3[%23, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
    %1543 = amdgpu.scaled_mfma 16x16x128 (%1517[2] * %1525) * (%1309[2] * %1387) + %1540 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1544 = amdgpu.scaled_mfma 16x16x128 (%1517[0] * %1524) * (%1309[1] * %1388) + %1331 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1545 = amdgpu.scaled_mfma 16x16x128 (%1517[2] * %1525) * (%1309[3] * %1389) + %1544 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1546 = vector.load %alloc_4[%169, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1547 = amdgpu.scaled_mfma 16x16x128 (%1517[1] * %1526) * (%1296[0] * %1382) + %1336 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1548 = arith.addi %33, %1533 overflow<nsw> : index
    %1549 = arith.select %35, %1548, %c2147483647 : index
    amdgpu.gather_to_lds %15[%1549], %alloc_3[%32, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
    %1550 = amdgpu.scaled_mfma 16x16x128 (%1517[3] * %1527) * (%1296[2] * %1383) + %1547 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1551 = amdgpu.scaled_mfma 16x16x128 (%1517[1] * %1526) * (%1296[1] * %1384) + %1338 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1552 = amdgpu.scaled_mfma 16x16x128 (%1517[3] * %1527) * (%1296[3] * %1385) + %1551 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1553 = vector.load %alloc_4[%178, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1554 = amdgpu.scaled_mfma 16x16x128 (%1517[1] * %1526) * (%1309[0] * %1386) + %1346 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1555 = arith.addi %42, %1533 overflow<nsw> : index
    %1556 = arith.select %44, %1555, %c2147483647 : index
    amdgpu.gather_to_lds %15[%1556], %alloc_3[%41, %12] : vector<16xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<128x128xi8, #gpu.address_space<workgroup>>
    %1557 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * 128 + s1 * 32 + s2 + 3)>()[%block_id_x, %49, %arg7]
    %1558 = arith.muli %1557, %c256 overflow<nsw> : index
    %1559 = arith.addi %1558, %51 overflow<nsw> : index
    amdgpu.gather_to_lds %57[%1559], %alloc[%54, %12] : vector<4xi8>, memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, memref<1024x1xi8, #gpu.address_space<workgroup>>
    %1560 = amdgpu.scaled_mfma 16x16x128 (%1517[3] * %1527) * (%1309[2] * %1387) + %1554 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1561 = amdgpu.scaled_mfma 16x16x128 (%1517[1] * %1526) * (%1309[1] * %1388) + %1348 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1562 = amdgpu.scaled_mfma 16x16x128 (%1517[3] * %1527) * (%1309[3] * %1389) + %1561 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1563 = vector.load %alloc_4[%178, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1564 = amdgpu.scaled_mfma 16x16x128 (%1519[0] * %1528) * (%1296[0] * %1382) + %1351 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1565 = amdgpu.scaled_mfma 16x16x128 (%1519[2] * %1529) * (%1296[2] * %1383) + %1564 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1566 = amdgpu.scaled_mfma 16x16x128 (%1519[0] * %1528) * (%1296[1] * %1384) + %1353 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1567 = amdgpu.scaled_mfma 16x16x128 (%1519[2] * %1529) * (%1296[3] * %1385) + %1566 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1568 = vector.load %alloc_4[%181, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1569 = amdgpu.scaled_mfma 16x16x128 (%1519[0] * %1528) * (%1309[0] * %1386) + %1356 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1570 = amdgpu.scaled_mfma 16x16x128 (%1519[2] * %1529) * (%1309[2] * %1387) + %1569 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1571 = amdgpu.scaled_mfma 16x16x128 (%1519[0] * %1528) * (%1309[1] * %1388) + %1358 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1572 = amdgpu.scaled_mfma 16x16x128 (%1519[2] * %1529) * (%1309[3] * %1389) + %1571 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1573 = vector.load %alloc_4[%181, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1574 = amdgpu.scaled_mfma 16x16x128 (%1519[1] * %1530) * (%1296[0] * %1382) + %1361 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1575 = amdgpu.scaled_mfma 16x16x128 (%1519[3] * %1531) * (%1296[2] * %1383) + %1574 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1576 = amdgpu.scaled_mfma 16x16x128 (%1519[1] * %1530) * (%1296[1] * %1384) + %1363 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1577 = amdgpu.scaled_mfma 16x16x128 (%1519[3] * %1531) * (%1296[3] * %1385) + %1576 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1578 = vector.load %alloc_4[%184, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1579 = amdgpu.scaled_mfma 16x16x128 (%1519[1] * %1530) * (%1309[0] * %1386) + %1366 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1580 = amdgpu.scaled_mfma 16x16x128 (%1519[3] * %1531) * (%1309[2] * %1387) + %1579 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1581 = amdgpu.scaled_mfma 16x16x128 (%1519[1] * %1530) * (%1309[1] * %1388) + %1368 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1582 = amdgpu.scaled_mfma 16x16x128 (%1519[3] * %1531) * (%1309[3] * %1389) + %1581 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
    %1583 = vector.load %alloc_4[%184, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    %1584 = vector.load %alloc_2[%c0, %187] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    %1585 = vector.load %alloc_2[%c0, %189] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    rocdl.sched.barrier 0
    amdgpu.memory_counter_wait load(5) ds(0)
    rocdl.s.barrier
    rocdl.sched.barrier 0
    scf.yield %1402, %1404, %1418, %1420, %1433, %1435, %1448, %1450, %1463, %1465, %1478, %1480, %1493, %1495, %1512, %1514, %1536, %1538, %1543, %1545, %1550, %1552, %1560, %1562, %1565, %1567, %1570, %1572, %1575, %1577, %1580, %1582, %1584, %1585, %1539, %1546, %1553, %1563, %1568, %1573, %1578, %1583, %1401, %1417, %1432, %1447, %1462, %1477, %1492, %1507, %1511, %1523 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi8>, vector<4xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<16xi8>, vector<4xi8>, vector<4xi8>
  }
  %198 = vector.bitcast %197#33 : vector<4xi8> to vector<4xf8E8M0FNU>
  %199 = vector.bitcast %197#51 : vector<4xi8> to vector<4xf8E8M0FNU>
  %200 = vector.bitcast %197#50 : vector<4xi8> to vector<4xf8E8M0FNU>
  %201 = vector.bitcast %197#32 : vector<4xi8> to vector<4xf8E8M0FNU>
  amdgpu.memory_counter_wait load(0) ds(0)
  rocdl.s.barrier
  %202 = vector.load %alloc_2[%c0, %195] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
  %203 = vector.bitcast %202 : vector<4xi8> to vector<4xf8E8M0FNU>
  %204 = vector.load %alloc_2[%c0, %196] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
  %205 = vector.bitcast %204 : vector<4xi8> to vector<4xf8E8M0FNU>
  %206 = vector.load %alloc_4[%191, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %207 = vector.load %alloc_4[%191, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %208 = vector.load %alloc_4[%192, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %209 = vector.load %alloc_4[%192, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %210 = vector.load %alloc_4[%193, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %211 = vector.load %alloc_4[%193, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %212 = vector.load %alloc_4[%194, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %213 = vector.load %alloc_4[%194, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %214 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 3968) floordiv 256)>()[%thread_id_x, %thread_id_y]
  %215 = affine.apply affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 128) floordiv 256) * 4096 + 2048)>()[%thread_id_x]
  %216 = arith.muli %214, %c4096 overflow<nsw> : index
  %217 = arith.addi %216, %215 overflow<nsw> : index
  %218 = arith.index_cast %217 : index to i32
  %219 = vector.broadcast %218 : i32 to vector<16xi32>
  %220 = arith.addi %219, %cst : vector<16xi32>
  %221 = arith.index_cast %220 : vector<16xi32> to vector<16xindex>
  %222 = arith.select %63, %221, %cst_0 : vector<16xi1>, vector<16xindex>
  %223 = vector.extract %222[0] : index from vector<16xindex>
  %224 = vector.load %70[%223] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %225 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 4032) floordiv 256)>()[%thread_id_x, %thread_id_y]
  %226 = affine.apply affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 16) * 256 + ((s0 mod 64) floordiv 16) * 256 - ((s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 192) floordiv 256) * 4096 + 3072)>()[%thread_id_x]
  %227 = arith.muli %225, %c4096 overflow<nsw> : index
  %228 = arith.addi %227, %226 overflow<nsw> : index
  %229 = arith.index_cast %228 : index to i32
  %230 = vector.broadcast %229 : i32 to vector<16xi32>
  %231 = arith.addi %230, %cst : vector<16xi32>
  %232 = arith.index_cast %231 : vector<16xi32> to vector<16xindex>
  %233 = arith.select %63, %232, %cst_0 : vector<16xi1>, vector<16xindex>
  %234 = vector.extract %233[0] : index from vector<16xindex>
  %235 = vector.load %70[%234] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %236 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 3968) floordiv 256 + 16)>()[%thread_id_x, %thread_id_y]
  %237 = arith.muli %236, %c4096 overflow<nsw> : index
  %238 = arith.addi %237, %215 overflow<nsw> : index
  %239 = arith.index_cast %238 : index to i32
  %240 = vector.broadcast %239 : i32 to vector<16xi32>
  %241 = arith.addi %240, %cst : vector<16xi32>
  %242 = arith.index_cast %241 : vector<16xi32> to vector<16xindex>
  %243 = arith.select %91, %242, %cst_0 : vector<16xi1>, vector<16xindex>
  %244 = vector.extract %243[0] : index from vector<16xindex>
  %245 = vector.load %70[%244] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %246 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 4032) floordiv 256 + 16)>()[%thread_id_x, %thread_id_y]
  %247 = arith.muli %246, %c4096 overflow<nsw> : index
  %248 = arith.addi %247, %226 overflow<nsw> : index
  %249 = arith.index_cast %248 : index to i32
  %250 = vector.broadcast %249 : i32 to vector<16xi32>
  %251 = arith.addi %250, %cst : vector<16xi32>
  %252 = arith.index_cast %251 : vector<16xi32> to vector<16xindex>
  %253 = arith.select %91, %252, %cst_0 : vector<16xi1>, vector<16xindex>
  %254 = vector.extract %253[0] : index from vector<16xindex>
  %255 = vector.load %70[%254] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %256 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 3968) floordiv 256 + 32)>()[%thread_id_x, %thread_id_y]
  %257 = arith.muli %256, %c4096 overflow<nsw> : index
  %258 = arith.addi %257, %215 overflow<nsw> : index
  %259 = arith.index_cast %258 : index to i32
  %260 = vector.broadcast %259 : i32 to vector<16xi32>
  %261 = arith.addi %260, %cst : vector<16xi32>
  %262 = arith.index_cast %261 : vector<16xi32> to vector<16xindex>
  %263 = arith.select %114, %262, %cst_0 : vector<16xi1>, vector<16xindex>
  %264 = vector.extract %263[0] : index from vector<16xindex>
  %265 = vector.load %70[%264] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %266 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 4032) floordiv 256 + 32)>()[%thread_id_x, %thread_id_y]
  %267 = arith.muli %266, %c4096 overflow<nsw> : index
  %268 = arith.addi %267, %226 overflow<nsw> : index
  %269 = arith.index_cast %268 : index to i32
  %270 = vector.broadcast %269 : i32 to vector<16xi32>
  %271 = arith.addi %270, %cst : vector<16xi32>
  %272 = arith.index_cast %271 : vector<16xi32> to vector<16xindex>
  %273 = arith.select %114, %272, %cst_0 : vector<16xi1>, vector<16xindex>
  %274 = vector.extract %273[0] : index from vector<16xindex>
  %275 = vector.load %70[%274] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %276 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 3968) floordiv 256 + 48)>()[%thread_id_x, %thread_id_y]
  %277 = arith.muli %276, %c4096 overflow<nsw> : index
  %278 = arith.addi %277, %215 overflow<nsw> : index
  %279 = arith.index_cast %278 : index to i32
  %280 = vector.broadcast %279 : i32 to vector<16xi32>
  %281 = arith.addi %280, %cst : vector<16xi32>
  %282 = arith.index_cast %281 : vector<16xi32> to vector<16xindex>
  %283 = arith.select %137, %282, %cst_0 : vector<16xi1>, vector<16xindex>
  %284 = vector.extract %283[0] : index from vector<16xindex>
  %285 = vector.load %70[%284] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %286 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16 + 4032) floordiv 256 + 48)>()[%thread_id_x, %thread_id_y]
  %287 = arith.muli %286, %c4096 overflow<nsw> : index
  %288 = arith.addi %287, %226 overflow<nsw> : index
  %289 = arith.index_cast %288 : index to i32
  %290 = vector.broadcast %289 : i32 to vector<16xi32>
  %291 = arith.addi %290, %cst : vector<16xi32>
  %292 = arith.index_cast %291 : vector<16xi32> to vector<16xindex>
  %293 = arith.select %137, %292, %cst_0 : vector<16xi1>, vector<16xindex>
  %294 = vector.extract %293[0] : index from vector<16xindex>
  %295 = vector.load %70[%294] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
  %296 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64 + 31)>()[%thread_id_x, %thread_id_y]
  %297 = arith.muli %296, %c256 overflow<nsw> : index
  %298 = arith.addi %297, %159 overflow<nsw> : index
  %299 = vector.load %163[%298] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
  %300 = vector.bitcast %299 : vector<4xi8> to vector<4xf8E8M0FNU>
  %301 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 mod 16 + ((s0 mod 64) floordiv 16) * 16) floordiv 64 + 63)>()[%thread_id_x, %thread_id_y]
  %302 = arith.muli %301, %c256 overflow<nsw> : index
  %303 = arith.addi %302, %159 overflow<nsw> : index
  %304 = vector.load %163[%303] : memref<?xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<4xi8>
  %305 = vector.bitcast %304 : vector<4xi8> to vector<4xf8E8M0FNU>
  amdgpu.memory_counter_wait load(0) ds(0)
  rocdl.s.barrier
  %306 = vector.bitcast %197#34 : vector<16xi8> to vector<32xf4E2M1FN>
  %307 = vector.bitcast %197#35 : vector<16xi8> to vector<32xf4E2M1FN>
  %308 = vector.bitcast %197#36 : vector<16xi8> to vector<32xf4E2M1FN>
  %309 = vector.bitcast %197#37 : vector<16xi8> to vector<32xf4E2M1FN>
  %310 = vector.bitcast %197#38 : vector<16xi8> to vector<32xf4E2M1FN>
  %311 = vector.bitcast %197#39 : vector<16xi8> to vector<32xf4E2M1FN>
  %312 = vector.bitcast %197#40 : vector<16xi8> to vector<32xf4E2M1FN>
  %313 = vector.bitcast %197#41 : vector<16xi8> to vector<32xf4E2M1FN>
  %314 = vector.bitcast %206 : vector<16xi8> to vector<32xf4E2M1FN>
  %315 = vector.bitcast %207 : vector<16xi8> to vector<32xf4E2M1FN>
  %316 = vector.bitcast %208 : vector<16xi8> to vector<32xf4E2M1FN>
  %317 = vector.bitcast %209 : vector<16xi8> to vector<32xf4E2M1FN>
  %318 = vector.bitcast %210 : vector<16xi8> to vector<32xf4E2M1FN>
  %319 = vector.bitcast %211 : vector<16xi8> to vector<32xf4E2M1FN>
  %320 = vector.bitcast %212 : vector<16xi8> to vector<32xf4E2M1FN>
  %321 = vector.bitcast %213 : vector<16xi8> to vector<32xf4E2M1FN>
  %322 = vector.bitcast %197#42 : vector<16xi8> to vector<32xf4E2M1FN>
  %323 = vector.bitcast %197#43 : vector<16xi8> to vector<32xf4E2M1FN>
  %324 = vector.bitcast %197#44 : vector<16xi8> to vector<32xf4E2M1FN>
  %325 = vector.bitcast %197#45 : vector<16xi8> to vector<32xf4E2M1FN>
  %326 = vector.bitcast %197#46 : vector<16xi8> to vector<32xf4E2M1FN>
  %327 = vector.bitcast %197#47 : vector<16xi8> to vector<32xf4E2M1FN>
  %328 = vector.bitcast %197#48 : vector<16xi8> to vector<32xf4E2M1FN>
  %329 = vector.bitcast %197#49 : vector<16xi8> to vector<32xf4E2M1FN>
  rocdl.sched.barrier 0
  %330 = vector.load %alloc[%c0, %187] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
  %331 = vector.bitcast %330 : vector<4xi8> to vector<4xf8E8M0FNU>
  %332 = vector.load %alloc[%c0, %189] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
  %333 = vector.bitcast %332 : vector<4xi8> to vector<4xf8E8M0FNU>
  %334 = vector.load %alloc_3[%169, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %335 = vector.load %alloc_3[%169, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %336 = vector.load %alloc_3[%178, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %337 = vector.load %alloc_3[%178, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %338 = vector.load %alloc_3[%181, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %339 = vector.load %alloc_3[%181, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %340 = vector.load %alloc_3[%184, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %341 = vector.load %alloc_3[%184, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %342 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %306) * (%200[0] * %322) + %197#0 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %343 = amdgpu.scaled_mfma 16x16x128 (%201[2] * %307) * (%200[2] * %323) + %342 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %344 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %306) * (%200[1] * %324) + %197#1 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %345 = amdgpu.scaled_mfma 16x16x128 (%201[2] * %307) * (%200[3] * %325) + %344 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %346 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %306) * (%199[0] * %326) + %197#2 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %347 = amdgpu.scaled_mfma 16x16x128 (%201[2] * %307) * (%199[2] * %327) + %346 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %348 = amdgpu.scaled_mfma 16x16x128 (%201[0] * %306) * (%199[1] * %328) + %197#3 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %349 = amdgpu.scaled_mfma 16x16x128 (%201[2] * %307) * (%199[3] * %329) + %348 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %350 = amdgpu.scaled_mfma 16x16x128 (%201[1] * %308) * (%200[0] * %322) + %197#4 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %351 = amdgpu.scaled_mfma 16x16x128 (%201[3] * %309) * (%200[2] * %323) + %350 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %352 = amdgpu.scaled_mfma 16x16x128 (%201[1] * %308) * (%200[1] * %324) + %197#5 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %353 = amdgpu.scaled_mfma 16x16x128 (%201[3] * %309) * (%200[3] * %325) + %352 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %354 = amdgpu.scaled_mfma 16x16x128 (%201[1] * %308) * (%199[0] * %326) + %197#6 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %355 = amdgpu.scaled_mfma 16x16x128 (%201[3] * %309) * (%199[2] * %327) + %354 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %356 = amdgpu.scaled_mfma 16x16x128 (%201[1] * %308) * (%199[1] * %328) + %197#7 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %357 = amdgpu.scaled_mfma 16x16x128 (%201[3] * %309) * (%199[3] * %329) + %356 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %358 = amdgpu.scaled_mfma 16x16x128 (%198[0] * %310) * (%200[0] * %322) + %197#8 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %359 = amdgpu.scaled_mfma 16x16x128 (%198[2] * %311) * (%200[2] * %323) + %358 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %360 = amdgpu.scaled_mfma 16x16x128 (%198[0] * %310) * (%200[1] * %324) + %197#9 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %361 = amdgpu.scaled_mfma 16x16x128 (%198[2] * %311) * (%200[3] * %325) + %360 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %362 = amdgpu.scaled_mfma 16x16x128 (%198[0] * %310) * (%199[0] * %326) + %197#10 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %363 = amdgpu.scaled_mfma 16x16x128 (%198[2] * %311) * (%199[2] * %327) + %362 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %364 = amdgpu.scaled_mfma 16x16x128 (%198[0] * %310) * (%199[1] * %328) + %197#11 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %365 = amdgpu.scaled_mfma 16x16x128 (%198[2] * %311) * (%199[3] * %329) + %364 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %366 = amdgpu.scaled_mfma 16x16x128 (%198[1] * %312) * (%200[0] * %322) + %197#12 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %367 = amdgpu.scaled_mfma 16x16x128 (%198[3] * %313) * (%200[2] * %323) + %366 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %368 = amdgpu.scaled_mfma 16x16x128 (%198[1] * %312) * (%200[1] * %324) + %197#13 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %369 = amdgpu.scaled_mfma 16x16x128 (%198[3] * %313) * (%200[3] * %325) + %368 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %370 = amdgpu.scaled_mfma 16x16x128 (%198[1] * %312) * (%199[0] * %326) + %197#14 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %371 = amdgpu.scaled_mfma 16x16x128 (%198[3] * %313) * (%199[2] * %327) + %370 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %372 = amdgpu.scaled_mfma 16x16x128 (%198[1] * %312) * (%199[1] * %328) + %197#15 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %373 = amdgpu.scaled_mfma 16x16x128 (%198[3] * %313) * (%199[3] * %329) + %372 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %374 = amdgpu.scaled_mfma 16x16x128 (%203[0] * %314) * (%200[0] * %322) + %197#16 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %375 = amdgpu.scaled_mfma 16x16x128 (%203[2] * %315) * (%200[2] * %323) + %374 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %376 = amdgpu.scaled_mfma 16x16x128 (%203[0] * %314) * (%200[1] * %324) + %197#17 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %377 = amdgpu.scaled_mfma 16x16x128 (%203[2] * %315) * (%200[3] * %325) + %376 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %378 = amdgpu.scaled_mfma 16x16x128 (%203[0] * %314) * (%199[0] * %326) + %197#18 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %379 = amdgpu.scaled_mfma 16x16x128 (%203[2] * %315) * (%199[2] * %327) + %378 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %380 = amdgpu.scaled_mfma 16x16x128 (%203[0] * %314) * (%199[1] * %328) + %197#19 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %381 = amdgpu.scaled_mfma 16x16x128 (%203[2] * %315) * (%199[3] * %329) + %380 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %382 = amdgpu.scaled_mfma 16x16x128 (%203[1] * %316) * (%200[0] * %322) + %197#20 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %383 = amdgpu.scaled_mfma 16x16x128 (%203[3] * %317) * (%200[2] * %323) + %382 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %384 = amdgpu.scaled_mfma 16x16x128 (%203[1] * %316) * (%200[1] * %324) + %197#21 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %385 = amdgpu.scaled_mfma 16x16x128 (%203[3] * %317) * (%200[3] * %325) + %384 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %386 = amdgpu.scaled_mfma 16x16x128 (%203[1] * %316) * (%199[0] * %326) + %197#22 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %387 = amdgpu.scaled_mfma 16x16x128 (%203[3] * %317) * (%199[2] * %327) + %386 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %388 = amdgpu.scaled_mfma 16x16x128 (%203[1] * %316) * (%199[1] * %328) + %197#23 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %389 = amdgpu.scaled_mfma 16x16x128 (%203[3] * %317) * (%199[3] * %329) + %388 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %390 = amdgpu.scaled_mfma 16x16x128 (%205[0] * %318) * (%200[0] * %322) + %197#24 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %391 = amdgpu.scaled_mfma 16x16x128 (%205[2] * %319) * (%200[2] * %323) + %390 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %392 = amdgpu.scaled_mfma 16x16x128 (%205[0] * %318) * (%200[1] * %324) + %197#25 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %393 = amdgpu.scaled_mfma 16x16x128 (%205[2] * %319) * (%200[3] * %325) + %392 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %394 = amdgpu.scaled_mfma 16x16x128 (%205[0] * %318) * (%199[0] * %326) + %197#26 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %395 = amdgpu.scaled_mfma 16x16x128 (%205[2] * %319) * (%199[2] * %327) + %394 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %396 = amdgpu.scaled_mfma 16x16x128 (%205[0] * %318) * (%199[1] * %328) + %197#27 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %397 = amdgpu.scaled_mfma 16x16x128 (%205[2] * %319) * (%199[3] * %329) + %396 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %398 = amdgpu.scaled_mfma 16x16x128 (%205[1] * %320) * (%200[0] * %322) + %197#28 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %399 = amdgpu.scaled_mfma 16x16x128 (%205[3] * %321) * (%200[2] * %323) + %398 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %400 = amdgpu.scaled_mfma 16x16x128 (%205[1] * %320) * (%200[1] * %324) + %197#29 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %401 = amdgpu.scaled_mfma 16x16x128 (%205[3] * %321) * (%200[3] * %325) + %400 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %402 = amdgpu.scaled_mfma 16x16x128 (%205[1] * %320) * (%199[0] * %326) + %197#30 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %403 = amdgpu.scaled_mfma 16x16x128 (%205[3] * %321) * (%199[2] * %327) + %402 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %404 = amdgpu.scaled_mfma 16x16x128 (%205[1] * %320) * (%199[1] * %328) + %197#31 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %405 = amdgpu.scaled_mfma 16x16x128 (%205[3] * %321) * (%199[3] * %329) + %404 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  amdgpu.memory_counter_wait load(0) ds(0)
  rocdl.s.barrier
  %406 = vector.load %alloc[%c0, %195] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
  %407 = vector.bitcast %406 : vector<4xi8> to vector<4xf8E8M0FNU>
  %408 = vector.load %alloc[%c0, %196] : memref<1024x1xi8, #gpu.address_space<workgroup>>, vector<4xi8>
  %409 = vector.bitcast %408 : vector<4xi8> to vector<4xf8E8M0FNU>
  %410 = vector.load %alloc_3[%191, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %411 = vector.load %alloc_3[%191, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %412 = vector.load %alloc_3[%192, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %413 = vector.load %alloc_3[%192, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %414 = vector.load %alloc_3[%193, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %415 = vector.load %alloc_3[%193, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %416 = vector.load %alloc_3[%194, %172] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  %417 = vector.load %alloc_3[%194, %176] : memref<128x128xi8, #gpu.address_space<workgroup>>, vector<16xi8>
  amdgpu.memory_counter_wait load(0) ds(0)
  rocdl.s.barrier
  %418 = vector.bitcast %334 : vector<16xi8> to vector<32xf4E2M1FN>
  %419 = vector.bitcast %335 : vector<16xi8> to vector<32xf4E2M1FN>
  %420 = vector.bitcast %336 : vector<16xi8> to vector<32xf4E2M1FN>
  %421 = vector.bitcast %337 : vector<16xi8> to vector<32xf4E2M1FN>
  %422 = vector.bitcast %338 : vector<16xi8> to vector<32xf4E2M1FN>
  %423 = vector.bitcast %339 : vector<16xi8> to vector<32xf4E2M1FN>
  %424 = vector.bitcast %340 : vector<16xi8> to vector<32xf4E2M1FN>
  %425 = vector.bitcast %341 : vector<16xi8> to vector<32xf4E2M1FN>
  %426 = vector.bitcast %410 : vector<16xi8> to vector<32xf4E2M1FN>
  %427 = vector.bitcast %411 : vector<16xi8> to vector<32xf4E2M1FN>
  %428 = vector.bitcast %412 : vector<16xi8> to vector<32xf4E2M1FN>
  %429 = vector.bitcast %413 : vector<16xi8> to vector<32xf4E2M1FN>
  %430 = vector.bitcast %414 : vector<16xi8> to vector<32xf4E2M1FN>
  %431 = vector.bitcast %415 : vector<16xi8> to vector<32xf4E2M1FN>
  %432 = vector.bitcast %416 : vector<16xi8> to vector<32xf4E2M1FN>
  %433 = vector.bitcast %417 : vector<16xi8> to vector<32xf4E2M1FN>
  %434 = vector.bitcast %224 : vector<16xi8> to vector<32xf4E2M1FN>
  %435 = vector.bitcast %235 : vector<16xi8> to vector<32xf4E2M1FN>
  %436 = vector.bitcast %245 : vector<16xi8> to vector<32xf4E2M1FN>
  %437 = vector.bitcast %255 : vector<16xi8> to vector<32xf4E2M1FN>
  %438 = vector.bitcast %265 : vector<16xi8> to vector<32xf4E2M1FN>
  %439 = vector.bitcast %275 : vector<16xi8> to vector<32xf4E2M1FN>
  %440 = vector.bitcast %285 : vector<16xi8> to vector<32xf4E2M1FN>
  %441 = vector.bitcast %295 : vector<16xi8> to vector<32xf4E2M1FN>
  rocdl.sched.barrier 0
  %442 = amdgpu.scaled_mfma 16x16x128 (%331[0] * %418) * (%300[0] * %434) + %343 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %443 = amdgpu.scaled_mfma 16x16x128 (%331[2] * %419) * (%300[2] * %435) + %442 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %444 = amdgpu.scaled_mfma 16x16x128 (%331[0] * %418) * (%300[1] * %436) + %345 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %445 = amdgpu.scaled_mfma 16x16x128 (%331[2] * %419) * (%300[3] * %437) + %444 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %446 = amdgpu.scaled_mfma 16x16x128 (%331[0] * %418) * (%305[0] * %438) + %347 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %447 = amdgpu.scaled_mfma 16x16x128 (%331[2] * %419) * (%305[2] * %439) + %446 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %448 = amdgpu.scaled_mfma 16x16x128 (%331[0] * %418) * (%305[1] * %440) + %349 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %449 = amdgpu.scaled_mfma 16x16x128 (%331[2] * %419) * (%305[3] * %441) + %448 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %450 = amdgpu.scaled_mfma 16x16x128 (%331[1] * %420) * (%300[0] * %434) + %351 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %451 = amdgpu.scaled_mfma 16x16x128 (%331[3] * %421) * (%300[2] * %435) + %450 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %452 = amdgpu.scaled_mfma 16x16x128 (%331[1] * %420) * (%300[1] * %436) + %353 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %453 = amdgpu.scaled_mfma 16x16x128 (%331[3] * %421) * (%300[3] * %437) + %452 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %454 = amdgpu.scaled_mfma 16x16x128 (%331[1] * %420) * (%305[0] * %438) + %355 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %455 = amdgpu.scaled_mfma 16x16x128 (%331[3] * %421) * (%305[2] * %439) + %454 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %456 = amdgpu.scaled_mfma 16x16x128 (%331[1] * %420) * (%305[1] * %440) + %357 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %457 = amdgpu.scaled_mfma 16x16x128 (%331[3] * %421) * (%305[3] * %441) + %456 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %458 = amdgpu.scaled_mfma 16x16x128 (%333[0] * %422) * (%300[0] * %434) + %359 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %459 = amdgpu.scaled_mfma 16x16x128 (%333[2] * %423) * (%300[2] * %435) + %458 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %460 = amdgpu.scaled_mfma 16x16x128 (%333[0] * %422) * (%300[1] * %436) + %361 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %461 = amdgpu.scaled_mfma 16x16x128 (%333[2] * %423) * (%300[3] * %437) + %460 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %462 = amdgpu.scaled_mfma 16x16x128 (%333[0] * %422) * (%305[0] * %438) + %363 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %463 = amdgpu.scaled_mfma 16x16x128 (%333[2] * %423) * (%305[2] * %439) + %462 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %464 = amdgpu.scaled_mfma 16x16x128 (%333[0] * %422) * (%305[1] * %440) + %365 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %465 = amdgpu.scaled_mfma 16x16x128 (%333[2] * %423) * (%305[3] * %441) + %464 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %466 = amdgpu.scaled_mfma 16x16x128 (%333[1] * %424) * (%300[0] * %434) + %367 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %467 = amdgpu.scaled_mfma 16x16x128 (%333[3] * %425) * (%300[2] * %435) + %466 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %468 = amdgpu.scaled_mfma 16x16x128 (%333[1] * %424) * (%300[1] * %436) + %369 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %469 = amdgpu.scaled_mfma 16x16x128 (%333[3] * %425) * (%300[3] * %437) + %468 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %470 = amdgpu.scaled_mfma 16x16x128 (%333[1] * %424) * (%305[0] * %438) + %371 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %471 = amdgpu.scaled_mfma 16x16x128 (%333[3] * %425) * (%305[2] * %439) + %470 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %472 = amdgpu.scaled_mfma 16x16x128 (%333[1] * %424) * (%305[1] * %440) + %373 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %473 = amdgpu.scaled_mfma 16x16x128 (%333[3] * %425) * (%305[3] * %441) + %472 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %474 = amdgpu.scaled_mfma 16x16x128 (%407[0] * %426) * (%300[0] * %434) + %375 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %475 = amdgpu.scaled_mfma 16x16x128 (%407[2] * %427) * (%300[2] * %435) + %474 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %476 = amdgpu.scaled_mfma 16x16x128 (%407[0] * %426) * (%300[1] * %436) + %377 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %477 = amdgpu.scaled_mfma 16x16x128 (%407[2] * %427) * (%300[3] * %437) + %476 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %478 = amdgpu.scaled_mfma 16x16x128 (%407[0] * %426) * (%305[0] * %438) + %379 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %479 = amdgpu.scaled_mfma 16x16x128 (%407[2] * %427) * (%305[2] * %439) + %478 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %480 = amdgpu.scaled_mfma 16x16x128 (%407[0] * %426) * (%305[1] * %440) + %381 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %481 = amdgpu.scaled_mfma 16x16x128 (%407[2] * %427) * (%305[3] * %441) + %480 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %482 = amdgpu.scaled_mfma 16x16x128 (%407[1] * %428) * (%300[0] * %434) + %383 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %483 = amdgpu.scaled_mfma 16x16x128 (%407[3] * %429) * (%300[2] * %435) + %482 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %484 = amdgpu.scaled_mfma 16x16x128 (%407[1] * %428) * (%300[1] * %436) + %385 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %485 = amdgpu.scaled_mfma 16x16x128 (%407[3] * %429) * (%300[3] * %437) + %484 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %486 = amdgpu.scaled_mfma 16x16x128 (%407[1] * %428) * (%305[0] * %438) + %387 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %487 = amdgpu.scaled_mfma 16x16x128 (%407[3] * %429) * (%305[2] * %439) + %486 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %488 = amdgpu.scaled_mfma 16x16x128 (%407[1] * %428) * (%305[1] * %440) + %389 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %489 = amdgpu.scaled_mfma 16x16x128 (%407[3] * %429) * (%305[3] * %441) + %488 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %490 = amdgpu.scaled_mfma 16x16x128 (%409[0] * %430) * (%300[0] * %434) + %391 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %491 = amdgpu.scaled_mfma 16x16x128 (%409[2] * %431) * (%300[2] * %435) + %490 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %492 = amdgpu.scaled_mfma 16x16x128 (%409[0] * %430) * (%300[1] * %436) + %393 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %493 = amdgpu.scaled_mfma 16x16x128 (%409[2] * %431) * (%300[3] * %437) + %492 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %494 = amdgpu.scaled_mfma 16x16x128 (%409[0] * %430) * (%305[0] * %438) + %395 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %495 = amdgpu.scaled_mfma 16x16x128 (%409[2] * %431) * (%305[2] * %439) + %494 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %496 = amdgpu.scaled_mfma 16x16x128 (%409[0] * %430) * (%305[1] * %440) + %397 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %497 = amdgpu.scaled_mfma 16x16x128 (%409[2] * %431) * (%305[3] * %441) + %496 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %498 = amdgpu.scaled_mfma 16x16x128 (%409[1] * %432) * (%300[0] * %434) + %399 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %499 = amdgpu.scaled_mfma 16x16x128 (%409[3] * %433) * (%300[2] * %435) + %498 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %500 = amdgpu.scaled_mfma 16x16x128 (%409[1] * %432) * (%300[1] * %436) + %401 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %501 = amdgpu.scaled_mfma 16x16x128 (%409[3] * %433) * (%300[3] * %437) + %500 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %502 = amdgpu.scaled_mfma 16x16x128 (%409[1] * %432) * (%305[0] * %438) + %403 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %503 = amdgpu.scaled_mfma 16x16x128 (%409[3] * %433) * (%305[2] * %439) + %502 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %504 = amdgpu.scaled_mfma 16x16x128 (%409[1] * %432) * (%305[1] * %440) + %405 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %505 = amdgpu.scaled_mfma 16x16x128 (%409[3] * %433) * (%305[3] * %441) + %504 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %506 = vector.extract_strided_slice %443 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %507 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4)>()[%thread_id_x, %block_id_x]
  %508 = arith.cmpi slt, %507, %arg5 : index
  %509 = arith.andi %62, %508 : i1
  %510 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%block_id_x]
  %511 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4)>()[%thread_id_x]
  %512 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16)>()[%thread_id_x, %thread_id_y]
  %513 = arith.muli %510, %arg6 overflow<nsw> : index
  %514 = arith.muli %511, %arg6 overflow<nsw> : index
  %515 = arith.addi %513, %64 overflow<nsw> : index
  %516 = arith.addi %514, %512 overflow<nsw> : index
  %reinterpret_cast_11 = memref.reinterpret_cast %4 to offset: [%515], sizes: [536870910], strides: [1] : memref<f32> to memref<536870910xf32, strided<[1], offset: ?>>
  %cast_12 = memref.cast %reinterpret_cast_11 : memref<536870910xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  %517 = amdgpu.fat_raw_buffer_cast %cast_12 validBytes(%c2147483643_i64) resetOffset : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>
  %518 = arith.select %509, %516, %c536870911 : index
  vector.store %506, %517[%518] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %519 = vector.extract_strided_slice %443 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %520 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1)>()[%thread_id_x, %block_id_x]
  %521 = arith.cmpi slt, %520, %arg5 : index
  %522 = arith.andi %62, %521 : i1
  %523 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1)>()[%thread_id_x]
  %524 = arith.muli %523, %arg6 overflow<nsw> : index
  %525 = arith.addi %524, %512 overflow<nsw> : index
  %526 = arith.select %522, %525, %c536870911 : index
  vector.store %519, %517[%526] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %527 = vector.extract_strided_slice %443 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %528 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 2)>()[%thread_id_x, %block_id_x]
  %529 = arith.cmpi slt, %528, %arg5 : index
  %530 = arith.andi %62, %529 : i1
  %531 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 2)>()[%thread_id_x]
  %532 = arith.muli %531, %arg6 overflow<nsw> : index
  %533 = arith.addi %532, %512 overflow<nsw> : index
  %534 = arith.select %530, %533, %c536870911 : index
  vector.store %527, %517[%534] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %535 = vector.extract_strided_slice %443 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %536 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3)>()[%thread_id_x, %block_id_x]
  %537 = arith.cmpi slt, %536, %arg5 : index
  %538 = arith.andi %62, %537 : i1
  %539 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3)>()[%thread_id_x]
  %540 = arith.muli %539, %arg6 overflow<nsw> : index
  %541 = arith.addi %540, %512 overflow<nsw> : index
  %542 = arith.select %538, %541, %c536870911 : index
  vector.store %535, %517[%542] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %543 = vector.extract_strided_slice %445 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %544 = arith.andi %90, %508 : i1
  %545 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 16)>()[%thread_id_x, %thread_id_y]
  %546 = arith.addi %514, %545 overflow<nsw> : index
  %547 = arith.select %544, %546, %c536870911 : index
  vector.store %543, %517[%547] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %548 = vector.extract_strided_slice %445 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %549 = arith.andi %90, %521 : i1
  %550 = arith.addi %524, %545 overflow<nsw> : index
  %551 = arith.select %549, %550, %c536870911 : index
  vector.store %548, %517[%551] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %552 = vector.extract_strided_slice %445 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %553 = arith.andi %90, %529 : i1
  %554 = arith.addi %532, %545 overflow<nsw> : index
  %555 = arith.select %553, %554, %c536870911 : index
  vector.store %552, %517[%555] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %556 = vector.extract_strided_slice %445 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %557 = arith.andi %90, %537 : i1
  %558 = arith.addi %540, %545 overflow<nsw> : index
  %559 = arith.select %557, %558, %c536870911 : index
  vector.store %556, %517[%559] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %560 = vector.extract_strided_slice %447 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %561 = arith.andi %113, %508 : i1
  %562 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 32)>()[%thread_id_x, %thread_id_y]
  %563 = arith.addi %514, %562 overflow<nsw> : index
  %564 = arith.select %561, %563, %c536870911 : index
  vector.store %560, %517[%564] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %565 = vector.extract_strided_slice %447 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %566 = arith.andi %113, %521 : i1
  %567 = arith.addi %524, %562 overflow<nsw> : index
  %568 = arith.select %566, %567, %c536870911 : index
  vector.store %565, %517[%568] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %569 = vector.extract_strided_slice %447 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %570 = arith.andi %113, %529 : i1
  %571 = arith.addi %532, %562 overflow<nsw> : index
  %572 = arith.select %570, %571, %c536870911 : index
  vector.store %569, %517[%572] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %573 = vector.extract_strided_slice %447 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %574 = arith.andi %113, %537 : i1
  %575 = arith.addi %540, %562 overflow<nsw> : index
  %576 = arith.select %574, %575, %c536870911 : index
  vector.store %573, %517[%576] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %577 = vector.extract_strided_slice %449 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %578 = arith.andi %136, %508 : i1
  %579 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64 - (s0 floordiv 16) * 16 + 48)>()[%thread_id_x, %thread_id_y]
  %580 = arith.addi %514, %579 overflow<nsw> : index
  %581 = arith.select %578, %580, %c536870911 : index
  vector.store %577, %517[%581] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %582 = vector.extract_strided_slice %449 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %583 = arith.andi %136, %521 : i1
  %584 = arith.addi %524, %579 overflow<nsw> : index
  %585 = arith.select %583, %584, %c536870911 : index
  vector.store %582, %517[%585] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %586 = vector.extract_strided_slice %449 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %587 = arith.andi %136, %529 : i1
  %588 = arith.addi %532, %579 overflow<nsw> : index
  %589 = arith.select %587, %588, %c536870911 : index
  vector.store %586, %517[%589] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %590 = vector.extract_strided_slice %449 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %591 = arith.andi %136, %537 : i1
  %592 = arith.addi %540, %579 overflow<nsw> : index
  %593 = arith.select %591, %592, %c536870911 : index
  vector.store %590, %517[%593] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %594 = vector.extract_strided_slice %451 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %595 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 16)>()[%thread_id_x, %block_id_x]
  %596 = arith.cmpi slt, %595, %arg5 : index
  %597 = arith.andi %62, %596 : i1
  %598 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 16)>()[%thread_id_x]
  %599 = arith.muli %598, %arg6 overflow<nsw> : index
  %600 = arith.addi %599, %512 overflow<nsw> : index
  %601 = arith.select %597, %600, %c536870911 : index
  vector.store %594, %517[%601] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %602 = vector.extract_strided_slice %451 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %603 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 17)>()[%thread_id_x, %block_id_x]
  %604 = arith.cmpi slt, %603, %arg5 : index
  %605 = arith.andi %62, %604 : i1
  %606 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 17)>()[%thread_id_x]
  %607 = arith.muli %606, %arg6 overflow<nsw> : index
  %608 = arith.addi %607, %512 overflow<nsw> : index
  %609 = arith.select %605, %608, %c536870911 : index
  vector.store %602, %517[%609] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %610 = vector.extract_strided_slice %451 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %611 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 18)>()[%thread_id_x, %block_id_x]
  %612 = arith.cmpi slt, %611, %arg5 : index
  %613 = arith.andi %62, %612 : i1
  %614 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 18)>()[%thread_id_x]
  %615 = arith.muli %614, %arg6 overflow<nsw> : index
  %616 = arith.addi %615, %512 overflow<nsw> : index
  %617 = arith.select %613, %616, %c536870911 : index
  vector.store %610, %517[%617] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %618 = vector.extract_strided_slice %451 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %619 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 19)>()[%thread_id_x, %block_id_x]
  %620 = arith.cmpi slt, %619, %arg5 : index
  %621 = arith.andi %62, %620 : i1
  %622 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 19)>()[%thread_id_x]
  %623 = arith.muli %622, %arg6 overflow<nsw> : index
  %624 = arith.addi %623, %512 overflow<nsw> : index
  %625 = arith.select %621, %624, %c536870911 : index
  vector.store %618, %517[%625] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %626 = vector.extract_strided_slice %453 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %627 = arith.andi %90, %596 : i1
  %628 = arith.addi %599, %545 overflow<nsw> : index
  %629 = arith.select %627, %628, %c536870911 : index
  vector.store %626, %517[%629] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %630 = vector.extract_strided_slice %453 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %631 = arith.andi %90, %604 : i1
  %632 = arith.addi %607, %545 overflow<nsw> : index
  %633 = arith.select %631, %632, %c536870911 : index
  vector.store %630, %517[%633] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %634 = vector.extract_strided_slice %453 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %635 = arith.andi %90, %612 : i1
  %636 = arith.addi %615, %545 overflow<nsw> : index
  %637 = arith.select %635, %636, %c536870911 : index
  vector.store %634, %517[%637] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %638 = vector.extract_strided_slice %453 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %639 = arith.andi %90, %620 : i1
  %640 = arith.addi %623, %545 overflow<nsw> : index
  %641 = arith.select %639, %640, %c536870911 : index
  vector.store %638, %517[%641] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %642 = vector.extract_strided_slice %455 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %643 = arith.andi %113, %596 : i1
  %644 = arith.addi %599, %562 overflow<nsw> : index
  %645 = arith.select %643, %644, %c536870911 : index
  vector.store %642, %517[%645] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %646 = vector.extract_strided_slice %455 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %647 = arith.andi %113, %604 : i1
  %648 = arith.addi %607, %562 overflow<nsw> : index
  %649 = arith.select %647, %648, %c536870911 : index
  vector.store %646, %517[%649] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %650 = vector.extract_strided_slice %455 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %651 = arith.andi %113, %612 : i1
  %652 = arith.addi %615, %562 overflow<nsw> : index
  %653 = arith.select %651, %652, %c536870911 : index
  vector.store %650, %517[%653] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %654 = vector.extract_strided_slice %455 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %655 = arith.andi %113, %620 : i1
  %656 = arith.addi %623, %562 overflow<nsw> : index
  %657 = arith.select %655, %656, %c536870911 : index
  vector.store %654, %517[%657] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %658 = vector.extract_strided_slice %457 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %659 = arith.andi %136, %596 : i1
  %660 = arith.addi %599, %579 overflow<nsw> : index
  %661 = arith.select %659, %660, %c536870911 : index
  vector.store %658, %517[%661] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %662 = vector.extract_strided_slice %457 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %663 = arith.andi %136, %604 : i1
  %664 = arith.addi %607, %579 overflow<nsw> : index
  %665 = arith.select %663, %664, %c536870911 : index
  vector.store %662, %517[%665] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %666 = vector.extract_strided_slice %457 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %667 = arith.andi %136, %612 : i1
  %668 = arith.addi %615, %579 overflow<nsw> : index
  %669 = arith.select %667, %668, %c536870911 : index
  vector.store %666, %517[%669] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %670 = vector.extract_strided_slice %457 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %671 = arith.andi %136, %620 : i1
  %672 = arith.addi %623, %579 overflow<nsw> : index
  %673 = arith.select %671, %672, %c536870911 : index
  vector.store %670, %517[%673] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %674 = vector.extract_strided_slice %459 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %675 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 32)>()[%thread_id_x, %block_id_x]
  %676 = arith.cmpi slt, %675, %arg5 : index
  %677 = arith.andi %62, %676 : i1
  %678 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 32)>()[%thread_id_x]
  %679 = arith.muli %678, %arg6 overflow<nsw> : index
  %680 = arith.addi %679, %512 overflow<nsw> : index
  %681 = arith.select %677, %680, %c536870911 : index
  vector.store %674, %517[%681] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %682 = vector.extract_strided_slice %459 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %683 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 33)>()[%thread_id_x, %block_id_x]
  %684 = arith.cmpi slt, %683, %arg5 : index
  %685 = arith.andi %62, %684 : i1
  %686 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 33)>()[%thread_id_x]
  %687 = arith.muli %686, %arg6 overflow<nsw> : index
  %688 = arith.addi %687, %512 overflow<nsw> : index
  %689 = arith.select %685, %688, %c536870911 : index
  vector.store %682, %517[%689] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %690 = vector.extract_strided_slice %459 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %691 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 34)>()[%thread_id_x, %block_id_x]
  %692 = arith.cmpi slt, %691, %arg5 : index
  %693 = arith.andi %62, %692 : i1
  %694 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 34)>()[%thread_id_x]
  %695 = arith.muli %694, %arg6 overflow<nsw> : index
  %696 = arith.addi %695, %512 overflow<nsw> : index
  %697 = arith.select %693, %696, %c536870911 : index
  vector.store %690, %517[%697] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %698 = vector.extract_strided_slice %459 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %699 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 35)>()[%thread_id_x, %block_id_x]
  %700 = arith.cmpi slt, %699, %arg5 : index
  %701 = arith.andi %62, %700 : i1
  %702 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 35)>()[%thread_id_x]
  %703 = arith.muli %702, %arg6 overflow<nsw> : index
  %704 = arith.addi %703, %512 overflow<nsw> : index
  %705 = arith.select %701, %704, %c536870911 : index
  vector.store %698, %517[%705] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %706 = vector.extract_strided_slice %461 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %707 = arith.andi %90, %676 : i1
  %708 = arith.addi %679, %545 overflow<nsw> : index
  %709 = arith.select %707, %708, %c536870911 : index
  vector.store %706, %517[%709] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %710 = vector.extract_strided_slice %461 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %711 = arith.andi %90, %684 : i1
  %712 = arith.addi %687, %545 overflow<nsw> : index
  %713 = arith.select %711, %712, %c536870911 : index
  vector.store %710, %517[%713] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %714 = vector.extract_strided_slice %461 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %715 = arith.andi %90, %692 : i1
  %716 = arith.addi %695, %545 overflow<nsw> : index
  %717 = arith.select %715, %716, %c536870911 : index
  vector.store %714, %517[%717] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %718 = vector.extract_strided_slice %461 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %719 = arith.andi %90, %700 : i1
  %720 = arith.addi %703, %545 overflow<nsw> : index
  %721 = arith.select %719, %720, %c536870911 : index
  vector.store %718, %517[%721] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %722 = vector.extract_strided_slice %463 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %723 = arith.andi %113, %676 : i1
  %724 = arith.addi %679, %562 overflow<nsw> : index
  %725 = arith.select %723, %724, %c536870911 : index
  vector.store %722, %517[%725] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %726 = vector.extract_strided_slice %463 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %727 = arith.andi %113, %684 : i1
  %728 = arith.addi %687, %562 overflow<nsw> : index
  %729 = arith.select %727, %728, %c536870911 : index
  vector.store %726, %517[%729] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %730 = vector.extract_strided_slice %463 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %731 = arith.andi %113, %692 : i1
  %732 = arith.addi %695, %562 overflow<nsw> : index
  %733 = arith.select %731, %732, %c536870911 : index
  vector.store %730, %517[%733] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %734 = vector.extract_strided_slice %463 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %735 = arith.andi %113, %700 : i1
  %736 = arith.addi %703, %562 overflow<nsw> : index
  %737 = arith.select %735, %736, %c536870911 : index
  vector.store %734, %517[%737] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %738 = vector.extract_strided_slice %465 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %739 = arith.andi %136, %676 : i1
  %740 = arith.addi %679, %579 overflow<nsw> : index
  %741 = arith.select %739, %740, %c536870911 : index
  vector.store %738, %517[%741] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %742 = vector.extract_strided_slice %465 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %743 = arith.andi %136, %684 : i1
  %744 = arith.addi %687, %579 overflow<nsw> : index
  %745 = arith.select %743, %744, %c536870911 : index
  vector.store %742, %517[%745] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %746 = vector.extract_strided_slice %465 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %747 = arith.andi %136, %692 : i1
  %748 = arith.addi %695, %579 overflow<nsw> : index
  %749 = arith.select %747, %748, %c536870911 : index
  vector.store %746, %517[%749] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %750 = vector.extract_strided_slice %465 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %751 = arith.andi %136, %700 : i1
  %752 = arith.addi %703, %579 overflow<nsw> : index
  %753 = arith.select %751, %752, %c536870911 : index
  vector.store %750, %517[%753] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %754 = vector.extract_strided_slice %467 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %755 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 48)>()[%thread_id_x, %block_id_x]
  %756 = arith.cmpi slt, %755, %arg5 : index
  %757 = arith.andi %62, %756 : i1
  %758 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 48)>()[%thread_id_x]
  %759 = arith.muli %758, %arg6 overflow<nsw> : index
  %760 = arith.addi %759, %512 overflow<nsw> : index
  %761 = arith.select %757, %760, %c536870911 : index
  vector.store %754, %517[%761] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %762 = vector.extract_strided_slice %467 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %763 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 49)>()[%thread_id_x, %block_id_x]
  %764 = arith.cmpi slt, %763, %arg5 : index
  %765 = arith.andi %62, %764 : i1
  %766 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 49)>()[%thread_id_x]
  %767 = arith.muli %766, %arg6 overflow<nsw> : index
  %768 = arith.addi %767, %512 overflow<nsw> : index
  %769 = arith.select %765, %768, %c536870911 : index
  vector.store %762, %517[%769] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %770 = vector.extract_strided_slice %467 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %771 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 50)>()[%thread_id_x, %block_id_x]
  %772 = arith.cmpi slt, %771, %arg5 : index
  %773 = arith.andi %62, %772 : i1
  %774 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 50)>()[%thread_id_x]
  %775 = arith.muli %774, %arg6 overflow<nsw> : index
  %776 = arith.addi %775, %512 overflow<nsw> : index
  %777 = arith.select %773, %776, %c536870911 : index
  vector.store %770, %517[%777] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %778 = vector.extract_strided_slice %467 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %779 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 51)>()[%thread_id_x, %block_id_x]
  %780 = arith.cmpi slt, %779, %arg5 : index
  %781 = arith.andi %62, %780 : i1
  %782 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 51)>()[%thread_id_x]
  %783 = arith.muli %782, %arg6 overflow<nsw> : index
  %784 = arith.addi %783, %512 overflow<nsw> : index
  %785 = arith.select %781, %784, %c536870911 : index
  vector.store %778, %517[%785] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %786 = vector.extract_strided_slice %469 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %787 = arith.andi %90, %756 : i1
  %788 = arith.addi %759, %545 overflow<nsw> : index
  %789 = arith.select %787, %788, %c536870911 : index
  vector.store %786, %517[%789] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %790 = vector.extract_strided_slice %469 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %791 = arith.andi %90, %764 : i1
  %792 = arith.addi %767, %545 overflow<nsw> : index
  %793 = arith.select %791, %792, %c536870911 : index
  vector.store %790, %517[%793] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %794 = vector.extract_strided_slice %469 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %795 = arith.andi %90, %772 : i1
  %796 = arith.addi %775, %545 overflow<nsw> : index
  %797 = arith.select %795, %796, %c536870911 : index
  vector.store %794, %517[%797] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %798 = vector.extract_strided_slice %469 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %799 = arith.andi %90, %780 : i1
  %800 = arith.addi %783, %545 overflow<nsw> : index
  %801 = arith.select %799, %800, %c536870911 : index
  vector.store %798, %517[%801] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %802 = vector.extract_strided_slice %471 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %803 = arith.andi %113, %756 : i1
  %804 = arith.addi %759, %562 overflow<nsw> : index
  %805 = arith.select %803, %804, %c536870911 : index
  vector.store %802, %517[%805] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %806 = vector.extract_strided_slice %471 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %807 = arith.andi %113, %764 : i1
  %808 = arith.addi %767, %562 overflow<nsw> : index
  %809 = arith.select %807, %808, %c536870911 : index
  vector.store %806, %517[%809] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %810 = vector.extract_strided_slice %471 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %811 = arith.andi %113, %772 : i1
  %812 = arith.addi %775, %562 overflow<nsw> : index
  %813 = arith.select %811, %812, %c536870911 : index
  vector.store %810, %517[%813] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %814 = vector.extract_strided_slice %471 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %815 = arith.andi %113, %780 : i1
  %816 = arith.addi %783, %562 overflow<nsw> : index
  %817 = arith.select %815, %816, %c536870911 : index
  vector.store %814, %517[%817] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %818 = vector.extract_strided_slice %473 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %819 = arith.andi %136, %756 : i1
  %820 = arith.addi %759, %579 overflow<nsw> : index
  %821 = arith.select %819, %820, %c536870911 : index
  vector.store %818, %517[%821] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %822 = vector.extract_strided_slice %473 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %823 = arith.andi %136, %764 : i1
  %824 = arith.addi %767, %579 overflow<nsw> : index
  %825 = arith.select %823, %824, %c536870911 : index
  vector.store %822, %517[%825] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %826 = vector.extract_strided_slice %473 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %827 = arith.andi %136, %772 : i1
  %828 = arith.addi %775, %579 overflow<nsw> : index
  %829 = arith.select %827, %828, %c536870911 : index
  vector.store %826, %517[%829] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %830 = vector.extract_strided_slice %473 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %831 = arith.andi %136, %780 : i1
  %832 = arith.addi %783, %579 overflow<nsw> : index
  %833 = arith.select %831, %832, %c536870911 : index
  vector.store %830, %517[%833] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %834 = vector.extract_strided_slice %475 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %835 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 64)>()[%thread_id_x, %block_id_x]
  %836 = arith.cmpi slt, %835, %arg5 : index
  %837 = arith.andi %62, %836 : i1
  %838 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 64)>()[%thread_id_x]
  %839 = arith.muli %838, %arg6 overflow<nsw> : index
  %840 = arith.addi %839, %512 overflow<nsw> : index
  %841 = arith.select %837, %840, %c536870911 : index
  vector.store %834, %517[%841] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %842 = vector.extract_strided_slice %475 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %843 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 65)>()[%thread_id_x, %block_id_x]
  %844 = arith.cmpi slt, %843, %arg5 : index
  %845 = arith.andi %62, %844 : i1
  %846 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 65)>()[%thread_id_x]
  %847 = arith.muli %846, %arg6 overflow<nsw> : index
  %848 = arith.addi %847, %512 overflow<nsw> : index
  %849 = arith.select %845, %848, %c536870911 : index
  vector.store %842, %517[%849] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %850 = vector.extract_strided_slice %475 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %851 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 66)>()[%thread_id_x, %block_id_x]
  %852 = arith.cmpi slt, %851, %arg5 : index
  %853 = arith.andi %62, %852 : i1
  %854 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 66)>()[%thread_id_x]
  %855 = arith.muli %854, %arg6 overflow<nsw> : index
  %856 = arith.addi %855, %512 overflow<nsw> : index
  %857 = arith.select %853, %856, %c536870911 : index
  vector.store %850, %517[%857] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %858 = vector.extract_strided_slice %475 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %859 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 67)>()[%thread_id_x, %block_id_x]
  %860 = arith.cmpi slt, %859, %arg5 : index
  %861 = arith.andi %62, %860 : i1
  %862 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 67)>()[%thread_id_x]
  %863 = arith.muli %862, %arg6 overflow<nsw> : index
  %864 = arith.addi %863, %512 overflow<nsw> : index
  %865 = arith.select %861, %864, %c536870911 : index
  vector.store %858, %517[%865] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %866 = vector.extract_strided_slice %477 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %867 = arith.andi %90, %836 : i1
  %868 = arith.addi %839, %545 overflow<nsw> : index
  %869 = arith.select %867, %868, %c536870911 : index
  vector.store %866, %517[%869] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %870 = vector.extract_strided_slice %477 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %871 = arith.andi %90, %844 : i1
  %872 = arith.addi %847, %545 overflow<nsw> : index
  %873 = arith.select %871, %872, %c536870911 : index
  vector.store %870, %517[%873] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %874 = vector.extract_strided_slice %477 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %875 = arith.andi %90, %852 : i1
  %876 = arith.addi %855, %545 overflow<nsw> : index
  %877 = arith.select %875, %876, %c536870911 : index
  vector.store %874, %517[%877] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %878 = vector.extract_strided_slice %477 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %879 = arith.andi %90, %860 : i1
  %880 = arith.addi %863, %545 overflow<nsw> : index
  %881 = arith.select %879, %880, %c536870911 : index
  vector.store %878, %517[%881] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %882 = vector.extract_strided_slice %479 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %883 = arith.andi %113, %836 : i1
  %884 = arith.addi %839, %562 overflow<nsw> : index
  %885 = arith.select %883, %884, %c536870911 : index
  vector.store %882, %517[%885] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %886 = vector.extract_strided_slice %479 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %887 = arith.andi %113, %844 : i1
  %888 = arith.addi %847, %562 overflow<nsw> : index
  %889 = arith.select %887, %888, %c536870911 : index
  vector.store %886, %517[%889] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %890 = vector.extract_strided_slice %479 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %891 = arith.andi %113, %852 : i1
  %892 = arith.addi %855, %562 overflow<nsw> : index
  %893 = arith.select %891, %892, %c536870911 : index
  vector.store %890, %517[%893] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %894 = vector.extract_strided_slice %479 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %895 = arith.andi %113, %860 : i1
  %896 = arith.addi %863, %562 overflow<nsw> : index
  %897 = arith.select %895, %896, %c536870911 : index
  vector.store %894, %517[%897] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %898 = vector.extract_strided_slice %481 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %899 = arith.andi %136, %836 : i1
  %900 = arith.addi %839, %579 overflow<nsw> : index
  %901 = arith.select %899, %900, %c536870911 : index
  vector.store %898, %517[%901] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %902 = vector.extract_strided_slice %481 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %903 = arith.andi %136, %844 : i1
  %904 = arith.addi %847, %579 overflow<nsw> : index
  %905 = arith.select %903, %904, %c536870911 : index
  vector.store %902, %517[%905] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %906 = vector.extract_strided_slice %481 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %907 = arith.andi %136, %852 : i1
  %908 = arith.addi %855, %579 overflow<nsw> : index
  %909 = arith.select %907, %908, %c536870911 : index
  vector.store %906, %517[%909] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %910 = vector.extract_strided_slice %481 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %911 = arith.andi %136, %860 : i1
  %912 = arith.addi %863, %579 overflow<nsw> : index
  %913 = arith.select %911, %912, %c536870911 : index
  vector.store %910, %517[%913] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %914 = vector.extract_strided_slice %483 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %915 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 80)>()[%thread_id_x, %block_id_x]
  %916 = arith.cmpi slt, %915, %arg5 : index
  %917 = arith.andi %62, %916 : i1
  %918 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 80)>()[%thread_id_x]
  %919 = arith.muli %918, %arg6 overflow<nsw> : index
  %920 = arith.addi %919, %512 overflow<nsw> : index
  %921 = arith.select %917, %920, %c536870911 : index
  vector.store %914, %517[%921] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %922 = vector.extract_strided_slice %483 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %923 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 81)>()[%thread_id_x, %block_id_x]
  %924 = arith.cmpi slt, %923, %arg5 : index
  %925 = arith.andi %62, %924 : i1
  %926 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 81)>()[%thread_id_x]
  %927 = arith.muli %926, %arg6 overflow<nsw> : index
  %928 = arith.addi %927, %512 overflow<nsw> : index
  %929 = arith.select %925, %928, %c536870911 : index
  vector.store %922, %517[%929] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %930 = vector.extract_strided_slice %483 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %931 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 82)>()[%thread_id_x, %block_id_x]
  %932 = arith.cmpi slt, %931, %arg5 : index
  %933 = arith.andi %62, %932 : i1
  %934 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 82)>()[%thread_id_x]
  %935 = arith.muli %934, %arg6 overflow<nsw> : index
  %936 = arith.addi %935, %512 overflow<nsw> : index
  %937 = arith.select %933, %936, %c536870911 : index
  vector.store %930, %517[%937] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %938 = vector.extract_strided_slice %483 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %939 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 83)>()[%thread_id_x, %block_id_x]
  %940 = arith.cmpi slt, %939, %arg5 : index
  %941 = arith.andi %62, %940 : i1
  %942 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 83)>()[%thread_id_x]
  %943 = arith.muli %942, %arg6 overflow<nsw> : index
  %944 = arith.addi %943, %512 overflow<nsw> : index
  %945 = arith.select %941, %944, %c536870911 : index
  vector.store %938, %517[%945] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %946 = vector.extract_strided_slice %485 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %947 = arith.andi %90, %916 : i1
  %948 = arith.addi %919, %545 overflow<nsw> : index
  %949 = arith.select %947, %948, %c536870911 : index
  vector.store %946, %517[%949] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %950 = vector.extract_strided_slice %485 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %951 = arith.andi %90, %924 : i1
  %952 = arith.addi %927, %545 overflow<nsw> : index
  %953 = arith.select %951, %952, %c536870911 : index
  vector.store %950, %517[%953] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %954 = vector.extract_strided_slice %485 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %955 = arith.andi %90, %932 : i1
  %956 = arith.addi %935, %545 overflow<nsw> : index
  %957 = arith.select %955, %956, %c536870911 : index
  vector.store %954, %517[%957] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %958 = vector.extract_strided_slice %485 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %959 = arith.andi %90, %940 : i1
  %960 = arith.addi %943, %545 overflow<nsw> : index
  %961 = arith.select %959, %960, %c536870911 : index
  vector.store %958, %517[%961] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %962 = vector.extract_strided_slice %487 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %963 = arith.andi %113, %916 : i1
  %964 = arith.addi %919, %562 overflow<nsw> : index
  %965 = arith.select %963, %964, %c536870911 : index
  vector.store %962, %517[%965] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %966 = vector.extract_strided_slice %487 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %967 = arith.andi %113, %924 : i1
  %968 = arith.addi %927, %562 overflow<nsw> : index
  %969 = arith.select %967, %968, %c536870911 : index
  vector.store %966, %517[%969] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %970 = vector.extract_strided_slice %487 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %971 = arith.andi %113, %932 : i1
  %972 = arith.addi %935, %562 overflow<nsw> : index
  %973 = arith.select %971, %972, %c536870911 : index
  vector.store %970, %517[%973] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %974 = vector.extract_strided_slice %487 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %975 = arith.andi %113, %940 : i1
  %976 = arith.addi %943, %562 overflow<nsw> : index
  %977 = arith.select %975, %976, %c536870911 : index
  vector.store %974, %517[%977] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %978 = vector.extract_strided_slice %489 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %979 = arith.andi %136, %916 : i1
  %980 = arith.addi %919, %579 overflow<nsw> : index
  %981 = arith.select %979, %980, %c536870911 : index
  vector.store %978, %517[%981] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %982 = vector.extract_strided_slice %489 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %983 = arith.andi %136, %924 : i1
  %984 = arith.addi %927, %579 overflow<nsw> : index
  %985 = arith.select %983, %984, %c536870911 : index
  vector.store %982, %517[%985] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %986 = vector.extract_strided_slice %489 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %987 = arith.andi %136, %932 : i1
  %988 = arith.addi %935, %579 overflow<nsw> : index
  %989 = arith.select %987, %988, %c536870911 : index
  vector.store %986, %517[%989] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %990 = vector.extract_strided_slice %489 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %991 = arith.andi %136, %940 : i1
  %992 = arith.addi %943, %579 overflow<nsw> : index
  %993 = arith.select %991, %992, %c536870911 : index
  vector.store %990, %517[%993] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %994 = vector.extract_strided_slice %491 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %995 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 96)>()[%thread_id_x, %block_id_x]
  %996 = arith.cmpi slt, %995, %arg5 : index
  %997 = arith.andi %62, %996 : i1
  %998 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 96)>()[%thread_id_x]
  %999 = arith.muli %998, %arg6 overflow<nsw> : index
  %1000 = arith.addi %999, %512 overflow<nsw> : index
  %1001 = arith.select %997, %1000, %c536870911 : index
  vector.store %994, %517[%1001] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1002 = vector.extract_strided_slice %491 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1003 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 97)>()[%thread_id_x, %block_id_x]
  %1004 = arith.cmpi slt, %1003, %arg5 : index
  %1005 = arith.andi %62, %1004 : i1
  %1006 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 97)>()[%thread_id_x]
  %1007 = arith.muli %1006, %arg6 overflow<nsw> : index
  %1008 = arith.addi %1007, %512 overflow<nsw> : index
  %1009 = arith.select %1005, %1008, %c536870911 : index
  vector.store %1002, %517[%1009] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1010 = vector.extract_strided_slice %491 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1011 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 98)>()[%thread_id_x, %block_id_x]
  %1012 = arith.cmpi slt, %1011, %arg5 : index
  %1013 = arith.andi %62, %1012 : i1
  %1014 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 98)>()[%thread_id_x]
  %1015 = arith.muli %1014, %arg6 overflow<nsw> : index
  %1016 = arith.addi %1015, %512 overflow<nsw> : index
  %1017 = arith.select %1013, %1016, %c536870911 : index
  vector.store %1010, %517[%1017] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1018 = vector.extract_strided_slice %491 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1019 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 99)>()[%thread_id_x, %block_id_x]
  %1020 = arith.cmpi slt, %1019, %arg5 : index
  %1021 = arith.andi %62, %1020 : i1
  %1022 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 99)>()[%thread_id_x]
  %1023 = arith.muli %1022, %arg6 overflow<nsw> : index
  %1024 = arith.addi %1023, %512 overflow<nsw> : index
  %1025 = arith.select %1021, %1024, %c536870911 : index
  vector.store %1018, %517[%1025] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1026 = vector.extract_strided_slice %493 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1027 = arith.andi %90, %996 : i1
  %1028 = arith.addi %999, %545 overflow<nsw> : index
  %1029 = arith.select %1027, %1028, %c536870911 : index
  vector.store %1026, %517[%1029] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1030 = vector.extract_strided_slice %493 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1031 = arith.andi %90, %1004 : i1
  %1032 = arith.addi %1007, %545 overflow<nsw> : index
  %1033 = arith.select %1031, %1032, %c536870911 : index
  vector.store %1030, %517[%1033] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1034 = vector.extract_strided_slice %493 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1035 = arith.andi %90, %1012 : i1
  %1036 = arith.addi %1015, %545 overflow<nsw> : index
  %1037 = arith.select %1035, %1036, %c536870911 : index
  vector.store %1034, %517[%1037] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1038 = vector.extract_strided_slice %493 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1039 = arith.andi %90, %1020 : i1
  %1040 = arith.addi %1023, %545 overflow<nsw> : index
  %1041 = arith.select %1039, %1040, %c536870911 : index
  vector.store %1038, %517[%1041] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1042 = vector.extract_strided_slice %495 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1043 = arith.andi %113, %996 : i1
  %1044 = arith.addi %999, %562 overflow<nsw> : index
  %1045 = arith.select %1043, %1044, %c536870911 : index
  vector.store %1042, %517[%1045] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1046 = vector.extract_strided_slice %495 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1047 = arith.andi %113, %1004 : i1
  %1048 = arith.addi %1007, %562 overflow<nsw> : index
  %1049 = arith.select %1047, %1048, %c536870911 : index
  vector.store %1046, %517[%1049] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1050 = vector.extract_strided_slice %495 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1051 = arith.andi %113, %1012 : i1
  %1052 = arith.addi %1015, %562 overflow<nsw> : index
  %1053 = arith.select %1051, %1052, %c536870911 : index
  vector.store %1050, %517[%1053] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1054 = vector.extract_strided_slice %495 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1055 = arith.andi %113, %1020 : i1
  %1056 = arith.addi %1023, %562 overflow<nsw> : index
  %1057 = arith.select %1055, %1056, %c536870911 : index
  vector.store %1054, %517[%1057] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1058 = vector.extract_strided_slice %497 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1059 = arith.andi %136, %996 : i1
  %1060 = arith.addi %999, %579 overflow<nsw> : index
  %1061 = arith.select %1059, %1060, %c536870911 : index
  vector.store %1058, %517[%1061] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1062 = vector.extract_strided_slice %497 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1063 = arith.andi %136, %1004 : i1
  %1064 = arith.addi %1007, %579 overflow<nsw> : index
  %1065 = arith.select %1063, %1064, %c536870911 : index
  vector.store %1062, %517[%1065] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1066 = vector.extract_strided_slice %497 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1067 = arith.andi %136, %1012 : i1
  %1068 = arith.addi %1015, %579 overflow<nsw> : index
  %1069 = arith.select %1067, %1068, %c536870911 : index
  vector.store %1066, %517[%1069] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1070 = vector.extract_strided_slice %497 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1071 = arith.andi %136, %1020 : i1
  %1072 = arith.addi %1023, %579 overflow<nsw> : index
  %1073 = arith.select %1071, %1072, %c536870911 : index
  vector.store %1070, %517[%1073] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1074 = vector.extract_strided_slice %499 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1075 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 112)>()[%thread_id_x, %block_id_x]
  %1076 = arith.cmpi slt, %1075, %arg5 : index
  %1077 = arith.andi %62, %1076 : i1
  %1078 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 112)>()[%thread_id_x]
  %1079 = arith.muli %1078, %arg6 overflow<nsw> : index
  %1080 = arith.addi %1079, %512 overflow<nsw> : index
  %1081 = arith.select %1077, %1080, %c536870911 : index
  vector.store %1074, %517[%1081] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1082 = vector.extract_strided_slice %499 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1083 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 113)>()[%thread_id_x, %block_id_x]
  %1084 = arith.cmpi slt, %1083, %arg5 : index
  %1085 = arith.andi %62, %1084 : i1
  %1086 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 113)>()[%thread_id_x]
  %1087 = arith.muli %1086, %arg6 overflow<nsw> : index
  %1088 = arith.addi %1087, %512 overflow<nsw> : index
  %1089 = arith.select %1085, %1088, %c536870911 : index
  vector.store %1082, %517[%1089] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1090 = vector.extract_strided_slice %499 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1091 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 114)>()[%thread_id_x, %block_id_x]
  %1092 = arith.cmpi slt, %1091, %arg5 : index
  %1093 = arith.andi %62, %1092 : i1
  %1094 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 114)>()[%thread_id_x]
  %1095 = arith.muli %1094, %arg6 overflow<nsw> : index
  %1096 = arith.addi %1095, %512 overflow<nsw> : index
  %1097 = arith.select %1093, %1096, %c536870911 : index
  vector.store %1090, %517[%1097] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1098 = vector.extract_strided_slice %499 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1099 = affine.apply affine_map<()[s0, s1] -> (s1 * 128 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 115)>()[%thread_id_x, %block_id_x]
  %1100 = arith.cmpi slt, %1099, %arg5 : index
  %1101 = arith.andi %62, %1100 : i1
  %1102 = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + 115)>()[%thread_id_x]
  %1103 = arith.muli %1102, %arg6 overflow<nsw> : index
  %1104 = arith.addi %1103, %512 overflow<nsw> : index
  %1105 = arith.select %1101, %1104, %c536870911 : index
  vector.store %1098, %517[%1105] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1106 = vector.extract_strided_slice %501 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1107 = arith.andi %90, %1076 : i1
  %1108 = arith.addi %1079, %545 overflow<nsw> : index
  %1109 = arith.select %1107, %1108, %c536870911 : index
  vector.store %1106, %517[%1109] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1110 = vector.extract_strided_slice %501 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1111 = arith.andi %90, %1084 : i1
  %1112 = arith.addi %1087, %545 overflow<nsw> : index
  %1113 = arith.select %1111, %1112, %c536870911 : index
  vector.store %1110, %517[%1113] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1114 = vector.extract_strided_slice %501 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1115 = arith.andi %90, %1092 : i1
  %1116 = arith.addi %1095, %545 overflow<nsw> : index
  %1117 = arith.select %1115, %1116, %c536870911 : index
  vector.store %1114, %517[%1117] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1118 = vector.extract_strided_slice %501 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1119 = arith.andi %90, %1100 : i1
  %1120 = arith.addi %1103, %545 overflow<nsw> : index
  %1121 = arith.select %1119, %1120, %c536870911 : index
  vector.store %1118, %517[%1121] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1122 = vector.extract_strided_slice %503 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1123 = arith.andi %113, %1076 : i1
  %1124 = arith.addi %1079, %562 overflow<nsw> : index
  %1125 = arith.select %1123, %1124, %c536870911 : index
  vector.store %1122, %517[%1125] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1126 = vector.extract_strided_slice %503 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1127 = arith.andi %113, %1084 : i1
  %1128 = arith.addi %1087, %562 overflow<nsw> : index
  %1129 = arith.select %1127, %1128, %c536870911 : index
  vector.store %1126, %517[%1129] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1130 = vector.extract_strided_slice %503 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1131 = arith.andi %113, %1092 : i1
  %1132 = arith.addi %1095, %562 overflow<nsw> : index
  %1133 = arith.select %1131, %1132, %c536870911 : index
  vector.store %1130, %517[%1133] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1134 = vector.extract_strided_slice %503 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1135 = arith.andi %113, %1100 : i1
  %1136 = arith.addi %1103, %562 overflow<nsw> : index
  %1137 = arith.select %1135, %1136, %c536870911 : index
  vector.store %1134, %517[%1137] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1138 = vector.extract_strided_slice %505 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1139 = arith.andi %136, %1076 : i1
  %1140 = arith.addi %1079, %579 overflow<nsw> : index
  %1141 = arith.select %1139, %1140, %c536870911 : index
  vector.store %1138, %517[%1141] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1142 = vector.extract_strided_slice %505 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1143 = arith.andi %136, %1084 : i1
  %1144 = arith.addi %1087, %579 overflow<nsw> : index
  %1145 = arith.select %1143, %1144, %c536870911 : index
  vector.store %1142, %517[%1145] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1146 = vector.extract_strided_slice %505 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1147 = arith.andi %136, %1092 : i1
  %1148 = arith.addi %1095, %579 overflow<nsw> : index
  %1149 = arith.select %1147, %1148, %c536870911 : index
  vector.store %1146, %517[%1149] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  %1150 = vector.extract_strided_slice %505 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
  %1151 = arith.andi %136, %1100 : i1
  %1152 = arith.addi %1103, %579 overflow<nsw> : index
  %1153 = arith.select %1151, %1152, %c536870911 : index
  vector.store %1150, %517[%1153] : memref<?xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf32>
  return
}
}
