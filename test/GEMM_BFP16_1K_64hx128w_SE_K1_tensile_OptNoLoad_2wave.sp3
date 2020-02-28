////////////////////////////////////////////////////////////////
///////////////implementation description///////////////////////
/////1. each thread group generate a 64X128. by multiply block A(64XK) and block B (KX128).
/////2. each thread group's input block A addressed by thread gourp idy,  block B addressed by thread gourp idx.
/////3. re-construct 8 waves into 2 dimensional  2 groups of 4 waves group(0) for fetch , group(1) for mac
/////3. each thread group has 8 waves. 4 waves are used for fetching 64x128x32 elements and other 4 waves are used for math
/////4. math waves use barrier to sync with fetch waves
/////5. each wave generate a 64X32,  by multiply block A(64XK) and block B (KX32)
/////6. in each loop, each wave multiply block A(64X32) and block B (32X32),  wave mem data load as belowing: 
///           Matrix A (K)                       Matrix B (N)
///           ---------32              --------4-------8------12-------16---------32---------- 64------------128----------
///           0   w0Ab0                -       |       |       |       |          |            |              |             |            |
///           4   w0Ab1                -       |       |       |       |          |            |              |             |            |
///           8   w0Ab2                -       |       |       |       |          |            |              |             |            |
///          12   w0Ab3                -       |       |       |       |          |            |              |             |            |
///      (M) 16   w1Ab0           (K)  - w0Bb0 | w0Bb1 | w0Bb2 | w0Bb3 | w0Bb[4-7]| w1Bb[0-7] |  w2Bb[0-7]  | w3Bb[0-7]
///           -                        -       |       |       |       |          |            |              |             |            |
///          ...                       -       |       |       |       |          |            |              |             |            |
///          32   w2Ab0                -       |       |       |       |          |            |              |             |            |
///          ...                      32       |       |       |       |          |            |              |             |            |
///          48   w3Ab0
///          ..
///          60   w3Ab3
//////////////////////////////////////////////////////////////////
//////sreg def/////////////

var sgprKernArgAddress = 0 
var sgprWorkGroup0 = 2
var sgprWorkGroup1 = 3
var sgprWorkGroup2 = 4
var sgprNumWorkGroups0=5
var sgprNumWorkGroups1=6
var sgprSrdA=8
var sgprSrdB=12
var sgprSrdC=16
var sgprSrdD=20
var sgprTensor2dSizeA= 24
var sgprTensor2dSizeB= 26
var sgprTensor2dSizeC= 28
var sgprSaveExecMask= 30
var sgprAddressD= 32
var sgprAddressC= 34
var sgprStridesD= 36
var sgprStridesC= 38
var sgprAlpha= 40
var sgprBeta= 41
var sgprSizesFree = 42
var sgprSizesSum  = 45
var sgprLoopCounters= 46
var sgprOrigLoopCounter= 47
var sgprStridesA= 48
var sgprStridesB= 50
var sgprAddressA= 52
var sgprAddressB= 54
var sgprShadowLimitA= 56
var sgprShadowLimitB= 58
var sgprOrigStaggerUIter= 60
var sgprStaggerUIter= 61
var sgprWrapUA= 62
var sgprWrapUB= 64
var sgprNumFullBlocks= 66
var sgprWgmRemainder1= 67
var sgprMagicNumberWgmRemainder1= 68
var sgprGlobalReadIncsA= 69
var sgprGlobalReadIncsB= 70
var sgprScalarGlobalReadOffsetA=71
var sgprScalarGlobalReadOffsetB=74
var sgprLocalWriteAddrA=76
var sgprLocalWriteAddrB=78
var hw_id = 80
var sgprFetchSubGrpId=82


/////vreg def////////////////

var vgprValuC=0
var vgprAcc=0
var vgprValuA_X0_I0=32
var vgprValuA_X0_I1=48
var vgprG2LA=48
var vgprValuB_X0_I0=68
var vgprG2LB=68
var vgprLocalWriteAddrA=76
var vgprLocalWriteAddrB=78
var vgprGlobalReadOfvarA=82
var vgprGlobalReadOfvarB=86
var vgprLocalReadAddrA=94
var vgprLocalReadAddrB=96
var vgprSerial=100
var vgprGlobalWriteOfvarC=104
var vgprTmp=106

////constant def/////////////
var varlds_pad            = 8
var varlds_pad_qw         = varlds_pad >> 2
var varlds_Asize_per_wr   = 256+varlds_pad                  //each load inst load one 32X4 block.    need contiunous 32X4X2=256    bytes in LDS
var varlds_Asize_per_wave = varlds_Asize_per_wr * 4   //each wave load 4 32X4 block one time.  need contiunous 32X4X4X2=1024 bytes in LDS
var varlds_Asize_per_wg   = varlds_Asize_per_wave * 4 //WG load 16 32X4 block(64X32) Matrix A to lds for pingpong.
var M_row_per_WG          = 64       //each WG process 64 row
var varlds_Bsize_per_wr   = 256+varlds_pad             //each load inst load one 32X4  block.    need contiunous 32X4X2=256     bytes in LDS
var varlds_Bsize_per_wave = varlds_Bsize_per_wr * 8   //each wave load seperate 32X64 block.    need contiunous 32X4X8X2=2048 bytes in LDS
var varlds_Bsize_per_wg   = varlds_Bsize_per_wave * 4  //WG load 64 32X4 block(32X256) Matrix B to lds for pingpong.
var varA_lds_base_addr    = 0
var varB_lds_base_addr    = varA_lds_base_addr + varlds_Asize_per_wg * 2  //in bytes

function v_regs(base, offset)
    var v_idx
    v_idx = base + offset
    return v[v_idx]
end

function s_regs(base, offset)
    var s_idx
    s_idx = base + offset
    return s[s_idx]
end

/******************************************/
/* 2GB limit - set offsets to -1 to exceed this and clamp */
/******************************************/
var BufferLimit=0x80000000

/******************************************/
/* Bits 127:96 of SRD.  Set DataFormat = 32 bit */
/******************************************/
var Srd127_96=0x0020000

var roundMaskVal=0xffff0000

shader main
  type(CS)

  user_sgpr_count(14)
  tgid_x_en(1)                                                  // s_tgid_x 
  tgid_y_en(1)                                                  // s_tgid_y
  tgid_z_en(1)                                                  // s_tgid_z 
  tidig_comp_cnt(2)


  /* Load Kernel Args */
  s_load_dwordx4 s[sgprAddressA:sgprAddressB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x28 // 
  s_load_dwordx4 s[sgprStridesA+0:sgprStridesB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50 // 
  s_load_dwordx4 s[sgprTensor2dSizeA+0:sgprTensor2dSizeB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x8 // 
  s_load_dword s[sgprSizesSum+0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x6c // 

  //s_load_dword s[sgprTensor2dSizeA+0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x8 // 
  //s_load_dword s[sgprTensor2dSizeA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xc // 
  //s_load_dword s[sgprTensor2dSizeB+0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10 // 
  //s_load_dword s[sgprTensor2dSizeB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x14 // 
  //s_load_dword s[sgprAddressA], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x28 // 
  //s_load_dword s[sgprAddressA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x2c // 
  //s_load_dword s[sgprAddressB], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x30 // 
  //s_load_dword s[sgprAddressB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x34 // 
  //s_load_dword s[sgprStridesA+0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50 // 
  //s_load_dword s[sgprStridesA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x54 // 
  //s_load_dword s[sgprStridesB+0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58 // 
  //s_load_dword s[sgprStridesB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x5c // 
  //s_load_dword s[sgprSizesSum+0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x6c // 


  s_mov_b32	        m0,0x3000					// LDS camp at 12288 bytes

  //vgprSerial  holds threadIdx 
  //vgprSerial+1  holds WaveFrontId (0-63)
  //vgprSerial+2  holds threadIdy  (wave0 =0 wave1 = 1)
  //vgprSerial+3  holds threadIdZ  (simd= 0,1,2,3)

  //HW_ID_REG
  //bit[0-3] //waveId
  //bit[5-4] //simdId
  //bit[11-8] //CuId

  v_mov_b32 	        v[vgprSerial], v0				//thread serial Id
  v_and_b32             v[vgprSerial+1], 0x3f, v0		        //threadId-x

  v_lshrrev_b32    	v2,  6,  v[vgprSerial]

  //Fetchid -- wave that fetches 16 rows in 64;; uses simdId 
  s_getreg_b32          s[hw_id], hwreg(HW_REG_HW_ID)
  v_and_b32             v4, 0x30, s[hw_id]
  v_lshrrev_b32         v[vgprSerial+2], 4, v4					//simdId
  v_readfirstlane_b32   s[sgprFetchSubGrpId], v[vgprSerial+2]
  v_and_b32             v[vgprSerial+3], 0xf, s[hw_id]				//waveId
  v_readfirstlane_b32   s[hw_id+1], v[vgprSerial+3]

  s_cmp_eq_u32     s[hw_id+1], 1 
  s_cbranch_scc0   wave0_entry_start

  s_waitcnt lgkmcnt(0)                               // wait for 144 bytes of kern args

//WorkgroupId agnostic address calculation
//s[workGroup0] provides MT0 tile number  that this workgroup working 
//s[workGroup1] provides MT1 tile number that this workgroup working 
// use tle number to generate start address of the tile that this workgroup allocated too


  //Global read addresses: address A resource descriptor set-up
  //sgpr[0-1] - base address
  //sgpr[2]   - limit
  //sgpr[3]   - attributes

  //calculate base address for  A 
  //1. multiply MT0 size with TileNumber passed in s[sgprWorkGroup0]
  //2. multiply [1] result with stride[0] store result into 64-bit
  //3. the above two steps gives starting address of tile that this workgroup working


  s_mov_b32        s[sgprSrdA+0], s[sgprAddressA+0]	        // SRD base = Address + tile_start
  s_mov_b32        s[sgprSrdA+1], s[sgprAddressA+1]	        // SRD base = Address + tile_start
  s_mov_b32        s[sgprSrdA+3], Srd127_96 			// set bits 127_96 in SRD
  s_sub_u32	   s[sgprShadowLimitA+0], s[sgprTensor2dSizeA], s84 // sub TileStart
  s_sub_u32	   s[sgprShadowLimitA+1], s[sgprTensor2dSizeA], s85 // sub TileStart
  s_lshl_b64	   s[sgprShadowLimitA+0:sgprShadowLimitA+1], s[sgprShadowLimitA+0:sgprShadowLimitA+1], 1  //convert to bytes
  s_add_u32 	   s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 4 			   // pad by 4
  s_addc_u32 	   s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 			   // pad by 4
  s_cmp_eq_u32     s[sgprShadowLimitA+1], 0 			//are we within 2^32 
  s_cselect_b32    s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit  // move shadow to real if we are within 2^32
  s_mov_b32	   s[sgprSrdA+2], BufferLimit
  

  s_mul_i32        s84,       s[sgprStridesA+0], 64	        //workGroup[0]*MT
  s_mul_i32        s84,       s[sgprWorkGroup0], s84	        //workGroup[0]*MT
  s_lshl_b32       s85,       s[sgprFetchSubGrpId], 4				// s85 = fetchId * x16 rows per wave
  s_mul_i32        s83,       s85, s[sgprStridesA+0] 		//wave start offset
  s_add_i32        s84,       s84, s83


          
  //tile offset assignment a : global read address 
  //LVCA= 16 
  //glvw = 1
  v_lshrrev_b32    v0,    4, v[vgprSerial+1]			//sub-groupin
  v_mul_lo_u32     v4,    s[sgprStridesA+0],  v0		//mul d1 lower
  v_and_b32        v1,    15, v[vgprSerial+1]
  v_lshlrev_b32	   v1,    1,  v1				// GLVW=1 (load_Dword) for GLVW=2(dwordx2) =2 GLVW=4 = 3 (dowrdx4) for BF16 ;; for f32 GLVW=1 (skip this isntruction) GLVW=1 (DWORDX2) GLVW=2 (DWORDX4)
  v_add_co_u32     v[vgprGlobalReadOfvarA+0], vcc, v4, v1	//accumulate d1 lower

  v_add_u32        v[vgprGlobalReadOfvarA+0], s84, v[vgprGlobalReadOfvarA+0]
  v_lshlrev_b32    v[vgprGlobalReadOfvarA+0], 0x1, v[vgprGlobalReadOfvarA+0]  // offset *= bytes/element (x2)  // convert into bytes offset

   
  // why subtract lds_Asize_per_wr - A_lds_size_wr is used as inst_offset in buffer_load_dword lds:1
  s_lshl_b32        s[sgprScalarGlobalReadOffsetA+0],   s[sgprStridesA+0],    3   // X16 = X4(4 lines) X2(conver to bytes).  each buffer load process 4 lines.
  s_sub_u32        s[sgprScalarGlobalReadOffsetA+0],  s[sgprScalarGlobalReadOffsetA+0], varlds_Asize_per_wr
  v_add_u32        v[vgprGlobalReadOfvarA+1], s[sgprScalarGlobalReadOffsetA+0], v[vgprGlobalReadOfvarA+0]
  v_add_u32        v[vgprGlobalReadOfvarA+2], s[sgprScalarGlobalReadOffsetA+0], v[vgprGlobalReadOfvarA+1]
  v_add_u32        v[vgprGlobalReadOfvarA+3], s[sgprScalarGlobalReadOffsetA+0], v[vgprGlobalReadOfvarA+2]

/* local write addresses: first offset a */
  s_mov_b32        s[sgprLocalWriteAddrA+0], varlds_Asize_per_wave
  s_mul_i32        s[sgprLocalWriteAddrA+0], s[sgprFetchSubGrpId], s[sgprLocalWriteAddrA+0] //lds start address of each wave in bytes

/******************************************/
/* global read addresses: addresses b */
/******************************************/
  s_mov_b32        s[sgprSrdB+0], s[sgprAddressB+0]             // SRD base = Address + tile_start
  s_mov_b32        s[sgprSrdB+1], s[sgprAddressB+1]             // SRD base = Address + tile_start
  s_mov_b32        s[sgprSrdB+3], Srd127_96                     // set bits 127_96 in SRD
  s_sub_u32        s[sgprShadowLimitB+0], s[sgprTensor2dSizeB], s84 // sub TileStart
  s_sub_u32        s[sgprShadowLimitB+1], s[sgprTensor2dSizeB], s85 // sub TileStart
  s_lshl_b64       s[sgprShadowLimitB+0:sgprShadowLimitB+1], s[sgprShadowLimitB+0:sgprShadowLimitB+1], 1  //convert to bytes
  s_add_u32        s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 4                         // pad by 4
  s_addc_u32       s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0                         // pad by 4
  s_cmp_eq_u32     s[sgprShadowLimitB+1], 0                     //are we within 2^32
  s_cselect_b32    s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit  // move shadow to real if we are within 2^32
  s_mov_b32        s[sgprSrdB+2], BufferLimit

  s_mul_i32        s84,       s[sgprStridesB+0], 128            //workGroup[0]*MT
  s_mul_i32        s84,       s[sgprWorkGroup1], s84            //workGroup[0]*MT
  s_mul_i32        s85,       32, s[sgprStridesB+0]
  s_mul_i32        s85,       s[sgprFetchSubGrpId], s85
  s_add_i32        s84,       s84, s85

  //tile offset assignment b : global read address
  /* LVCA= 16 */
  //glvw = 1
  v_lshrrev_b32    v2,     4,  v[vgprSerial+1]
  v_and_b32        v3,     15, v[vgprSerial+1]
  v_lshlrev_b32	   v3,     1,  v3
  v_mul_lo_u32     v4,     s[sgprStridesB+0], v2                        //mul d1 lower
  v_add_co_u32     v[vgprGlobalReadOfvarB+0], vcc, v4,  v3              //accumulate d1 lower
  v_add_u32        v[vgprGlobalReadOfvarB+0], s84,  v[vgprGlobalReadOfvarB+0]           //accumulate d1 lower
  v_lshlrev_b32    v[vgprGlobalReadOfvarB+0], 0x1, v[vgprGlobalReadOfvarB+0]  // offset *= bytes/element


  s_lshl_b32       s[sgprScalarGlobalReadOffsetB+0], s[sgprStridesB+0],    3   // X16 = X4(4 lines) X2(conver to bytes).  each buffer load process 4 lines.
  s_sub_u32        s[sgprScalarGlobalReadOffsetB+0], s[sgprScalarGlobalReadOffsetB+0], varlds_Bsize_per_wr

// X16 = X4(4 lines) X2(conver to bytes).  each buffer load process 4 lines.
  for var i = 1; i < 8; i++
    v_add_u32         v_regs(vgprGlobalReadOfvarB,i), s[sgprScalarGlobalReadOffsetB+0],  v_regs(vgprGlobalReadOfvarB,i-1)
  end


/* local write addresses: first offset b */
  s_mov_b32        s[sgprLocalWriteAddrB+0], varlds_Bsize_per_wave
  s_mul_i32        s[sgprLocalWriteAddrB+0], s[sgprFetchSubGrpId], s[sgprLocalWriteAddrB+0]
  s_add_i32        s[sgprLocalWriteAddrB+0], s[sgprLocalWriteAddrB+0], varB_lds_base_addr



  //////////////preload to LDS///////////
  //Fetch latency is about ~1000 of cycles
  //v_mfma_f32_32x32x4bf16 => latency is 64 cycles 
  // we need deep software pipelining to hide read latency = MAC latency
  // unroll Depth = 32 
  // global prefetch next 2 unroll loop iteration
  // wait for first 32 iteration data to arrive
  // prefetch from  LDS A, B 
  // unroll loop start
  // 16 * 64 miMAC cycles 
  // LDS prefetch next 8 micro-iteration 

  // prefetch: Global to Local D2LDS
  // fetch 64x32 elements => each load fetch  4x32 elements / Wave
  // TODO : Convert x4 fetch

  s_barrier

  s_mov_b32        m0,  s[sgprLocalWriteAddrA+0] //lds input offset
  s_add_i32        s[sgprLocalWriteAddrA+1], s[sgprLocalWriteAddrA+0], varlds_Asize_per_wg

  //Fetch A 16x32 elements/wave 
  buffer_load_dword v[vgprG2LA+0],  v[vgprGlobalReadOfvarA+0],  s[sgprSrdA:sgprSrdA+3], 0 offen:1 lds:1 offset:0
  buffer_load_dword v[vgprG2LA+1],  v[vgprGlobalReadOfvarA+1],  s[sgprSrdA:sgprSrdA+3], 0 offen:1 lds:1 offset:varlds_Asize_per_wr * 1
  buffer_load_dword v[vgprG2LA+2],  v[vgprGlobalReadOfvarA+2],  s[sgprSrdA:sgprSrdA+3], 0 offen:1 lds:1 offset:varlds_Asize_per_wr * 2
  buffer_load_dword v[vgprG2LA+3],  v[vgprGlobalReadOfvarA+3],  s[sgprSrdA:sgprSrdA+3], 0 offen:1 lds:1 offset:varlds_Asize_per_wr * 3

  s_mov_b32        m0,  s[sgprLocalWriteAddrB+0]     //lds offset B
  s_add_i32        s[sgprLocalWriteAddrB+1], s[sgprLocalWriteAddrB+0], varlds_Bsize_per_wg

  // prefetch: Global to Local D2LDS
  // fetch 64x32 elements => each load fetch  4x32 elements / Wave
  // TODO : Convert x4 fetch
  buffer_load_dword v[vgprG2LB+0],  v[vgprGlobalReadOfvarB+0],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:0
  buffer_load_dword v[vgprG2LB+1],  v[vgprGlobalReadOfvarB+1],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr * 1
  buffer_load_dword v[vgprG2LB+2],  v[vgprGlobalReadOfvarB+2],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr * 2
  buffer_load_dword v[vgprG2LB+3],  v[vgprGlobalReadOfvarB+3],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr * 3
  buffer_load_dword v[vgprG2LB+4],  v[vgprGlobalReadOfvarB+4],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr * 4
  buffer_load_dword v[vgprG2LB+5],  v[vgprGlobalReadOfvarB+5],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr * 5
  buffer_load_dword v[vgprG2LB+6],  v[vgprGlobalReadOfvarB+6],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr * 6
  buffer_load_dword v[vgprG2LB+7],  v[vgprGlobalReadOfvarB+7],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr * 7

  // update LDS pointer For Fetch A[1]
  s_mov_b32  m0, s[sgprLocalWriteAddrA+1]
  //increment 32 elements to fetch next k=32 elements of tile 64x32
  v_add_u32	v[vgprGlobalReadOfvarA+0], 64, v[vgprGlobalReadOfvarA+0]
  v_add_u32	v[vgprGlobalReadOfvarA+1], 64, v[vgprGlobalReadOfvarA+1]
  v_add_u32	v[vgprGlobalReadOfvarA+2], 64, v[vgprGlobalReadOfvarA+2]
  v_add_u32	v[vgprGlobalReadOfvarA+3], 64, v[vgprGlobalReadOfvarA+3]
  v_add_u32	v[vgprGlobalReadOfvarB+0], 64, v[vgprGlobalReadOfvarB+0]
  v_add_u32	v[vgprGlobalReadOfvarB+1], 64, v[vgprGlobalReadOfvarB+1]
  v_add_u32	v[vgprGlobalReadOfvarB+2], 64, v[vgprGlobalReadOfvarB+2]
  v_add_u32	v[vgprGlobalReadOfvarB+3], 64, v[vgprGlobalReadOfvarB+3]
  v_add_u32	v[vgprGlobalReadOfvarB+4], 64, v[vgprGlobalReadOfvarB+4]
  v_add_u32	v[vgprGlobalReadOfvarB+5], 64, v[vgprGlobalReadOfvarB+5]
  v_add_u32	v[vgprGlobalReadOfvarB+6], 64, v[vgprGlobalReadOfvarB+6]
  v_add_u32	v[vgprGlobalReadOfvarB+7], 64, v[vgprGlobalReadOfvarB+7]

  //Fetch 2nd unroll loop iteration (2nd 32 k indices)
  buffer_load_dword v[vgprG2LA+0],  v[vgprGlobalReadOfvarA+0],  s[sgprSrdA:sgprSrdA+3], 0 offen:1 lds:1 offset:0
  buffer_load_dword v[vgprG2LA+1],  v[vgprGlobalReadOfvarA+1],  s[sgprSrdA:sgprSrdA+3], 0 offen:1 lds:1 offset:varlds_Asize_per_wr*1
  buffer_load_dword v[vgprG2LA+2],  v[vgprGlobalReadOfvarA+2],  s[sgprSrdA:sgprSrdA+3], 0 offen:1 lds:1 offset:varlds_Asize_per_wr*2
  buffer_load_dword v[vgprG2LA+3],  v[vgprGlobalReadOfvarA+3],  s[sgprSrdA:sgprSrdA+3], 0 offen:1 lds:1 offset:varlds_Asize_per_wr*3

  // update LDS pointer For Fetch B[1]
  s_mov_b32         m0,  s[sgprLocalWriteAddrB+1] //lds input offset
  s_nop 0

  buffer_load_dword v[vgprG2LB+0],  v[vgprGlobalReadOfvarB+0],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:0
  buffer_load_dword v[vgprG2LB+1],  v[vgprGlobalReadOfvarB+1],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*1
  buffer_load_dword v[vgprG2LB+2],  v[vgprGlobalReadOfvarB+2],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*2
  buffer_load_dword v[vgprG2LB+3],  v[vgprGlobalReadOfvarB+3],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*3
  buffer_load_dword v[vgprG2LB+4],  v[vgprGlobalReadOfvarB+4],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*4
  buffer_load_dword v[vgprG2LB+5],  v[vgprGlobalReadOfvarB+5],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*5
  buffer_load_dword v[vgprG2LB+6],  v[vgprGlobalReadOfvarB+6],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*6
  buffer_load_dword v[vgprG2LB+7],  v[vgprGlobalReadOfvarB+7],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*7

  //increment 32 elements to fetch next k=32 elements of tile 64x32
  v_add_u32	v[vgprGlobalReadOfvarA+0], 64, v[vgprGlobalReadOfvarA+0]
  v_add_u32	v[vgprGlobalReadOfvarA+1], 64, v[vgprGlobalReadOfvarA+1]
  v_add_u32	v[vgprGlobalReadOfvarA+2], 64, v[vgprGlobalReadOfvarA+2]
  v_add_u32	v[vgprGlobalReadOfvarA+3], 64, v[vgprGlobalReadOfvarA+3]
  v_add_u32	v[vgprGlobalReadOfvarB+0], 64, v[vgprGlobalReadOfvarB+0]
  v_add_u32	v[vgprGlobalReadOfvarB+1], 64, v[vgprGlobalReadOfvarB+1]
  v_add_u32	v[vgprGlobalReadOfvarB+2], 64, v[vgprGlobalReadOfvarB+2]
  v_add_u32	v[vgprGlobalReadOfvarB+3], 64, v[vgprGlobalReadOfvarB+3]
  v_add_u32	v[vgprGlobalReadOfvarB+4], 64, v[vgprGlobalReadOfvarB+4]
  v_add_u32	v[vgprGlobalReadOfvarB+5], 64, v[vgprGlobalReadOfvarB+5]
  v_add_u32	v[vgprGlobalReadOfvarB+6], 64, v[vgprGlobalReadOfvarB+6]
  v_add_u32	v[vgprGlobalReadOfvarB+7], 64, v[vgprGlobalReadOfvarB+7]
  s_mov_b32     m0, s[sgprLocalWriteAddrA+0]

  s_waitcnt vmcnt(20)
  s_barrier
  s_waitcnt vmcnt(12)
  s_barrier

  s_lshr_b32       s[sgprLoopCounters+0], s[sgprSizesSum+0], 5 // s[sgprLoopCounters+0] = s[sgprSizesSum+0] / 32
  s_sub_u32	   s[sgprLoopCounters+0], 0x0, s[sgprLoopCounters+0]
  s_cmp_eq_u32     s[sgprLoopCounters+0], 0x0            // numIter0I == 0
  s_cbranch_scc1   label_0006                           // Dont enter Loop

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_0005:

  s_barrier

  //Fetch A for Unroll iteration# (u+2) for 32 k indices
  buffer_load_dword v[vgprG2LA+0],  v[vgprGlobalReadOfvarA+0],  s[sgprSrdA:sgprSrdA+3], 0 offen:1 lds:1 offset:0
  buffer_load_dword v[vgprG2LA+1],  v[vgprGlobalReadOfvarA+1],  s[sgprSrdA:sgprSrdA+3], 0 offen:1 lds:1 offset:varlds_Asize_per_wr*1
  buffer_load_dword v[vgprG2LA+2],  v[vgprGlobalReadOfvarA+2],  s[sgprSrdA:sgprSrdA+3], 0 offen:1 lds:1 offset:varlds_Asize_per_wr*2
  buffer_load_dword v[vgprG2LA+3],  v[vgprGlobalReadOfvarA+3],  s[sgprSrdA:sgprSrdA+3], 0 offen:1 lds:1 offset:varlds_Asize_per_wr*3

  s_mov_b32     m0, s[sgprLocalWriteAddrB+0]
  s_nop 0
  //Fetch B for Unroll iteration# (u+2) for 32 k indices
  buffer_load_dword v[vgprG2LB+0],  v[vgprGlobalReadOfvarB+0],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:0
  buffer_load_dword v[vgprG2LB+1],  v[vgprGlobalReadOfvarB+1],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*1
  buffer_load_dword v[vgprG2LB+2],  v[vgprGlobalReadOfvarB+2],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*2
  buffer_load_dword v[vgprG2LB+3],  v[vgprGlobalReadOfvarB+3],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*3
  buffer_load_dword v[vgprG2LB+4],  v[vgprGlobalReadOfvarB+4],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*4
  buffer_load_dword v[vgprG2LB+5],  v[vgprGlobalReadOfvarB+5],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*5
  buffer_load_dword v[vgprG2LB+6],  v[vgprGlobalReadOfvarB+6],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*6
  buffer_load_dword v[vgprG2LB+7],  v[vgprGlobalReadOfvarB+7],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*7

  s_waitcnt vmcnt(20)
  s_setprio 1
  s_barrier
  //increment 32 elements to fetch next k=32 elements of tile 64x32
  v_add_u32	v[vgprGlobalReadOfvarA+0], 64, v[vgprGlobalReadOfvarA+0]
  v_add_u32	v[vgprGlobalReadOfvarA+1], 64, v[vgprGlobalReadOfvarA+1]
  v_add_u32	v[vgprGlobalReadOfvarA+2], 64, v[vgprGlobalReadOfvarA+2]
  v_add_u32	v[vgprGlobalReadOfvarA+3], 64, v[vgprGlobalReadOfvarA+3]
  v_add_u32	v[vgprGlobalReadOfvarB+0], 64, v[vgprGlobalReadOfvarB+0]
  v_add_u32	v[vgprGlobalReadOfvarB+1], 64, v[vgprGlobalReadOfvarB+1]
  v_add_u32	v[vgprGlobalReadOfvarB+2], 64, v[vgprGlobalReadOfvarB+2]
  v_add_u32	v[vgprGlobalReadOfvarB+3], 64, v[vgprGlobalReadOfvarB+3]
  s_setprio 0
  s_waitcnt vmcnt(12)
  s_setprio 1
  s_barrier
  v_add_u32	v[vgprGlobalReadOfvarB+4], 64, v[vgprGlobalReadOfvarB+4]
  v_add_u32	v[vgprGlobalReadOfvarB+5], 64, v[vgprGlobalReadOfvarB+5]
  v_add_u32	v[vgprGlobalReadOfvarB+6], 64, v[vgprGlobalReadOfvarB+6]
  v_add_u32	v[vgprGlobalReadOfvarB+7], 64, v[vgprGlobalReadOfvarB+7]
  s_setprio 0

  s_mov_b32     m0, s[sgprLocalWriteAddrA+1]
  s_add_u32     s[sgprLoopCounters+0], s[sgprLoopCounters+0], 0x1		//inc CounterL

  //*************************
  // Unroll Looop 1
  //**************************
  s_barrier

  //Fetch A for Unroll iteration# (u+3) for 32 k indices
  buffer_load_dword v[vgprG2LA+0],  v[vgprGlobalReadOfvarA+0],  s[sgprSrdA:sgprSrdA+3], 0 offen:1 lds:1 offset:0
  buffer_load_dword v[vgprG2LA+1],  v[vgprGlobalReadOfvarA+1],  s[sgprSrdA:sgprSrdA+3], 0 offen:1 lds:1 offset:varlds_Asize_per_wr*1
  buffer_load_dword v[vgprG2LA+2],  v[vgprGlobalReadOfvarA+2],  s[sgprSrdA:sgprSrdA+3], 0 offen:1 lds:1 offset:varlds_Asize_per_wr*2
  buffer_load_dword v[vgprG2LA+3],  v[vgprGlobalReadOfvarA+3],  s[sgprSrdA:sgprSrdA+3], 0 offen:1 lds:1 offset:varlds_Asize_per_wr*3

  s_mov_b32     m0, s[sgprLocalWriteAddrB+1]
  s_nop 0
  //Fetch B for Unroll iteration# (u+2) for 32 k indices
  buffer_load_dword v[vgprG2LB+0],  v[vgprGlobalReadOfvarB+0],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:0
  buffer_load_dword v[vgprG2LB+1],  v[vgprGlobalReadOfvarB+1],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*1
  buffer_load_dword v[vgprG2LB+2],  v[vgprGlobalReadOfvarB+2],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*2
  buffer_load_dword v[vgprG2LB+3],  v[vgprGlobalReadOfvarB+3],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*3
  buffer_load_dword v[vgprG2LB+4],  v[vgprGlobalReadOfvarB+4],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*4
  buffer_load_dword v[vgprG2LB+5],  v[vgprGlobalReadOfvarB+5],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*5
  buffer_load_dword v[vgprG2LB+6],  v[vgprGlobalReadOfvarB+6],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*6
  buffer_load_dword v[vgprG2LB+7],  v[vgprGlobalReadOfvarB+7],  s[sgprSrdB:sgprSrdB+3], 0 offen:1 lds:1 offset:varlds_Bsize_per_wr*7

  s_waitcnt vmcnt(20)
  s_barrier
  s_setprio 1	//raise the wave priority for simd co-execution
  //increment 32 elements to fetch next k=32 elements of tile 64x32
  v_add_u32	v[vgprGlobalReadOfvarA+0], 64, v[vgprGlobalReadOfvarA+0]
  v_add_u32	v[vgprGlobalReadOfvarA+1], 64, v[vgprGlobalReadOfvarA+1]
  v_add_u32	v[vgprGlobalReadOfvarA+2], 64, v[vgprGlobalReadOfvarA+2]
  v_add_u32	v[vgprGlobalReadOfvarA+3], 64, v[vgprGlobalReadOfvarA+3]
  v_add_u32	v[vgprGlobalReadOfvarB+0], 64, v[vgprGlobalReadOfvarB+0]
  v_add_u32	v[vgprGlobalReadOfvarB+1], 64, v[vgprGlobalReadOfvarB+1]
  v_add_u32	v[vgprGlobalReadOfvarB+2], 64, v[vgprGlobalReadOfvarB+2]
  v_add_u32	v[vgprGlobalReadOfvarB+3], 64, v[vgprGlobalReadOfvarB+3]
  s_setprio 0
  s_waitcnt vmcnt(12)
  s_barrier
  s_setprio 1
  v_add_u32	v[vgprGlobalReadOfvarB+4], 64, v[vgprGlobalReadOfvarB+4]
  v_add_u32	v[vgprGlobalReadOfvarB+5], 64, v[vgprGlobalReadOfvarB+5]
  v_add_u32	v[vgprGlobalReadOfvarB+6], 64, v[vgprGlobalReadOfvarB+6]
  v_add_u32	v[vgprGlobalReadOfvarB+7], 64, v[vgprGlobalReadOfvarB+7]
  s_setprio 0

  s_mov_b32     m0, s[sgprLocalWriteAddrA+0]
  s_add_u32     s[sgprLoopCounters+0], s[sgprLoopCounters+0], 0x1		//inc CounterL

  s_cmp_eq_i32  s[sgprLoopCounters+0], -0x2					// CounterL=0x2
  s_cbranch_scc0  label_0005

label_0006:
  s_waitcnt vmcnt(8)
  s_barrier
  s_waitcnt vmcnt(0)
  s_barrier

s_endpgm



wave0_entry_start:

  //  accvgpr init
  //  4*32=128CYCLES
  // hide all mem loatency with ACC VGPR initialization
  for var i =0; i < 32; i++
      v_accvgpr_write v_regs(0,i), 0, 0 
  end      

  //s_load_dword s[sgprAddressD], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x18 // 
  //s_load_dword s[sgprAddressD+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x1c // 
  //s_load_dword s[sgprAddressC], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x20 // 
  //s_load_dword s[sgprAddressC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x24 // 
  //s_load_dword s[sgprTensor2dSizeC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x4 // 
  s_load_dwordx2 s[sgprTensor2dSizeC+0:sgprTensor2dSizeC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0 // 
  s_load_dwordx4 s[sgprAlpha:sgprSizesFree+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x38 // 
  s_load_dwordx4 s[sgprStridesD+0:sgprStridesC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40 // 
  s_load_dwordx4 s[sgprAddressD:sgprAddressC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x18 // 
  //s_load_dwordx4 s[sgprAlpha:sgprSizesFree+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x38 // 
  //s_load_dwordx2 s[sgprTensor2dSizeC+0:sgprTensor2dSizeC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0 // 
  //s_load_dword s[sgprSizesFree+2], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68 // 
  //s_load_dword s[sgprOrigStaggerUIter], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x70 //
  //s_load_dword s[sgprNumWorkGroups0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x74 // 
  //s_load_dword s[sgprNumWorkGroups1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x78 // 
  //s_load_dword s[sgprNumFullBlocks], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x84 // 
  //s_load_dword s[sgprWgmRemainder1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x88 // 
  //s_load_dword s[sgprMagicNumberWgmRemainder1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x8c // 
  //s_load_dword s[sgprStridesD+0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40 // 
  //s_load_dword s[sgprStridesD+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x44 // 
  //s_load_dword s[sgprStridesC+0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x48 // 
  //s_load_dword s[sgprStridesC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x4c // 
  //s_load_dword s[sgprSizesFree+0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60 // 
  //s_load_dword s[sgprSizesFree+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x64 // 

   
/******************************************/
/* Local Read Addresses                   */
/******************************************/
 
  //32 lanes for holding M elements
  v_and_b32        v1, v[vgprSerial+1],  0x1f
  v_mul_lo_u32     v[vgprLocalReadAddrA+0],  16, v1
  v_lshrrev_b32    v1, 2, v1
  v_mul_lo_u32     v1, varlds_pad_qw, v1
  v_add_u32        v[vgprLocalReadAddrA+0],  v1,  v[vgprLocalReadAddrA+0]

/* local read addresses: final offsets a */

  v_lshrrev_b32    v1, 5,  v[vgprSerial+1]
  v_add_u32        v[vgprLocalReadAddrA+0], v1, v[vgprLocalReadAddrA+0]
  v_lshlrev_b32    v[vgprLocalReadAddrA+0], 2, v[vgprLocalReadAddrA+0]  //convert to bytes
  v_add_u32        v[vgprLocalReadAddrA+0], varA_lds_base_addr,  v[vgprLocalReadAddrA+0]
  v_add_u32        v[vgprLocalReadAddrA+1], varlds_Asize_per_wg, v[vgprLocalReadAddrA+0]

   s_barrier

/******************************************/
/* Local Read Addresses Offset B          */
/******************************************/

  //32 lanes for holding N elements
  v_and_b32        v1,  v[vgprSerial+1],     0x1f
  v_mul_lo_u32     v[vgprLocalReadAddrB+0],  16, v1
  v_lshrrev_b32    v1,  2,  v1
  v_mul_lo_u32     v1, varlds_pad_qw, v1
  v_add_u32        v[vgprLocalReadAddrB+0],  v1,  v[vgprLocalReadAddrB+0]

/* local read addresses: final offsets b */

  v_lshrrev_b32    v1,                  5,      v[vgprSerial+1]
  v_add_u32        v[vgprLocalReadAddrB+0],  v1, v[vgprLocalReadAddrB+0]
  v_lshlrev_b32    v[vgprLocalReadAddrB+0],  2, v[vgprLocalReadAddrB+0]  //convert to bytes

  s_mul_i32        s84,  s[sgprFetchSubGrpId],    varlds_Bsize_per_wave
  v_add_u32        v[vgprLocalReadAddrB+0], s84, v[vgprLocalReadAddrB+0]
  v_add_u32        v[vgprLocalReadAddrB+0],  varB_lds_base_addr,   v[vgprLocalReadAddrB+0]
  v_add_u32        v[vgprLocalReadAddrB+1], varlds_Bsize_per_wg, v[vgprLocalReadAddrB+0]

  s_waitcnt lgkmcnt(0)
/*****************************************************************/
/* Global Address C calculation */
/*****************************************************************/

  s_mov_b32	   s[sgprSrdC+0], s[sgprAddressC+0]         // init SRD base address (lower)
  s_mov_b32	   s[sgprSrdC+1], s[sgprAddressC+1]         // init SRD base address (lower)
  s_mov_b32        s[sgprSrdC+2], 0x80000000 		    // limit at 2*31
  s_mov_b32        s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in SRD

  s_mov_b32	   s[sgprSrdD+0], s[sgprAddressD+0]         // init SRD base address (lower)
  s_mov_b32        s[sgprSrdD+1], s[sgprAddressD+1]	    // set const_stride=4 for idxen
  s_mov_b32        s[sgprSrdD+2], 0x80000000 		    // limit at 2*31
  s_mov_b32        s[sgprSrdD+3], Srd127_96		    // Set bits 127_96 in SRD

  s_mul_i32	  s86, 0x80, s[sgprWorkGroup1]		    // <- wg1*MT1
  s_mul_hi_u32    s85, s86,  s[sgprStridesC+0]
  s_mul_i32       s84, s86,  s[sgprStridesC+0]		    // scale by stride
  s_lshl_b64      s[84:85],  s[84:85], 1		    // scale by bpe 
  s_add_u32       s[sgprSrdC+0], s[sgprSrdC+0], s84         
  s_addc_u32      s[sgprSrdC+1], s[sgprSrdC+1], s85         
  s_add_u32       s[sgprSrdD+0], s[sgprSrdD+0], s84         
  s_addc_u32      s[sgprSrdD+1], s[sgprSrdD+1], s85         

  s_mul_hi_u32    s85, s[sgprWorkGroup2], s[sgprStridesC+1] 	//Scale s[sgprWorkGroup2] by stride
  s_mul_i32       s84, s[sgprWorkGroup2], s[sgprStridesC+1] 	//Scale s[sgprWorkGroup2] by stride
  s_lshl_b64      s[84:85],  s[84:85], 1		        // scale by bpe 
  s_add_u32       s[sgprSrdC+0], s[sgprSrdC+0], s84         
  s_addc_u32      s[sgprSrdC+1], s[sgprSrdC+1], s85         
  s_add_u32       s[sgprSrdD+0], s[sgprSrdD+0], s84         
  s_addc_u32      s[sgprSrdD+1], s[sgprSrdD+1], s85         

  //use threadIdz to re-map threadIdx using for calculating waveId start off set for MAC wves to store 'C'
  v_lshlrev_b32   v[vgprSerial], 6,    v[vgprSerial+2]	  	// threadIdx = simdId<<6
  v_add_u32       v[vgprSerial], v[vgprSerial+1], v[vgprSerial] // threadIdx = threadIdx + 0-63
  v_mul_lo_u32    v4, 0x20, v[vgprSerial+2]	        	// scale by sub-tile-size 32 (B-tile/128)
  v_mul_lo_u32    v3, v4,   s[sgprStridesC+0] 			// wavestart vgpr
  v_and_b32	  v4, 0x1f, v[vgprSerial]                       // vectorStaticDiv vgprTmp = vgprSerial % 32
  v_mul_lo_u32    v5, v4,   s[sgprStridesC+0]			// rowstart VGPR
  v_and_b32	  v4, 0x3f, v[vgprSerial]                       // vectorStaticDiv vgprTmp = vgprSerial % 32
  v_lshrrev_b32	  v6, 5,    v4	 		 	        // vectorStaticDiv vgprTmp = vgprSerial / 32 
  v_lshlrev_b32   v6, 2,    v6			   		// glvw=2
  v_add_u32	  v[vgprTmp+1], v3,  v5

  s_mul_i32	  s84, 0x40, s[sgprWorkGroup0]			// s84 = wgp0*MT0
  v_add_co_u32    v[vgprTmp], vcc, s84, v6
  v_add_lshl_u32  v[vgprGlobalWriteOfvarC], v[vgprTmp], v[vgprTmp+1], 0x1	// c base_addr = wave_start+row_start scaled by BPE

  //sync point to load waves
  s_barrier
  //Read A Elements from LDS into VREGS...  
  // one time burst read of A elements due to latency delay bbetween A and  B for first fetch
  // if elements are in L2 hit , you still have some latency between A and B for first fetch
  //  hide the latency by issuing bursts of A (fiqure out how many from profiler / thread trace)

  ds_read_b32       v[vgprValuA_X0_I0+0],  v[vgprLocalReadAddrA+0]  offset:0
  ds_read_b32       v[vgprValuA_X0_I0+1],  v[vgprLocalReadAddrA+0]  offset:varlds_Asize_per_wr * 8 
  ds_read_b32       v[vgprValuA_X0_I0+2],  v[vgprLocalReadAddrA+0]  offset:8
  ds_read_b32       v[vgprValuA_X0_I0+3],  v[vgprLocalReadAddrA+0]  offset:varlds_Asize_per_wr * 8 + 8
  ds_read_b32       v[vgprValuA_X0_I0+4],  v[vgprLocalReadAddrA+0]  offset:16
  ds_read_b32       v[vgprValuA_X0_I0+5],  v[vgprLocalReadAddrA+0]  offset:varlds_Asize_per_wr * 8 + 16
  ds_read_b32       v[vgprValuA_X0_I0+6],  v[vgprLocalReadAddrA+0]  offset:24 
  ds_read_b32       v[vgprValuA_X0_I0+7],  v[vgprLocalReadAddrA+0]  offset:varlds_Asize_per_wr * 8 + 24

  // Read B elements from LDS into VREGS
  // different strategy;; read as minimum as possible.  we need to get into unroll loop as soon as possible
  // Issue read for 8x32  B elements 
  ds_read_b32       v[vgprValuA_X0_I0+8],  v[vgprLocalReadAddrA+0]  offset:32
  ds_read_b32       v[vgprValuA_X0_I0+9],  v[vgprLocalReadAddrA+0]  offset:varlds_Asize_per_wr * 8  + 32
  ds_read_b32       v[vgprValuA_X0_I0+10],  v[vgprLocalReadAddrA+0]  offset:40
  ds_read_b32       v[vgprValuA_X0_I0+11],  v[vgprLocalReadAddrA+0]  offset:varlds_Asize_per_wr * 8 + 40
  ds_read_b32       v[vgprValuA_X0_I0+12],  v[vgprLocalReadAddrA+0]  offset:48
  ds_read_b32       v[vgprValuA_X0_I0+13],  v[vgprLocalReadAddrA+0]  offset:varlds_Asize_per_wr * 8 + 48
  ds_read_b32       v[vgprValuA_X0_I0+14],  v[vgprLocalReadAddrA+0]  offset:56
  ds_read_b32       v[vgprValuA_X0_I0+15],  v[vgprLocalReadAddrA+0]  offset:varlds_Asize_per_wr * 8 + 56
  s_barrier
  ds_read_b32       v[vgprValuB_X0_I0+0],  v[vgprLocalReadAddrB+0]  offset:0
  ds_read_b32       v[vgprValuB_X0_I0+1],  v[vgprLocalReadAddrB+0]  offset:8
  ds_read_b32       v[vgprValuB_X0_I0+2],  v[vgprLocalReadAddrB+0]  offset:16
  ds_read_b32       v[vgprValuB_X0_I0+3],  v[vgprLocalReadAddrB+0]  offset:24

  s_lshr_b32       s[sgprLoopCounters+0], s[sgprSizesSum+0], 5 // s[sgprLoopCounters+0] = s[sgprSizesSum+0] / 32
  s_sub_u32	   s[sgprLoopCounters+0], 0x0, s[sgprLoopCounters+0]
  s_cmp_eq_u32     s[sgprLoopCounters+0], 0x0            // numIter0I == 0
  s_cbranch_scc1   label_0004                           // Dont enter Loop

  var p =0
  var k =1
  s_waitcnt lgkmcnt(3)
 
  v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+0], v_regs(vgprValuB_X0_I0, 0+p*8),  v[vgprAcc+0] 
  ds_read_b32       v_regs(vgprValuB_X0_I0,4+0*8),  v_regs(vgprLocalReadAddrB,0)  offset:32
  ds_read_b32       v_regs(vgprValuB_X0_I0,5+0*8),  v_regs(vgprLocalReadAddrB,0)  offset:40
  ds_read_b32       v_regs(vgprValuB_X0_I0,6+0*8),  v_regs(vgprLocalReadAddrB,0)  offset:48
  ds_read_b32       v_regs(vgprValuB_X0_I0,7+0*8),  v_regs(vgprLocalReadAddrB,0)  offset:56
  v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+1],   v_regs(vgprValuB_X0_I0, 0+p*8), v[vgprAcc+16] 
  s_barrier
  s_waitcnt lgkmcnt(6)
  v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+2], v_regs(vgprValuB_X0_I0, 1+p*8),  v[vgprAcc+0] 
  v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+3],   v_regs(vgprValuB_X0_I0, 1+p*8), v[vgprAcc+16] 
  s_waitcnt lgkmcnt(5)
  v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+4], v_regs(vgprValuB_X0_I0, 2+p*8),  v[vgprAcc+0] 
  v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+5], v_regs(vgprValuB_X0_I0, 2+p*8), v[vgprAcc+16] 
  s_waitcnt lgkmcnt(4)
  v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+6], v_regs(vgprValuB_X0_I0, 3+p*8),  v[vgprAcc+0] 
  v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+7], v_regs(vgprValuB_X0_I0, 3+p*8),  v[vgprAcc+16] 
  s_waitcnt lgkmcnt(0)

  //****************************************
  // End of  32x32 (block 0 of 64x32)
  //****************************************

  //****************************************
  // Begin of  32x32 (block 1 of 64x32)
  //****************************************

  v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+8],  v_regs(vgprValuB_X0_I0, 4+p*8),  v[vgprAcc+0] 
  v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+9],  v_regs(vgprValuB_X0_I0, 4+p*8),  v[vgprAcc+16] 
  s_setprio 0 //lower wave priority other wave to execute valu instructions
  v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+10], v_regs(vgprValuB_X0_I0, 5+p*8),  v[vgprAcc+0] 
  s_barrier
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+0),  v_regs(vgprLocalReadAddrA,k)  offset:0
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+1),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8
  v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+11], v_regs(vgprValuB_X0_I0, 5+p*8),  v[vgprAcc+16] 
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+2),  v_regs(vgprLocalReadAddrA,k)  offset:8
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+3),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 8
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+4),  v_regs(vgprLocalReadAddrA,k)  offset:16
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+5),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 16
  v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+12], v_regs(vgprValuB_X0_I0, 6+p*8),  v[vgprAcc+0] 
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+6),  v_regs(vgprLocalReadAddrA,k)  offset:24
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+7),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 24
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+8),  v_regs(vgprLocalReadAddrA,k)  offset:32
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+9),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 32
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+10),  v_regs(vgprLocalReadAddrA,k)  offset:40
  v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+13], v_regs(vgprValuB_X0_I0, 6+p*8), v[vgprAcc+16] 
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+11),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 40
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+12),  v_regs(vgprLocalReadAddrA,k)  offset:48
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+13),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 48
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+14),  v_regs(vgprLocalReadAddrA,k)  offset:56
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+15),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 56
  v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+14], v_regs(vgprValuB_X0_I0, 7+p*8),  v[vgprAcc+0] 
  s_barrier
  ds_read_b32       v_regs(vgprValuB_X0_I0,0+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:0
  ds_read_b32       v_regs(vgprValuB_X0_I0,1+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:8
  v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+15], v_regs(vgprValuB_X0_I0, 7+p*8),  v[vgprAcc+16] 
  ds_read_b32       v_regs(vgprValuB_X0_I0,2+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:16
  ds_read_b32       v_regs(vgprValuB_X0_I0,3+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:24
  ds_read_b32       v_regs(vgprValuB_X0_I0,4+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:32
  ds_read_b32       v_regs(vgprValuB_X0_I0,5+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:40
  s_setprio 1
  s_add_u32     s[sgprLoopCounters+0], s[sgprLoopCounters+0], 0x1		//inc CounterL

  p =1
  k =0
  s_waitcnt lgkmcnt(5)
 
  v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+0+16*p], v_regs(vgprValuB_X0_I0, 0+p*8),  v[vgprAcc+0] 
  ds_read_b32       v_regs(vgprValuB_X0_I0,6+p*8),  v_regs(vgprLocalReadAddrB,p)  offset:48
  ds_read_b32       v_regs(vgprValuB_X0_I0,7+p*8),  v_regs(vgprLocalReadAddrB,p)  offset:56
  v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+1+16*p],   v_regs(vgprValuB_X0_I0, 0+p*8), v[vgprAcc+16] 
  s_barrier
  s_waitcnt lgkmcnt(6)
  v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+2+16*p], v_regs(vgprValuB_X0_I0, 1+p*8),  v[vgprAcc+0] 
  v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+3+16*p],   v_regs(vgprValuB_X0_I0, 1+p*8), v[vgprAcc+16] 
  s_waitcnt lgkmcnt(5)
  v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+4+16*p], v_regs(vgprValuB_X0_I0, 2+p*8),  v[vgprAcc+0] 
  v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+5+16*p], v_regs(vgprValuB_X0_I0, 2+p*8), v[vgprAcc+16] 
  s_waitcnt lgkmcnt(4)
  v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+6+16*p], v_regs(vgprValuB_X0_I0, 3+p*8),  v[vgprAcc+0] 
  v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+7+16*p], v_regs(vgprValuB_X0_I0, 3+p*8),  v[vgprAcc+16] 
  s_waitcnt lgkmcnt(0)

  //****************************************
  // End of  32x32 (block 0 of 64x32)
  //****************************************

  //****************************************
  // Begin of  32x32 (block 1 of 64x32)
  //****************************************

  v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+8+16*p],  v_regs(vgprValuB_X0_I0, 4+p*8),  v[vgprAcc+0] 
  v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+9+16*p],  v_regs(vgprValuB_X0_I0, 4+p*8),  v[vgprAcc+16] 
  s_setprio 0 //lower wave priority other wave to execute valu instructions
  v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+10+16*p], v_regs(vgprValuB_X0_I0, 5+p*8),  v[vgprAcc+0] 
  s_barrier
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+0),  v_regs(vgprLocalReadAddrA,k)  offset:0
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+1),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8
  v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+11+16*p], v_regs(vgprValuB_X0_I0, 5+p*8),  v[vgprAcc+16] 
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+2),  v_regs(vgprLocalReadAddrA,k)  offset:8
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+3),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 8
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+4),  v_regs(vgprLocalReadAddrA,k)  offset:16
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+5),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 16
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+6),  v_regs(vgprLocalReadAddrA,k)  offset:24
  v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+12+16*p], v_regs(vgprValuB_X0_I0, 6+p*8),  v[vgprAcc+0] 
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+7),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 24
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+8),  v_regs(vgprLocalReadAddrA,k)  offset:32
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+9),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 32
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+10),  v_regs(vgprLocalReadAddrA,k)  offset:40
  v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+13+16*p], v_regs(vgprValuB_X0_I0, 6+p*8), v[vgprAcc+16] 
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+11),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 40
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+12),  v_regs(vgprLocalReadAddrA,k)  offset:48
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+13),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 48
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+14),  v_regs(vgprLocalReadAddrA,k)  offset:56
  ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+15),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 56
  v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+14+16*p], v_regs(vgprValuB_X0_I0, 7+p*8),  v[vgprAcc+0] 
  s_barrier
  ds_read_b32       v_regs(vgprValuB_X0_I0,0+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:0
  ds_read_b32       v_regs(vgprValuB_X0_I0,1+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:8
  v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+15+16*p], v_regs(vgprValuB_X0_I0, 7+p*8),  v[vgprAcc+16] 
  ds_read_b32       v_regs(vgprValuB_X0_I0,3+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:24
  ds_read_b32       v_regs(vgprValuB_X0_I0,4+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:32
  ds_read_b32       v_regs(vgprValuB_X0_I0,5+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:40
  ds_read_b32       v_regs(vgprValuB_X0_I0,2+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:16
  s_setprio 1
  s_add_u32     s[sgprLoopCounters+0], s[sgprLoopCounters+0], 0x1		//inc CounterL




/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_0001:

  //A matrix tile 64 rows
  //B matrix tile 128 columns
  //B matrix tile 128 columns split across 4 SIMD
  //A matrix tile split into 2 tiles 
  // unroll loop  k/32 times * 2 
  // each mfma caculates k=4 unroll iterations
  // 8 mfma for k=32  for each 32x32 ThreadTile.. (2 for 64 A rows)

  //unrolling double 
  for  p =0; p < 2; p++
      k = 1-p

     s_waitcnt lgkmcnt(5)

     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+16*p+0], v_regs(vgprValuB_X0_I0, 0+p*8),  v[vgprAcc+0] 
     ds_read_b32       v_regs(vgprValuB_X0_I0,6+p*8),  v_regs(vgprLocalReadAddrB,p)  offset:48
     ds_read_b32       v_regs(vgprValuB_X0_I0,7+p*8),  v_regs(vgprLocalReadAddrB,p)  offset:56
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+16*p+1],   v_regs(vgprValuB_X0_I0, 0+p*8), v[vgprAcc+16] 
     s_barrier
     s_waitcnt lgkmcnt(6)
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+16*p+2], v_regs(vgprValuB_X0_I0, 1+p*8),  v[vgprAcc+0] 
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+16*p+3],   v_regs(vgprValuB_X0_I0, 1+p*8), v[vgprAcc+16] 
     s_waitcnt lgkmcnt(5)
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+16*p+4], v_regs(vgprValuB_X0_I0, 2+p*8),  v[vgprAcc+0] 
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+16*p+5], v_regs(vgprValuB_X0_I0, 2+p*8), v[vgprAcc+16] 
     s_waitcnt lgkmcnt(4)
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+16*p+6], v_regs(vgprValuB_X0_I0, 3+p*8),  v[vgprAcc+0] 
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+16*p+7], v_regs(vgprValuB_X0_I0, 3+p*8),  v[vgprAcc+16] 
     s_waitcnt lgkmcnt(0)

     //****************************************
     // End of  32x32 (block 0 of 64x32)
     //****************************************

     //****************************************
     // Begin of  32x32 (block 1 of 64x32)
     //****************************************

     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+16*p+8],  v_regs(vgprValuB_X0_I0, 4+p*8),  v[vgprAcc+0] 
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+16*p+9],  v_regs(vgprValuB_X0_I0, 4+p*8),  v[vgprAcc+16] 
     s_setprio 0 //lower wave priority other wave to execute valu instructions
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+16*p+10], v_regs(vgprValuB_X0_I0, 5+p*8),  v[vgprAcc+0] 
     s_barrier
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+0),  v_regs(vgprLocalReadAddrA,k)  offset:0
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+1),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+16*p+11], v_regs(vgprValuB_X0_I0, 5+p*8),  v[vgprAcc+16] 
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+2),  v_regs(vgprLocalReadAddrA,k)  offset:8
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+3),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 8
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+4),  v_regs(vgprLocalReadAddrA,k)  offset:16
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+5),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 16
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+16*p+12], v_regs(vgprValuB_X0_I0, 6+p*8),  v[vgprAcc+0] 
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+6),  v_regs(vgprLocalReadAddrA,k)  offset:24
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+7),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 24
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+8),  v_regs(vgprLocalReadAddrA,k)  offset:32
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+9),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 32
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+10),  v_regs(vgprLocalReadAddrA,k)  offset:40
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+16*p+13], v_regs(vgprValuB_X0_I0, 6+p*8), v[vgprAcc+16] 
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+11),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 40
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+12),  v_regs(vgprLocalReadAddrA,k)  offset:48
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+13),  v_regs(vgprLocalReadAddrA,k)  offset:varlds_Asize_per_wr * 8 + 48
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+14),  v_regs(vgprLocalReadAddrA,k)  offset:56
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16*k+15),  v_regs(vgprLocalReadAddrA,k)  offset:56+varlds_Asize_per_wr * 8
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+16*p+14], v_regs(vgprValuB_X0_I0, 7+p*8),  v[vgprAcc+0] 
     s_barrier
     ds_read_b32       v_regs(vgprValuB_X0_I0,0+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:0
     ds_read_b32       v_regs(vgprValuB_X0_I0,1+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:8
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+16*p+15], v_regs(vgprValuB_X0_I0, 7+p*8),  v[vgprAcc+16] 
     ds_read_b32       v_regs(vgprValuB_X0_I0,2+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:16
     ds_read_b32       v_regs(vgprValuB_X0_I0,3+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:24
     ds_read_b32       v_regs(vgprValuB_X0_I0,4+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:32
     ds_read_b32       v_regs(vgprValuB_X0_I0,5+k*8),  v_regs(vgprLocalReadAddrB,k)  offset:40
     s_setprio 1
     s_add_u32     s[sgprLoopCounters+0], s[sgprLoopCounters+0], 0x1		//inc CounterL
   end
   s_cmp_eq_i32  s[sgprLoopCounters+0], -0x2					// CounterL=0x2
   s_cbranch_scc0  label_0001

label_0002:


/*****************************************************************/
/*  NoLoadLoop - Begin
/*****************************************************************/

  //unrolling double 
     s_waitcnt lgkmcnt(5)
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0] ,  v[vgprValuA_X0_I0+0], v_regs(vgprValuB_X0_I0, 0+0*8),  v[vgprAcc+0] 
     ds_read_b32       v_regs(vgprValuB_X0_I0,6+0*8),  v_regs(vgprLocalReadAddrB,0)  offset:48
     ds_read_b32       v_regs(vgprValuB_X0_I0,7+0*8),  v_regs(vgprLocalReadAddrB,0)  offset:56
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16],  v[vgprValuA_X0_I0+1], v_regs(vgprValuB_X0_I0, 0+0*8), v[vgprAcc+16] 
     s_waitcnt lgkmcnt(6)
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0],   v[vgprValuA_X0_I0+2], v_regs(vgprValuB_X0_I0, 1+0*8),  v[vgprAcc+0] 
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16],  v[vgprValuA_X0_I0+3], v_regs(vgprValuB_X0_I0, 1+0*8), v[vgprAcc+16] 
     s_waitcnt lgkmcnt(5)
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0],   v[vgprValuA_X0_I0+4], v_regs(vgprValuB_X0_I0, 2+0*8),  v[vgprAcc+0] 
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16],  v[vgprValuA_X0_I0+5], v_regs(vgprValuB_X0_I0, 2+0*8), v[vgprAcc+16] 
     s_waitcnt lgkmcnt(4)
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0],   v[vgprValuA_X0_I0+6], v_regs(vgprValuB_X0_I0, 3+0*8),  v[vgprAcc+0] 
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16],  v[vgprValuA_X0_I0+7], v_regs(vgprValuB_X0_I0, 3+0*8),  v[vgprAcc+16] 
     s_waitcnt lgkmcnt(0)

     //****************************************
     // End of  32x32 (block 0 of 64x32)
     //****************************************

     //****************************************
     // Begin of  32x32 (block 1 of 64x32)
     //****************************************

     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+8], v_regs(vgprValuB_X0_I0, 4+0*8),  v[vgprAcc+0] 
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+9], v_regs(vgprValuB_X0_I0, 4+0*8),  v[vgprAcc+16] 
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+10], v_regs(vgprValuB_X0_I0, 5+0*8),  v[vgprAcc+0] 
     s_barrier
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16+0),  v_regs(vgprLocalReadAddrA,1)  offset:0
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16+1),  v_regs(vgprLocalReadAddrA,1)  offset:varlds_Asize_per_wr * 8
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+11], v_regs(vgprValuB_X0_I0, 5+0*8),  v[vgprAcc+16] 
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16+2),  v_regs(vgprLocalReadAddrA,1)  offset:8
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16+3),  v_regs(vgprLocalReadAddrA,1)  offset:varlds_Asize_per_wr * 8 + 8
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16+4),  v_regs(vgprLocalReadAddrA,1)  offset:16
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16+5),  v_regs(vgprLocalReadAddrA,1)  offset:varlds_Asize_per_wr * 8 + 16
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0],   v[vgprValuA_X0_I0+12], v_regs(vgprValuB_X0_I0, 6+0*8),  v[vgprAcc+0] 
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16+6),  v_regs(vgprLocalReadAddrA,1)  offset:24
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16+7),  v_regs(vgprLocalReadAddrA,1)  offset:varlds_Asize_per_wr * 8 + 24
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16+8),  v_regs(vgprLocalReadAddrA,1)  offset:32
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16+9),  v_regs(vgprLocalReadAddrA,1)  offset:varlds_Asize_per_wr * 8 + 32
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16+10), v_regs(vgprLocalReadAddrA,1)  offset:40
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16],  v[vgprValuA_X0_I0+13],   v_regs(vgprValuB_X0_I0, 6+0*8), v[vgprAcc+16] 
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16+11), v_regs(vgprLocalReadAddrA,1)  offset:varlds_Asize_per_wr * 8 + 40
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16+12), v_regs(vgprLocalReadAddrA,1)  offset:48
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16+13), v_regs(vgprLocalReadAddrA,1)  offset:varlds_Asize_per_wr * 8 + 48
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16+14), v_regs(vgprLocalReadAddrA,1)  offset:56
     ds_read_b32       v_regs(vgprValuA_X0_I0, 16+15), v_regs(vgprLocalReadAddrA,1)  offset:56 + varlds_Asize_per_wr * 8
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0],   v[vgprValuA_X0_I0+14], v_regs(vgprValuB_X0_I0, 7+0*8),  v[vgprAcc+0] 
     s_barrier
     ds_read_b32       v_regs(vgprValuB_X0_I0,0+1*8),  v_regs(vgprLocalReadAddrB,1)  offset:0
     ds_read_b32       v_regs(vgprValuB_X0_I0,1+1*8),  v_regs(vgprLocalReadAddrB,1)  offset:8
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16],  v[vgprValuA_X0_I0+15], v_regs(vgprValuB_X0_I0, 7+0*8),  v[vgprAcc+16] 
     ds_read_b32       v_regs(vgprValuB_X0_I0,2+1*8),  v_regs(vgprLocalReadAddrB,1)  offset:16
     ds_read_b32       v_regs(vgprValuB_X0_I0,3+1*8),  v_regs(vgprLocalReadAddrB,1)  offset:24
     ds_read_b32       v_regs(vgprValuB_X0_I0,4+1*8),  v_regs(vgprLocalReadAddrB,1)  offset:32
     ds_read_b32       v_regs(vgprValuB_X0_I0,5+1*8),  v_regs(vgprLocalReadAddrB,1)  offset:40

     //second unroll Loop 
     s_waitcnt lgkmcnt(5)

     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+16+0], v_regs(vgprValuB_X0_I0, 0+1*8),  v[vgprAcc+0] 
     ds_read_b32       v_regs(vgprValuB_X0_I0,6+1*8),  v_regs(vgprLocalReadAddrB,1)  offset:48
     ds_read_b32       v_regs(vgprValuB_X0_I0,7+1*8),  v_regs(vgprLocalReadAddrB,1)  offset:56
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+16+1],   v_regs(vgprValuB_X0_I0, 0+1*8), v[vgprAcc+16] 
     s_waitcnt lgkmcnt(6)
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+16+2], v_regs(vgprValuB_X0_I0, 1+1*8),  v[vgprAcc+0] 
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+16+3],   v_regs(vgprValuB_X0_I0, 1+1*8), v[vgprAcc+16] 
     s_waitcnt lgkmcnt(5)
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+16+4], v_regs(vgprValuB_X0_I0, 2+1*8),  v[vgprAcc+0] 
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]  ,   v[vgprValuA_X0_I0+16+5],   v_regs(vgprValuB_X0_I0, 2+1*8), v[vgprAcc+16] 
     s_waitcnt lgkmcnt(4)
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+16+6], v_regs(vgprValuB_X0_I0, 3+1*8),  v[vgprAcc+0] 
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]   ,   v[vgprValuA_X0_I0+16+7], v_regs(vgprValuB_X0_I0, 3+1*8),  v[vgprAcc+16] 

     //****************************************
     // End of  32x32 (block 0 of 64x32)
     //****************************************
     //****************************************
     // Begin of  32x32 (block 1 of 64x32)
     //****************************************
     s_waitcnt lgkmcnt(0)
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+16+8], v_regs(vgprValuB_X0_I0, 4+1*8),  v[vgprAcc+0] 
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+16+10], v_regs(vgprValuB_X0_I0, 5+1*8),  v[vgprAcc+0] 
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+16+12], v_regs(vgprValuB_X0_I0, 6+1*8),  v[vgprAcc+0] 
     v_mfma_f32_32x32x4bf16   v[vgprAcc+0]   ,   v[vgprValuA_X0_I0+16+14], v_regs(vgprValuB_X0_I0, 7+1*8),  v[vgprAcc+0] 

      s_mov_b32		s84, roundMaskVal 
      v_xor_b32		v[vgprTmp+1], v[vgprTmp],v[vgprTmp]
      v_or_b32		v[vgprTmp+1], s84, v[vgprTmp+1]

     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]   ,   v[vgprValuA_X0_I0+16+9], v_regs(vgprValuB_X0_I0, 4+1*8),  v[vgprAcc+16] 
      for var j = 0; j < 8; j++
          v_accvgpr_read v_regs(vgprValuC, j),v_regs(j,0*16), 0
      end
      v_lshrrev_b32	v[vgprValuC+0], 16, v[vgprValuC+0]
      v_and_or_b32  	v[vgprValuC+0], v[vgprValuC+1], v[vgprTmp+1], v[vgprValuC+0]
      for var j = 2; j < 6; j+=2
          v_lshrrev_b32	   v[vgprValuC+j], 16, v[vgprValuC+j]
          v_and_or_b32     v[vgprValuC+(j/2)], v[vgprValuC+j+1],v[vgprTmp+1], v[vgprValuC+j]
      end
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]   ,   v[vgprValuA_X0_I0+16+11], v_regs(vgprValuB_X0_I0, 5+1*8),  v[vgprAcc+16] 
      buffer_store_dwordx2 v[vgprValuC+0:vgprValuC+1],v[vgprGlobalWriteOfvarC],  s[sgprSrdD:sgprSrdD+3], 0  offset:64*0+0 offen:1 // store C
      for var j = 6; j < 8; j+=2
          v_lshrrev_b32	   v[vgprValuC+j], 16, v[vgprValuC+j]
          v_and_or_b32     v[vgprValuC+(j/2)], v[vgprValuC+j+1],v[vgprTmp+1], v[vgprValuC+j]
      end
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]   ,   v[vgprValuA_X0_I0+16+13], v_regs(vgprValuB_X0_I0, 6+1*8),  v[vgprAcc+16] 
      buffer_store_dwordx2 v[vgprValuC+2:vgprValuC+3],v[vgprGlobalWriteOfvarC],  s[sgprSrdD:sgprSrdD+3], 0 offset:64*0+16 offen:1 // store C
      for var j = 8; j < 12; j++
          v_accvgpr_read v_regs(vgprValuC, j),v_regs(j,0*16), 0
      end
     v_mfma_f32_32x32x4bf16   v[vgprAcc+16]   ,   v[vgprValuA_X0_I0+16+15], v_regs(vgprValuB_X0_I0, 7+1*8),  v[vgprAcc+16] 
      for var j = 8; j < 12; j+=2
          v_lshrrev_b32	   v[vgprValuC+j], 16, v[vgprValuC+j]
          v_and_or_b32     v[vgprValuC+(j/2)], v[vgprValuC+j+1],v[vgprTmp+1], v[vgprValuC+j]
      end
      buffer_store_dwordx2 v[vgprValuC+4:vgprValuC+5],v[vgprGlobalWriteOfvarC],  s[sgprSrdD:sgprSrdD+3], 0 offset:64*0+32 offen:1 // store C
      for var j = 12; j < 16; j++
          v_accvgpr_read v_regs(vgprValuC, j),v_regs(j,0*16), 0
      end
      for var j = 12; j < 16; j+=2
          v_lshrrev_b32	   v[vgprValuC+j], 16, v[vgprValuC+j]
          v_and_or_b32     v[vgprValuC+(j/2)], v[vgprValuC+j+1],v[vgprTmp+1], v[vgprValuC+j]
      end
      buffer_store_dwordx2 v[vgprValuC+6:vgprValuC+7],v[vgprGlobalWriteOfvarC],  s[sgprSrdD:sgprSrdD+3], 0 offset:64*0+48 offen:1 // store C


store_c:
  
 for var i = 1; i < 2; i++
      for var j = 0; j < 8; j++
          v_accvgpr_read v_regs(vgprValuC, j),v_regs(j,i*16), 0
      end

      v_lshrrev_b32	v[vgprValuC+0], 16, v[vgprValuC+0]
      v_and_or_b32  	v[vgprValuC+0], v[vgprValuC+1], v[vgprTmp+1], v[vgprValuC+0]

      for var j = 2; j < 6; j+=2
          v_lshrrev_b32	   v[vgprValuC+j], 16, v[vgprValuC+j]
          v_and_or_b32     v[vgprValuC+(j/2)], v[vgprValuC+j+1],v[vgprTmp+1], v[vgprValuC+j]
      end

      buffer_store_dwordx2 v[vgprValuC+0:vgprValuC+1],v[vgprGlobalWriteOfvarC],  s[sgprSrdD:sgprSrdD+3], 0  offset:64*i+0 offen:1 // store C

      for var j = 6; j < 8; j+=2
          v_lshrrev_b32	   v[vgprValuC+j], 16, v[vgprValuC+j]
          v_and_or_b32     v[vgprValuC+(j/2)], v[vgprValuC+j+1],v[vgprTmp+1], v[vgprValuC+j]
      end

      for var j = 8; j < 12; j++
          v_accvgpr_read v_regs(vgprValuC, j),v_regs(j,i*16), 0
      end

      buffer_store_dwordx2 v[vgprValuC+2:vgprValuC+3],v[vgprGlobalWriteOfvarC],  s[sgprSrdD:sgprSrdD+3], 0 offset:64*i+16 offen:1 // store C

      for var j = 8; j < 12; j+=2
          v_lshrrev_b32	   v[vgprValuC+j], 16, v[vgprValuC+j]
          v_and_or_b32     v[vgprValuC+(j/2)], v[vgprValuC+j+1],v[vgprTmp+1], v[vgprValuC+j]
      end

      for var j = 12; j < 16; j++
          v_accvgpr_read v_regs(vgprValuC, j),v_regs(j,i*16), 0
      end

      buffer_store_dwordx2 v[vgprValuC+4:vgprValuC+5],v[vgprGlobalWriteOfvarC],  s[sgprSrdD:sgprSrdD+3], 0 offset:64*i+32 offen:1 // store C
      for var j = 12; j < 16; j+=2
          v_lshrrev_b32	   v[vgprValuC+j], 16, v[vgprValuC+j]
          v_and_or_b32     v[vgprValuC+(j/2)], v[vgprValuC+j+1],v[vgprTmp+1], v[vgprValuC+j]
      end
      buffer_store_dwordx2 v[vgprValuC+6:vgprValuC+7],v[vgprGlobalWriteOfvarC],  s[sgprSrdD:sgprSrdD+3], 0 offset:64*i+48 offen:1 // store C

 end

label_0004:
  s_waitcnt        0  
  s_endpgm                                          



end
