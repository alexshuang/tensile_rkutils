from core.layers.layer import Layer, get_compute_flops
from core.structs.auxiliary import LayerResults
from core.auxiliary import get_dt_size, prod
from core.const import *
import copy
import numpy as np


def get_variant(inputs, direction='fwd'):  # figure out which BNorm variant we fall under
    h, w, n = inputs.in_dims[H_IND], inputs.in_dims[W_IND], inputs.in_dims[N_IND]
    if direction == 'fwd':
        variant = 0 if h * w <= 512 else 3 if h * w <= 1024 else 1 if h * w * n < 33554432 else 2
    else:
        variant = 0 if h * w <= 512 else 3 if h * w <= 1024 and n < 32 else 1 if h * w * n < 33554432 else 2
    return variant


def get_num_loop_iter(inputs, variant, is_2nd_red_loop=False, is_update_loop=False, direction='fwd'):
    MIO_BN_GRP0 = DEFAULT_WG_SIZE
    h, w, n = inputs.in_dims[H_IND], inputs.in_dims[W_IND], inputs.in_dims[N_IND]
    MIO_BN_HW = h * w
    MIO_BN_NHW = MIO_BN_HW * n
    if variant == 0:  # below is from MIOpenBatchNormFwdTrainSpatial.cl code
        MIO_BN_SEGTMP = (MIO_BN_HW * (MIO_BN_GRP0 / MIO_BN_HW))
        MIO_BN_SEGMENT = MIO_BN_NHW if MIO_BN_SEGTMP > MIO_BN_NHW else MIO_BN_SEGTMP
        MIO_BN_NLOOP = np.floor((MIO_BN_NHW + MIO_BN_SEGMENT - 1) / MIO_BN_SEGMENT)
        return MIO_BN_NLOOP
    elif variant == 1:
        MIO_MAX_READ = 3 if MIO_BN_HW >= 4096 and direction == 'fwd' else 2
        GRPRD = (MIO_BN_GRP0 * 4)
        MIO_BN_REM4 = (MIO_BN_NHW - ((MIO_BN_NHW / GRPRD) * GRPRD))
        MIO_BN_LESS4 = (MIO_BN_NHW - MIO_BN_REM4)
        MIO_BN_REM = (MIO_BN_NHW - ((MIO_BN_NHW / MIO_BN_GRP0) * MIO_BN_GRP0))
        MIO_BN_LESS = (MIO_BN_NHW - MIO_BN_REM)
        MIO_BN_CHUNK = (MIO_MAX_READ * MIO_BN_GRP0)
        MIO_BN_REMOUT = (MIO_BN_NHW - ((MIO_BN_NHW / MIO_BN_CHUNK) * MIO_BN_CHUNK))
        MIO_BN_LESSOUT = (MIO_BN_NHW - MIO_BN_REMOUT)
        if is_update_loop:
            nested_iters = MIO_BN_LESSOUT/MIO_BN_CHUNK * (2 * MIO_MAX_READ) + (2 * MIO_MAX_READ if MIO_BN_REMOUT else 0)  # + one iteration at the end to handle remainder
            iters = np.floor(MIO_BN_LESS/MIO_BN_GRP0 if MIO_BN_REM == 0 else nested_iters)
            return iters
        else:
            iters = np.floor(MIO_BN_LESS4/GRPRD if MIO_BN_HW >= 4096 or is_2nd_red_loop else MIO_BN_LESS/MIO_BN_GRP0)
            return iters + 1 if MIO_BN_REM4 or MIO_BN_REM else iters  # + one iteration at the end to handle remainder
    else:
        return n


def get_unroll_factors_fwd(variant, loop_iters, best_case=False):
    if best_case:
        return min(loop_iters[0], BEST_CASE_UNROLL_FACTOR), min(loop_iters[1], BEST_CASE_UNROLL_FACTOR)
    else:
        if variant == 0:
            return 1, 1
        elif variant == 1:
            return 1, 1
        else:
            return BNORM_UNROLL_FACTOR, 1


def get_unroll_factors_bwd(variant, loop_iters, best_case=False):
    if best_case:
        return min(loop_iters[0], BEST_CASE_UNROLL_FACTOR), \
               min(loop_iters[1], BEST_CASE_UNROLL_FACTOR), \
               min(loop_iters[2], BEST_CASE_UNROLL_FACTOR)
    else:
        if variant == 0:
            return BNORM_UNROLL_FACTOR, 1, 1
        elif variant == 1:
            return 1, 1, 1
        else:
            return 2 * BNORM_UNROLL_FACTOR, BNORM_UNROLL_FACTOR, BNORM_UNROLL_FACTOR


class BatchNorm(Layer):
    def __init__(self, sys_config, rnn_bn=False):
        super(BatchNorm, self).__init__(sys_config)
        self.rnn_bn = rnn_bn
        self.hw_cfg = self.sys_cfg.hw_cfg
        self.sw_opt = self.sys_cfg.sw_opt

    def get_dpp_lds_reduction_cc(self, variant, inputs, max_unroll_factor=2, direction='fwd'):  # lds reduce loop unrolled 2 times
        if direction == 'bwd':
            return DPP_LDS_CC_BWD
        elif variant in [0, 1]:
            return DPP_LDS_CC_VAR0_VAR1
        else:
            num_waves = inputs.in_dims[H_IND] * inputs.in_dims[W_IND] / WI_PER_WAVE
            dpp_cc = DPP_CC_PER_WAVE * np.ceil(num_waves / SIMD_PER_CU)  # dpp reduction cycles
            lds_cc = LDS_CC_PER_WAVE * num_waves/max_unroll_factor + np.floor(num_waves / SIMD_PER_CU) * LDS_CC_PER_WAVE  # lds reduction cycles
            return dpp_cc + lds_cc

    def get_latency_cc(self, loop_iter, num_alu_ops=0, max_unroll_factor=2, l2_hit_rate=0.0, lat_hiding_ratio=1.0):
        # Compute residual cycles from latency
        average_latency = l2_hit_rate * self.hw_cfg.l2_hit_latency + (1 - l2_hit_rate) * self.hw_cfg.l2_miss_latency
        main_alu_cc_per_iter = num_alu_ops * CC_PER_ALU_INSTR
        addr_calc_alu_cc_per_unroll_iter = ADDR_CALC_OPS_NUM * CC_PER_ALU_INSTR * max_unroll_factor
        load_issue_cc_per_unroll_iter = max_unroll_factor * CC_PER_ALU_INSTR
        cc_per_unroll_iter = addr_calc_alu_cc_per_unroll_iter + load_issue_cc_per_unroll_iter
        if cc_per_unroll_iter < average_latency:
            latency_exposed = (average_latency - cc_per_unroll_iter * lat_hiding_ratio) * (loop_iter / max_unroll_factor)
        else:
            latency_exposed = 0

        return latency_exposed + (cc_per_unroll_iter * (1.0 - lat_hiding_ratio) + main_alu_cc_per_iter) * loop_iter

    def fprop(self, inputs, inference=False):
        best_case = not self.sw_opt.useCalibrated
        variant = get_variant(inputs)
        reduce_loop_iter = get_num_loop_iter(inputs, variant)
        update_loop_iter = get_num_loop_iter(inputs, variant, is_update_loop=True)
        hbm_rd_bw = 0
        hbm_wr_bw = 0
        reduce_unroll, update_unroll = get_unroll_factors_fwd(variant, (reduce_loop_iter, update_loop_iter), best_case)

        data_size = get_dt_size(self.sw_opt)
        if inference:
            total_ops = APPLY_BNORM_OPS_NUM
        else:
            total_ops = REDUCE_MEAN_VAR_OPS_NUM + APPLY_BNORM_OPS_NUM
        _inputs = copy.deepcopy(inputs)
        if not self.rnn_bn:
            _inputs.in_dims[C_IND] = 1  # Normalization is performed over NHW channels
        alu_cc = self.get_alu_cc(_inputs.in_dims, total_ops)

        # simulates input reads in reduction loop and writes in update loop
        l2_hit_rate_wr = min(self.hw_cfg.l2_size / (prod(inputs.out_dims) * data_size), 1.0)
        bw_rd_mem_cc, bw_wr_mem_cc, _ = self.get_mem_cc(inputs.in_dims, inputs.in_dims, inputs.l2_hit_rate_act, l2_hit_rate_wr, inputs.l3_hit_rate_act)
        hbm_rd_bw += (1 - inputs.l2_hit_rate_act) * (1 - inputs.l3_hit_rate_act) * prod(inputs.in_dims) * data_size
        hbm_wr_bw += (1 - l2_hit_rate_wr) * prod(inputs.out_dims) * data_size

        addr_calc_cc = ADDR_CALC_OPS_NUM * CC_PER_ALU_INSTR * get_num_loop_iter(inputs, variant)
        sqrt_cc = MAD_SQRT_ETC_CC
        startup_cc = self.hw_cfg.inst_fetch_bubble
        degrade_factor = DEGRADE_FACTOR

        if variant == 0:  # always 16 Waves per WG
            lat_rd_mem_cc_reduce = self.get_latency_cc(reduce_loop_iter, num_alu_ops=REDUCE_MEAN_VAR_OPS_NUM, max_unroll_factor=reduce_unroll, l2_hit_rate=inputs.l2_hit_rate_act)
            lat_rd_mem_cc_update = 0
            # simulates spills to global memory and minibatch not fitting in registers resulting in additional reads and writes
            bw_spill_rd_mem_cc, bw_spill_wr_mem_cc, _ = self.get_mem_cc(inputs.in_dims, inputs.in_dims, inputs.l2_hit_rate_act, l2_hit_rate_wr, l3_hit_rate_act=0.0)  # simulates reads and writes of spills
            hbm_rd_bw += (1 - inputs.l2_hit_rate_act) * (1 - inputs.l3_hit_rate_act) * prod(inputs.in_dims) * data_size
            hbm_wr_bw += (1 - l2_hit_rate_wr) * prod(inputs.out_dims) * data_size
            lat_spill_rd_mem_cc = self.get_latency_cc(reduce_loop_iter, num_alu_ops=APPLY_BNORM_OPS_NUM, max_unroll_factor=1, l2_hit_rate=inputs.l2_hit_rate_act)
            if inputs.in_dims[N_IND] * inputs.in_dims[H_IND] * inputs.in_dims[W_IND] < SPILL_THRES_FWD:  # this means there are no spills
                bw_spill_rd_mem_cc, bw_spill_wr_mem_cc = 0, 0  # no spills
                lat_spill_rd_mem_cc = 0
                addr_calc_cc *= 1 if best_case else SIMD_PER_CU * PIPELINE_PUSHBACK_FACTOR  # pipeline pushbacks when no spills effectively reducing bandwidth by a factor of SIMD_PER_CU
                bw_wr_mem_cc *= 1 if best_case else SIMD_PER_CU * PIPELINE_PUSHBACK_FACTOR  # pipeline pushbacks when no spills effectively reducing bandwidth by a factor of SIMD_PER_CU
            sqrt_cc = MAD_SQRT_ETC_CC_VAR0_VAR1
            startup_cc = INST_FETCH_BUBBLE_VAR0_VAR1_BWD
        elif variant == 1:  # always 16 Waves per WG
            lat_rd_mem_cc_reduce = self.get_latency_cc(reduce_loop_iter, num_alu_ops=REDUCE_MEAN_VAR_OPS_NUM, max_unroll_factor=reduce_unroll, l2_hit_rate=inputs.l2_hit_rate_act, lat_hiding_ratio=0.0)
            lat_rd_mem_cc_update = self.get_latency_cc(update_loop_iter, num_alu_ops=APPLY_BNORM_OPS_NUM, max_unroll_factor=update_unroll, l2_hit_rate=inputs.l2_hit_rate_act, lat_hiding_ratio=0.0)
            bw_spill_rd_mem_cc, bw_spill_wr_mem_cc = 0, 0  # no spills
            lat_spill_rd_mem_cc = 0
            sqrt_cc = MAD_SQRT_ETC_CC_VAR0_VAR1
            startup_cc = INST_FETCH_BUBBLE_VAR0_VAR1_BWD
            degrade_factor = 0.0  # calibrated pretty accurately as it is
        else:  # variant 3: varying number of Waves per WG
            lat_rd_mem_cc_reduce = self.get_latency_cc(reduce_loop_iter, num_alu_ops=REDUCE_MEAN_VAR_OPS_NUM, max_unroll_factor=reduce_unroll, l2_hit_rate=inputs.l2_hit_rate_act)
            lat_rd_mem_cc_update = 0
            # simulates spills to global memory and minibatch not fitting in registers resulting in additional reads and writes
            bw_spill_rd_mem_cc, bw_spill_wr_mem_cc, _ = self.get_mem_cc(inputs.in_dims, inputs.in_dims, inputs.l2_hit_rate_act, l2_hit_rate_wr, l3_hit_rate_act=0.0)  # simulates reads and writes of spills
            hbm_rd_bw += (1 - inputs.l2_hit_rate_act) * (1 - inputs.l3_hit_rate_act) * prod(inputs.in_dims) * data_size
            hbm_wr_bw += (1 - l2_hit_rate_wr) * prod(inputs.out_dims) * data_size
            lat_spill_rd_mem_cc = self.get_latency_cc(reduce_loop_iter, num_alu_ops=APPLY_BNORM_OPS_NUM, max_unroll_factor=1, l2_hit_rate=inputs.l2_hit_rate_act)

        if best_case:
            lat_spill_rd_mem_cc, bw_spill_rd_mem_cc, bw_spill_wr_mem_cc, degrade_factor = 0, 0, 0, 0

        reduce_mem_cc = max(bw_rd_mem_cc + addr_calc_cc, lat_rd_mem_cc_reduce) + bw_spill_wr_mem_cc
        update_mem_cc = max(bw_spill_rd_mem_cc + addr_calc_cc, lat_spill_rd_mem_cc, lat_rd_mem_cc_update) + bw_wr_mem_cc + RAMPDOWN_CC
        mem_cc = reduce_mem_cc + update_mem_cc
        if inference and self.sw_opt.kernel_fusion:
            cycles = alu_cc
        else:
            dpp_lds_red_cc = self.get_dpp_lds_reduction_cc(variant, inputs)
            alu_cc += dpp_lds_red_cc
            cycles_clean = startup_cc + reduce_mem_cc + dpp_lds_red_cc + sqrt_cc + update_mem_cc
            cycles = cycles_clean + cycles_clean * degrade_factor
        alu_util_factor = (alu_cc / cycles) * 100
        speedup = 1
        flop = get_compute_flops(_inputs.in_dims, total_ops)
        res = LayerResults(alu_util_factor, alu_util_factor, speedup, cycles, flop, self.sys_cfg.hw_cfg.num_cu,
                           hbm_rd_bw=hbm_rd_bw, hbm_wr_bw=hbm_wr_bw, alu_cc=alu_cc,
                           mem_cc=mem_cc)  # Ashish added
        if CALIBRATING:
            print("===FORWARD PASS===FORWARD PASS===FORWARD PASS=========================")
            print("reduce loop cc: {}".format(reduce_mem_cc))
            print("update loop cc: {}".format(update_mem_cc))
            print("dpp_lds_red_cc: {}".format(dpp_lds_red_cc))
            print("fetch_bubble: {}".format(startup_cc))
            print("total: {} cc".format(cycles_clean))
            print("total degrade: {} cc".format(cycles))
            print("total: {} us".format(cycles/CALIBRATION_CLOCK))
            print("Reduce loop iters: {}".format(reduce_loop_iter))
            print("Update loop iters: {}".format(update_loop_iter))
            print("==================================")
        return res

    def bprop(self, inputs):
        #--- Reduce 1: Calculate/Reduce MEAN and VARIANCE  --- Global/Shared mem: (1 read for x_in)
        #--- Reduce 2: Calculate/Reduce D_SCALE and D_BIAS --- Global/Shared mem: (1 read for x_in, 1 read for dy_in)
        #--- Update:   Calculate and save D_X              --- Global/Shared mem: (1 read for x_in, 1 read for dy_in, 1 write for dx_out)
        best_case = not self.sw_opt.useCalibrated
        variant = get_variant(inputs, direction='bwd')
        reduce1_iter = get_num_loop_iter(inputs, variant, direction='bwd')
        reduce2_iter = get_num_loop_iter(inputs, variant, is_2nd_red_loop=True, direction='bwd')  # different from loop 1 only for variant 1
        update_iter = get_num_loop_iter(inputs, variant, is_update_loop=True, direction='bwd')
        reduce1_unroll, reduce2_unroll, update_unroll = get_unroll_factors_bwd(variant, (reduce1_iter, reduce2_iter, update_iter), best_case)
        degrade_factor = DEGRADE_FACTOR
        data_size = 2 if self.sw_opt.fp16_inputs or self.sw_opt.bf16_inputs else (4 if self.sw_opt.fp32_inputs else 8)
        total_ops = REDUCE_MEAN_VAR_OPS_NUM + APPLY_BNORM_OPS_NUM
        _inputs = copy.deepcopy(inputs)
        hbm_rd_bw = 0
        hbm_wr_bw = 0

        if not self.rnn_bn:
            _inputs.in_dims[C_IND] = 1  # Normalization is performed over NHW channels
        alu_cc = self.get_alu_cc(_inputs.in_dims, total_ops)

        l2_hit_rate_wr = min(self.hw_cfg.l2_size / (prod(inputs.out_dims) * data_size), 1.0)
        spills = inputs.in_dims[N_IND] * inputs.in_dims[H_IND] * inputs.in_dims[W_IND] >= SPILL_THRES_BWD

        # REDUCE 1:
        reduce1_addr_calc_cc = ADDR_CALC_OPS_NUM * CC_PER_ALU_INSTR * reduce1_iter
        reduce1_bw_rd_mem_cc = self.get_mem_cc(inputs.in_dims, inputs.in_dims, inputs.l2_hit_rate_act, l2_hit_rate_wr, inputs.l3_hit_rate_act)[0]
        hbm_rd_bw += (1 - inputs.l2_hit_rate_act) * (1 - inputs.l3_hit_rate_act) * prod(inputs.in_dims) * data_size
        hbm_wr_bw += (1 - l2_hit_rate_wr) * prod(inputs.out_dims) * data_size
        reduce1_lat_rd_mem_cc = self.get_latency_cc(reduce1_iter, num_alu_ops=REDUCE_MEAN_VAR_OPS_NUM, max_unroll_factor=reduce1_unroll, l2_hit_rate=inputs.l2_hit_rate_act)
        reduce1_bw_spill_wr_mem_cc = 0 if not spills else self.get_mem_cc(inputs.in_dims, inputs.in_dims, inputs.l2_hit_rate_act, l2_hit_rate_wr, l3_hit_rate_act=0.0)[1]

        # REDUCE 2:
        # ATTN MIGHT NEED TO ACCOUNT FOR lat_hiding_ratio=0.0
        num_global_access = 2  # number of reads from global memory (1 for x_in, 1 for dy_in)
        in_dims_adj = [inputs.in_dims[0] * num_global_access, inputs.in_dims[1], inputs.in_dims[2], inputs.in_dims[3]]  # adjusted to reflect the number of global accesses
        reduce2_addr_calc_cc = num_global_access * ADDR_CALC_OPS_NUM * CC_PER_ALU_INSTR * reduce2_iter
        reduce2_bw_rd_mem_cc = self.get_mem_cc(inputs.in_dims, inputs.in_dims, inputs.l2_hit_rate_act, l2_hit_rate_wr, inputs.l3_hit_rate_act)[0]
        reduce2_lat_rd_mem_cc = self.get_latency_cc(reduce2_iter, num_alu_ops=REDUCE_DS_DB_OPS_NUM, max_unroll_factor=reduce2_unroll, l2_hit_rate=inputs.l2_hit_rate_act)
        reduce2_bw_spill_rd_mem_cc = 0 if not spills else self.get_mem_cc(inputs.in_dims, inputs.in_dims, inputs.l2_hit_rate_act, l2_hit_rate_wr, inputs.l3_hit_rate_act)[0]
        reduce2_bw_spill_wr_mem_cc = 0 if not spills else self.get_mem_cc(in_dims_adj, in_dims_adj, inputs.l2_hit_rate_act, l2_hit_rate_wr, l3_hit_rate_act=0.0)[1]
        reduce2_addr_calc_cc = reduce2_addr_calc_cc / num_global_access if not spills else reduce2_addr_calc_cc

        # UPDATE:
        update_addr_calc_cc = reduce2_addr_calc_cc
        update_bw_spill_rd_mem_cc = 0 if not spills else self.get_mem_cc(in_dims_adj, in_dims_adj, inputs.l2_hit_rate_act, l2_hit_rate_wr, inputs.l3_hit_rate_act)[0]
        update_lat_spill_rd_mem_cc = 0 if not spills else self.get_latency_cc(update_iter, num_alu_ops=CALC_DX_OUT_OPS_NUM, max_unroll_factor=update_unroll, l2_hit_rate=inputs.l2_hit_rate_act)
        update_bw_wr_mem_cc = self.get_mem_cc(inputs.in_dims, inputs.in_dims, inputs.l2_hit_rate_act, l2_hit_rate_wr, inputs.l3_hit_rate_act)[1]
        hbm_rd_bw += (1 - inputs.l2_hit_rate_act) * (1 - inputs.l3_hit_rate_act) * prod(inputs.in_dims) * data_size
        hbm_wr_bw += (1 - l2_hit_rate_wr) * prod(inputs.out_dims) * data_size

        if variant == 0:
            rampdown_cc = RAMPDOWN_CC
        elif variant == 1:  # spills == True for Var 1, however there are no actual spills, thus reset extra spill writes, spill reads stay the same as they indicate global reads anyway
            reduce1_bw_spill_wr_mem_cc = 0
            reduce2_bw_spill_wr_mem_cc = 0
            rampdown_cc = RAMPDOWN_CC * update_iter / update_unroll
        else:
            rampdown_cc = RAMPDOWN_CC * update_iter / update_unroll

        if best_case:
            reduce1_bw_spill_wr_mem_cc, reduce2_bw_spill_rd_mem_cc, reduce2_bw_spill_wr_mem_cc, update_bw_spill_rd_mem_cc, update_lat_spill_rd_mem_cc, degrade_factor = 0, 0, 0, 0, 0, 0

        startup_cc = INST_FETCH_BUBBLE_VAR0_VAR1_BWD
        reduce1_mem_cc = max(reduce1_bw_rd_mem_cc + reduce1_addr_calc_cc, reduce1_lat_rd_mem_cc) + reduce1_bw_spill_wr_mem_cc  #ATTN MIGHT NEED TO ADD ADDR CALC TO SPILL WRITES as well
        reduce2_mem_cc = max(reduce2_bw_rd_mem_cc + reduce2_bw_spill_rd_mem_cc + reduce2_addr_calc_cc, reduce2_lat_rd_mem_cc) + reduce2_bw_spill_wr_mem_cc  # ATTN MIGHT NEED TO ADD ADDR CALC TO SPILL WRITES as well
        update_mem_cc = max(update_bw_spill_rd_mem_cc + update_addr_calc_cc, update_lat_spill_rd_mem_cc) + update_bw_wr_mem_cc + rampdown_cc  # ATTN MIGHT NEED TO ADD ADDR CALC TO SPILL WRITES as well
        dpp_lds_red_cc = self.get_dpp_lds_reduction_cc(variant, inputs, direction='bwd')
        alu_cc += dpp_lds_red_cc

        cycles_clean = startup_cc + reduce1_mem_cc + reduce2_mem_cc + dpp_lds_red_cc + update_mem_cc
        cycles = cycles_clean + cycles_clean * degrade_factor
        alu_util_factor = (alu_cc / cycles) * 100
        speedup = 1
        flop = get_compute_flops(_inputs.in_dims, total_ops)
        res = LayerResults(alu_util_factor, alu_util_factor, speedup, cycles, flop, self.sys_cfg.hw_cfg.num_cu,
                           hbm_rd_bw=hbm_rd_bw, hbm_wr_bw=hbm_wr_bw)
        if CALIBRATING:
            print("===BACKWARD PASS===BACKWARD PASS===BACKWARD PASS=========================")
            print("reduce loop MEAN/VAR cc: {}".format(reduce1_mem_cc))
            print("reduce loop DS/DB cc: {}".format(reduce2_mem_cc))
            print("update loop cc: {}".format(update_mem_cc))
            print("dpp_lds_red_cc: {}".format(dpp_lds_red_cc))
            print("fetch_bubble: {}".format(startup_cc))
            print("total: {} cc".format(cycles_clean))
            print("total degrade: {} cc".format(cycles))
            print("total: {} us".format(cycles / CALIBRATION_CLOCK))
            print("Reduce loop iters: {}".format(reduce1_iter))
            print("Update loop iters: {}".format(update_iter))
            print("==================================")
        return res
