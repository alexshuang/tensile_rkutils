import math
import copy
from core.gemm import GEMM
from core.const import *
from core.auxiliary import get_dl_macs_per_cu, get_native_macs_per_cu, get_dt_size
import heapq


def get_compute_flops(inputs, ops):
    total_flops = inputs[0] * inputs[1] * inputs[2] * inputs[3] * ops
    return total_flops


def convert_format(inputs, filt, format='nhwc'):  # Input format assumed to be NCHW
    in_dims = [0] * 4
    out_dims = [0] * 4
    strides = [0] * 4
    pads = [0] * 4
    if format == 'nhwc':
        in_dims[0] = inputs.in_dims[N_IND]
        in_dims[1] = inputs.in_dims[H_IND]
        in_dims[2] = inputs.in_dims[W_IND]
        in_dims[3] = inputs.in_dims[C_IND]
        out_dims[0] = inputs.out_dims[N_IND]
        out_dims[1] = inputs.out_dims[H_IND]
        out_dims[2] = inputs.out_dims[W_IND]
        out_dims[3] = inputs.out_dims[C_IND]
        strides[0] = 0
        strides[1] = filt.strides[0]
        strides[2] = filt.strides[1]
        strides[3] = 0
        pads[0] = 0
        pads[1] = filt.pads[0]
        pads[2] = filt.pads[1]
        pads[3] = 0
    elif format == 'chwn':
        in_dims[0] = inputs.in_dims[C_IND]
        in_dims[1] = inputs.in_dims[H_IND]
        in_dims[2] = inputs.in_dims[W_IND]
        in_dims[3] = inputs.in_dims[N_IND]
        out_dims[0] = inputs.out_dims[C_IND]
        out_dims[1] = inputs.out_dims[H_IND]
        out_dims[2] = inputs.out_dims[W_IND]
        out_dims[3] = inputs.out_dims[N_IND]
        strides[0] = 0
        strides[1] = filt.strides[0]
        strides[2] = filt.strides[1]
        strides[3] = 0
        pads[0] = 0
        pads[1] = filt.pads[0]
        pads[2] = filt.pads[1]
        pads[3] = 0
    else:
        assert 0

    return in_dims, out_dims, strides, pads


class Layer:
    def __init__(self, sys_config):
        self.sys_cfg = sys_config

    def layer_type(self):
        return self.__class__.__name__

    def fprop(self, inputs, inference=False):  # Forward propagation
        # Base class with no implementation
        raise NotImplementedError

    def bprop(self, inputs): # Backward propagation
        # Base class with no implementation
        raise NotImplementedError

    def get_alu_cc(self, inputs, ops, mai_accelerated=0, num_cu=0):
        flops = get_compute_flops(inputs, ops)
        hw_flops = self.get_hw_flops(mai_accelerated, num_cu)
        alu_cc = flops / hw_flops
        return math.ceil(alu_cc)

    def get_mem_cc(self, inputs, outputs, l2_hit_rate_act_rd=0.0, l2_hit_rate_act_wr=0.0, l3_hit_rate_act=0.0,
                   weights=0, reduced_bitwidth=0, num_cu=0, hbm_bw=0):
        hw_cfg = self.sys_cfg.hw_cfg
        sw_opt = self.sys_cfg.sw_opt
        if reduced_bitwidth:
            data_size = reduced_bitwidth / 8  # in bytes
        else:
            data_size = get_dt_size(sw_opt)
        num_cu_util = num_cu if num_cu > 0 else hw_cfg.num_cu_util
        hbm_bw_util = hbm_bw if hbm_bw > 0 else hw_cfg.hbm_bw
        total_act_rd_per_cu = inputs[N_IND] * inputs[C_IND] * inputs[H_IND] * inputs[W_IND] * data_size / num_cu_util
        total_wt_rd_per_cu = 0
        if weights:
            wt_size = data_size
            for i in range(len(weights.dims)):
                wt_size *= weights.dims[i]
            total_wt_rd_per_cu = wt_size
        total_wr_per_cu = outputs[N_IND] * outputs[C_IND] * outputs[H_IND] * outputs[W_IND] * data_size / num_cu_util

        num_cu_per_se = hw_cfg.num_cu // hw_cfg.num_se_per_cluster
        l2_bw_per_se = hw_cfg.l2_read_buses_per_se * hw_cfg.l2_read_bus_width
        l2_rd_bw_per_cu = min(l2_bw_per_se / num_cu_per_se, hw_cfg.l1_bw) if hw_cfg.l1_bw else l2_bw_per_se / num_cu_per_se
        l3_rd_bw_per_cu = min(hw_cfg.l3_bw / num_cu_util, l2_rd_bw_per_cu)
        hbm_rd_bw_per_cu = min(hbm_bw_util / num_cu_util, l2_rd_bw_per_cu)
        l2_wr_bw_per_cu = min(hw_cfg.l2_write_buses_per_se * hw_cfg.l2_write_bus_width / num_cu_per_se, hw_cfg.l1_bw) if hw_cfg.l1_bw else \
                            hw_cfg.l2_write_buses_per_se * hw_cfg.l2_write_bus_width / num_cu_per_se
        l3_wr_bw_per_cu = min(hw_cfg.l3_bw / num_cu_util, l2_wr_bw_per_cu)
        hbm_wr_bw_per_cu = min(hbm_bw_util / num_cu_util, l2_wr_bw_per_cu)
        effective_rd_bw = l2_hit_rate_act_rd * l2_rd_bw_per_cu + \
                          (1 - l2_hit_rate_act_rd) * l3_hit_rate_act * l3_rd_bw_per_cu + \
                          (1 - l2_hit_rate_act_rd) * (1 - l3_hit_rate_act) * hbm_rd_bw_per_cu
        effective_wr_bw = l2_hit_rate_act_wr * l2_wr_bw_per_cu + \
                          (1 - l2_hit_rate_act_wr) * l3_hit_rate_act * l3_wr_bw_per_cu + \
                          (1 - l2_hit_rate_act_wr) * (1 - l3_hit_rate_act) * hbm_wr_bw_per_cu

        if sw_opt.kernel_fusion and self.layer_type() in {'BN', 'Activation'}:
            act_rd_mem_cc, act_wr_mem_cc, wgt_rd_mem_cc = 0, 0, 0
        else:
            act_rd_mem_cc = math.ceil(total_act_rd_per_cu / effective_rd_bw)
            act_wr_mem_cc = math.ceil(total_wr_per_cu / effective_wr_bw)
            wgt_rd_mem_cc = math.ceil(total_wt_rd_per_cu / hbm_rd_bw_per_cu)

        return act_rd_mem_cc, act_wr_mem_cc, wgt_rd_mem_cc

    def get_hw_flops(self, mai_accelerated=0, num_cu=0):
        hw_cfg = self.sys_cfg.hw_cfg
        sw_opt = self.sys_cfg.sw_opt
        if mai_accelerated:
            hw_flops = 2 * get_dl_macs_per_cu(hw_cfg, sw_opt)
        else:
            hw_flops = 2 * get_native_macs_per_cu(hw_cfg, sw_opt)
        hw_flops *= (num_cu if num_cu > 0 else hw_cfg.num_cu_util)
        return hw_flops

    def get_gemm_hbm_bw(self, m, n, k, l2_hit_rate_wt, l2_hit_rate_act, l3_hit_rate_act, act='A'):
        in_size = get_dt_size(self.sys_cfg.sw_opt)
        hbm_rd_bw = (1 - l2_hit_rate_wt) * (n * k * in_size if act == 'A' else m * k * in_size)
        hbm_rd_bw += (1 - l2_hit_rate_act) * (1 - l3_hit_rate_act) * (m * k * in_size if act == 'A' else n * k * in_size)
        act_size = m * n * in_size
        hbm_wr_bw = 0 if act_size <= self.sys_cfg.hw_cfg.l2_size else (act_size - self.sys_cfg.hw_cfg.l2_size)
        return hbm_rd_bw, hbm_wr_bw

    def gcd(self,x,y):
        hcf = 1
        for i in range(1, (x if x < y else y) + 1):
            if ((x % i == 0) and (y % i == 0)):
                hcf = i
        return hcf

    def get_workgroup_defs(self, hw_cfg, sw_opt):
        validWorkGroups = []
        validThreadTileSides = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        validDepthU = [8, 16, 32, 64, 128]
        validMacroTiles = []
        validThreadTiles = []
        for numThreads in range(64, 1025, 64):
            for nsg in [1]: # assuming LSU as 1 for now. Others will be added later. , 2, 4, 8, 16, 32, 64, 96, 128, 256]:
                for sg0 in range(1, numThreads // nsg + 1):
                    sg1 = numThreads // nsg // sg0
                    if sg0 * sg1 * nsg == numThreads: # and (sg0 > 3 and sg1 > 3):
                        for i in validThreadTileSides:
                            for j in validThreadTileSides:
                                for k in validDepthU:
                                    #input fits in LDS and o/p fits in VGPRs and gfx906 compliance (GRVW > 1 => VW > 1 => TT0*TT1 GCD > 1)
                                    if (sg0*i + sg1*j)*k < hw_cfg.lds_size and \
                                            (sg0*sg1*i*j*get_dt_size(sw_opt)) < hw_cfg.arch_vgpr_size_per_cu - 65536 \
                                            and self.gcd(i, j) > 1:
                                        validMacroTiles.append([sg0*i, sg1*j, k])
                                        threadTile = [i, j]
                                        validThreadTiles.append(threadTile)
                                        workGroup = [sg0, sg1, nsg]
                                        validWorkGroups.append(workGroup)
        return validMacroTiles, len(validMacroTiles), validThreadTiles, validWorkGroups

    def reset_instr_block(self, hw_cfg):
        hw_cfg.dl_instr_large_block[INSTR_M_IND] = 1
        hw_cfg.dl_instr_large_block[INSTR_N_IND] = 1
        hw_cfg.dl_instr_large_block[INSTR_K_IND] = 1
        hw_cfg.dl_instr_large_block[INSTR_NUM_BLOCKS_IND] = 1

    def perform_gemm(self, m, n, k, l2_hit_rate_act=0, l2_hit_rate_wt=0, l3_hit_rate_act=0, act='A', is_c_mat_used=False,
                     batched_gemm=False, num_cu_per_batch=0, implicit_gemm=None, prnn_opt_en=False, winograd=None,
                     tpu_partition_scheme='', stashed_weights=False):
        hw_cfg = self.sys_cfg.hw_cfg
        sw_opt = self.sys_cfg.sw_opt
        gemm_res_list = []
        gemm_cycles_list = []
        Ashish_flag = 0
        # Run GEMM for config with different num_cu and pick the best config
        for i in range(int(hw_cfg.num_cu)):
            hw_cfg_mod = copy.deepcopy(hw_cfg)
            hw_cfg_mod.num_cu_util -= i
            hw_cfg_mod.num_cu -= i
            if hw_cfg_mod.num_cu_util == 0 or hw_cfg_mod.num_cu == 0:
                break
            gemm = GEMM(hw_cfg_mod, self.sys_cfg.hw_opt, sw_opt,
                        l2_hit_rate_act=l2_hit_rate_act, l2_hit_rate_wt=l2_hit_rate_wt,
                        l3_hit_rate_act=l3_hit_rate_act, act=act, batched_gemm=batched_gemm, winograd=winograd,
                        num_cu_per_batch=num_cu_per_batch, implicit_gemm=implicit_gemm, prnn_opt_en=prnn_opt_en,
                        tpu_partition_scheme=tpu_partition_scheme, stashed_weights=stashed_weights)

            if hw_cfg_mod.dl_instr_large_block[INSTR_M_IND] == 1 and hw_cfg_mod.dl_instr_large_block[INSTR_N_IND] == 1 and \
                    hw_cfg_mod.dl_instr_large_block[INSTR_K_IND] == 1 and hw_cfg_mod.dl_instr_large_block[INSTR_NUM_BLOCKS_IND] == 1 and \
                    Ashish_flag == 1:
                validmacroTiles, num_workgroups, validThreadTiles, validWorkGroups = self.get_workgroup_defs(hw_cfg_mod, self.sys_cfg.sw_opt)
                gemm_cycles_tensile_list = []
                gemm_tensile_list = []
                for i in range(num_workgroups):
                    hw_cfg_mod.dl_instr_large_block[INSTR_M_IND] = validmacroTiles[i][0]
                    hw_cfg_mod.dl_instr_large_block[INSTR_N_IND] = validmacroTiles[i][1]
                    hw_cfg_mod.dl_instr_large_block[INSTR_K_IND] = validmacroTiles[i][2]
                    hw_cfg_mod.dl_instr_small_block[INSTR_M_IND] = validmacroTiles[i][0]
                    hw_cfg_mod.dl_instr_small_block[INSTR_N_IND] = validmacroTiles[i][1]
                    hw_cfg_mod.dl_instr_small_block[INSTR_K_IND] = validmacroTiles[i][2]
                    hw_cfg_mod.dl_other_instr_block[INSTR_M_IND] = validmacroTiles[i][0]
                    hw_cfg_mod.dl_other_instr_block[INSTR_N_IND] = validmacroTiles[i][1]
                    hw_cfg_mod.dl_other_instr_block[INSTR_K_IND] = validmacroTiles[i][2]
                    res = gemm.perform_gemm(m, n, k, is_c_mat_used, hw_cfg.num_cu_util)
                    res.threadTile = ("%d_%d" % (validThreadTiles[i][0], validThreadTiles[i][1]))
                    res.workGroup = ("%d_%d" % (validWorkGroups[i][0], validWorkGroups[i][1]))
                    gemm_tensile_list.append(res)
                    gemm_cycles_tensile_list.append(res.cycles)
                min_vals = heapq.nsmallest(len(gemm_cycles_tensile_list), gemm_cycles_tensile_list)
                min_idx = gemm_cycles_tensile_list.index(min_vals[0])
                gemm_cycles_list.append(gemm_cycles_tensile_list[min_idx])
                gemm_res_list.append(gemm_tensile_list[min_idx])
                self.reset_instr_block(hw_cfg_mod)
                #break
            else:
                res = gemm.perform_gemm(m, n, k, is_c_mat_used, hw_cfg.num_cu_util)
                gemm_res_list.append(res)
                gemm_cycles_list.append(res.cycles)
            if res.chip_util_factor >= 95 or batched_gemm:
                break

        min_vals = heapq.nsmallest(len(gemm_cycles_list), gemm_cycles_list)
        min_idx = gemm_cycles_list.index(min_vals[0])
        for i in range(len(min_vals) - 1):
            perf_diff = ((min_vals[i + 1] - min_vals[0]) / min_vals[0]) * 100
            curr_idx = gemm_cycles_list.index(min_vals[i + 1])
            if perf_diff < self.sys_cfg.sw_opt.PerfThreshold and gemm_res_list[curr_idx].num_cu_util > gemm_res_list[min_idx].num_cu_util:
                min_idx = curr_idx
        assert gemm_cycles_list[min_idx] < INVALID_LARGE_VAL
        return gemm_res_list[min_idx]