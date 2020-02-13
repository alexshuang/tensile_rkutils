from core.layers.layer import Layer
from core.structs.auxiliary import LayerResults
from core.auxiliary import get_dt_size
from core.const import *
import copy


class Gemm(Layer):
    def __init__(self, sys_config, dims, is_c_mat_used=False, act='A', wgrad_dims=None, dgrad_dims=None,
                 batched_gemm=False, num_cu_per_batch=1):
        super(Gemm, self).__init__(sys_config)
        self.m = dims[0]
        self.n = dims[1]
        self.k = dims[2]
        self.is_c_mat_used = is_c_mat_used
        self.act = act
        self.wgrad_dims = wgrad_dims
        self.dgrad_dims = dgrad_dims
        self.batched_gemm = batched_gemm
        self.num_cu_per_batch = num_cu_per_batch

    def fprop(self, inputs, inference=False):
        gemm_res = self.perform_gemm(self.m, self.n, self.k, inputs.l2_hit_rate_act, inputs.l2_hit_rate_wt,
                                     inputs.l3_hit_rate_act, act=self.act, is_c_mat_used=self.is_c_mat_used,
                                     batched_gemm=self.batched_gemm, num_cu_per_batch=self.num_cu_per_batch,
                                     tpu_partition_scheme=inputs.tpu_partition_scheme)

        flop = 2 * self.m * self.n * self.k

        hbm_rd_bw, hbm_wr_bw = self.get_gemm_hbm_bw(self.m, self.n, self.k, inputs.l2_hit_rate_wt, inputs.l2_hit_rate_act, inputs.l3_hit_rate_act, self.act)
        layer_res = LayerResults(gemm_res.alu_util_factor, gemm_res.chip_util_factor, gemm_res.speedup, gemm_res.cycles, flop, gemm_res.num_cu_util,
                                 self.m, self.n, self.k, gemm_res.num_a_blocks, gemm_res.num_b_blocks, gemm_res.num_partitions,
                                 hbm_rd_bw, hbm_wr_bw, tpu_sub_res=gemm_res.tpu_sub_res,
                                 alu_cc=gemm_res.alu_cc, mem_cc=gemm_res.mem_cc,num_rounds=gemm_res.num_rounds, #Ashish added
                                 total_blocks=gemm_res.total_blocks, num_cu_util_trail=gemm_res.num_cu_util_trail, #Ashish added
                                 num_a_blocks_trail=gemm_res.num_a_blocks_trail, num_b_blocks_trail=gemm_res.num_b_blocks_trail, #Ashish added
                                 num_partitions_trail=gemm_res.num_partitions_trail, cycles_trail=gemm_res.cycles_trail, wr_cc=gemm_res.wr_cc,
                                 main_instr=gemm_res.main_instr, threadTile=gemm_res.threadTile, workGroup=gemm_res.workGroup,
                                 unroll_factor=gemm_res.unroll_factor, unroll_factor_trail=gemm_res.unroll_factor_trail,
                                 num_rounds_trail=gemm_res.num_rounds_trail, alu_cc_trail=gemm_res.alu_cc_trail,
                                 mem_cc_trail=gemm_res.mem_cc_trail, wr_cc_trail=gemm_res.wr_cc_trail)  # Ashish added
        return layer_res

    def bprop(self, inputs):
        in_size = get_dt_size(self.sys_cfg.sw_opt)

        # 1. Compute weight gradients: dW[l] = A[l-1] . dZ[l]
        weights_size = (self.wgrad_dims[GEMM_N_IND] * self.wgrad_dims[GEMM_K_IND] * in_size)  # dZ[l] becomes weight matrix for WGRAD
        inputs.l2_hit_rate_act = 1 if self.sys_cfg.hw_cfg.tpu_en else 0  # ML Chiplet assumes external DMA makes all activations resident in L2 before feeding into PEs
        if weights_size <= self.sys_cfg.hw_cfg.l2_size:
            inputs.l2_hit_rate_wt = HIGH_L2_HIT_RATE
        else:
            inputs.l2_hit_rate_wt = self.sys_cfg.hw_cfg.l2_size * 0.8 / weights_size

        m = self.wgrad_dims[GEMM_M_IND]
        n = self.wgrad_dims[GEMM_N_IND]
        k = self.wgrad_dims[GEMM_K_IND]

        gemm_res = self.perform_gemm(m, n, k, inputs.l2_hit_rate_act, inputs.l2_hit_rate_wt, inputs.l3_hit_rate_act, act='A',
                                     is_c_mat_used=False, batched_gemm=self.batched_gemm, num_cu_per_batch=self.num_cu_per_batch)

        flop_wgrad = 2 * m * n * k
        hbm_rd_bw_wgrad, hbm_wr_bw_wgrad = self.get_gemm_hbm_bw(m, n, k, inputs.l2_hit_rate_wt, inputs.l2_hit_rate_act, inputs.l3_hit_rate_act, act='A')
        wgrad_res = LayerResults(gemm_res.alu_util_factor, gemm_res.chip_util_factor, gemm_res.speedup, gemm_res.cycles, flop_wgrad,
                                 gemm_res.num_cu_util, m, n, k, gemm_res.num_a_blocks, gemm_res.num_b_blocks,
                                 gemm_res.num_partitions, hbm_rd_bw_wgrad, hbm_wr_bw_wgrad, 'weight_gradient', tpu_sub_res=gemm_res.tpu_sub_res,
                                 alu_cc=gemm_res.alu_cc, mem_cc=gemm_res.mem_cc, num_rounds=gemm_res.num_rounds,  # Ashish added
                                 total_blocks=gemm_res.total_blocks, num_cu_util_trail=gemm_res.num_cu_util_trail,  # Ashish added
                                 num_a_blocks_trail=gemm_res.num_a_blocks_trail, num_b_blocks_trail=gemm_res.num_b_blocks_trail,  # Ashish added
                                 num_partitions_trail=gemm_res.num_partitions_trail, cycles_trail=gemm_res.cycles_trail, wr_cc=gemm_res.wr_cc,
                                 threadTile=gemm_res.threadTile, workGroup=gemm_res.workGroup, unroll_factor=gemm_res.unroll_factor,
                                 unroll_factor_trail=gemm_res.unroll_factor_trail, num_rounds_trail=gemm_res.num_rounds_trail,
                                 alu_cc_trail=gemm_res.alu_cc_trail, mem_cc_trail=gemm_res.mem_cc_trail, wr_cc_trail=gemm_res.wr_cc_trail)  # Ashish added

        # 2. Compute data gradients: dA[l-1] = dZ[l] . W
        if not inputs.last_node:
            inputs.l2_hit_rate_act = inputs.l2_hit_rate_wt  # dZ[l] is now activation for DGRAD
            inputs.l2_hit_rate_wt = 0
            m = self.dgrad_dims[GEMM_M_IND]
            n = self.dgrad_dims[GEMM_N_IND]
            k = self.dgrad_dims[GEMM_K_IND]

            gemm_res = self.perform_gemm(m, n, k, inputs.l2_hit_rate_act, inputs.l2_hit_rate_wt, inputs.l3_hit_rate_act, act='A',
                                         is_c_mat_used=False, batched_gemm=self.batched_gemm, num_cu_per_batch=self.num_cu_per_batch,
                                         stashed_weights=True if self.sys_cfg.hw_cfg.tpu_en else False)

            flop_dgrad = 2 * m * n * k
            hbm_rd_bw_dgrad, hbm_wr_bw_dgrad = self.get_gemm_hbm_bw(m, n, k, inputs.l2_hit_rate_wt, inputs.l2_hit_rate_act,
                                                                    inputs.l3_hit_rate_act, act='A')
            dgrad_res = LayerResults(gemm_res.alu_util_factor, gemm_res.chip_util_factor, gemm_res.speedup, gemm_res.cycles, flop_dgrad,
                                     gemm_res.num_cu_util, m, n, k, gemm_res.num_a_blocks, gemm_res.num_b_blocks,
                                     gemm_res.num_partitions, hbm_rd_bw_dgrad, hbm_wr_bw_dgrad, 'data_gradient', tpu_sub_res=gemm_res.tpu_sub_res,
                                     alu_cc=gemm_res.alu_cc, mem_cc=gemm_res.mem_cc, num_rounds=gemm_res.num_rounds,  # Ashish added
                                     total_blocks=gemm_res.total_blocks, num_cu_util_trail=gemm_res.num_cu_util_trail,  # Ashish added
                                     num_a_blocks_trail=gemm_res.num_a_blocks_trail, num_b_blocks_trail=gemm_res.num_b_blocks_trail,  # Ashish added
                                     num_partitions_trail=gemm_res.num_partitions_trail, cycles_trail=gemm_res.cycles_trail, wr_cc=gemm_res.wr_cc,
                                     threadTile=gemm_res.threadTile, workGroup=gemm_res.workGroup, unroll_factor=gemm_res.unroll_factor,
                                     unroll_factor_trail=gemm_res.unroll_factor_trail, num_rounds_trail=gemm_res.num_rounds_trail,
                                     alu_cc_trail=gemm_res.alu_cc_trail, mem_cc_trail=gemm_res.mem_cc_trail, wr_cc_trail=gemm_res.wr_cc_trail)  # Ashish added

            cycles = wgrad_res.cycles + dgrad_res.cycles
            flop = flop_wgrad + flop_dgrad
            alu_util_factor = (wgrad_res.alu_util_factor * wgrad_res.cycles +
                               dgrad_res.alu_util_factor * dgrad_res.cycles) / cycles
            chip_util_factor = (wgrad_res.chip_util_factor * wgrad_res.cycles +
                               dgrad_res.chip_util_factor * dgrad_res.cycles) / cycles
            hbm_rd_bw = hbm_rd_bw_wgrad + hbm_rd_bw_dgrad
            hbm_wr_bw = hbm_wr_bw_wgrad + hbm_wr_bw_dgrad
            layer_results = LayerResults(alu_util_factor, chip_util_factor, gemm_res.speedup, cycles, flop, gemm_res.num_cu_util,
                                         hbm_rd_bw=hbm_rd_bw, hbm_wr_bw=hbm_wr_bw)
            layer_results.populate_sub_results(wgrad_res)
            layer_results.populate_sub_results(dgrad_res)
        else:
            layer_results = wgrad_res

        return layer_results
