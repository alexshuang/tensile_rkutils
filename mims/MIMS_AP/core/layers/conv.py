from core.layers.layer import Layer
from core.structs.auxiliary import LayerResults, FilterParam
from core.gemm import ImplicitGemm, Winograd
from core.const import *
from core.auxiliary import get_dt_size
import copy
import math


def im2col(n, c, h, w, k, r, s, pads, strides):
    p = math.floor((abs(h - r + 2 * pads[0]) / strides[0]) + 1)
    q = math.floor((abs(w - s + 2 * pads[1]) / strides[1]) + 1)

    M = n * p * q
    N = k
    K = c * r * s
    return M, N, K


def convert_format(inputs, filt, fmt='nhwc'):  # Input format assumed to be NCHW
        in_dims = [0] * 4
        out_dims = [0] * 4
        if fmt == 'nhwc':
            in_dims = [inputs.in_dims[i] for i in [N_IND, H_IND, W_IND, C_IND]]
            out_dims = [inputs.out_dims[i] for i in [N_IND, H_IND, W_IND, C_IND]]
        elif fmt == 'chwn':
            in_dims = [inputs.in_dims[i] for i in [C_IND, H_IND, W_IND, N_IND]]
            out_dims = [inputs.out_dims[i] for i in [C_IND, H_IND, W_IND, N_IND]]
        elif fmt == 'cnhw':
            in_dims = [inputs.in_dims[i] for i in [C_IND, N_IND, H_IND, W_IND]]
            out_dims = [inputs.out_dims[i] for i in [C_IND, N_IND, H_IND, W_IND]]
        strides = [0, filt.strides[0], filt.strides[1], 0]
        pads = [0, filt.pads[0], filt.pads[1], 0]

        return in_dims, out_dims, strides, pads


class Conv(Layer):
    def __init__(self, sys_config, filter_params):
        super(Conv, self).__init__(sys_config)
        self.filt = filter_params

    def get_compute_flops(self, inputs, ops):
        n = self.filt.dims[F_C_IND] * self.filt.dims[F_R_IND] * self.filt.dims[F_S_IND]
        flops_per_instance = 2 * n - 1
        num_instances_per_filter = (abs(inputs.in_dims[H_IND] - self.filt.dims[F_R_IND] + 2 * self.filt.pads[0]) //
                                    self.filt.strides[0]) + 1
        num_instances_per_filter *= (abs(inputs.in_dims[W_IND] - self.filt.dims[F_S_IND] + 2 * self.filt.pads[1]) //
                                     self.filt.strides[1]) + 1
        flops_per_filter = num_instances_per_filter * flops_per_instance
        total_flops = flops_per_filter * self.filt.dims[F_K_IND] * inputs.in_dims[N_IND]
        return total_flops

    def get_alu_cc(self, inputs, ops, mai_accelerated=0, num_cu=0):
        flops = self.get_compute_flops(inputs, ops)
        hw_flops = self.get_hw_flops(mai_accelerated)
        alu_cc = math.ceil(flops / hw_flops)
        return alu_cc

    def perform_implicit_gemm(self, m, n, k, inputs, hw_cfg, filt, input_format='nhwc', stashed_weights=False):
        # Perform nchw -> nhwc before applying implicit gemm
        in_dims, out_dims, strides, pads = convert_format(inputs, filt, fmt=input_format)
        implicit_gemm = ImplicitGemm(enable=True, in_dims=in_dims, out_dims=out_dims,
                                     filt_dims=filt.dims, strides=strides, pads=pads, format=input_format)
        if filt.group > 1:  # For batched implicit GEMM case launch a single gemm per SE/slice
            batched_gemm = True
            num_cu_per_batch = self.sys_cfg.hw_cfg.num_cu // self.sys_cfg.hw_cfg.num_se_per_cluster
        else:
            batched_gemm = False
            num_cu_per_batch = 1
        gemm_res = self.perform_gemm(m, n, k, inputs.l2_hit_rate_act, inputs.l2_hit_rate_wt, inputs.l3_hit_rate_act,
                                     act='A', batched_gemm=batched_gemm, num_cu_per_batch=num_cu_per_batch,
                                     implicit_gemm=implicit_gemm, tpu_partition_scheme=inputs.tpu_partition_scheme,
                                     stashed_weights=stashed_weights)
        if filt.group > 1:
            num_rounds = filt.group // self.sys_cfg.hw_cfg.num_se_per_cluster
            gemm_res.cycles *= num_rounds
            #gemm_res.tpu_sub_res.vgpr_util_bytes_wt *= num_rounds
            gemm_res.num_cu_util *= self.sys_cfg.hw_cfg.num_se_per_cluster
        return gemm_res

    def perform_conv_as_gemm(self, hw_cfg, inputs):
        use_winograd = False  # Winograd disabled; Implicit GEMM for all conv cases
        if self.filt.dims[F_R_IND] == 3 and self.filt.dims[F_S_IND] == 3 and use_winograd:
            assert self.filt.group == 1
            winograd = Winograd(self.sys_cfg.hw_cfg, self.sys_cfg.sw_opt,
                                enable=True, m=[2, 2], r=[3, 3], split_tile=False)
            m = math.ceil((inputs.out_dims[H_IND] / 2) * (inputs.out_dims[W_IND] / 2) * inputs.out_dims[N_IND])
            n = self.filt.dims[F_K_IND]
            k = inputs.in_dims[C_IND]

            use_winograd = winograd.use_winograd(m, n)
            if use_winograd:  # 3x3 Convolution implemented by Winograd F(2,3) algorithm (https://arxiv.org/pdf/1509.09308.pdf)
                gemm_res = self.perform_gemm(m, n, k, inputs.l2_hit_rate_act, inputs.l2_hit_rate_wt, inputs.l3_hit_rate_act,
                                             act='A', winograd=winograd)
                tile_size = 16
                flop = 2*m*n*k*tile_size/2.25
        if not use_winograd:
            group = self.filt.group
            filt_dims = self.filt.dims
            filt_dims[F_K_IND] //= group
            mod_inputs = copy.deepcopy(inputs)
            mod_inputs.in_dims[C_IND] //= group
            mod_inputs.out_dims[C_IND] //= group
            m, n, k = im2col(mod_inputs.in_dims[N_IND], mod_inputs.in_dims[C_IND], mod_inputs.in_dims[H_IND], mod_inputs.in_dims[W_IND],
                             filt_dims[F_K_IND], filt_dims[F_R_IND], filt_dims[F_S_IND], self.filt.pads, self.filt.strides)
            if self.sys_cfg.hw_cfg.tpu_en:
                input_format = 'cnhw'
            else:
                input_format = 'nhwc'
            gemm_res = self.perform_implicit_gemm(m, n, k, mod_inputs, hw_cfg, self.filt, input_format,
                                                  stashed_weights=True if self.sys_cfg.hw_cfg.tpu_en else False)
            flop = 2 * m * n * k * group
            hbm_rd_bw, hbm_wr_bw = self.get_gemm_hbm_bw(m, n, k, inputs.l2_hit_rate_wt, inputs.l2_hit_rate_act,
                                                        inputs.l3_hit_rate_act, act='A')

        layer_res = LayerResults(gemm_res.alu_util_factor, gemm_res.chip_util_factor, gemm_res.speedup, gemm_res.cycles,
                                 flop, gemm_res.num_cu_util, m, n, k, gemm_res.num_a_blocks, gemm_res.num_b_blocks,
                                 gemm_res.num_partitions, hbm_rd_bw=hbm_rd_bw, hbm_wr_bw=hbm_wr_bw, tpu_sub_res=gemm_res.tpu_sub_res, 
                                 alu_cc=gemm_res.alu_cc, mem_cc=gemm_res.mem_cc, num_rounds=gemm_res.num_rounds,  # Ashish added
                                 total_blocks=gemm_res.total_blocks, num_cu_util_trail=gemm_res.num_cu_util_trail,  # Ashish added
                                 num_a_blocks_trail=gemm_res.num_a_blocks_trail, num_b_blocks_trail=gemm_res.num_b_blocks_trail,  # Ashish added
                                 num_partitions_trail=gemm_res.num_partitions_trail, cycles_trail=gemm_res.cycles_trail, wr_cc=gemm_res.wr_cc,
                                 main_instr=gemm_res.main_instr, threadTile=gemm_res.threadTile, workGroup=gemm_res.workGroup,
                                 unroll_factor=gemm_res.unroll_factor, unroll_factor_trail=gemm_res.unroll_factor_trail,
                                 num_rounds_trail=gemm_res.num_rounds_trail, alu_cc_trail=gemm_res.alu_cc_trail,
                                 mem_cc_trail=gemm_res.mem_cc_trail, wr_cc_trail=gemm_res.wr_cc_trail)  # Ashish added

        return layer_res

    def perform_conv_as_conv(self, hw_cfg, inputs):
        ops = 1
        alu_cc = self.get_alu_cc(inputs, ops)
        mem_cc = sum(self.get_mem_cc(inputs.in_dims, inputs.out_dims, inputs.l2_hit_rate_act, l3_hit_rate_act=inputs.l3_hit_rate_act))
        cycles = max(alu_cc, mem_cc)
        alu_util_factor = (alu_cc / cycles) * 100
        speedup = 1
        flop = self.get_compute_flops(inputs, ops)
        res = LayerResults(alu_util_factor, alu_util_factor, speedup, cycles, flop, self.sys_cfg.hw_cfg.num_cu)
        return res

    def fprop(self, inputs, inference=False):
        hw_cfg = self.sys_cfg.hw_cfg
        res = self.perform_conv_as_gemm(hw_cfg, inputs)
        return res

    def bprop(self, inputs):
        hw_cfg = self.sys_cfg.hw_cfg
        in_size = get_dt_size(self.sys_cfg.sw_opt)
        wgrad_inputs = copy.deepcopy(inputs)
        dgrad_inputs = copy.deepcopy(inputs)
        # 1. Compute weight gradients: dW[l] = A[l-1] conv dZ[l]
        weights_size = in_size
        for i in range(len(wgrad_inputs.in_dims)):
            weights_size *= wgrad_inputs.in_dims[i]  # dZ[l] becomes weights matrix for WGRAD
        wgrad_inputs.l2_hit_rate_act = 1 if self.sys_cfg.hw_cfg.tpu_en else 0  # ML Chiplet assumes external DMA makes all activations resident in L2 before feeding into PEs
        if weights_size <= self.sys_cfg.hw_cfg.l2_size:
            wgrad_inputs.l2_hit_rate_wt = HIGH_L2_HIT_RATE
        else:
            wgrad_inputs.l2_hit_rate_wt = self.sys_cfg.hw_cfg.l2_size * 0.8 / weights_size

        # For stride > 1 (stride-1) zeros are inserted in the filter after every pixel.
        extra_pad_w = 1 if (wgrad_inputs.out_dims[W_IND] - self.filt.dims[F_R_IND] + 2 * self.filt.pads[0]) % \
                           self.filt.strides[0] else 0
        extra_pad_h = 1 if (wgrad_inputs.out_dims[H_IND] - self.filt.dims[F_S_IND] + 2 * self.filt.pads[1]) % \
                           self.filt.strides[1] else 0
        wgrad_inputs.in_dims[W_IND] = wgrad_inputs.in_dims[W_IND] + (self.filt.strides[0] - 1) * (wgrad_inputs.in_dims[W_IND] - 1) + \
                                      extra_pad_w
        wgrad_inputs.in_dims[H_IND] = wgrad_inputs.in_dims[H_IND] + (self.filt.strides[1] - 1) * (wgrad_inputs.in_dims[H_IND] - 1) + \
                                      extra_pad_h

        # For Wgrad, input image's N acts as C, C acts as N; out image's C acts as K, N acts as C
        m = wgrad_inputs.out_dims[C_IND] * self.filt.dims[2] * self.filt.dims[3]
        n = self.filt.dims[0]
        k = wgrad_inputs.in_dims[N_IND] * wgrad_inputs.in_dims[H_IND] * wgrad_inputs.in_dims[W_IND]

        # Pad filter (in this case input image). For stride > 1 (stride-1) zeros are inserted in the filter after every pixel plus
        # filter is padded with 2*(r-p-1) on both dimensions
        wgrad_filt = FilterParam(dims=[wgrad_inputs.in_dims[C_IND], wgrad_inputs.in_dims[N_IND], wgrad_inputs.in_dims[H_IND], wgrad_inputs.in_dims[W_IND]],
                                 pads=[0, 0, 0, 0], strides=[1, 1, 1, 1], group=self.filt.group)

        wgrad_inputs.out_dims = self.filt.dims
        wgrad_inputs.in_dims = inputs.out_dims
        assert(self.filt.group == 1)  # TODO: Support group conv backward
        if self.sys_cfg.hw_cfg.tpu_en:
            input_format = 'cnhw'
        else:
            input_format = 'nhwc'
        gemm_res = self.perform_implicit_gemm(m, n, k, wgrad_inputs, hw_cfg, wgrad_filt, input_format)
        wgrad_flop = 2 * m * n * k
        hbm_rd_bw_wgrad, hbm_wr_bw_wgrad = self.get_gemm_hbm_bw(m, n, k, inputs.l2_hit_rate_wt, inputs.l2_hit_rate_act,
                                                                inputs.l3_hit_rate_act, act='A')
        wgrad_res = LayerResults(gemm_res.alu_util_factor, gemm_res.chip_util_factor, gemm_res.speedup, gemm_res.cycles, wgrad_flop, gemm_res.num_cu_util, m, n, k,
                                 gemm_res.num_a_blocks, gemm_res.num_b_blocks, gemm_res.num_partitions,
                                 hbm_rd_bw=hbm_rd_bw_wgrad, hbm_wr_bw=hbm_wr_bw_wgrad, op_name='weight_gradient', tpu_sub_res=gemm_res.tpu_sub_res,
                                 wr_cc=gemm_res.wr_cc, main_instr=gemm_res.main_instr, unroll_factor_trail=gemm_res.unroll_factor_trail,
                                 unroll_factor=gemm_res.unroll_factor, num_rounds_trail=gemm_res.num_rounds_trail)

        if not inputs.last_node:
            # 2. Compute data gradients: dA[l-1] = dZ[l] conv W
            k = self.filt.dims[0]
            c = self.filt.dims[1]
            self.filt.dims[0] = c
            self.filt.dims[1] = k
            inputs.l2_hit_rate_act = inputs.l2_hit_rate_wt  # dZ[l] is now activation for DGRAD
            inputs.l2_hit_rate_wt = 0.0

            # Pad input image. For stride > 1 (stride-1) zeros are inserted in the input image after every pixel plus
            # input image in padded with 2*(r-p-1) on both dimensions
            extra_pad_w = 1 if (dgrad_inputs.out_dims[W_IND] - self.filt.dims[F_R_IND] + 2 * self.filt.pads[0]) % \
                               self.filt.strides[0] else 0
            extra_pad_h = 1 if (dgrad_inputs.out_dims[H_IND] - self.filt.dims[F_S_IND] + 2 * self.filt.pads[1]) % \
                               self.filt.strides[1] else 0
            dgrad_inputs.in_dims[W_IND] = dgrad_inputs.in_dims[W_IND] + ((self.filt.strides[0] - 1) * (dgrad_inputs.in_dims[W_IND] - 1)) + \
                                    2 * (self.filt.dims[F_R_IND] - self.filt.pads[0] - 1) + extra_pad_w
            dgrad_inputs.in_dims[H_IND] = dgrad_inputs.in_dims[H_IND] + ((self.filt.strides[1] - 1) * (dgrad_inputs.in_dims[H_IND] - 1)) + \
                                    2 * (self.filt.dims[F_S_IND] - self.filt.pads[1] - 1) + extra_pad_h
            self.filt.pads = 2 * [0]
            self.filt.strides = 2 * [1]

            dgrad_res = self.perform_conv_as_gemm(hw_cfg, dgrad_inputs)
            dgrad_res.op_name = 'data_gradient'
            cycles = wgrad_res.cycles + dgrad_res.cycles
            alu_util_factor = (wgrad_res.alu_util_factor * wgrad_res.cycles +
                               dgrad_res.alu_util_factor * dgrad_res.cycles) / cycles
            chip_util_factor = (wgrad_res.chip_util_factor * wgrad_res.cycles +
                               dgrad_res.chip_util_factor * dgrad_res.cycles) / cycles
            flop = wgrad_flop + dgrad_res.flop
            layer_results = LayerResults(alu_util_factor, chip_util_factor, gemm_res.speedup, cycles, flop)
            layer_results.populate_sub_results(wgrad_res)
            layer_results.populate_sub_results(dgrad_res)
        else:
            layer_results = wgrad_res
        return layer_results
