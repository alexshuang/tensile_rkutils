from core.layers.layer import Layer
from core.structs.auxiliary import LayerResults
from core.const import *
import math


class Pooling(Layer):
    def __init__(self, sys_config, filt_params):
        super(Pooling, self).__init__(sys_config)
        self.filt = filt_params
        self.hw_cfg = self.sys_cfg.hw_cfg
        self.sw_opt = self.sys_cfg.sw_opt

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

    def fprop(self, inputs, inference=False):
        ops = 1
        alu_cc = self.get_alu_cc(inputs, ops)
        mem_cc = sum(self.get_mem_cc(inputs.in_dims, inputs.out_dims, inputs.l2_hit_rate_act, l3_hit_rate_act=inputs.l3_hit_rate_act))
        if inference and self.sw_opt.kernel_fusion:
            cycles = alu_cc
        else:
            cycles = max(alu_cc, mem_cc)
        alu_util_factor = (alu_cc / cycles) * 100
        speedup = 1
        flop = self.get_compute_flops(inputs, ops)
        res = LayerResults(alu_util_factor, alu_util_factor, speedup, cycles, flop, self.sys_cfg.hw_cfg.num_cu,
                           alu_cc=alu_cc, mem_cc=mem_cc)  # Ashish added
        return res

    def bprop(self, inputs):
        ops = 1
        flop = inputs.in_dims[0] * inputs.in_dims[1] * inputs.in_dims[2] * inputs.in_dims[3] * ops
        alu_cc = flop / self.get_hw_flops(mai_accelerated=0)
        mem_cc = sum(self.get_mem_cc(inputs.in_dims, inputs.out_dims, inputs.l2_hit_rate_act, l3_hit_rate_act=inputs.l3_hit_rate_act))
        cycles = max(alu_cc, mem_cc)
        alu_util_factor = (alu_cc / cycles) * 100
        speedup = 1
        res = LayerResults(alu_util_factor, alu_util_factor, speedup, cycles, flop, self.sys_cfg.hw_cfg.num_cu,
                           alu_cc=alu_cc, mem_cc=mem_cc)  # Ashish added
        return res