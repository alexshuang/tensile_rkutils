from core.layers.layer import Layer, get_compute_flops
from core.structs.auxiliary import LayerResults
from core.auxiliary import prod, get_dt_size
from core.const import *

class Activation(Layer):
    def __init__(self, sys_config, act='relu'):
        super(Activation, self).__init__(sys_config)
        self.act = act
        self.hw_cfg = self.sys_cfg.hw_cfg
        self.sw_opt = self.sys_cfg.sw_opt

    def fprop(self, inputs, inference=False):
        ops = 1 if inference and self.sw_opt.kernel_fusion else 5  # ~4 address calculation and 1 activation
        alu_cc = self.get_alu_cc(inputs.in_dims, ops)
        data_size = get_dt_size(self.sys_cfg.sw_opt)
        l2_hit_rate_wr = min(self.hw_cfg.l2_size / (prod(inputs.out_dims) * data_size), 1.0)
        mem_cc = sum(self.get_mem_cc(inputs.in_dims, inputs.out_dims, l2_hit_rate_act_rd=inputs.l2_hit_rate_act,
                                     l2_hit_rate_act_wr=l2_hit_rate_wr, l3_hit_rate_act=inputs.l3_hit_rate_act))
        if inference and self.sw_opt.kernel_fusion:
            cycles = alu_cc
        else:
            cycles = max(alu_cc, mem_cc)
        alu_util_factor = (alu_cc / cycles) * 100
        speedup = 1
        flop = get_compute_flops(inputs.in_dims, ops)
        hbm_rd_bw = (1 - inputs.l2_hit_rate_act) * (1 - inputs.l3_hit_rate_act) * prod(inputs.in_dims) * data_size
        hbm_wr_bw = (1 - l2_hit_rate_wr) * prod(inputs.out_dims) * data_size
        res = LayerResults(alu_util_factor, alu_util_factor, speedup, cycles, flop, self.sys_cfg.hw_cfg.num_cu,
                           hbm_rd_bw=hbm_rd_bw, hbm_wr_bw=hbm_wr_bw,
                           alu_cc=alu_cc, mem_cc=mem_cc)  # Ashish added
        if CALIBRATING:
            print("===FORWARD PASS===FORWARD PASS===FORWARD PASS=========================")
            print("cc: {}".format(cycles))
            print("total: {} us".format(cycles / CALIBRATION_CLOCK))
            print("==================================")
        return res

    def bprop(self, inputs):
        return self.fprop(inputs)

