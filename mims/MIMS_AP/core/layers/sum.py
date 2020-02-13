from core.layers.layer import Layer, get_compute_flops
from core.structs.auxiliary import LayerResults


class Sum(Layer):
    def __init__(self, sys_config):
        super(Sum, self).__init__(sys_config)
        self.sw_opt = sys_config.sw_opt

    def fprop(self, inputs, inference=False):
        ops = 1
        alu_cc = self.get_alu_cc(inputs.in_dims, ops)
        in_dims = [inputs.in_dims[0] * 2, inputs.in_dims[1], inputs.in_dims[2], inputs.in_dims[3]]
        mem_cc = sum(self.get_mem_cc(in_dims, inputs.out_dims, inputs.l2_hit_rate_act, l3_hit_rate_act=inputs.l3_hit_rate_act))
        if inference and self.sw_opt.kernel_fusion:
            cycles = alu_cc
        else:
            cycles = max(alu_cc, mem_cc)
        alu_util_factor = (alu_cc / cycles) * 100
        speedup = 1
        flop = get_compute_flops(inputs.in_dims, ops)
        res = LayerResults(alu_util_factor, alu_util_factor, speedup, cycles, flop, self.sys_cfg.hw_cfg.num_cu,
                           alu_cc=alu_cc, mem_cc=mem_cc)  # Ashish added
        return res

    def bprop(self, inputs):
        res = LayerResults(cycles=1) #Ashish added
        return res
