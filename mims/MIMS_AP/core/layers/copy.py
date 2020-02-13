from core.layers.layer import Layer
from core.structs.auxiliary import LayerResults


class Copy(Layer):
    def __init__(self, sys_config):
        super(Copy, self).__init__(sys_config)

    def fprop(self, inputs, inference=False):
        mem_cc = sum(self.get_mem_cc(inputs.in_dims, inputs.out_dims, inputs.l2_hit_rate_act, l3_hit_rate_act=inputs.l3_hit_rate_act))
        alu_util_factor = 0.0
        speedup = 1
        flop = 0
        res = LayerResults(alu_util_factor, alu_util_factor, speedup, mem_cc, flop, self.sys_cfg.hw_cfg.num_cu)
        return res

    def bprop(self, inputs):
        pass
