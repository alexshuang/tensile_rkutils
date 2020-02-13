from core.layers.layer import Layer, get_compute_flops
from core.structs.auxiliary import LayerResults


class AllReduce(Layer):
    def __init__(self, sys_config):
        super(AllReduce, self).__init__(sys_config)

    def fprop(self, inputs, inference=False):
        ops = 1
        alu_cc = self.get_alu_cc(inputs.in_dims, ops)
        mem_cc = sum(self.get_mem_cc(inputs.in_dims, inputs.out_dims, inputs.l2_hit_rate_act,
                                     l3_hit_rate_act=inputs.l3_hit_rate_act, reduced_bitwidth=self.sys_cfg.sw_opt.mgpu_num_grad_bits))
        cycles = max(alu_cc, mem_cc)
        alu_util_factor = (alu_cc / cycles) * 100
        speedup = 1
        flop = get_compute_flops(inputs.in_dims, ops)
        res = LayerResults(alu_util_factor, alu_util_factor, speedup, cycles, flop, self.sys_cfg.hw_cfg.num_cu,
                           alu_cc=alu_cc, mem_cc=mem_cc)  # Ashish added
        return res

    def bprop(self, inputs):
        pass