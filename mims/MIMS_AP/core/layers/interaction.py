from core.layers.layer import Layer, get_compute_flops
from core.auxiliary import get_dt_size, prod
from core.structs.auxiliary import LayerResults


class Interaction(Layer):
    def __init__(self, sys_config):
        super(Interaction, self).__init__(sys_config)

    def fprop(self, inputs, inference=False):
        ops = 1
        alu_cc = self.get_alu_cc(inputs.in_dims, ops)
        # Reads of two input tensors simulated by out_dims
        data_size = get_dt_size(self.sys_cfg.sw_opt)
        l2_hit_rate_wr = min(self.sys_cfg.hw_cfg.l2_size / (prod(inputs.out_dims) * data_size), 1.0)
        mem_cc = sum(self.get_mem_cc(inputs.in_dims, inputs.out_dims, inputs.l2_hit_rate_act, l2_hit_rate_wr,
                                     l3_hit_rate_act=inputs.l3_hit_rate_act))
        cycles = max(alu_cc, mem_cc)
        alu_util_factor = (alu_cc / cycles) * 100
        speedup = 1
        flop = get_compute_flops(inputs.in_dims, ops)
        hbm_rd_bw = (1 - inputs.l2_hit_rate_act) * (1 - inputs.l3_hit_rate_act) * prod(inputs.in_dims) * data_size
        hbm_wr_bw = (1 - l2_hit_rate_wr) * prod(inputs.out_dims) * data_size
        res = LayerResults(alu_util_factor, alu_util_factor, speedup, cycles, flop, self.sys_cfg.hw_cfg.num_cu,
                           hbm_rd_bw=hbm_rd_bw, hbm_wr_bw=hbm_wr_bw,
                           alu_cc=alu_cc, mem_cc=mem_cc)  # Ashish added
        return res

    def bprop(self, inputs):
        pass
