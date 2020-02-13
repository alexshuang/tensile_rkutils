from core.layers.layer import Layer, get_compute_flops
from core.auxiliary import get_dt_size
from core.structs.auxiliary import LayerResults


class All2all(Layer):
    def __init__(self, sys_config, allreduce_cfg, mgpu_backend):
        super(All2all, self).__init__(sys_config)
        self.allreduce_cfg = allreduce_cfg
        self.mgpu_backend = mgpu_backend

    def fprop(self, inputs, inference=False):
        assert (self.sys_cfg.sw_opt.multi_gpu)
        num_elements = inputs.in_dims[0] * inputs.in_dims[1] * inputs.in_dims[2] * inputs.in_dims[3]
        scatter_cc = self.mgpu_backend.get_scatter_cc(num_elements)
        gather_cc = self.mgpu_backend.get_gather_cc(num_elements)
        cycles = scatter_cc + gather_cc
        # TODO: These are approx. numbers for now. Add topology specific calculations later.
        hbm_rd_bw = 2 * num_elements * get_dt_size(self.sys_cfg.sw_opt)
        res = LayerResults(alu_util_factor=0, speedup=1, cycles=cycles, flop=0, num_cu_util=self.allreduce_cfg.hw_cfg.num_cu,
                           hbm_rd_bw=hbm_rd_bw, hbm_wr_bw=hbm_rd_bw)
        return res

    def bprop(self, inputs):
        assert (self.sys_cfg.sw_opt.multi_gpu)
        num_elements = inputs.in_dims[0] * inputs.in_dims[1] * inputs.in_dims[2] * inputs.in_dims[3]
        scatter_cc = self.mgpu_backend.get_scatter_cc(num_elements)
        gather_cc = self.mgpu_backend.get_gather_cc(num_elements)
        cycles = scatter_cc + gather_cc
        # TODO: These are approx. numbers for now. Add topology specific calculations later.
        hbm_rd_bw = 2 * num_elements * get_dt_size(self.sys_cfg.sw_opt)
        res = LayerResults(alu_util_factor=0, speedup=1, cycles=cycles, flop=0, num_cu_util=self.allreduce_cfg.hw_cfg.num_cu,
                           hbm_rd_bw=hbm_rd_bw, hbm_wr_bw=hbm_rd_bw)
        return res


class Allgather(Layer):
    def __init__(self, sys_config, allreduce_cfg, mgpu_backend):
        super(Allgather, self).__init__(sys_config)
        self.allreduce_cfg = allreduce_cfg
        self.mgpu_backend = mgpu_backend

    def fprop(self, inputs, inference=False):
        #assert (self.sys_cfg.sw_opt.multi_gpu)
        num_elements = inputs.in_dims[0] * inputs.in_dims[1] * inputs.in_dims[2] * inputs.in_dims[3]
        cycles = self.mgpu_backend.get_scatter_cc(num_elements)
        # TODO: These are approx. numbers for now. Add topology specific calculations later.
        hbm_rd_bw = num_elements * get_dt_size(self.sys_cfg.sw_opt)
        res = LayerResults(cycles=cycles, num_cu_util=self.allreduce_cfg.hw_cfg.num_cu,
                           hbm_rd_bw=hbm_rd_bw, hbm_wr_bw=hbm_rd_bw)
        return res

    def bprop(self, inputs):
        #assert (self.sys_cfg.sw_opt.multi_gpu)
        num_elements = inputs.in_dims[0] * inputs.in_dims[1] * inputs.in_dims[2] * inputs.in_dims[3]
        cycles = self.mgpu_backend.get_scatter_cc(num_elements)
        # TODO: These are approx. numbers for now. Add topology specific calculations later.
        hbm_rd_bw = num_elements * get_dt_size(self.sys_cfg.sw_opt)
        res = LayerResults(cycles=cycles, num_cu_util=self.allreduce_cfg.hw_cfg.num_cu,
                           hbm_rd_bw=hbm_rd_bw, hbm_wr_bw=hbm_rd_bw)
        return res
