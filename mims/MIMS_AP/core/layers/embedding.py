from core.layers.layer import Layer, get_compute_flops
from core.structs.auxiliary import LayerResults
from core.auxiliary import prod
import math
import copy

EMB_BS_IND = 0
EMB_NUM_TABLES_IND = 1
EMB_NUM_LUT_INDICES_IND = 2
EMB_DIM_IND = 3

# ----- Currently this layer is specifically hardcoded for Phantom's sparseNN network------ #


class Embedding(Layer):
    def __init__(self, sys_config):
        super(Embedding, self).__init__(sys_config)
        self.hw_cfg = sys_config.hw_cfg
        self.sw_opt = sys_config.sw_opt
        self.raw_hbm_bw = self.hw_cfg.hbm_bus_width * self.hw_cfg.hbm_freq / 8  # in GB/s
        if self.hw_cfg.chiplet_mode_en:
            self.raw_hbm_bw /= 2  # Assumption of 2 die chiplet

    def num_cu_to_saturate_hbm(self):
        num_cu_per_se = self.hw_cfg.num_cu_util / self.hw_cfg.num_se_per_cluster
        l2_bw_per_se = self.hw_cfg.l2_read_buses_per_se * self.hw_cfg.l2_read_bus_width
        bw_per_cu = self.hw_cfg.l1_bw if self.hw_cfg.l1_bw else math.ceil(l2_bw_per_se / num_cu_per_se)
        num_cu_required = math.ceil(l2_bw_per_se / bw_per_cu) * self.hw_cfg.num_se_per_cluster
        return num_cu_required

    # hardcode random_util_factor for sparseNN for convenience
    # numbers for 'self.hw_cfg.chiplet_mode_en' are for MI-200 with HBM2E @ 3.2 Gbps
    def get_hbm_random_util_factor(self, P):
        random_util_factor = 0.6
        if P == 40:
            random_util_factor = 0.20 if self.hw_cfg.chiplet_mode_en else 0.25
        elif P == 60:
            random_util_factor = 0.38 if self.hw_cfg.chiplet_mode_en else 0.5
        elif P == 100:
            random_util_factor = 0.6 if self.hw_cfg.chiplet_mode_en else 0.6
        else:
            assert 'Unknown P value'
        return random_util_factor

    def fprop(self, inputs, inference=False):
        addr_calc_ops_p = 4
        addr_calc_ops_e = 4
        pooling_ops = 1
        total_ops = addr_calc_ops_p + addr_calc_ops_e + pooling_ops
        data_size = 2 if self.sw_opt.fp16_inputs or self.sw_opt.bf16_inputs else (4 if self.sw_opt.fp32_inputs else 8)
        hbm_rd_bw = 0
        hbm_wr_bw = 0

        num_cu_required = self.num_cu_to_saturate_hbm()

        # ALU cycles for pooling
        alu_cc = self.get_alu_cc(inputs.in_dims, total_ops, num_cu=num_cu_required)

        # memory cycles which emulates reads of P indices and result writes
        p_in_dims = [inputs.in_dims[EMB_BS_IND], inputs.in_dims[EMB_NUM_TABLES_IND], inputs.in_dims[EMB_NUM_LUT_INDICES_IND], 2] if self.sw_opt.fp16_inputs or self.sw_opt.bf16_inputs else \
                    [inputs.in_dims[EMB_BS_IND], inputs.in_dims[EMB_NUM_TABLES_IND], inputs.in_dims[EMB_NUM_LUT_INDICES_IND], 1]  # This is to emulate FP32 reads for P indices

        p_mem_cc = (prod(p_in_dims) * data_size / (self.raw_hbm_bw * 0.85)) * self.hw_cfg.gpu_freq  # P indices fetch should have regular access pattern
        wr_mem_cc = (prod(inputs.out_dims) * data_size / (self.raw_hbm_bw * 0.85)) * self.hw_cfg.gpu_freq
        hbm_rd_bw += prod(p_in_dims) * data_size
        hbm_wr_bw += prod(inputs.out_dims) * data_size

        # reduced effective HBM BW due to random accesses
        hbm_bw = self.raw_hbm_bw * self.get_hbm_random_util_factor(inputs.in_dims[EMB_NUM_LUT_INDICES_IND])
        # memory cycles to fetch P entries from embedding tables
        e_mem_cc = (prod(inputs.in_dims) * data_size / hbm_bw) * self.hw_cfg.gpu_freq
        hbm_rd_bw += prod(inputs.in_dims) * data_size

        cycles = max(alu_cc, p_mem_cc + e_mem_cc + wr_mem_cc)
        alu_util_factor = (alu_cc / cycles) * 100
        chip_util_factor = alu_util_factor * (num_cu_required / self.hw_cfg.num_cu_util)
        speedup = 1
        flop = get_compute_flops(inputs.in_dims, total_ops)
        res = LayerResults(alu_util_factor, chip_util_factor, speedup, cycles, flop, num_cu_required,
                           hbm_rd_bw=hbm_rd_bw, hbm_wr_bw=hbm_wr_bw,
                           alu_cc=alu_cc, mem_cc=e_mem_cc+p_mem_cc+wr_mem_cc)  # Ashish added
        return res

    def bprop(self, inputs):
        ops = 4
        data_size = 2 if self.sw_opt.fp16_inputs or self.sw_opt.bf16_inputs else (4 if self.sw_opt.fp32_inputs else 8)
        num_cu_required = self.num_cu_to_saturate_hbm()
        hbm_rd_bw = 0
        hbm_wr_bw = 0

        # ALU cycles
        alu_cc = self.get_alu_cc(inputs.out_dims, ops, num_cu=num_cu_required)

        # memory cycles
        in_dims = copy.deepcopy(inputs.out_dims)
        in_dims[EMB_NUM_LUT_INDICES_IND] += 1
        hbm_bw = self.raw_hbm_bw * self.get_hbm_random_util_factor(inputs.out_dims[EMB_NUM_LUT_INDICES_IND]/2)  # in GB/s
        rd_mem_cc = (prod(in_dims) * data_size / hbm_bw) * self.hw_cfg.gpu_freq
        wr_mem_cc = (prod(inputs.out_dims) * data_size / hbm_bw) * self.hw_cfg.gpu_freq
        hbm_rd_bw += prod(in_dims) * data_size
        hbm_wr_bw += prod(inputs.out_dims) * data_size
        mem_cc = rd_mem_cc + wr_mem_cc

        cycles = max(alu_cc, mem_cc)
        alu_util_factor = (alu_cc / cycles) * 100
        chip_util_factor = alu_util_factor * (num_cu_required / self.hw_cfg.num_cu_util)
        speedup = 1
        flop = get_compute_flops(in_dims, ops)
        res = LayerResults(alu_util_factor, chip_util_factor, speedup, cycles, flop, num_cu_required,
                           hbm_rd_bw=hbm_rd_bw, hbm_wr_bw=hbm_wr_bw)
        return res
