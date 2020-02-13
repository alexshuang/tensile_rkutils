import copy
from math import ceil
from core.structs.hw_config import HWConfig
from core.structs.hw_options import HWOptions
from core.structs.sw_options import SWOptions


class FullConfig:
    def __init__(self, hw_cfg_container, hw_opt_container, sw_opt_container, default_cfg, from_ui=True, cfg_type=''):
        self.default_cfg = default_cfg
        default_hw_opt_map = default_cfg.hw_opt.get_attr_default_val_map() if default_cfg else None
        default_sw_opt_map = default_cfg.sw_opt.get_attr_default_val_map() if default_cfg else None
        self.hw_cfg = HWConfig(hw_cfg_container) if from_ui else hw_cfg_container
        self.hw_opt = HWOptions(hw_opt_container, default_hw_opt_map) if from_ui else hw_opt_container
        self.sw_opt = SWOptions(sw_opt_container, default_sw_opt_map) if from_ui else sw_opt_container
        self.cfg_type = cfg_type

        self._set_derived_attr_for_mgpu_and_chiplet()

    def _set_derived_attr_for_mgpu_and_chiplet(self):
        if self.cfg_type == 'sys':
            if self.sw_opt.multi_gpu:
                if self.hw_cfg.chiplet_mode_en and not self.hw_cfg.chiplet_strong_scaling:
                    self.sw_opt.mgpu_gpu_count *= self.hw_cfg.num_cu_clusters
                if self.sw_opt.mgpu_weak_scaling:
                    self.sw_opt.batch_size *= (self.sw_opt.mgpu_gpu_count * self.sw_opt.mgpu_multi_node)
                    if self.hw_cfg.tpu_en:
                        self.sw_opt.batch_size *= self.hw_cfg.num_xpe_clusters
        elif self.cfg_type == 'gpu':
            if self.sw_opt.multi_gpu and not self.sw_opt.mgpu_weak_scaling:
                self.sw_opt.batch_size //= (self.sw_opt.mgpu_gpu_count * self.sw_opt.mgpu_multi_node)

            if self.hw_cfg.chiplet_mode_en:
                if self.hw_cfg.chiplet_strong_scaling:
                    self.sw_opt.batch_size = self.sw_opt.batch_size / self.hw_cfg.num_cu_clusters
                self.hw_cfg.num_cu = self.hw_cfg.num_cu / self.hw_cfg.num_cu_clusters
                self.hw_cfg.num_cu_util = self.hw_cfg.num_cu
                self.hw_cfg.hbm_bw = self.hw_cfg.hbm_bw / self.hw_cfg.num_cu_clusters
                if self.sw_opt.multi_gpu:
                    self.sw_opt.mgpu_xgmi_link_count = ceil(
                        self.sw_opt.mgpu_xgmi_link_count / self.hw_cfg.num_cu_clusters)

                self.hw_cfg.num_cu_clusters = 1
                assert self.sw_opt.training  # Currently chiplet mode only supported for training
        elif self.cfg_type == 'tpu' and self.hw_cfg.tpu_en:
            self.set_tpu_cfg()

    def set_tpu_cfg(self):
        self.hw_cfg.num_cu = self.hw_cfg.num_xpe
        self.hw_cfg.num_cu_util = self.hw_cfg.num_xpe
        self.hw_cfg.num_cu_clusters = self.hw_cfg.num_xpe_clusters
        self.hw_cfg.num_cu_per_cluster = self.hw_cfg.num_xpe_per_cluster
        self.hw_cfg.num_se_per_cluster = self.hw_cfg.num_slice_per_cluster
        self.hw_cfg.chiplet_mode_en = self.hw_cfg.chiplet_mode_en_tpu
        self.hw_cfg.chiplet_strong_scaling = self.hw_cfg.chiplet_strong_scaling_tpu
        self.hw_cfg.int8_dl_macs_per_cu = self.hw_cfg.int8_dl_macs_per_xpe
        self.hw_cfg.fp16_dl_macs_per_cu = self.hw_cfg.fp16_dl_macs_per_xpe
        self.hw_cfg.fp32_dl_macs_per_cu = self.hw_cfg.fp32_dl_macs_per_xpe
        self.hw_cfg.fp64_dl_macs_per_cu = self.hw_cfg.fp64_dl_macs_per_xpe
        self.hw_cfg.dl_instr_large_block = self.hw_cfg.dl_instr_large_block_tpu
        self.hw_cfg.dl_instr_small_block = self.hw_cfg.dl_instr_small_block_tpu
        self.hw_cfg.dl_other_instr_block = self.hw_cfg.dl_other_instr_block_tpu
        self.hw_cfg.legacy_int8_macs_per_cu = self.hw_cfg.legacy_int8_macs_per_xpe
        self.hw_cfg.legacy_fp16_macs_per_cu = self.hw_cfg.legacy_fp16_macs_per_xpe
        self.hw_cfg.legacy_fp32_macs_per_cu = self.hw_cfg.legacy_fp32_macs_per_xpe
        self.hw_cfg.arch_vgpr_size_per_cu = self.hw_cfg.arch_vgpr_size_per_xpe
        self.hw_cfg.accum_vgpr_size_per_cu = self.hw_cfg.accum_vgpr_size_per_xpe
        self.hw_cfg.total_vgpr_size_per_cu = self.hw_cfg.total_vgpr_size_per_xpe
        self.hw_cfg.arch_vgpr_rd_ports_per_cu = self.hw_cfg.arch_vgpr_rd_ports_per_xpe
        self.hw_cfg.arch_vgpr_rd_bw_per_cu = self.hw_cfg.arch_vgpr_rd_bw_per_xpe
        self.hw_cfg.arch_vgpr_wr_ports_per_cu = self.hw_cfg.arch_vgpr_wr_ports_per_xpe
        self.hw_cfg.arch_vgpr_wr_bw_per_cu = self.hw_cfg.arch_vgpr_wr_bw_per_xpe
        self.hw_cfg.vgpr_rd_bw = self.hw_cfg.vgpr_rd_bw_xpe
        self.hw_cfg.vgpr_wr_bw = self.hw_cfg.vgpr_wr_bw_xpe
        self.hw_cfg.lds_bw = self.hw_cfg.lds_bw_xpe
        self.hw_cfg.lds_size = self.hw_cfg.lds_size_xpe
        self.hw_cfg.l1_bw = self.hw_cfg.l1_bw_xpe
        self.hw_cfg.l1_size = self.hw_cfg.l1_size_xpe
        self.hw_cfg.l1_cache_line_size = self.hw_cfg.l1_cache_line_size_xpe
        self.hw_cfg.l2_read_buses_per_se = self.hw_cfg.l2_read_buses_per_slice
        self.hw_cfg.l2_read_bus_width = self.hw_cfg.l2_read_bus_width_slice
        self.hw_cfg.l2_write_buses_per_se = self.hw_cfg.l2_write_buses_per_slice
        self.hw_cfg.l2_write_bus_width = self.hw_cfg.l2_write_bus_width_slice
        self.hw_cfg.l2_size = self.hw_cfg.l2_size_tpu
        self.hw_cfg.l2_cache_line_size = self.hw_cfg.l2_cache_line_size_tpu
        self.hw_cfg.l2_hit_latency = self.hw_cfg.l2_hit_latency_tpu
        self.hw_cfg.l2_miss_latency = self.hw_cfg.l2_miss_latency_tpu
        self.hw_cfg.l2_df_bw = self.hw_cfg.l2_df_bw_tpu
        self.hw_cfg.gpu_freq = self.hw_cfg.tpu_freq
        self.hw_cfg.num_gmi_links = self.hw_cfg.num_gmi_links_tpu
        self.hw_cfg.gmi_link_bw = self.hw_cfg.gmi_link_bw_tpu

        if self.sw_opt.multi_gpu and not self.sw_opt.mgpu_weak_scaling:
            self.sw_opt.batch_size //= (self.sw_opt.mgpu_gpu_count * self.sw_opt.mgpu_multi_node)

        if self.hw_cfg.chiplet_mode_en:
            if self.hw_cfg.chiplet_strong_scaling:
                self.sw_opt.batch_size = self.sw_opt.batch_size / self.hw_cfg.num_cu_clusters
            self.hw_cfg.num_cu = self.hw_cfg.num_cu / self.hw_cfg.num_cu_clusters
            self.hw_cfg.num_cu_util = self.hw_cfg.num_cu
            self.hw_cfg.num_cu_clusters = 1
            assert self.sw_opt.training  # Currently chiplet mode only supported for training


    def is_dirty(self):
        return self.hw_opt.is_dirty() or self.sw_opt.is_dirty()

    def reset_to_defaults(self):
        return self.hw_opt.reset_to_defaults() or self.sw_opt.reset_to_defaults()

    def gen_allreduce_config(self, num_allreduce_cu=0):
        hw_cfg = copy.deepcopy(self.hw_cfg)
        hw_opt = copy.deepcopy(self.hw_opt)
        sw_opt = copy.deepcopy(self.sw_opt)
        sw_opt.total_xgmi_bw = self.sw_opt.mgpu_xgmi_link_bw * self.sw_opt.mgpu_xgmi_link_count / self.hw_cfg.gpu_freq  # bytes/cc
        sw_opt.total_mgpu_multi_node_bw = self.sw_opt.mgpu_multi_node_link_bw * self.sw_opt.mgpu_multi_node_link_count * 0.8 / self.hw_cfg.gpu_freq  # bytes/cc

        # Config for running allReduce operation on GPU
        if sw_opt.mgpu_sdma_xfer:
            hw_cfg.num_cu_util = 8  # CUs only perform reduce; scatter/gather phases handled by SDMA engines
            assert (num_allreduce_cu < hw_cfg.num_cu_util)
            hw_cfg.num_cu_util -= num_allreduce_cu
            hw_cfg.total_gmi_bw = self.hw_cfg.gmi_efficiency_sdma * self.hw_cfg.total_gmi_bw
            sw_opt.total_xgmi_bw = sw_opt.total_xgmi_bw * 0.94  # XGMI SDMA efficiency
            hw_cfg.hbm_bw = max(sw_opt.total_xgmi_bw, hw_cfg.total_gmi_bw)  # SDMAs will utilize HBM BW to saturate xgmi/gmi links
        else:  # CUs perform scatter/gather phases along with reduce operation
            hw_cfg.total_gmi_bw = self.hw_cfg.gmi_efficiency_cu * self.hw_cfg.total_gmi_bw
            sw_opt.total_xgmi_bw = sw_opt.total_xgmi_bw * self.sw_opt.mgpu_xgmi_link_bw_eff  # XGMI CU efficiency
            gmi_or_xgmi_bw = max(hw_cfg.total_gmi_bw, sw_opt.total_xgmi_bw)
            # l2_df_bw: L2 to Data fabric BW
            l2_df_bw_per_cu = self.hw_cfg.l2_df_bw / self.hw_cfg.num_cu
            num_cu_allreduce_xgmi = ceil(gmi_or_xgmi_bw / l2_df_bw_per_cu)
            num_cu_allreduce_xgmi = num_cu_allreduce_xgmi + 1 if num_cu_allreduce_xgmi % 2 else num_cu_allreduce_xgmi
            hw_cfg.num_cu_util = num_cu_allreduce_xgmi
            assert (num_allreduce_cu < hw_cfg.num_cu_util)
            hw_cfg.num_cu_util -= num_allreduce_cu
            hw_cfg.hbm_bw = (self.hw_cfg.hbm_bw / self.hw_cfg.num_cu) * hw_cfg.num_cu_util  # CUs will utilize HBM BW to saturate xgmi links
            # For reduced num_cu used for allReduce compared to minimum required, total XGMI/GMI BW is reduced accordingly
            sw_opt.total_xgmi_bw = min((hw_cfg.num_cu_util / num_cu_allreduce_xgmi), 1) * sw_opt.total_xgmi_bw

        return FullConfig(hw_cfg, hw_opt, sw_opt, default_cfg=self.default_cfg, from_ui=False)
