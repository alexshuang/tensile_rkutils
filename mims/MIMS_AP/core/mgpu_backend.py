from core.layers import *
import math
import matplotlib.pyplot as plt
import numpy as np

PLOT_TRANFER_COST_GRAPH = 0


class Topology:
    def __init__(self, gpu_cfg, allreduce_cfg, alpha, mgpu_multi_node_alpha, beta_gmi, beta_xgmi, beta_mgpu_multi_node):
        self.gpu_cfg = gpu_cfg
        self.allreduce_cfg = allreduce_cfg
        self.alpha = alpha
        self.mgpu_multi_node_alpha = mgpu_multi_node_alpha
        self.beta_gmi = beta_gmi
        self.beta_xgmi = beta_xgmi
        self.beta_mgpu_multi_node = beta_mgpu_multi_node

    def get_link_bw_curve_factor(self, n):
        msg_size_range_high = 0
        lbc_range_high = 0
        lbc_range_low = 0
        for msg_size_range_low, lbc_range_low in self.gpu_cfg.hw_cfg.ext_link_bw_curve.items():
            if n >= msg_size_range_low:
                if lbc_range_high:
                    # linear interpolation between the ranges
                    lbc_range_low = lbc_range_low + (n - msg_size_range_low) * ((lbc_range_high - lbc_range_low) / (msg_size_range_high - msg_size_range_low))
                break
            else:
                lbc_range_high = lbc_range_low
                msg_size_range_high = msg_size_range_low
        return lbc_range_low

    def get_buffer_copy_cycles(self, buf_size):
        cycles = math.ceil(buf_size / self.allreduce_cfg.hw_cfg.hbm_bw)
        return cycles

    def get_reduce_cycles(self, rd_buf_size, wt_buf_size, num_buf, persistent=False):
        allred = AllReduce(self.allreduce_cfg)
        layer_params = LayerInputParam(in_dims=[1, num_buf, 1, rd_buf_size],
                                       out_dims=([0, 0, 0, 0] if persistent else [1, 1, 1, wt_buf_size]))
        layer_result = allred.fprop(layer_params)
        return layer_result.cycles

    def get_mcm_reduce_cycles(self, n):
        data_size = self.gpu_cfg.sw_opt.mgpu_num_grad_bits / 8
        num_cu_clusters = 2  # Assumption: Dual die GPU
        num_gmi_links = self.allreduce_cfg.hw_cfg.num_gmi_links

        total_cycles_list = []
        for buffer in range(1, 4+1):  # Buffering
            mcm_comm_steps = buffer
            msg_size = math.ceil(n / (mcm_comm_steps * num_cu_clusters * num_gmi_links)) * data_size
            beta_gmi = self.beta_gmi / self.get_link_bw_curve_factor(msg_size)
            buffer_copy_cc = self.get_buffer_copy_cycles(msg_size * (num_cu_clusters-1))
            if self.gpu_cfg.sw_opt.mgpu_sdma_xfer:
                epsilon = 2 * buffer_copy_cc  # multiply by 2 corresponds to read from remote plus write in local
            else:
                epsilon = 0
            mcm_scatter_cc_per_step = self.alpha + msg_size * beta_gmi + epsilon
            mcm_scatter_cc = mcm_comm_steps * mcm_scatter_cc_per_step
            mcm_reduce_cc = self.get_reduce_cycles(math.ceil(n / num_cu_clusters), math.ceil(n / num_cu_clusters), num_cu_clusters)
            if buffer == 1:
                mcm_reducescatter_cc = mcm_scatter_cc + mcm_reduce_cc
            else:
                mcm_reducescatter_cc = max(mcm_scatter_cc, mcm_reduce_cc) + mcm_scatter_cc_per_step
            msg_size = (n / (num_cu_clusters * num_gmi_links)) * data_size
            beta_gmi = self.beta_gmi / self.get_link_bw_curve_factor(msg_size)
            mcm_gather_cc = self.alpha + msg_size * beta_gmi + epsilon
            total_cycles_list.append(mcm_reducescatter_cc + mcm_gather_cc)
        min_vals = heapq.nsmallest(len(total_cycles_list), total_cycles_list)
        min_idx = total_cycles_list.index(min_vals[0])

        return total_cycles_list[min_idx]

    def get_multi_node_cycles(self, num_elements):
        num_gpu = self.gpu_cfg.sw_opt.mgpu_gpu_count
        num_nodes = self.gpu_cfg.sw_opt.mgpu_multi_node
        num_cu_clusters = 2 if self.gpu_cfg.hw_cfg.chiplet_mode_en else 1  # Assumption: Dual die chiplet
        data_size = self.gpu_cfg.sw_opt.mgpu_num_grad_bits / 8
        available_onchip_mem = self.allreduce_cfg.hw_cfg.arch_vgpr_size_per_cu * 0.8 * self.allreduce_cfg.hw_cfg.num_cu_util
        # Multi-GPU and multi-node system
        gradient_aggr_cycles = 0
        if num_nodes > 1:
            assert (num_nodes == 2)  # Work with 2 nodes for now. Expand later
            if 'default' in self.allreduce_cfg.sw_opt.mgpu_multi_node_mgpu_all_reduce_algo:
                msg_size = math.ceil(num_elements / self.allreduce_cfg.sw_opt.mgpu_multi_node_link_count) * data_size
                beta_mgpu_multi_node = self.beta_mgpu_multi_node / self.get_link_bw_curve_factor(msg_size)
                scatter_cc = self.mgpu_multi_node_alpha + msg_size * beta_mgpu_multi_node
                num_gpu_per_node = 4
                reduce_cc = self.get_reduce_cycles(math.ceil(num_elements / num_gpu_per_node),
                                                   math.ceil(num_elements / num_gpu_per_node),
                                                   1)  # assuming previous steps results still in onchip memory
                # Broadcast among all GPUs in a node
                msg_size = math.ceil(num_elements / num_gpu_per_node) * data_size
                beta_xgmi = self.beta_xgmi / self.get_link_bw_curve_factor(msg_size)
                bcast_cc = 2 * (self.alpha + msg_size * beta_xgmi)  # needs 2 hops for non-connected GPUs
                gradient_aggr_cycles = (scatter_cc + reduce_cc + bcast_cc)
            elif 'X8' in self.allreduce_cfg.sw_opt.mgpu_multi_node_mgpu_all_reduce_algo:
                comm_steps = int(math.log2(num_gpu * num_cu_clusters * num_nodes))
                # 1. CU based system scatter reduce
                sys_reducescatter_cc = 0
                scatter_recursive_rate = 0
                for i in range(1, comm_steps + 1):
                    scatter_recursive_rate = math.pow(2, i)
                    msg_size = (num_elements / scatter_recursive_rate) * data_size
                    beta_xgmi = self.beta_xgmi / 2 if i <= 2 else self.beta_xgmi  # GPUs in first two steps are connected by 2 xgmi links
                    beta = beta_xgmi / self.get_link_bw_curve_factor(msg_size)
                    buffer_copy_cc = self.get_buffer_copy_cycles(msg_size)
                    if self.gpu_cfg.sw_opt.mgpu_sdma_xfer:
                        epsilon = 2 * buffer_copy_cc  # multiply by 2 corresponds to read from remote plus write in local
                    else:
                        epsilon = 0
                    alpha = self.mgpu_multi_node_alpha if i == comm_steps else self.alpha
                    sys_scatter_cc = alpha + msg_size * beta + epsilon
                    reduce_rd_size = math.ceil(num_elements / scatter_recursive_rate)
                    num_rd_buf = 1 if reduce_rd_size * data_size < available_onchip_mem and i > 1 else 2
                    sys_reduce_cc = self.get_reduce_cycles(reduce_rd_size, reduce_rd_size, num_rd_buf)
                    sys_reducescatter_cc += (sys_scatter_cc + sys_reduce_cc)

                # 2. System gather
                sys_gather_cc = 0
                for i in range(0, comm_steps):
                    gather_recursive_rate = math.pow(2, i)
                    msg_size = (gather_recursive_rate * num_elements / scatter_recursive_rate) * data_size
                    beta = self.beta_xgmi / self.get_link_bw_curve_factor(msg_size)
                    buffer_copy_cc = self.get_buffer_copy_cycles(msg_size)
                    if self.gpu_cfg.sw_opt.mgpu_sdma_xfer:
                        epsilon = 2 * buffer_copy_cc  # multiply by 2 corresponds to read from remote plus write in local
                    else:
                        epsilon = 0
                    alpha = self.mgpu_multi_node_alpha if i == 0 else self.alpha
                    sys_gather_cc += (alpha + msg_size * beta + epsilon)

                gradient_aggr_cycles = sys_reducescatter_cc + sys_gather_cc
        return gradient_aggr_cycles


class ChordalRing(Topology):
    def __init__(self, gpu_cfg, allreduce_cfg, alpha, mgpu_multi_node_alpha, beta_gmi, beta_xgmi, beta_mgpu_multi_node):
        super(ChordalRing, self).__init__(gpu_cfg, allreduce_cfg, alpha, mgpu_multi_node_alpha, beta_gmi, beta_xgmi, beta_mgpu_multi_node)

    def get_scatter_cc(self, num_elements):
        num_gpu = self.gpu_cfg.sw_opt.mgpu_gpu_count
        data_size = self.gpu_cfg.sw_opt.mgpu_num_grad_bits / 8
        num_mem_channels = 32

        if self.gpu_cfg.sw_opt.mgpu_xgmi_link_count == self.gpu_cfg.sw_opt.mgpu_gpu_count-1:
            residual_msg_size = 0
        else:
            # Each link gets extra payload in case of 'almost' fully connected chordal ring.
            # 6/32 load per link due imbalance between links and memory channels(32)
            residual_msg_size = (6 * num_elements / (num_gpu * num_mem_channels)) * data_size
        msg_size = (num_elements / num_gpu) * data_size + residual_msg_size
        beta_xgmi = self.beta_xgmi / self.get_link_bw_curve_factor(msg_size)
        buffer_copy_cc = self.get_buffer_copy_cycles(msg_size * (num_gpu - 1))
        if self.gpu_cfg.sw_opt.mgpu_sdma_xfer:
            epsilon = 2 * buffer_copy_cc  # multiply by 2 corresponds to read from remote plus write in local
        else:
            epsilon = 0
        scatter_cc = self.alpha + msg_size * beta_xgmi + epsilon
        beta_xgmi = self.beta_xgmi / self.get_link_bw_curve_factor(residual_msg_size)
        if residual_msg_size:
            scatter_cc += self.alpha + residual_msg_size * beta_xgmi + epsilon  # 2 hops for un-connected GPU
        return scatter_cc

    def get_gather_cc(self, num_elements):
        return self.get_scatter_cc(num_elements)  # Gather phase identical to scatter phase

    def CR8G_algo(self, num_elements):
        dual_cr = 0
        num_gpu = self.gpu_cfg.sw_opt.mgpu_gpu_count
        num_xgmi_link_per_cluster = self.gpu_cfg.sw_opt.mgpu_xgmi_link_count
        num_cu_clusters = 2 if self.gpu_cfg.hw_cfg.chiplet_mode_en else 1  # Assumption: Dual die chiplet
        data_size = self.gpu_cfg.sw_opt.mgpu_num_grad_bits / 8
        num_gmi_links = self.gpu_cfg.hw_cfg.num_gmi_links

        if self.gpu_cfg.hw_cfg.chiplet_mode_en and not dual_cr:
            # 8 multi-die GPUs connected with 3 or 4 xgmi links per die
            assert (num_gpu == 8)
            # 1. MCM Reduce
            mcm_reduce_cc = self.get_mcm_reduce_cycles(num_elements)

            # 2. System Reduce
            # 2.1 System Scatter Reduce
            if self.gpu_cfg.sw_opt.mgpu_xgmi_link_count == 4:
                msg_size = (num_elements / num_gpu) * data_size
            else:
                assert (self.gpu_cfg.sw_opt.mgpu_xgmi_link_count == 3)
                msg_size = (num_elements / num_gpu + num_elements / (num_gpu * num_cu_clusters * 2)) * data_size
            beta_xgmi = self.beta_xgmi / self.get_link_bw_curve_factor(msg_size)
            buffer_copy_cc = self.get_buffer_copy_cycles(msg_size * (num_gpu / num_cu_clusters - 1))
            if self.gpu_cfg.sw_opt.mgpu_sdma_xfer:
                epsilon = 2 * buffer_copy_cc  # multiply by 2 corresponds to read from remote plus write in local
            else:
                epsilon = 0
            sys_scatter_cc = self.alpha + msg_size * beta_xgmi + epsilon
            if self.gpu_cfg.sw_opt.mgpu_xgmi_link_count == 4:
                sys_scatter_remainder_cc = 0
            else:
                msg_size = (num_elements / (num_gpu * num_cu_clusters * 2)) * data_size
                beta_xgmi = self.beta_xgmi / self.get_link_bw_curve_factor(msg_size)
                buffer_copy_cc = self.get_buffer_copy_cycles(msg_size)
                if self.gpu_cfg.sw_opt.mgpu_sdma_xfer:
                    epsilon_remainder = 2 * buffer_copy_cc
                else:
                    epsilon_remainder = 0
                sys_scatter_remainder_cc = self.alpha + msg_size * beta_xgmi + epsilon_remainder
            sys_reduce_cc = self.get_reduce_cycles(math.ceil(num_elements / num_gpu), math.ceil(num_elements / num_gpu),
                                                   num_gpu / num_cu_clusters)
            if self.gpu_cfg.sw_opt.mgpu_xgmi_link_count == 3:
                sys_reduce_cc += self.get_reduce_cycles(math.ceil(num_elements / (num_gpu * 2)), math.ceil(num_elements / (num_gpu * 2)), 1)

            # 2.2 MCM Reduce
            mcm_reduce_cc_1 = self.get_mcm_reduce_cycles(num_elements / num_gpu)

            # 2.3 System Gather
            msg_size = (num_elements / num_gpu) * data_size
            beta_xgmi = self.beta_xgmi / self.get_link_bw_curve_factor(msg_size)
            sys_gather_cc = self.alpha + msg_size * beta_xgmi + epsilon

            # 2.4 MCM Gather
            msg_size = (num_xgmi_link_per_cluster * num_elements / (num_gpu * num_gmi_links)) * data_size
            beta_gmi = self.beta_gmi / self.get_link_bw_curve_factor(msg_size)
            buffer_copy_cc = self.get_buffer_copy_cycles(msg_size * (num_cu_clusters - 1))
            if self.gpu_cfg.sw_opt.mgpu_sdma_xfer:
                epsilon = 2 * buffer_copy_cc
            else:
                epsilon = 0
            mcm_gather_cc_2 = self.alpha + msg_size * beta_gmi + epsilon

            gradient_aggr_cycles = mcm_reduce_cc + sys_scatter_cc + sys_scatter_remainder_cc + sys_reduce_cc + \
                                   mcm_reduce_cc_1 + sys_gather_cc + mcm_gather_cc_2
        else:
            # Step1: Scatter (num_gpu-1) chunks
            sys_scatter_cc = self.get_scatter_cc(num_elements)
            # Step2: Reduce chunks by launching kernel on subset of CUs
            sys_reduce_cc = self.get_reduce_cycles(math.ceil(num_elements / num_gpu), math.ceil(num_elements / num_gpu), num_gpu)
            # Step3: Gather chunks using SDMA engines
            sys_gather_cc = sys_scatter_cc

            gradient_aggr_cycles = sys_scatter_cc + sys_reduce_cc + sys_gather_cc
            if dual_cr:
                msg_size = (num_elements / num_gmi_links) * data_size
                beta_gmi = self.beta_gmi / self.get_link_bw_curve_factor(msg_size)
                sys_scatter_cc = self.alpha + msg_size * beta_gmi
                sys_reduce_cc = self.get_reduce_cycles(num_elements, num_elements, 1)
                gradient_aggr_cycles += (sys_scatter_cc + sys_reduce_cc)

        return gradient_aggr_cycles


class HybridMeshCube(Topology):
    def __init__(self, gpu_cfg, allreduce_cfg, alpha, mgpu_multi_node_alpha, beta_gmi, beta_xgmi, beta_mgpu_multi_node):
        super(HybridMeshCube, self).__init__(gpu_cfg, allreduce_cfg, alpha, mgpu_multi_node_alpha, beta_gmi, beta_xgmi, beta_mgpu_multi_node)

    def get_scatter_cc(self, num_elements, num_unidir_rings):
        num_cu_clusters = 2 if self.gpu_cfg.hw_cfg.chiplet_mode_en else 1  # Assumption: Dual die chiplet
        num_gpu = self.gpu_cfg.sw_opt.mgpu_gpu_count * num_cu_clusters
        data_size = self.gpu_cfg.sw_opt.mgpu_num_grad_bits / 8
        comm_steps = num_gpu - 1

        # Step1: Each WG scatters chunk to ring-connected neighbor GPU
        msg_size = (num_elements / (num_gpu * num_unidir_rings)) * data_size
        beta_xgmi = self.beta_xgmi / self.get_link_bw_curve_factor(msg_size)
        if self.gpu_cfg.sw_opt.mgpu_sdma_xfer:
            assert (0)  # SDMA transfer not feasible for HCM
        scatter_cc_per_step = self.alpha + msg_size * beta_xgmi
        scatter_cc = scatter_cc_per_step * comm_steps
        return scatter_cc

    def get_gather_cc(self, num_elements, num_unidir_rings):
        return self.get_scatter_cc(num_elements, num_unidir_rings)  # Gather phase identical to scatter phase

    def ring_algo(self, num_elements):
        num_gpu = self.gpu_cfg.sw_opt.mgpu_gpu_count
        num_cu_clusters = 2 if self.gpu_cfg.hw_cfg.chiplet_mode_en else 1  # Assumption: Dual die chiplet
        data_size = self.gpu_cfg.sw_opt.mgpu_num_grad_bits / 8
        available_onchip_mem = self.allreduce_cfg.hw_cfg.arch_vgpr_size_per_cu * 0.8 * self.allreduce_cfg.hw_cfg.num_cu_util

        if self.gpu_cfg.hw_cfg.chiplet_mode_en:
            if num_gpu == 8:
                assert ('FC' not in self.gpu_cfg.sw_opt.mgpu_topology)
                num_unidir_rings = self.gpu_cfg.sw_opt.mgpu_xgmi_link_count * num_cu_clusters
                num_gpu *= num_cu_clusters
            elif num_gpu == 4:
                num_unidir_rings = 4
                num_gpu *= num_cu_clusters
                raise NotImplementedError
        else:
            num_unidir_rings = self.gpu_cfg.sw_opt.mgpu_xgmi_link_count


        comm_steps = num_gpu - 1
        # Step1: Each WG scatters chunk to ring-connected neighbor GPU
        sys_scatter_cc = self.get_scatter_cc(num_elements, num_unidir_rings)
        sys_scatter_cc_per_step = sys_scatter_cc / comm_steps
        # Step2: Reduce chunks from neighbouring GPUs
        reduce_work_size = num_unidir_rings * (num_elements / (num_gpu * num_unidir_rings))
        sys_reduce_cc_first_step = self.get_reduce_cycles(reduce_work_size, reduce_work_size, 2)
        sys_reducescatter_cc_first_step = sys_reduce_cc_first_step + sys_scatter_cc_per_step
        sys_reduce_cc_per_step = self.get_reduce_cycles(reduce_work_size, reduce_work_size, 1)
        sys_reducescatter_cc_per_step = sys_reduce_cc_per_step + sys_scatter_cc_per_step
        if reduce_work_size * data_size < available_onchip_mem:
            sys_reducescatter_cc = sys_reducescatter_cc_first_step + sys_reducescatter_cc_per_step * (comm_steps - 1)
        else:
            sys_reducescatter_cc = sys_reducescatter_cc_first_step * comm_steps
        # Step3: AllGather
        sys_gather_cc = self.get_gather_cc(num_elements, num_unidir_rings)

        gradient_aggr_cycles = sys_reducescatter_cc + sys_gather_cc
        return gradient_aggr_cycles


class X8(Topology):
    def __init__(self, gpu_cfg, allreduce_cfg, alpha, mgpu_multi_node_alpha, beta_gmi, beta_xgmi, beta_mgpu_multi_node):
        super(X8, self).__init__(gpu_cfg, allreduce_cfg, alpha, mgpu_multi_node_alpha, beta_gmi, beta_xgmi, beta_mgpu_multi_node)

    #def get_scatter_cc(self, num_elements):

    def RHDD_algo(self, num_elements):
        num_gpu = self.gpu_cfg.sw_opt.mgpu_gpu_count
        num_cu_clusters = 2 if self.gpu_cfg.hw_cfg.chiplet_mode_en else 1  # Assumption: Dual die chiplet
        data_size = self.gpu_cfg.sw_opt.mgpu_num_grad_bits / 8
        available_onchip_mem = self.allreduce_cfg.hw_cfg.arch_vgpr_size_per_cu * 0.8 * self.allreduce_cfg.hw_cfg.num_cu_util

        if self.gpu_cfg.hw_cfg.chiplet_mode_en:
            comm_steps = int(math.log2(num_gpu * num_cu_clusters))

            # First step in scatter reduce and last step in allGather approximated with mcm_reduce function
            mcm_reduce_cc = self.get_mcm_reduce_cycles(num_elements)
            # 1. CU based system scatter reduce
            sys_reducescatter_cc = 0
            scatter_recursive_rate = 0
            for i in range(2, comm_steps + 1):
                scatter_recursive_rate = math.pow(2, i)
                msg_size = (num_elements / scatter_recursive_rate) * data_size
                beta = self.beta_xgmi / self.get_link_bw_curve_factor(msg_size)
                buffer_copy_cc = self.get_buffer_copy_cycles(msg_size)
                if self.gpu_cfg.sw_opt.mgpu_sdma_xfer:
                    epsilon = 2 * buffer_copy_cc  # multiply by 2 corresponds to read from remote plus write in local
                else:
                    epsilon = 0
                sys_scatter_cc = self.alpha + msg_size * beta + epsilon
                reduce_rd_size = math.ceil(num_elements / scatter_recursive_rate)
                num_rd_buf = 1 if reduce_rd_size * data_size < available_onchip_mem else 2
                sys_reduce_cc = self.get_reduce_cycles(reduce_rd_size, reduce_rd_size, num_rd_buf)
                sys_reducescatter_cc += (sys_scatter_cc + sys_reduce_cc)

            # 2. System gather
            sys_gather_cc = 0
            for i in range(0, comm_steps - 1):
                gather_recursive_rate = math.pow(2, i)
                msg_size = (gather_recursive_rate * num_elements / scatter_recursive_rate) * data_size
                beta = self.beta_xgmi / self.get_link_bw_curve_factor(msg_size)
                buffer_copy_cc = self.get_buffer_copy_cycles(msg_size)
                if self.gpu_cfg.sw_opt.mgpu_sdma_xfer:
                    epsilon = 2 * buffer_copy_cc  # multiply by 2 corresponds to read from remote plus write in local
                else:
                    epsilon = 0
                sys_gather_cc += self.alpha + msg_size * beta + epsilon

            gradient_aggr_cycles = sys_reducescatter_cc + sys_gather_cc + mcm_reduce_cc
            return gradient_aggr_cycles
        else:
            raise NotImplementedError


class HierarchicalQuad(Topology):
    def __init__(self, gpu_cfg, allreduce_cfg, alpha, mgpu_multi_node_alpha, beta_gmi, beta_xgmi, beta_mgpu_multi_node):
        super(HierarchicalQuad, self).__init__(gpu_cfg, allreduce_cfg, alpha, mgpu_multi_node_alpha, beta_gmi, beta_xgmi, beta_mgpu_multi_node)

    def HReduce_algo(self, num_elements):
        num_gpu = self.gpu_cfg.sw_opt.mgpu_gpu_count
        data_size = self.gpu_cfg.sw_opt.mgpu_num_grad_bits / 8

        if self.gpu_cfg.hw_cfg.chiplet_mode_en:
            assert (num_gpu == 8)  # Currently works for 8 connected MI-200
            # 1. Intra-quad reduce and gather
            quad = 4
            total_cycles_list = []
            for buffer in range(1, 4 + 1):  # Buffering
                comm_steps = buffer
                msg_size = math.ceil(num_elements / (comm_steps * quad)) * data_size
                beta_xgmi = self.beta_xgmi / self.get_link_bw_curve_factor(msg_size)
                buffer_copy_cc = self.get_buffer_copy_cycles(msg_size * (quad - 1))
                if self.gpu_cfg.sw_opt.mgpu_sdma_xfer:
                    epsilon = 2 * buffer_copy_cc  # multiply by 2 corresponds to read from remote plus write in local
                else:
                    epsilon = 0
                scatter_cc_per_step = self.alpha + msg_size * beta_xgmi + epsilon
                scatter_cc = comm_steps * scatter_cc_per_step
                reduce_cc = self.get_reduce_cycles(math.ceil(num_elements / quad), math.ceil(num_elements / quad), quad)
                if buffer == 1:
                    reducescatter_cc = scatter_cc + reduce_cc
                else:
                    reducescatter_cc = max(scatter_cc, reduce_cc) + scatter_cc_per_step
                msg_size = (num_elements / quad) * data_size
                beta_xgmi = self.beta_xgmi / self.get_link_bw_curve_factor(msg_size)
                gather_cc = self.alpha + msg_size * beta_xgmi + self.get_buffer_copy_cycles(msg_size)
                total_cycles_list.append(reducescatter_cc + gather_cc)
            min_vals = heapq.nsmallest(len(total_cycles_list), total_cycles_list)
            min_idx = total_cycles_list.index(min_vals[0])
            intra_quad_reduce1 = total_cycles_list[min_idx]

            # 2. Inter-quad reduce
            # 2.1 partial inter-quad scatter reduce
            msg_size = math.ceil(num_elements / quad) * data_size
            beta_xgmi = self.beta_xgmi / self.get_link_bw_curve_factor(msg_size)
            buffer_copy_cc = self.get_buffer_copy_cycles(msg_size)
            if self.gpu_cfg.sw_opt.mgpu_sdma_xfer:
                epsilon = 2 * buffer_copy_cc  # multiply by 2 corresponds to read from remote plus write in local
            else:
                epsilon = 0
            scatter_cc = self.alpha + msg_size * beta_xgmi + epsilon
            reduce_cc = self.get_reduce_cycles(math.ceil(num_elements / quad), math.ceil(num_elements / quad), 2)
            inter_quad_reduce = scatter_cc + reduce_cc
            # 2.2 full intra-quad scatter reduce
            reduce_cc = self.get_reduce_cycles(math.ceil(num_elements / quad), math.ceil(num_elements / quad),
                                               1)  # assuming previous steps results still in onchip memory
            intra_quad_reduce2 = scatter_cc + reduce_cc

            # 3. Inter-quad gather
            inter_quad_gather = scatter_cc
            # 4. Intra-quad broadcast
            msg_size = math.ceil(2 * num_elements / quad) * data_size
            beta_xgmi = self.beta_xgmi / self.get_link_bw_curve_factor(msg_size)
            buffer_copy_cc = self.get_buffer_copy_cycles(msg_size * (quad - 1))
            if self.gpu_cfg.sw_opt.mgpu_sdma_xfer:
                epsilon = 2 * buffer_copy_cc  # multiply by 2 corresponds to read from remote plus write in local
            else:
                epsilon = 0
            intra_quad_bcast = self.alpha + msg_size * beta_xgmi + epsilon

            gradient_aggr_cycles = intra_quad_reduce1 + inter_quad_reduce + intra_quad_reduce2 + inter_quad_gather + intra_quad_bcast
            return gradient_aggr_cycles
        else:
            raise NotImplementedError


class MGPUBackend:
    def __init__(self, gpu_cfg, allreduce_cfg):
        self.gpu_cfg = gpu_cfg
        self.allreduce_cfg = allreduce_cfg
        hw_cfg = allreduce_cfg.hw_cfg
        sw_opt = allreduce_cfg.sw_opt
        self.alpha = sw_opt.mgpu_startup_lat * 1e-6 * hw_cfg.gpu_freq * 1e9
        self.mgpu_multi_node_alpha = sw_opt.mgpu_mn_startup_lat * 1e-6 * hw_cfg.gpu_freq * 1e9
        self.beta_gmi = 1 / (hw_cfg.total_gmi_bw / hw_cfg.num_gmi_links) if hw_cfg.total_gmi_bw > 0 else 0  # cycles/B
        self.beta_xgmi = 1 / (sw_opt.total_xgmi_bw / sw_opt.mgpu_xgmi_link_count) if sw_opt.total_xgmi_bw > 0 else 0  # cycles/B
        self.beta_mgpu_multi_node = 1 / (sw_opt.total_mgpu_multi_node_bw / sw_opt.mgpu_multi_node_link_count) if sw_opt.mgpu_multi_node > 1 else 0  # cycles/B

        # Build graph for total transfer cost vs message size
        pow_of_2_range = 21  # 2M
        num_elements_array = [2**i for i in range(8, pow_of_2_range+1)]  #256 to 2M in increments of power of 2
        num_elements_array.extend([2**pow_of_2_range + i*2**20 for i in range(1, 10)])  #5M to 11M

        self.transfer_cost_curve = {}
        data_size = self.gpu_cfg.sw_opt.mgpu_num_grad_bits / 8
        for i in range(len(num_elements_array)):
            cycles = self.get_weight_update_cycles(num_elements_array[i])
            self.transfer_cost_curve.update({num_elements_array[i]*data_size:int(cycles)})
        if PLOT_TRANFER_COST_GRAPH:
            plt.subplot(121)
            plt.xlabel('transfer size in bytes')
            plt.ylabel('transfer cycles')
            plt.plot(self.transfer_cost_curve.keys(), self.transfer_cost_curve.values(), '-*')

            plt.subplot(122)
            plt.xlabel('gradient transfer size in bytes')
            plt.ylabel('gradient transfer cycles')
            gradient = np.gradient(np.array(list(self.transfer_cost_curve.values())), np.array(list(self.transfer_cost_curve.keys())))
            plt.plot(self.transfer_cost_curve.keys(), gradient, '-*')
            plt.show()
            a = 3

    def get_weight_update_cycles(self, num_elements):
        # Gradient aggregation across multiple GPUs/MCM dies
        gradient_aggr_cycles = 0
        topology = Topology(self.gpu_cfg, self.allreduce_cfg, self.alpha, self.mgpu_multi_node_alpha,
                            self.beta_gmi, self.beta_xgmi, self.beta_mgpu_multi_node)
        if self.gpu_cfg.sw_opt.multi_gpu:
            if 'CR' in self.gpu_cfg.sw_opt.mgpu_topology:  # Chordal Ring
                cr_topology = ChordalRing(self.gpu_cfg, self.allreduce_cfg, self.alpha, self.mgpu_multi_node_alpha,
                                          self.beta_gmi, self.beta_xgmi, self.beta_mgpu_multi_node)
                if 'CR8G' in self.gpu_cfg.sw_opt.mgpu_all_reduce_algo:
                    gradient_aggr_cycles = cr_topology.CR8G_algo(num_elements)
                else:
                    raise NotImplementedError
            elif self.gpu_cfg.sw_opt.mgpu_topology in ['HMC']:  # Hybrid Cube Mesh
                hmc_topology = HybridMeshCube(self.gpu_cfg, self.allreduce_cfg, self.alpha, self.mgpu_multi_node_alpha,
                                              self.beta_gmi, self.beta_xgmi, self.beta_mgpu_multi_node)
                if 'Ring' in self.gpu_cfg.sw_opt.mgpu_all_reduce_algo:
                    gradient_aggr_cycles = hmc_topology.ring_algo(num_elements)
                else:
                    raise NotImplementedError
            elif 'X8' in self.gpu_cfg.sw_opt.mgpu_topology:
                x8_topology = X8(self.gpu_cfg, self.allreduce_cfg, self.alpha, self.mgpu_multi_node_alpha,
                                 self.beta_gmi, self.beta_xgmi, self.beta_mgpu_multi_node)
                if 'RHDD' in self.gpu_cfg.sw_opt.mgpu_all_reduce_algo:
                    gradient_aggr_cycles = x8_topology.RHDD_algo(num_elements)
                else:
                    raise NotImplementedError
            elif 'HQ' in self.gpu_cfg.sw_opt.mgpu_topology:  # Hierarchical Quad
                hq_topology = HierarchicalQuad(self.gpu_cfg, self.allreduce_cfg, self.alpha, self.mgpu_multi_node_alpha,
                                               self.beta_gmi, self.beta_xgmi, self.beta_mgpu_multi_node)
                if 'H. Reduce' in self.gpu_cfg.sw_opt.mgpu_all_reduce_algo:  # Hierarchical Reduce
                    gradient_aggr_cycles = hq_topology.HReduce_algo(num_elements)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        elif self.gpu_cfg.hw_cfg.chiplet_mode_en:
            gradient_aggr_cycles = topology.get_mcm_reduce_cycles(num_elements)

        gradient_aggr_cycles += topology.get_multi_node_cycles(num_elements)
        # Gradient update
        in_dims = [1, 1, 1, num_elements]
        out_dims = [1, 1, 1, num_elements]
        layer_params = LayerInputParam(in_dims, out_dims)
        sum_layer = Sum(self.gpu_cfg)
        layer_result = sum_layer.fprop(layer_params)
        weight_update_cycles = layer_result.cycles

        return gradient_aggr_cycles + weight_update_cycles

    def get_scatter_cc(self, num_elements):
        scatter_cycles = 0
        if self.gpu_cfg.sw_opt.multi_gpu:
            if 'CR' in self.gpu_cfg.sw_opt.mgpu_topology:  # Chordal Ring
                cr_topology = ChordalRing(self.gpu_cfg, self.allreduce_cfg, self.alpha, self.mgpu_multi_node_alpha,
                                          self.beta_gmi, self.beta_xgmi, self.beta_mgpu_multi_node)
                if 'CR8G' in self.gpu_cfg.sw_opt.mgpu_all_reduce_algo:
                    scatter_cycles = cr_topology.get_scatter_cc(num_elements)
                else:
                    raise NotImplementedError
            elif self.gpu_cfg.sw_opt.mgpu_topology in ['HMC']:  # Hybrid Cube Mesh
                hmc_topology = HybridMeshCube(self.gpu_cfg, self.allreduce_cfg, self.alpha, self.mgpu_multi_node_alpha,
                                              self.beta_gmi, self.beta_xgmi, self.beta_mgpu_multi_node)
                if 'Ring' in self.gpu_cfg.sw_opt.mgpu_all_reduce_algo:
                    num_gpu = self.gpu_cfg.sw_opt.mgpu_gpu_count
                    num_cu_clusters = 2 if self.gpu_cfg.hw_cfg.chiplet_mode_en else 1  # Assumption: Dual die chiplet
                    if self.gpu_cfg.hw_cfg.chiplet_mode_en:
                        if num_gpu == 8:
                            assert ('FC' not in self.gpu_cfg.sw_opt.mgpu_topology)
                            num_unidir_rings = self.gpu_cfg.sw_opt.mgpu_xgmi_link_count * num_cu_clusters
                            num_gpu *= num_cu_clusters
                        elif num_gpu == 4:
                            num_unidir_rings = 4
                            num_gpu *= num_cu_clusters
                            raise NotImplementedError
                    else:
                        num_unidir_rings = self.gpu_cfg.sw_opt.mgpu_xgmi_link_count
                    scatter_cycles = hmc_topology.get_scatter_cc(num_elements, num_unidir_rings)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        return scatter_cycles

    def get_gather_cc(self, num_elements):
        gather_cycles = 0
        if self.gpu_cfg.sw_opt.multi_gpu:
            if 'CR' in self.gpu_cfg.sw_opt.mgpu_topology:  # Chordal Ring
                cr_topology = ChordalRing(self.gpu_cfg, self.allreduce_cfg, self.alpha, self.mgpu_multi_node_alpha,
                                          self.beta_gmi, self.beta_xgmi, self.beta_mgpu_multi_node)
                if 'CR8G' in self.gpu_cfg.sw_opt.mgpu_all_reduce_algo:
                    gather_cycles = cr_topology.get_gather_cc(num_elements)
                else:
                    raise NotImplementedError
            elif self.gpu_cfg.sw_opt.mgpu_topology in ['HMC']:  # Hybrid Cube Mesh
                hmc_topology = HybridMeshCube(self.gpu_cfg, self.allreduce_cfg, self.alpha, self.mgpu_multi_node_alpha,
                                              self.beta_gmi, self.beta_xgmi, self.beta_mgpu_multi_node)
                if 'Ring' in self.gpu_cfg.sw_opt.mgpu_all_reduce_algo:
                    num_gpu = self.gpu_cfg.sw_opt.mgpu_gpu_count
                    num_cu_clusters = 2 if self.gpu_cfg.hw_cfg.chiplet_mode_en else 1  # Assumption: Dual die chiplet
                    if self.gpu_cfg.hw_cfg.chiplet_mode_en:
                        if num_gpu == 8:
                            assert ('FC' not in self.gpu_cfg.sw_opt.mgpu_topology)
                            num_unidir_rings = self.gpu_cfg.sw_opt.mgpu_xgmi_link_count * num_cu_clusters
                            num_gpu *= num_cu_clusters
                        elif num_gpu == 4:
                            num_unidir_rings = 4
                            num_gpu *= num_cu_clusters
                            raise NotImplementedError
                    else:
                        num_unidir_rings = self.gpu_cfg.sw_opt.mgpu_xgmi_link_count
                    gather_cycles = hmc_topology.get_gather_cc(num_elements, num_unidir_rings)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        return gather_cycles
