import math
import copy


def get_last_layer_ind(results):
    layer_ind = 0
    for layer_ind, res in enumerate(list(reversed(results))):
        if 'sub' not in res.source:
            break
    last_layer_ind = len(results) - layer_ind - 1
    return last_layer_ind


class GraphResultOptimizer:
    def __init__(self, fwd_nw_results, bwd_nw_results, sys_cfg, mgpu_backend):
        self.fwd_nw_results = fwd_nw_results
        self.bwd_nw_results = bwd_nw_results
        self.sys_cfg = sys_cfg
        self.mgpu_backend = mgpu_backend

    def clear_transfer_fields(self, exec_list, bwd_nw_results):
        exec_ind = 0
        for layer_ind, layr in enumerate(bwd_nw_results):
            if layer_ind == exec_list[exec_ind]:
                exec_ind += 1
                layr.delta_weight_transfer_cycles = 0
                layr.transfer_cycles_exposed = 0

    # Prepare execution unit lists on which optimum operation scheduling would be applied;
    # Layers like 'Communication' create auto sync points which breaks graph into execution lists
    def get_node_exec_lists(self):
        exec_lists = []
        sub_list = []
        for layer_ind, layr in enumerate(self.bwd_nw_results):
            if layr.layer_type in ['Communication', 'Embedding']:
                if len(sub_list):
                    exec_lists.append(sub_list)
                sub_list = []
                exec_lists.append([layer_ind])
            else:
                sub_list.append(layer_ind)
        if len(sub_list):
            exec_lists.append(sub_list)
        return exec_lists

    # get cycles for the gang scheduled nodes in the execution list
    def get_exec_list_cycles(self, exec_list, gang_sch_sizes, bwd_nw_results):
        total_cycles = 0
        overlap_bwd_comp_cycles = 0
        delta_wt_transfer_cycles = 0
        total_transfer_cycles_exposed = 0
        gang_sch_ind = 0
        gang_sch_size_ind = 0
        gang_weight_size = 0
        compute_cycles = 0
        last_layer_ind = get_last_layer_ind(self.bwd_nw_results)
        data_size = self.sys_cfg.sw_opt.mgpu_num_grad_bits / 8
        exec_ind = 0
        self.clear_transfer_fields(exec_list, bwd_nw_results)

        for layer_ind, layr in enumerate(bwd_nw_results):
            if layer_ind == exec_list[exec_ind]:
                exec_ind += 1
                layr_cycles = 0
                if delta_wt_transfer_cycles and 'sub' not in layr.source:
                    overlap_bwd_comp_cycles += layr.cycles
                else:
                    layr_cycles = layr.cycles
                if layr.layer_type in ['Gemm', 'Conv', 'Rnn', 'Attention', 'Communication', 'Embedding'] and \
                                        not self.sys_cfg.sw_opt.rl and layer_ind:
                    gang_sch_ind += 1
                    compute_cycles += layr.cycles
                    if gang_sch_ind >= gang_sch_sizes[gang_sch_size_ind]:
                        compute_cycles = max(compute_cycles, overlap_bwd_comp_cycles)
                        layr_cycles = max(compute_cycles, delta_wt_transfer_cycles)
                        if delta_wt_transfer_cycles > overlap_bwd_comp_cycles:
                            transfer_cycles_exposed = delta_wt_transfer_cycles - overlap_bwd_comp_cycles
                        else:
                            transfer_cycles_exposed = 0
                        delta_wt_transfer_cycles = 0
                        overlap_bwd_comp_cycles = 0
                        compute_cycles = 0
                        layr.transfer_cycles_exposed = transfer_cycles_exposed
                    else:
                        layr_cycles = 0
                        transfer_cycles_exposed = 0
                    if layer_ind == last_layer_ind:
                        layr_cycles += layr.delta_weight_transfer_cycles
                    else:
                        total_transfer_cycles_exposed += transfer_cycles_exposed
                if 'sub' not in layr.source:
                    if layr.weightSize:
                        gang_weight_size += layr.weightSize
                        layr.delta_weight_transfer_cycles = 0
                        if gang_sch_ind >= gang_sch_sizes[gang_sch_size_ind]:
                            delta_wt_transfer_cycles = self.mgpu_backend.get_weight_update_cycles(gang_weight_size / data_size)
                            gang_sch_ind = 0
                            gang_weight_size = 0
                            gang_sch_size_ind += 1
                            layr.delta_weight_transfer_cycles = delta_wt_transfer_cycles
                    total_cycles += layr_cycles
        total_cycles += delta_wt_transfer_cycles
        total_transfer_cycles_exposed += delta_wt_transfer_cycles

        return total_cycles, total_transfer_cycles_exposed

    def get_num_transfer_nodes(self, exec_list):
        exec_ind = 0
        num_wt_nodes = 0
        for layer_ind, layr in enumerate(self.bwd_nw_results):
            if layer_ind == exec_list[exec_ind]:
                exec_ind += 1
                if (layr.weightSize and 'sub' not in layr.source) or layr.layer_type in ['Communication', 'Embedding']:
                    num_wt_nodes = num_wt_nodes + 1
        return num_wt_nodes

    # Run different layer scheduling strategies on the execution lists with the goal to maximise performance within the list
    def get_optimum_cycles(self, exec_lists):
        total_cycles = 0
        total_transfer_cycles_exposed = 0
        for exec_list in exec_lists:
            exec_list.append(-1)  # Marker to specify end of list
            exec_list_cycles = 0
            transfer_cycles_exposed = 0
            num_wt_nodes = self.get_num_transfer_nodes(exec_list)
            bwd_nw_results_copy = copy.deepcopy(self.bwd_nw_results)
            # Gang multiple nodes together to increase transfer buffer size
            for gang_sch_size in range(1, int(math.ceil(num_wt_nodes/2))+1):
                gang_sch_sizes = [gang_sch_size] * int(num_wt_nodes // gang_sch_size)
                gang_sch_sizes[-1] += num_wt_nodes % gang_sch_size
                curr_list_cycles, curr_transfer_cycles_exposed = self.get_exec_list_cycles(exec_list, gang_sch_sizes, bwd_nw_results_copy)
                if not exec_list_cycles or curr_list_cycles < exec_list_cycles:
                    exec_list_cycles = curr_list_cycles
                    transfer_cycles_exposed = curr_transfer_cycles_exposed
                    self.bwd_nw_results = copy.deepcopy(bwd_nw_results_copy)
            total_cycles += exec_list_cycles
            total_transfer_cycles_exposed += transfer_cycles_exposed
        return total_cycles, total_transfer_cycles_exposed

    def get_total_nn_cycles(self):
        fwd_cycles = 0
        bwd_cycles = 0
        transfer_cycles_exposed = 0
        # Forward network results
        for layr in self.fwd_nw_results:
            if 'sub' not in layr.source:
                fwd_cycles += getattr(layr, 'cycles')

        # Backward network results
        # Apply different layer scheduling schemes to best hide the transfer cycles cost and maximise overall performance
        if self.sys_cfg.sw_opt.training:
            exec_lists = self.get_node_exec_lists()
            bwd_cycles, transfer_cycles_exposed = self.get_optimum_cycles(exec_lists)

        return fwd_cycles, bwd_cycles, transfer_cycles_exposed, self.fwd_nw_results, self.bwd_nw_results


# class GraphInferenceOptimizer:
#    def __init__(self, ):