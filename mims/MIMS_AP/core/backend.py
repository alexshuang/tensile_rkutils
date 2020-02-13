from core.results import Results
from core.mgpu_backend import MGPUBackend
import numpy as np
from core.layers import *
from core.structs.auxiliary import *
from core.auxiliary import get_act_hit_rates, print_node_info, prod, get_dt_size
from core.const import *


class MIMSBackend:
    def __init__(self, opnodes_list, gpu_cfg, allreduce_cfg, mgpu_backend):
        self.opnodes_list = opnodes_list
        self.gpu_cfg = gpu_cfg
        self.allreduce_cfg = allreduce_cfg
        self.mgpu_backend = mgpu_backend

    def modify_graph_param(self, param='batch_size', val=0):
        for op_id, op in enumerate(self.opnodes_list):
            if param == 'batch_size':
                op.inputs[IMG_IND].dims[N_IND] = val
                op.outputs[IMG_IND].dims[N_IND] = val

    def modify_node_param(self, target_node_id, comp_id=0, val=0):
        for node_id, op in enumerate(self.opnodes_list):
            if node_id == target_node_id:
                op.inputs[IMG_IND].dims[comp_id] = val
                op.outputs[IMG_IND].dims[comp_id] = val

    # In case of ML Chipet inference roughly estimate how many layers needs applying moderate/conservative policy
    def get_top_k_layers_estimate(self, sorted_layers_list, tpu_sram_stats, nw_results, vgpr_util_bytes_per_cu):
        top_k = 1
        for layer_ind in sorted_layers_list:
            ideal_weights_per_cu = nw_results[layer_ind].weightSize // self.gpu_cfg.hw_cfg.num_cu
            diff = tpu_sram_stats[0][layer_ind] - ideal_weights_per_cu
            vgpr_util_bytes_per_cu -= diff
            if vgpr_util_bytes_per_cu < self.gpu_cfg.hw_cfg.arch_vgpr_size_per_cu:
                break
            else:
                top_k += 1
        return top_k

    def eval_graph(self, direction='forward'):
        nw_results = []
        if self.gpu_cfg.sw_opt.rl:  # Modify batch size for re-inforcement learning
            if direction == 'backward':
                batch_size = self.gpu_cfg.sw_opt.rl_cpu_threads * self.gpu_cfg.sw_opt.rl_tmax
                self.modify_graph_param(param='batch_size', val=batch_size)
            else:
                batch_size = self.gpu_cfg.sw_opt.rl_cpu_threads
                self.modify_graph_param(param='batch_size', val=batch_size)

        if (self.gpu_cfg.sw_opt.multi_gpu or self.gpu_cfg.hw_cfg.chiplet_mode_en) and direction == 'backward' and not self.gpu_cfg.hw_cfg.tpu_en:
            self.gpu_cfg.hw_cfg.num_cu_util -= self.allreduce_cfg.hw_cfg.num_cu_util
            self.gpu_cfg.hw_cfg.hbm_bw -= self.allreduce_cfg.hw_cfg.hbm_bw

        node_list = self.opnodes_list if direction == 'forward' else list(reversed(self.opnodes_list))

        if self.gpu_cfg.hw_cfg.tpu_en:
            success = False
            # First try the aggresive strategy which may result into pinned weights being duplicated across XPEs
            nw_results, vgpr_util_bytes_per_cu, tpu_sram_stats, success = self.traverse_graph(node_list, direction, 'aggresive')
            if vgpr_util_bytes_per_cu > self.gpu_cfg.hw_cfg.arch_vgpr_size_per_cu:  # Try moderate partitioning policy
                # Try applying moderate policy to top K 'bad' layers;
                # moderate_lv0: Initially 'bad' layers are ones utilizing too much SRAM for weights, So first try sorting layers based on SRAM weight utilization
                sorted_layers_list1 = sorted(tpu_sram_stats[0], key=tpu_sram_stats[0].__getitem__, reverse=True)
                initial_top_k = self.get_top_k_layers_estimate(sorted_layers_list1, tpu_sram_stats, nw_results, vgpr_util_bytes_per_cu)
                for top_k in range(initial_top_k, len(sorted_layers_list1)+1):
                    moderate_policy_op_list = sorted_layers_list1[0:top_k]
                    print("Applying moderate_lv0 with top K=", top_k, "layers")
                    nw_results, vgpr_util_bytes_per_cu, tpu_sram_stats, success = \
                        self.traverse_graph(node_list, direction, 'moderate_lv0', moderate_policy_op_list)
                    if success:
                        print("TPU Inference Partition Scheme: moderate_lv0 with top K=", top_k, "layers")
                        break
                    else:  # moderate_lv1: Try partitioning activation matrix as well along with weight matrix
                        new_moderate_policy_op_list = []
                        # Sort layers based on SRAM result area utilization
                        sorted_layers_list2 = sorted(tpu_sram_stats[1], key=tpu_sram_stats[1].__getitem__, reverse=True)
                        new_moderate_policy_op_list.append(sorted_layers_list2[0])
                        new_moderate_policy_op_list += [layer for layer in moderate_policy_op_list if layer not in new_moderate_policy_op_list]
                        print("Applying moderate_lv1 with top K=", top_k, "layers")
                        nw_results, vgpr_util_bytes_per_cu, tpu_sram_stats, success = \
                            self.traverse_graph(node_list, direction, 'moderate_lv1', new_moderate_policy_op_list)
                        if not success: # moderate_lv2: Try further partitioning activation matrix along with weight matrix
                            print("Applying moderate_lv2 with top K=", top_k, "layers")
                            nw_results, vgpr_util_bytes_per_cu, tpu_sram_stats, success = \
                                self.traverse_graph(node_list, direction, 'moderate_lv2', new_moderate_policy_op_list)
                            if success:
                                print("TPU Inference Partition Scheme: moderate_lv2 with top K=", top_k, "layers")
                                break
                        else:
                            print("TPU Inference Partition Scheme: moderate_lv1 with top K=", top_k, "layers")
                            break
            else:
                print("TPU Inference Partition Scheme: Aggresive")

            if not success:
                print("Applying conservative inference partition scheme")
                nw_results, vgpr_util_bytes_per_cu, _, success = self.traverse_graph(node_list, direction, 'conservative')
                if not success:
                    assert 'None of the partition strategy worked'
                else:
                    print("TPU Inference Partition Scheme: Conservative")

        else:
            nw_results, _, _, _ = self.traverse_graph(node_list, direction)

        return nw_results

    def traverse_graph(self, node_list, direction='forward', top_tpu_partition_scheme='', moderate_policy_op_list=None):
        nw_results = []
        total_vgpr_util_bytes = 0
        curr_vgpr_util_bytes_res = 0
        tpu_sram_wt_util_dict = {}
        tpu_sram_res_util_dict = {}
        success = True
        cached_results_forward = []
        cached_results_backward = []
        for op_id, op in enumerate(node_list):
            print_node_info(op, op_id + 1, len(node_list))
            is_last_node = op_id + 1 == len(node_list)
            layer_type = op.op_type
            in_dims = copy.deepcopy(op.inputs[IMG_IND].dims if direction == 'forward' else op.outputs[IMG_IND].dims)
            out_dims = copy.deepcopy(op.outputs[IMG_IND].dims if direction == 'forward' else op.inputs[IMG_IND].dims)
            in_dims.extend(np.ones(4 - len(in_dims)))  # Force in_dims and out_dims as 4D
            out_dims.extend(np.ones(4 - len(out_dims)))

            filt_dims = np.zeros(4)
            pad_dims = np.zeros(4)
            stride_dims = np.zeros(4)
            data_size = get_dt_size(self.gpu_cfg.sw_opt)
            weights_size = prod(op.inputs[FILT_IND].dims) * data_size if layer_type == "Conv" else 0
            total_act_size = out_dims[N_IND] * out_dims[W_IND] * out_dims[H_IND] * out_dims[C_IND] * data_size
            # total_act_size = np.prod(out_dims) * data_size
            channels_or_seq_len = in_dims[C_IND]
            height_or_input_sz = in_dims[H_IND]
            width_or_hidden_sz = in_dims[W_IND]
            l2_hit_rate_wgt = 0
            l2_hit_rate_act = 0.0
            l3_hit_rate_act = 0.0
            # Estimate l2 hit rate
            if op_id == 0 or self.gpu_cfg.hw_opt.worst_case_perf:
                if len(node_list) == 1: #Ashish added
                    sw_opt = self.gpu_cfg.sw_opt #Ashish added
                    element_size = 2 * data_size if (sw_opt.fp16_inputs or sw_opt.int8_inputs or sw_opt.bf16_inputs) else data_size # and nw_results[-1].num_partitions > 1 else data_size #Ashish added
                    l2_hit_rate_act, l3_hit_rate_act = get_act_hit_rates(prod(in_dims) * element_size, weights_size, self.gpu_cfg.hw_cfg, self.gpu_cfg.sw_opt, self.gpu_cfg.hw_opt.cache_scale) #Ashish added
                else: #Ashish added
                    l2_hit_rate_act = STREAM_L2_HIT_RATE
            else:
                sw_opt = self.gpu_cfg.sw_opt
                element_size = 2 * data_size if (sw_opt.fp16_inputs or sw_opt.int8_inputs or sw_opt.bf16_inputs) and nw_results[-1].num_partitions > 1 else data_size
                l2_hit_rate_act, l3_hit_rate_act = get_act_hit_rates(prod(in_dims) * element_size, weights_size, self.gpu_cfg.hw_cfg, self.gpu_cfg.sw_opt, self.gpu_cfg.hw_opt.cache_scale)
                if op.parents == []:
                    l2_hit_rate_act, l3_hit_rate_act = 0.0, 0.0

            # TPU partition strategy
            tpu_partition_scheme = top_tpu_partition_scheme
            if 'moderate' in tpu_partition_scheme:
                if op_id not in moderate_policy_op_list:
                    tpu_partition_scheme = 'aggresive'

            if layer_type == "Conv":
                filt_dims = copy.deepcopy(op.inputs[FILT_IND].dims)
                if 'pads' not in op.attributes:
                    op.attributes['pads'] = [0, 0, 0, 0]
                group = op.attributes['group'] if 'group' in op.attributes else 1

                filt_params = FilterParam(filt_dims, [op.attributes['pads'][2], op.attributes['pads'][3]],
                                          op.attributes['strides'], group)
                pad_dims = copy.deepcopy(op.attributes['pads'])
                stride_dims = copy.deepcopy(op.attributes['strides'])

                layer_params = LayerInputParam(in_dims, out_dims, l2_hit_rate_wgt, l2_hit_rate_act, l3_hit_rate_act,
                                               last_node=is_last_node, tpu_partition_scheme=tpu_partition_scheme)
                conv = Conv(self.gpu_cfg, filt_params)

                if direction == 'forward':
                    if (op_id != 0 and sw_opt.disable_l2_optimization):# and self.gpu_cfg.hw_cfg.dl_inst_large_block[INSTR_M_IND] == 1
                            #and self.gpu_cfg.hw_cfg.dl_inst_large_block[INSTR_N_IND] == 1
                            #and self.gpu_cfg.hw_cfg.dl_inst_large_block[INSTR_K_IND] == 1
                            #and self.gpu_cfg.hw_cfg.dl_inst_large_block[INSTR_NUM_BLOCKS_IND] == 1):
                        cached_result_hit_flag = False
                        for ref_result in cached_results_forward:
                            if channels_or_seq_len == ref_result.channels_or_rnn_seq_length and width_or_hidden_sz == ref_result.width_or_rnn_hidden_sz \
                                    and height_or_input_sz == ref_result.height_or_rnn_input_sz and filt_dims[F_R_IND] == ref_result.fWidth \
                                    and filt_dims[F_S_IND] == ref_result.fHeight and filt_dims[F_K_IND] == ref_result.oChannels \
                                    and pad_dims[0] == ref_result.padW and pad_dims[1] == ref_result.padH and stride_dims[0] == ref_result.strideW \
                                    and stride_dims[1] == ref_result.strideH and out_dims[W_IND] == ref_result.oWidth and out_dims[H_IND] == ref_result.oHeight \
                                    and layer_type == ref_result.layer_type:
                                temp = copy.deepcopy(ref_result)
                                temp.source = 'layer' + str(op_id)
                                nw_results.append(temp)
                                #nw_results[len(nw_results) - 1].source = 'layer' + str(op_id)
                                cached_result_hit_flag = True
                                break
                        if cached_result_hit_flag == True:
                            continue
                    layer_result = conv.fprop(layer_params, not self.gpu_cfg.sw_opt.training)

                else:
                    layer_result = conv.bprop(layer_params)
            elif layer_type.lower() in ["batchnormalization", "lrn"]:
                layer_params = LayerInputParam(in_dims, out_dims, 0, l2_hit_rate_act, l3_hit_rate_act)
                bn = BatchNorm(self.gpu_cfg, rnn_bn=False)
                if direction == 'forward':
                    if (op_id != 0 and sw_opt.disable_l2_optimization):  # and self.gpu_cfg.hw_cfg.dl_inst_large_block[INSTR_M_IND] == 1
                        # and self.gpu_cfg.hw_cfg.dl_inst_large_block[INSTR_N_IND] == 1
                        # and self.gpu_cfg.hw_cfg.dl_inst_large_block[INSTR_K_IND] == 1
                        # and self.gpu_cfg.hw_cfg.dl_inst_large_block[INSTR_NUM_BLOCKS_IND] == 1):
                        cached_result_hit_flag = False
                        for ref_result in cached_results_forward:
                            if channels_or_seq_len == ref_result.channels_or_rnn_seq_length and width_or_hidden_sz == ref_result.width_or_rnn_hidden_sz \
                                    and height_or_input_sz == ref_result.height_or_rnn_input_sz and layer_type == ref_result.layer_type:
                                temp = copy.deepcopy(ref_result)
                                temp.source = 'layer' + str(op_id)
                                nw_results.append(temp)
                                cached_result_hit_flag = True
                                break
                        if cached_result_hit_flag == True:
                            continue
                    layer_result = bn.fprop(layer_params, not self.gpu_cfg.sw_opt.training)
                else:
                    # layer_params.in_dims[0] *= 2  # Simulates 2 reads in bprop (input x and partial derivative dy)
                    layer_result = bn.bprop(layer_params)
            elif layer_type.lower() in ["relu", "sigmoid", "tanh"]:
                layer_params = LayerInputParam(in_dims, out_dims, 0, l2_hit_rate_act, l3_hit_rate_act)
                act = Activation(self.gpu_cfg)
                if direction == 'forward':
                    if (op_id != 0 and sw_opt.disable_l2_optimization):  # and self.gpu_cfg.hw_cfg.dl_inst_large_block[INSTR_M_IND] == 1
                        # and self.gpu_cfg.hw_cfg.dl_inst_large_block[INSTR_N_IND] == 1
                        # and self.gpu_cfg.hw_cfg.dl_inst_large_block[INSTR_K_IND] == 1
                        # and self.gpu_cfg.hw_cfg.dl_inst_large_block[INSTR_NUM_BLOCKS_IND] == 1):
                        cached_result_hit_flag = False
                        for ref_result in cached_results_forward:
                            if channels_or_seq_len == ref_result.channels_or_rnn_seq_length and width_or_hidden_sz == ref_result.width_or_rnn_hidden_sz \
                                    and height_or_input_sz == ref_result.height_or_rnn_input_sz and layer_type == ref_result.layer_type:
                                temp = copy.deepcopy(ref_result)
                                temp.source = 'layer' + str(op_id)
                                nw_results.append(temp)
                                cached_result_hit_flag = True
                                break
                        if cached_result_hit_flag == True:
                            continue
                    layer_result = act.fprop(layer_params, not self.gpu_cfg.sw_opt.training)
                else:
                    layer_result = act.bprop(layer_params)
            elif layer_type.lower() in ["maxpool", "averagepool"]:
                pool_out_dims = out_dims
                if layer_type == "MaxPool" and direction == 'forward' and self.gpu_cfg.sw_opt.training:
                    pool_out_dims = 2 * out_dims  # In case of training max pool layer needs to save indices of max values
                layer_params = LayerInputParam(in_dims, pool_out_dims, 0, l2_hit_rate_act, l3_hit_rate_act)
                if 'pads' not in op.attributes:
                    op.attributes['pads'] = [0, 0, 0, 0]
                filt_params = FilterParam([1, in_dims[C_IND], op.attributes['kernel_shape'][0],
                                           op.attributes['kernel_shape'][1]],
                                          [op.attributes['pads'][2], op.attributes['pads'][3]],
                                           op.attributes['strides'])
                pool = Pooling(self.gpu_cfg, filt_params)
                if direction == 'forward':
                    layer_result = pool.fprop(layer_params, not self.gpu_cfg.sw_opt.training)
                else:
                    layer_result = pool.bprop(layer_params)
                filt_dims[F_R_IND] = copy.deepcopy(op.attributes['kernel_shape'][0])
                filt_dims[F_S_IND] = copy.deepcopy(op.attributes['kernel_shape'][1])
                pad_dims = copy.deepcopy(op.attributes['pads'])
                stride_dims = copy.deepcopy(op.attributes['strides'])
            elif layer_type.lower() in ["sum", "add"]:
                layer_params = LayerInputParam(in_dims, out_dims, 0, l2_hit_rate_act, l3_hit_rate_act)
                sum = Sum(self.gpu_cfg)
                if direction == 'forward':
                    if (op_id != 0 and sw_opt.disable_l2_optimization):  # and self.gpu_cfg.hw_cfg.dl_inst_large_block[INSTR_M_IND] == 1
                        # and self.gpu_cfg.hw_cfg.dl_inst_large_block[INSTR_N_IND] == 1
                        # and self.gpu_cfg.hw_cfg.dl_inst_large_block[INSTR_K_IND] == 1
                        # and self.gpu_cfg.hw_cfg.dl_inst_large_block[INSTR_NUM_BLOCKS_IND] == 1):
                        cached_result_hit_flag = False
                        for ref_result in cached_results_forward:
                            if channels_or_seq_len == ref_result.channels_or_rnn_seq_length and width_or_hidden_sz == ref_result.width_or_rnn_hidden_sz \
                                    and height_or_input_sz == ref_result.height_or_rnn_input_sz and layer_type == ref_result.layer_type:
                                temp = copy.deepcopy(ref_result)
                                temp.source = 'layer' + str(op_id)
                                nw_results.append(temp)
                                cached_result_hit_flag = True
                                break
                        if cached_result_hit_flag == True:
                            continue
                    layer_result = sum.fprop(layer_params, not self.gpu_cfg.sw_opt.training)
                else:
                    layer_result = LayerResults()  # Sum layer passes the gradients to both branches equally
            elif layer_type == "Gemm":
                a_trans = 0
                b_trans = 0
                if 'transA' in op.attributes:
                    a_trans = op.attributes['transA']
                if 'transB' in op.attributes:
                    b_trans = op.attributes['transB']

                m = op.inputs[0].dims[1] if a_trans else op.inputs[0].dims[0]
                n = op.inputs[1].dims[0] if b_trans else op.inputs[1].dims[1]
                k = op.inputs[0].dims[0] if a_trans else op.inputs[0].dims[1]
                _k = op.inputs[1].dims[1] if b_trans else op.inputs[1].dims[0]
                assert(k == _k)
                ############### TODO: Temporary Hack for sparseNN; Remove Later #############################
                #if n == 1024 and k == 32:
                #    l2_hit_rate_act = 0.0
                weights_size = n * k * data_size
                layer_params = LayerInputParam(in_dims, out_dims, l2_hit_rate_wgt, l2_hit_rate_act, l3_hit_rate_act,
                                               last_node=is_last_node, tpu_partition_scheme=tpu_partition_scheme)
                is_c_mat_used = True if len(op.inputs) > 2 and len(op.inputs[2].dims) > 0 else False

                if direction == 'forward':
                    gemm = Gemm(self.gpu_cfg, (m, n, k), is_c_mat_used)
                    layer_result = gemm.fprop(layer_params, not self.gpu_cfg.sw_opt.training)
                else:
                    wgrad_dims = [0] * 3
                    dgrad_dims = [0] * 3
                    wgrad_dims[GEMM_M_IND] = k
                    wgrad_dims[GEMM_N_IND] = n
                    wgrad_dims[GEMM_K_IND] = m
                    dgrad_dims[GEMM_M_IND] = m
                    dgrad_dims[GEMM_N_IND] = k
                    dgrad_dims[GEMM_K_IND] = n
                    gemm = Gemm(self.gpu_cfg, (m, n, k), is_c_mat_used, wgrad_dims=wgrad_dims, dgrad_dims=dgrad_dims)
                    layer_result = gemm.bprop(layer_params)

            elif layer_type.lower() in ['rnn', 'lstm', 'gru']:
                total_act_size = out_dims[RNN_OUT_SEQ_LEN_IND] * out_dims[RNN_OUT_BS_IND] * out_dims[RNN_OUT_HIDDEN_SZ_IND] * out_dims[RNN_OUT_NUM_DIR_IND] * data_size
                channels_or_seq_len = in_dims[RNN_IN_SEQ_LEN_IND]
                height_or_input_sz = in_dims[RNN_IN_SZ_IND]
                width_or_hidden_sz = op.inputs[RNN_IN_WT_IND].dims[RNN_IN_WT_HIDDEN_SZ_IND]
                layer_type = 'Bidirectional' + layer_type if out_dims[RNN_OUT_NUM_DIR_IND] == 2 else layer_type

                rnn = Rnn(self.gpu_cfg)
                layer_params = LayerInputParam(op, 0, l2_hit_rate_wgt, last_node=is_last_node, tpu_partition_scheme=tpu_partition_scheme)
                if direction == 'forward':
                    layer_result = rnn.fprop(layer_params, not self.gpu_cfg.sw_opt.training)
                else:
                    layer_result = rnn.bprop(layer_params)
                in_weights_size = data_size
                rc_weights_size = data_size
                for i in range(len(op.inputs[RNN_IN_WT_IND].dims)):
                    in_weights_size *= op.inputs[RNN_IN_WT_IND].dims[i]
                for i in range(len(op.inputs[RNN_RC_WT_IND].dims)):
                    rc_weights_size *= op.inputs[RNN_RC_WT_IND].dims[i]
                weights_size = in_weights_size + rc_weights_size
                if layer_type == 'Lstm':
                    total_act_size *= 4
                elif layer_type == 'Gru':
                    total_act_size *= 3

            elif layer_type == 'Attention':
                if op.attributes['type'] == 'bahdanau':  # Reference: https://arxiv.org/pdf/1409.0473.pdf
                    attention = BahdanuAttention(self.gpu_cfg)
                    layer_params = LayerInputParam(op, 0, l2_hit_rate_wgt)
                    if direction == 'forward':
                        layer_result = attention.fprop(layer_params, not self.gpu_cfg.sw_opt.training)
                    else:
                        layer_result = attention.bprop(layer_params)
                    #weights_size = hidden_weight_sz + output_weight_sz
                    #total_act_size = hidden_act_sz + output_act_sz
                elif op.attributes['type'] == 'multihead':  # Reference: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
                    attention = MultiHeadAttention(self.gpu_cfg)
                    layer_params = LayerInputParam(op, 0, l2_hit_rate_wgt, last_node=is_last_node, tpu_partition_scheme=tpu_partition_scheme)
                    if direction == 'forward':
                        layer_result = attention.fprop(layer_params, not self.gpu_cfg.sw_opt.training)
                    else:
                        layer_result = attention.bprop(layer_params)

                    dmodel = op.attributes['dmodel']
                    dk = op.attributes['dk']
                    dv = op.attributes['dv']
                    nhead = op.attributes['nhead']
                    batch_sz = op.inputs[0].dims[ATTN_BS_IND]
                    q_seq_len = op.inputs[0].dims[ATTN_SEQ_LEN_IND]
                    kv_seq_len = op.inputs[0].dims[ATTN_SEQ_LEN_IND]

                    q_in_proj_wt_size = dk * dmodel * data_size * nhead * 3
                    q_in_proj_act_size = batch_sz * q_seq_len * dmodel * data_size * nhead * 3
                    # qkv_dotprod_wt_size does not count as it created by input projections
                    qkv_dotprod_act_size = 3 * (q_seq_len * kv_seq_len * data_size * batch_sz * nhead)  # activations from 3 operations: Q*K, softmax(Q*K), QK*V
                    nhead_concat_act_size = batch_sz * q_seq_len * dv * data_size
                    q_out_proj_wt_size = dmodel * dv * data_size
                    q_out_proj_act_size = q_seq_len * batch_sz * dmodel * data_size
                    weights_size = q_in_proj_wt_size + q_out_proj_wt_size
                    total_act_size = q_in_proj_act_size + qkv_dotprod_act_size + nhead_concat_act_size + q_out_proj_act_size
                else:
                    raise NotImplementedError
            elif layer_type == "Embedding":
                if direction == 'forward':
                    lut_indices_in_dims = [out_dims[N_IND], out_dims[C_IND], op.inputs[0].dims[EMB_NUM_LUT_INDICES_IND], out_dims[H_IND]]  # [batch size, num tables, num lookup indices, embedding dim]
                    lut_indices_out_dims = [out_dims[N_IND], out_dims[C_IND], 1, out_dims[H_IND]]  # [batch size, num tables, 1, embedding dim]
                else:
                    lut_indices_out_dims = [in_dims[N_IND], in_dims[C_IND], op.inputs[0].dims[EMB_NUM_LUT_INDICES_IND], in_dims[H_IND]]  # [batch size, num tables, num lookup indices, embedding dim]
                    lut_indices_in_dims = [in_dims[N_IND], in_dims[C_IND], 1, in_dims[H_IND]]  # [batch size, num tables, 1, embedding dim]
                if op.attributes['op'] == "pooling":
                    embedding = Embedding(self.gpu_cfg)
                    layer_params = LayerInputParam(lut_indices_in_dims, lut_indices_out_dims, 0, l2_hit_rate_act=0,
                                                   l3_hit_rate_act=0)
                    if direction == 'forward':
                        layer_result = embedding.fprop(layer_params, not self.gpu_cfg.sw_opt.training)
                    else:
                        if op.attributes['backward_update'] == 'adagrad':
                            lut_indices_out_dims[2] *= 2  # emulate adagrad gradient scheme updating momentum values
                        layer_result = embedding.bprop(layer_params)
                else:
                    raise NotImplementedError
            elif layer_type == 'Communication':
                layer_params = LayerInputParam(in_dims, out_dims, 0, l2_hit_rate_act, l3_hit_rate_act)
                if op.attributes['op'] == 'all2all':
                    all2all = All2all(self.gpu_cfg, self.allreduce_cfg, self.mgpu_backend)
                    if direction == 'forward':
                        layer_result = all2all.fprop(layer_params, not self.gpu_cfg.sw_opt.training)
                    else:
                        layer_result = all2all.bprop(layer_params)
                elif op.attributes['op'] == 'allgather':
                    allgather = Allgather(self.gpu_cfg, self.allreduce_cfg, self.mgpu_backend)
                    if direction == 'forward':
                        layer_result = allgather.fprop(layer_params, not self.gpu_cfg.sw_opt.training)
                    else:
                        layer_result = allgather.bprop(layer_params)
                else:
                    raise NotImplementedError
            elif layer_type == "Softmax":
                softmax = Softmax(self.gpu_cfg)
                layer_params = LayerInputParam(in_dims, out_dims, 0, l2_hit_rate_act, l3_hit_rate_act)
                if direction == 'forward':
                    layer_result = softmax.fprop(layer_params, not self.gpu_cfg.sw_opt.training)
                else:
                    layer_result = softmax.bprop(layer_params)
            elif layer_type == "Concat":
                layer_params = LayerInputParam(in_dims, out_dims, 0, l2_hit_rate_act, l3_hit_rate_act)
                concat = Concat(self.gpu_cfg)
                if direction == 'forward':
                    layer_result = concat.fprop(layer_params, not self.gpu_cfg.sw_opt.training)
                else:
                    layer_result = concat.bprop(layer_params)
            elif layer_type == "Interaction":
                interaction = Interaction(self.gpu_cfg)
                layer_params = LayerInputParam(in_dims, out_dims, 0, l2_hit_rate_act, l3_hit_rate_act)
                if direction == 'forward':
                    layer_result = interaction.fprop(layer_params, not self.gpu_cfg.sw_opt.training)
                else:
                    layer_result = LayerResults(op_name='interaction')
            elif layer_type == "PixelShuffle":
                layer_params = LayerInputParam(in_dims, out_dims, 0, l2_hit_rate_act, l3_hit_rate_act)
                pixelShuffle = pixelShuffle(self.gpu_cfg)
                if direction == 'forward':
                    layer_result = pixelShuffle.fprop(layer_params, not self.gpu_cfg.sw_opt.training)
                else:
                    layer_result = pixelShuffle.bprop(layer_params)
            elif layer_type.lower() in ["transpose", "reshape", "flatten"]:  # currently implemented as activations
                layer_params = LayerInputParam(in_dims, out_dims, 0, l2_hit_rate_act, l3_hit_rate_act)
                mod_gpu_cfg = copy.deepcopy(self.gpu_cfg)
                mod_gpu_cfg.sw_opt.kernel_fusion = False  # Assumption is these ops cannot be fused
                act = Activation(mod_gpu_cfg)
                if direction == 'forward':
                    layer_result = act.fprop(layer_params)
                else:
                    layer_result = act.bprop(layer_params)
            elif layer_type.lower() in ["dropout", "pad"]:
                layer_result = LayerResults(op_name=layer_type)
            else:
                raise NotImplementedError

            delta_weight_transfer_cycles = 0
            allreduce_num_cu = self.allreduce_cfg.hw_cfg.num_cu_util
            if direction == 'backward' and weights_size and ((self.gpu_cfg.sw_opt.multi_gpu and not self.gpu_cfg.sw_opt.optimize_graph) or
                                                             (self.gpu_cfg.hw_cfg.chiplet_mode_en and not self.gpu_cfg.sw_opt.multi_gpu)):
                transfer_data_size = self.gpu_cfg.sw_opt.mgpu_num_grad_bits / 8
                delta_weight_transfer_cycles = self.mgpu_backend.get_weight_update_cycles(weights_size / transfer_data_size)  # input: number of allreduce buffer elements

            # For TPU/ML chiplet case check if the internal SRAMs capacity utilization is exceeded;
            # If it is, then change the partitioning strategy in the next run
            if self.gpu_cfg.hw_cfg.tpu_en and weights_size and direction == 'forward':
                #print('node: ', op_id+1, 'wt_bytes: ', layer_result.tpu_sub_res.vgpr_util_bytes_wt, 'res_bytes: ', layer_result.tpu_sub_res.vgpr_util_bytes_res)
                # Maintain dictionary for sram usage stats
                tpu_sram_wt_util_dict[op_id] = layer_result.tpu_sub_res.vgpr_util_bytes_wt
                tpu_sram_res_util_dict[op_id] = layer_result.tpu_sub_res.vgpr_util_bytes_res
                total_vgpr_util_bytes += layer_result.tpu_sub_res.vgpr_util_bytes_wt
                if curr_vgpr_util_bytes_res < layer_result.tpu_sub_res.vgpr_util_bytes_res:
                    total_vgpr_util_bytes -= curr_vgpr_util_bytes_res
                    curr_vgpr_util_bytes_res = layer_result.tpu_sub_res.vgpr_util_bytes_res
                    total_vgpr_util_bytes += curr_vgpr_util_bytes_res
                if total_vgpr_util_bytes > self.gpu_cfg.hw_cfg.arch_vgpr_size_per_cu and 'moderate' in top_tpu_partition_scheme:
                    success = False
                    break

            if layer_result.cycles == 0:
                gflops = 0
            else:
                gflops = layer_result.flop*(self.gpu_cfg.hw_cfg.gpu_freq)/layer_result.cycles

            #print(self.gpu_cfg.sw_opt.a_blocks, ' ', self.gpu_cfg.sw_opt.b_blocks, ' . . . . . . ', gflops)

            result = Results(source='layer' + str(op_id), layer_type=layer_type, batch_size=self.gpu_cfg.sw_opt.batch_size,
                             iChannels=channels_or_seq_len, iWidth=width_or_hidden_sz,
                             iHeight=height_or_input_sz, fWidth=filt_dims[F_R_IND], fHeight=filt_dims[F_S_IND],
                             padW=pad_dims[0], padH=pad_dims[1], strideW=stride_dims[0], strideH=stride_dims[1],
                             oChannels=filt_dims[F_K_IND], oWidth=out_dims[W_IND], oHeight=out_dims[H_IND],
                             weight_size=weights_size, activation_size=total_act_size,
                             M=layer_result.m, N=layer_result.n, K=layer_result.k,
                             num_a_blocks=layer_result.num_a_blocks, num_b_blocks=layer_result.num_b_blocks,
                             num_partitions=layer_result.num_partitions, num_cu_utilized=layer_result.num_cu_util,
                             alu_utilization=layer_result.alu_util_factor, chip_utilization=layer_result.chip_util_factor,
                             cycles=layer_result.cycles, gflops=gflops, dram_rd_bw=layer_result.hbm_rd_bw, dram_wr_bw=layer_result.hbm_wr_bw,
                             delta_weight_transfer_cycles=delta_weight_transfer_cycles,
                             flop=layer_result.flop, allreduce_num_cu=allreduce_num_cu if direction == 'backward' else 0,
                             tpu_partition_scheme=tpu_partition_scheme if weights_size else None,
                             alu_cc=layer_result.alu_cc, mem_cc=layer_result.mem_cc, wr_cc=layer_result.wr_cc, num_rounds=layer_result.num_rounds,
                             total_blocks=layer_result.total_blocks, num_cu_util_trail=layer_result.num_cu_util_trail,
                             num_a_blocks_trail=layer_result.num_a_blocks_trail, num_b_blocks_trail=layer_result.num_b_blocks_trail,
                             num_partitions_trail=layer_result.num_partitions_trail, cycles_trail=layer_result.cycles_trail,
                             main_instr=layer_result.main_instr, threadTile=layer_result.threadTile, workGroup=layer_result.workGroup,
                             num_rounds_trail=layer_result.num_rounds_trail, unroll_factor=layer_result.unroll_factor,
                             unroll_factor_trail=layer_result.unroll_factor_trail, alu_cc_trail=layer_result.alu_cc_trail,
                             mem_cc_trail=layer_result.mem_cc_trail, wr_cc_trail=layer_result.wr_cc_trail)  # Ashish added

            nw_results.append(result)
            if(direction == 'forward'):
                cached_results_forward.append(result)
            else:
                cached_results_backward.append(result)
            if len(layer_result.sub_results) > 0:
                for i in range(len(layer_result.sub_results)):  # sub_results a.k.a e.g. dgrad and wgrad
                    result = Results(source='sub_layer' + str(op_id), layer_type=layer_result.sub_results[i].op_name,
                                     batch_size=self.gpu_cfg.sw_opt.batch_size,
                                     iChannels=channels_or_seq_len, iWidth=width_or_hidden_sz,
                                     iHeight=height_or_input_sz, fWidth=filt_dims[F_R_IND],
                                     fHeight=filt_dims[F_S_IND],
                                     padW=pad_dims[0], padH=pad_dims[1], strideW=stride_dims[0],
                                     strideH=stride_dims[1],
                                     oChannels=filt_dims[F_K_IND], oWidth=out_dims[W_IND], oHeight=out_dims[H_IND],
                                     weight_size=weights_size, activation_size=total_act_size,
                                     M=layer_result.sub_results[i].m, N=layer_result.sub_results[i].n, K=layer_result.sub_results[i].k,
                                     num_a_blocks=layer_result.sub_results[i].num_a_blocks,
                                     num_b_blocks=layer_result.sub_results[i].num_b_blocks,
                                     num_partitions=layer_result.sub_results[i].num_partitions,
                                     num_cu_utilized=layer_result.sub_results[i].num_cu_util,
                                     alu_utilization=layer_result.sub_results[i].alu_util_factor,
                                     chip_utilization=layer_result.sub_results[i].chip_util_factor,
                                     cycles=layer_result.sub_results[i].cycles,
                                     dram_rd_bw=layer_result.sub_results[i].hbm_rd_bw, dram_wr_bw=layer_result.sub_results[i].hbm_wr_bw,
                                     delta_weight_transfer_cycles=delta_weight_transfer_cycles,
                                     flop=layer_result.sub_results[i].flop)
                    nw_results.append(result)

        tpu_sram_stats = [tpu_sram_wt_util_dict, tpu_sram_res_util_dict]
        return nw_results, total_vgpr_util_bytes, tpu_sram_stats, success

