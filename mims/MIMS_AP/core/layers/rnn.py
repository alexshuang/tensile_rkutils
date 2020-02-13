from core.layers.layer import Layer, get_compute_flops
from core.layers.activation import Activation
from core.layers.batchnorm import BatchNorm
from core.structs.auxiliary import LayerResults, LayerInputParam, TpuSubResults
from core.const import *
from core.auxiliary import get_act_hit_rates, get_dt_size
import copy


class Rnn(Layer):
    def __init__(self, sys_config):
        super(Rnn, self).__init__(sys_config)

    def fprop(self, inputs, inference=False):
        hw_cfg = self.sys_cfg.hw_cfg
        sw_opt = self.sys_cfg.sw_opt
        # For RNN the original operation struct 'op' is overloaded on top of input.in_dims
        op = copy.deepcopy(inputs.in_dims)
        in_size = get_dt_size(sw_opt)
        total_act_size = op.inputs[IMG_IND].dims[RNN_IN_SEQ_LEN_IND] * op.inputs[IMG_IND].dims[RNN_IN_BS_IND] * \
                         op.inputs[IMG_IND].dims[RNN_IN_SZ_IND] * in_size
        in_filt_dims = copy.deepcopy(op.inputs[RNN_IN_WT_IND].dims)
        rc_filt_dims = copy.deepcopy(op.inputs[RNN_RC_WT_IND].dims)
        num_dir = copy.deepcopy(in_filt_dims[RNN_IN_WT_NUM_DIR_IND])
        ts = copy.deepcopy(op.inputs[RNN_IN_SEQ_IND].dims[RNN_IN_SEQ_LEN_IND])
        in_weights_size = in_size
        rc_weights_size = in_size
        for i in range(len(in_filt_dims)):
            in_weights_size *= in_filt_dims[i]
        for i in range(len(rc_filt_dims)):
            rc_weights_size *= rc_filt_dims[i]
        # Perform input weight GEMM; Assumption is all timesteps are combined together in one big GEMM
        m = copy.deepcopy(op.inputs[RNN_IN_WT_IND].dims[RNN_IN_WT_HIDDEN_SZ_IND])
        k = copy.deepcopy(op.inputs[RNN_IN_WT_IND].dims[RNN_IN_WT_SZ_IND])
        n = copy.deepcopy(op.inputs[RNN_IN_SEQ_IND].dims[RNN_IN_BS_IND] * ts)
        # in_l2_hit_rate_act = (self.sys_config.hw_cfg.l2_size - in_weights_size*in_size) / total_act_size
        in_l2_hit_rate_act, in_l3_hit_rate_act = get_act_hit_rates(total_act_size, in_weights_size, hw_cfg, sw_opt)
        is_c_mat_used = True if len(op.inputs) > 3 and len(op.inputs[3].dims) > 0 else False
        gemm_res = self.perform_gemm(m, n, k, in_l2_hit_rate_act, inputs.l2_hit_rate_wt, in_l3_hit_rate_act, act='B',
                                     is_c_mat_used=is_c_mat_used, tpu_partition_scheme=inputs.tpu_partition_scheme)
        flop = 2 * m * n * k * num_dir
        in_layer_result = LayerResults(alu_util_factor=gemm_res.alu_util_factor, chip_util_factor=gemm_res.chip_util_factor,
                                       speedup=gemm_res.speedup, cycles=gemm_res.cycles * num_dir, flop=flop,
                                       num_cu_util=gemm_res.num_cu_util, m=m, n=n, k=k,
                                       num_a_blocks=gemm_res.num_a_blocks, num_b_blocks=gemm_res.num_b_blocks,
                                       num_partitions=gemm_res.num_partitions, op_name='input_gemm', tpu_sub_res=gemm_res.tpu_sub_res,
                                       alu_cc=gemm_res.alu_cc, mem_cc=gemm_res.mem_cc, num_rounds=gemm_res.num_rounds,# Ashish added
                                       total_blocks=gemm_res.total_blocks, num_cu_util_trail=gemm_res.num_cu_util_trail,# Ashish added
                                       num_a_blocks_trail=gemm_res.num_a_blocks_trail, #Ashish added
                                       num_b_blocks_trail=gemm_res.num_b_blocks_trail,  # Ashish added
                                       num_partitions_trail=gemm_res.num_partitions_trail, cycles_trail=gemm_res.cycles_trail,
                                       wr_cc=gemm_res.wr_cc)  # Ashish added

        # Perform recurrent weight GEMM
        m = copy.deepcopy(op.inputs[RNN_RC_WT_IND].dims[RNN_RC_WT_HIDDEN_SZ_IND_1])
        k = copy.deepcopy(op.inputs[RNN_RC_WT_IND].dims[RNN_RC_WT_HIDDEN_SZ_IND_2])
        n = copy.deepcopy(op.inputs[RNN_IN_SEQ_IND].dims[RNN_IN_BS_IND])
        rc_out_dims = 4 * [0]
        rc_out_dims[0] = 1
        rc_out_dims[1] = 1
        rc_out_dims[2] = copy.deepcopy(op.inputs[RNN_RC_WT_IND].dims[RNN_RC_WT_HIDDEN_SZ_IND_1])
        rc_out_dims[3] = copy.deepcopy(op.inputs[RNN_IN_SEQ_IND].dims[RNN_IN_BS_IND])
        # rc_l2_hit_rate_act = min((self.sys_config.hw_cfg.l2_size - rc_weights_size * in_size) / (n * k * in_size), 1)
        rc_l2_hit_rate_act, rc_l3_hit_rate_act = get_act_hit_rates(n * k * in_size, rc_weights_size, hw_cfg, sw_opt)
        l2_hit_rate_wt = min((hw_cfg.l2_size - n * k * in_size) / (m * k * in_size), 1)
        gemm_res = self.perform_gemm(m, n, k, rc_l2_hit_rate_act, l2_hit_rate_wt, rc_l3_hit_rate_act, act='B',
                                     is_c_mat_used=is_c_mat_used, prnn_opt_en=True, tpu_partition_scheme=inputs.tpu_partition_scheme)
        flop = 2 * m * n * k * ts * num_dir
        rc_layer_result = LayerResults(alu_util_factor=gemm_res.alu_util_factor, chip_util_factor=gemm_res.chip_util_factor,
                                       speedup=gemm_res.speedup, cycles=gemm_res.cycles * num_dir * ts, flop=flop,
                                       num_cu_util=gemm_res.num_cu_util, m=m, n=n, k=k,
                                       num_a_blocks=gemm_res.num_a_blocks, num_b_blocks=gemm_res.num_b_blocks,
                                       num_partitions=gemm_res.num_partitions, op_name='recurrent_gemm',
                                       tpu_sub_res=gemm_res.tpu_sub_res,
                                       alu_cc=gemm_res.alu_cc, mem_cc=gemm_res.mem_cc, num_rounds=gemm_res.num_rounds,  # Ashish added
                                       total_blocks=gemm_res.total_blocks, num_cu_util_trail=gemm_res.num_cu_util_trail,  # Ashish added
                                       num_a_blocks_trail=gemm_res.num_a_blocks_trail, num_b_blocks_trail=gemm_res.num_b_blocks_trail,  # Ashish added
                                       num_partitions_trail=gemm_res.num_partitions_trail, cycles_trail=gemm_res.cycles_trail,
                                       wr_cc=gemm_res.wr_cc)  # Ashish added )

        # Perform Relu and batch norm if specified
        act_layer_result = LayerResults()
        bn_layer_result = LayerResults()
        if 'activations' in op.attributes:
            dims = 4 * [0]
            dims[N_IND] = copy.deepcopy(op.inputs[RNN_IN_SEQ_IND].dims[RNN_IN_BS_IND])
            dims[C_IND] = 1
            dims[H_IND] = copy.deepcopy(op.inputs[RNN_RC_WT_IND].dims[RNN_RC_WT_HIDDEN_SZ_IND_1])
            dims[W_IND] = 1
            layer_params = LayerInputParam(dims, rc_out_dims, 0, rc_l2_hit_rate_act, rc_l3_hit_rate_act)
            act = Activation(self.sys_cfg)
            act_layer_result = act.fprop(layer_params, not self.sys_cfg.sw_opt.training)
            act_layer_result.cycles *= (ts * num_dir)
            act_layer_result.flop *= (ts * num_dir)
            act_layer_result.op_name = 'activation'
        if 'batchnorm' in op.attributes:
            # Assumption is batchnorm is performed on the input GEMM result only
            dims = 4 * [0]
            dims[N_IND] = copy.deepcopy(op.inputs[RNN_IN_SEQ_IND].dims[RNN_IN_BS_IND] * ts)
            dims[C_IND] = 1
            dims[H_IND] = copy.deepcopy(op.inputs[RNN_RC_WT_IND].dims[RNN_RC_WT_HIDDEN_SZ_IND_1])
            dims[W_IND] = 1
            layer_params = LayerInputParam(dims, rc_out_dims, 0, in_l2_hit_rate_act, in_l3_hit_rate_act)
            bn = BatchNorm(self.sys_cfg)
            bn_layer_result = bn.fprop(layer_params, not self.sys_cfg.sw_opt.training)
            bn_layer_result.cycles *= num_dir
            bn_layer_result.flop *= num_dir
            bn_layer_result.op_name = 'batch_norm'

        # Total cycles
        cycles = in_layer_result.cycles + rc_layer_result.cycles + act_layer_result.cycles + bn_layer_result.cycles
        flop = in_layer_result.flop + rc_layer_result.flop + act_layer_result.flop + bn_layer_result.flop
        # Average ALU utilization factor
        alu_util_factor = ((in_layer_result.cycles * in_layer_result.alu_util_factor) +
                           (rc_layer_result.cycles * rc_layer_result.alu_util_factor) +
                           (act_layer_result.cycles * act_layer_result.alu_util_factor) +
                           (bn_layer_result.cycles * bn_layer_result.alu_util_factor)) / cycles
        chip_util_factor = ((in_layer_result.cycles * in_layer_result.chip_util_factor) +
                           (rc_layer_result.cycles * rc_layer_result.chip_util_factor) +
                           (act_layer_result.cycles * act_layer_result.chip_util_factor) +
                           (bn_layer_result.cycles * bn_layer_result.chip_util_factor)) / cycles
        speedup = 1  # TODO: Compute later
        if self.sys_cfg.hw_cfg.tpu_en:
            vgpr_util_bytes_wt = in_layer_result.tpu_sub_res.vgpr_util_bytes_wt * num_dir + \
                                 rc_layer_result.tpu_sub_res.vgpr_util_bytes_wt * num_dir
            vgpr_util_bytes_res = max(in_layer_result.tpu_sub_res.vgpr_util_bytes_res,
                                      rc_layer_result.tpu_sub_res.vgpr_util_bytes_res)
            tpu_sub_res = TpuSubResults(vgpr_util_bytes_wt, vgpr_util_bytes_res)
        else:
            tpu_sub_res = TpuSubResults()
        layer_result = LayerResults(alu_util_factor=alu_util_factor, chip_util_factor=chip_util_factor, speedup=speedup,
                                    cycles=cycles, flop=flop, num_cu_util=rc_layer_result.num_cu_util,
                                    m=rc_layer_result.m, n=rc_layer_result.n, k=rc_layer_result.k,
                                    num_a_blocks=rc_layer_result.num_a_blocks, num_b_blocks=rc_layer_result.num_b_blocks,
                                    num_partitions=rc_layer_result.num_partitions, op_name='Rnn', tpu_sub_res=tpu_sub_res)
        layer_result.populate_sub_results(in_layer_result)
        layer_result.populate_sub_results(rc_layer_result)
        if 'activations' in op.attributes:
            layer_result.populate_sub_results(act_layer_result)
        if 'batchnorm' in op.attributes:
            layer_result.populate_sub_results(bn_layer_result)

        return layer_result

    def bprop(self, inputs):
        hw_cfg = self.sys_cfg.hw_cfg
        sw_opt = self.sys_cfg.sw_opt
        op = copy.deepcopy(inputs.in_dims)
        in_size = get_dt_size(sw_opt)
        num_dir = copy.deepcopy(op.inputs[RNN_IN_WT_IND].dims[RNN_IN_WT_NUM_DIR_IND])
        ts = copy.deepcopy(op.inputs[RNN_IN_SEQ_IND].dims[RNN_IN_SEQ_LEN_IND])

        # Compute dL/dh(t-1) GEMM
        m = copy.deepcopy(op.inputs[RNN_RC_WT_IND].dims[RNN_RC_WT_HIDDEN_SZ_IND_1])
        k = copy.deepcopy(op.inputs[RNN_RC_WT_IND].dims[RNN_RC_WT_HIDDEN_SZ_IND_2])
        n = copy.deepcopy(op.inputs[RNN_IN_SEQ_IND].dims[RNN_IN_BS_IND])
        l2_hit_rate_act = 0  # activations here are pre-activation hidden states from forward pass
        l3_hit_rate_act = 0
        l2_hit_rate_wt = min((hw_cfg.l2_size - n * k * in_size) / (m * k * in_size), 1)
        dh_gemm_res = self.perform_gemm(m, n, k, l2_hit_rate_act, l2_hit_rate_wt, l3_hit_rate_act, act='B')

        dh_flop = 2 * m * n * k * num_dir * ts
        dh_gemm_res.cycles *= num_dir * ts
        dh_layer_result = LayerResults(alu_util_factor=dh_gemm_res.alu_util_factor, chip_util_factor=dh_gemm_res.chip_util_factor,
                                       speedup=dh_gemm_res.speedup, cycles=dh_gemm_res.cycles, flop=dh_flop,
                                       num_cu_util=dh_gemm_res.num_cu_util, m=m, n=n, k=k,
                                       num_a_blocks=dh_gemm_res.num_a_blocks, num_b_blocks=dh_gemm_res.num_b_blocks,
                                       num_partitions=dh_gemm_res.num_partitions, op_name='dh_gemm',
                                       alu_cc=dh_gemm_res.alu_cc, mem_cc=dh_gemm_res.mem_cc, num_rounds=dh_gemm_res.num_rounds,# Ashish added
                                       total_blocks=dh_gemm_res.total_blocks, num_cu_util_trail=dh_gemm_res.num_cu_util_trail,# Ashish added
                                       num_a_blocks_trail=dh_gemm_res.num_a_blocks_trail, #Ashish added
                                       num_b_blocks_trail=dh_gemm_res.num_b_blocks_trail,  # Ashish added
                                       num_partitions_trail=dh_gemm_res.num_partitions_trail, cycles_trail=dh_gemm_res.cycles_trail,
                                       wr_cc=dh_gemm_res.wr_cc)  # Ashish added

        # Computing deactive(st) = 1 - st**2
        ops = 2
        m = copy.deepcopy(op.inputs[RNN_RC_WT_IND].dims[RNN_RC_WT_HIDDEN_SZ_IND_1])
        n = copy.deepcopy(op.inputs[RNN_IN_SEQ_IND].dims[RNN_IN_BS_IND])
        l2_hit_rate_act, l3_hit_rate_act = get_act_hit_rates(activation_size=m * n * in_size, weights_size=0,
                                                              hw_config=hw_cfg, sw_opt=sw_opt)
        alu_cc = self.get_alu_cc([m, n, 1, 1], ops)
        mem_cc = sum(self.get_mem_cc([m, n, 1, 1], [m, n, 1, 1], l2_hit_rate_act, l3_hit_rate_act=l3_hit_rate_act))
        ds_cycles = max(alu_cc, mem_cc)
        ds_alu_util_factor = (alu_cc / ds_cycles) * 100
        speedup = 1
        ds_flop = get_compute_flops([m, n, 1, 1], ops) * num_dir * ts
        ds_cycles *= num_dir * ts
        ds_layer_result = LayerResults(alu_util_factor=ds_alu_util_factor, chip_util_factor=ds_alu_util_factor, speedup=speedup,
                                       cycles=ds_cycles, flop=ds_flop, num_cu_util=self.sys_cfg.hw_cfg.num_cu, op_name='deactiv_st')

        # Compute dL/dWrecur GEMM
        m = copy.deepcopy(op.inputs[RNN_RC_WT_IND].dims[RNN_RC_WT_HIDDEN_SZ_IND_1])
        k = copy.deepcopy(op.inputs[RNN_IN_SEQ_IND].dims[RNN_IN_BS_IND])
        n = copy.deepcopy(op.inputs[RNN_RC_WT_IND].dims[RNN_RC_WT_HIDDEN_SZ_IND_2])

        l2_hit_rate_act = 0  # activations here are dL/dh(t-1) during previous steps
        l3_hit_rate_act = 0
        l2_hit_rate_wt = 0  # weights here are post activation hidden states saved from forward pass
        dwrecur_gemm_res = self.perform_gemm(m, n, k, l2_hit_rate_act, l2_hit_rate_wt, l3_hit_rate_act, act='A')
        dwrecur_flop = 2 * m * n * k * num_dir * ts
        dwrecur_gemm_res.cycles *= num_dir * ts
        dwrecur_layer_result = LayerResults(alu_util_factor=dwrecur_gemm_res.alu_util_factor, chip_util_factor=dwrecur_gemm_res.chip_util_factor,
                                            speedup=dwrecur_gemm_res.speedup, cycles=dwrecur_gemm_res.cycles,
                                            flop=dwrecur_flop, num_cu_util=dwrecur_gemm_res.num_cu_util, m=m, n=n, k=k,
                                            num_a_blocks=dwrecur_gemm_res.num_a_blocks, num_b_blocks=dwrecur_gemm_res.num_b_blocks,
                                            num_partitions=dwrecur_gemm_res.num_partitions, op_name='dWrecur_gemm',
                                            alu_cc=dwrecur_gemm_res.alu_cc, mem_cc=dwrecur_gemm_res.mem_cc,
                                            num_rounds=dwrecur_gemm_res.num_rounds,  # Ashish added
                                            total_blocks=dwrecur_gemm_res.total_blocks,
                                            num_cu_util_trail=dwrecur_gemm_res.num_cu_util_trail,  # Ashish added
                                            num_a_blocks_trail=dwrecur_gemm_res.num_a_blocks_trail,  # Ashish added
                                            num_b_blocks_trail=dwrecur_gemm_res.num_b_blocks_trail,  # Ashish added
                                            num_partitions_trail=dwrecur_gemm_res.num_partitions_trail, #Ashish added
                                            cycles_trail=dwrecur_gemm_res.cycles_trail,
                                            wr_cc=dwrecur_gemm_res.wr_cc)  # Ashish added

        # Compute dL/dWinput GEMM
        if not inputs.last_node:
            m = copy.deepcopy(op.inputs[RNN_RC_WT_IND].dims[RNN_RC_WT_HIDDEN_SZ_IND_1])
            k = copy.deepcopy(op.inputs[RNN_IN_SEQ_IND].dims[RNN_IN_BS_IND]) * ts
            n = copy.deepcopy(op.inputs[RNN_IN_WT_IND].dims[RNN_IN_WT_SZ_IND])

            l2_hit_rate_act = 0  # activations here are dL/dh(t-1) during previous steps
            l3_hit_rate_act = 0
            l2_hit_rate_wt = 0  # weights here are inputs at all timesteps saved from forward pass
            dwin_gemm_res = self.perform_gemm(m, n, k, l2_hit_rate_act, l2_hit_rate_wt, l3_hit_rate_act, act='A')
            dwin_flop = 2 * m * n * k * num_dir
            dwin_gemm_res.cycles *= num_dir
            dwin_layer_result = LayerResults(alu_util_factor=dwin_gemm_res.alu_util_factor, chip_util_factor=dwin_gemm_res.chip_util_factor,
                                             speedup=dwin_gemm_res.speedup, cycles=dwin_gemm_res.cycles,
                                             flop=dwin_flop, num_cu_util=dwin_gemm_res.num_cu_util, m=m, n=n, k=k,
                                             num_a_blocks=dwin_gemm_res.num_a_blocks, num_b_blocks=dwin_gemm_res.num_b_blocks,
                                             num_partitions=dwin_gemm_res.num_partitions, op_name='dWinput_gemm',
                                             alu_cc=dwin_gemm_res.alu_cc, mem_cc=dwin_gemm_res.mem_cc,
                                             num_rounds=dwin_gemm_res.num_rounds,  # Ashish added
                                             total_blocks=dwin_gemm_res.total_blocks,
                                             num_cu_util_trail=dwin_gemm_res.num_cu_util_trail,  # Ashish added
                                             num_a_blocks_trail=dwin_gemm_res.num_a_blocks_trail,  # Ashish added
                                             num_b_blocks_trail=dwin_gemm_res.num_b_blocks_trail,  # Ashish added
                                             num_partitions_trail=dwin_gemm_res.num_partitions_trail, #Ashish added
                                             cycles_trail=dwin_gemm_res.cycles_trail,
                                             wr_cc=dwin_gemm_res.wr_cc)  # Ashish added
        else:
            dwin_layer_result = LayerResults()
            dwin_gemm_res = LayerResults()
            dwin_flop = 0

        cycles = dwrecur_gemm_res.cycles + dwin_gemm_res.cycles + dh_gemm_res.cycles + ds_cycles
        flop = dwrecur_flop + dwin_flop + dh_flop + ds_flop
        alu_util_factor = ((dwrecur_gemm_res.cycles * dwrecur_gemm_res.alu_util_factor) +
                           (dwin_gemm_res.cycles * dwin_gemm_res.alu_util_factor) +
                           (dh_gemm_res.cycles * dh_gemm_res.alu_util_factor) +
                           (ds_cycles * ds_alu_util_factor)) / cycles
        chip_util_factor = ((dwrecur_gemm_res.cycles * dwrecur_gemm_res.chip_util_factor) +
                           (dwin_gemm_res.cycles * dwin_gemm_res.chip_util_factor) +
                           (dh_gemm_res.cycles * dh_gemm_res.chip_util_factor) +
                           (ds_cycles * ds_alu_util_factor)) / cycles
        speedup = 1  # TODO:Compute later
        layer_result = LayerResults(alu_util_factor=alu_util_factor, chip_util_factor=chip_util_factor, speedup=speedup,
                                    cycles=cycles, flop=flop, alu_cc=alu_cc,
                                    mem_cc=mem_cc)  # Ashish added
        layer_result.populate_sub_results(dwrecur_layer_result)
        layer_result.populate_sub_results(dwin_layer_result)
        layer_result.populate_sub_results(dh_layer_result)
        layer_result.populate_sub_results(ds_layer_result)
        return layer_result
