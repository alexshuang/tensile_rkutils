from core.layers.layer import Layer
from core.structs.auxiliary import LayerResults
from core.const import *
import math
import copy
from core.auxiliary import get_act_hit_rates, prod, get_dt_size
from core.structs.auxiliary import LayerInputParam, TpuSubResults
from core.layers.softmax import Softmax
from core.layers.concat import Concat
from core.layers.gemm import Gemm


class BahdanuAttention(Layer):
    def __init__(self, sys_config):
        super(BahdanuAttention, self).__init__(sys_config)

    def fprop(self, inputs, inference=False):
        input_sz = op.inputs[0].dims[ATTN_HIDDEN_SZ_IND] + op.inputs[1].dims[ATTN_HIDDEN_SZ_IND]
        hidden_sz = op.attributes['hidden_size'][0]
        batch_sz = op.inputs[0].dims[ATTN_BS_IND]
        # Hidden layer
        m = batch_sz
        n = hidden_sz
        k = input_sz
        hidden_weight_sz = n * k * data_size
        hidden_act_sz = m * k * data_size
        if hidden_weight_sz < self.sys_cfg.hw_cfg.l2_size:
            l2_hit_rate_wgt = 1.0
        else:
            l2_hit_rate_wgt = self.sys_cfg.hw_cfg.l2_size / hidden_weight_sz
            l2_hit_rate_act, l3_hit_rate_act = get_act_hit_rates(hidden_act_sz, hidden_weight_sz,
                                                                 self.sys_cfg.hw_cfg, self.sys_cfg.sw_opt)
        layer_params = LayerInputParam(in_dims, out_dims, l2_hit_rate_wgt, l2_hit_rate_act,
                                       l3_hit_rate_act)
        is_c_mat_used = False
        if direction == 'forward':
            gemm = Gemm(self.sys_cfg, (m, n, k), is_c_mat_used)
            layer_result = gemm.fprop(layer_params, not self.sys_cfg.sw_opt.training)
        else:
            wgrad_dims = [0] * 3
            dgrad_dims = [0] * 3
            wgrad_dims[GEMM_M_IND] = k
            wgrad_dims[GEMM_N_IND] = n
            wgrad_dims[GEMM_K_IND] = m
            dgrad_dims[GEMM_M_IND] = m
            dgrad_dims[GEMM_N_IND] = k
            dgrad_dims[GEMM_K_IND] = n
            gemm = Gemm(self.sys_cfg, (m, n, k), is_c_mat_used, wgrad_dims=wgrad_dims,
                        dgrad_dims=dgrad_dims)
            layer_result = gemm.bprop(layer_params)

        # Output layer
        m = batch_sz
        n = 1
        k = hidden_sz
        output_weight_sz = n * k * data_size
        output_act_sz = m * k * data_size
        if output_weight_sz < self.sys_cfg.hw_cfg.l2_size:
            l2_hit_rate_wgt = 1.0
        else:
            l2_hit_rate_wgt = self.sys_cfg.hw_cfg.l2_size / output_weight_sz
        l2_hit_rate_act = 0.0
        layer_params = LayerInputParam(in_dims, out_dims, l2_hit_rate_wgt, l2_hit_rate_act,
                                       l3_hit_rate_act)
        is_c_mat_used = False
        if direction == 'forward':
            gemm = Gemm(self.sys_cfg, (m, n, k), is_c_mat_used)
            sub_layer_result = gemm.fprop(layer_params, not self.sys_cfg.sw_opt.training)
        else:
            wgrad_dims = [0] * 3
            dgrad_dims = [0] * 3
            wgrad_dims[GEMM_M_IND] = k
            wgrad_dims[GEMM_N_IND] = n
            wgrad_dims[GEMM_K_IND] = m
            dgrad_dims[GEMM_M_IND] = m
            dgrad_dims[GEMM_N_IND] = k
            dgrad_dims[GEMM_K_IND] = n
            gemm = Gemm(self.sys_cfg, (m, n, k), is_c_mat_used, wgrad_dims=wgrad_dims,
                        dgrad_dims=dgrad_dims)
            sub_layer_result = gemm.bprop(layer_params)

        cycles = layer_result.cycles + sub_layer_result.cycles
        layer_result.alu_util_factor = (layer_result.cycles * layer_result.alu_util_factor +
                                        sub_layer_result.cycles * sub_layer_result.alu_util_factor) / cycles
        layer_result.chip_util_factor = (layer_result.cycles * layer_result.chip_util_factor +
                                         sub_layer_result.cycles * sub_layer_result.chip_util_factor) / cycles
        layer_result.cycles = cycles * (op.inputs[0].dims[ATTN_SEQ_LEN_IND] ** 2)  # Quadratic of sequence length

    def bprop(self, inputs):
        pass


class MultiHeadAttnParams:
    def __init__(self, batch_sz, dmodel, dk, dv, q_seq_len, kv_seq_len, nhead, is_self_attn):
        self.batch_sz = batch_sz
        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv
        self.q_seq_len = q_seq_len
        self.kv_seq_len = kv_seq_len
        self.nhead = nhead
        self.is_self_attn = is_self_attn


class MultiHeadAttention(Layer):
    def __init__(self, sys_config):
        super(MultiHeadAttention, self).__init__(sys_config)

    def compute_input_proj(self, mha_params, in_dims, out_dims, l2_hit_rate_act=0.0, l2_hit_rate_wt=0.0, direction='forward', tpu_partition_scheme=''):
        m = mha_params.batch_sz * mha_params.q_seq_len
        # Assumption: In case of inference input projection is done with weight matrix [dmodel x dmodel] and then split into nheads
        n = mha_params.dk if self.sys_cfg.sw_opt.training else mha_params.dmodel
        k = mha_params.dmodel
        layer_params = LayerInputParam(in_dims, out_dims, l2_hit_rate_wt, l2_hit_rate_act, tpu_partition_scheme=tpu_partition_scheme)

        if direction == 'forward':
            gemm = Gemm(self.sys_cfg, (m, n, k), is_c_mat_used=False)
            q_in_proj_result = gemm.fprop(layer_params, not self.sys_cfg.sw_opt.training)
        else:
            wgrad_dims = [0] * 3
            dgrad_dims = [0] * 3
            wgrad_dims[GEMM_M_IND] = k
            wgrad_dims[GEMM_N_IND] = n
            wgrad_dims[GEMM_K_IND] = m
            dgrad_dims[GEMM_M_IND] = m
            dgrad_dims[GEMM_N_IND] = k
            dgrad_dims[GEMM_K_IND] = n
            gemm = Gemm(self.sys_cfg, (m, n, k), is_c_mat_used=False, wgrad_dims=wgrad_dims, dgrad_dims=dgrad_dims)
            q_in_proj_result = gemm.bprop(layer_params)

        q_in_proj_result.op_name = 'input projection GEMM'
        q_in_proj_result.cycles *= (mha_params.nhead * 3 if self.sys_cfg.sw_opt.training else 3)
        q_in_proj_result.flop *= (mha_params.nhead * 3 if self.sys_cfg.sw_opt.training else 3)
        q_in_proj_result.hbm_rd_bw *= (mha_params.nhead * 3 if self.sys_cfg.sw_opt.training else 3)
        q_in_proj_result.hbm_wr_bw *= (mha_params.nhead * 3 if self.sys_cfg.sw_opt.training else 3)
        return q_in_proj_result

    def compute_qk_dotprod(self, mha_params, in_dims, out_dims, l2_hit_rate_act=0.0, l2_hit_rate_wt=0.0, direction='forward', tpu_partition_scheme=''):
        m = mha_params.q_seq_len
        n = mha_params.kv_seq_len
        k = mha_params.dk
        num_cu_per_batch = self.sys_cfg.hw_cfg.num_cu // self.sys_cfg.hw_cfg.num_se_per_cluster

        layer_params = LayerInputParam(in_dims, out_dims, l2_hit_rate_wt, l2_hit_rate_act, tpu_partition_scheme=tpu_partition_scheme)
        if direction == 'forward':
            gemm = Gemm(self.sys_cfg, (m, n, k), is_c_mat_used=False, batched_gemm=True, num_cu_per_batch=num_cu_per_batch)
            qk_dotprod_result = gemm.fprop(layer_params, not self.sys_cfg.sw_opt.training)
        else:
            wgrad_dims = [0] * 3
            dgrad_dims = [0] * 3
            wgrad_dims[GEMM_M_IND] = k
            wgrad_dims[GEMM_N_IND] = n
            wgrad_dims[GEMM_K_IND] = m
            dgrad_dims[GEMM_M_IND] = m
            dgrad_dims[GEMM_N_IND] = k
            dgrad_dims[GEMM_K_IND] = n
            gemm = Gemm(self.sys_cfg, (m, n, k), is_c_mat_used=False, wgrad_dims=wgrad_dims, dgrad_dims=dgrad_dims,
                        batched_gemm=True, num_cu_per_batch=num_cu_per_batch)
            qk_dotprod_result = gemm.bprop(layer_params)

        qk_dotprod_result.op_name = 'qk GEMM'
        qk_dotprod_result.num_cu_util *= self.sys_cfg.hw_cfg.num_se_per_cluster
        num_batches = math.ceil(mha_params.batch_sz / self.sys_cfg.hw_cfg.num_se_per_cluster)
        qk_dotprod_result.cycles *= num_batches * mha_params.nhead
        qk_dotprod_result.flop *= mha_params.nhead * mha_params.batch_sz
        qk_dotprod_result.hbm_rd_bw *= mha_params.nhead * mha_params.batch_sz
        qk_dotprod_result.hbm_wr_bw *= mha_params.nhead * mha_params.batch_sz
        return qk_dotprod_result

    def compute_qkv_dotprod(self, mha_params, in_dims, out_dims, l2_hit_rate_act=0.0, l2_hit_rate_wt=0.0, direction='forward', tpu_partition_scheme=''):
        m = mha_params.q_seq_len
        n = mha_params.dv
        k = mha_params.kv_seq_len
        num_cu_per_batch = self.sys_cfg.hw_cfg.num_cu // self.sys_cfg.hw_cfg.num_se_per_cluster
        layer_params = LayerInputParam(in_dims, out_dims, l2_hit_rate_wt, l2_hit_rate_act, tpu_partition_scheme=tpu_partition_scheme)

        if direction == 'forward':
            gemm = Gemm(self.sys_cfg, (m, n, k), is_c_mat_used=False, batched_gemm=True, num_cu_per_batch=num_cu_per_batch)
            qkv_dotprod_result = gemm.fprop(layer_params, not self.sys_cfg.sw_opt.training)
        else:
            wgrad_dims = [0] * 3
            dgrad_dims = [0] * 3
            wgrad_dims[GEMM_M_IND] = k
            wgrad_dims[GEMM_N_IND] = n
            wgrad_dims[GEMM_K_IND] = m
            dgrad_dims[GEMM_M_IND] = m
            dgrad_dims[GEMM_N_IND] = k
            dgrad_dims[GEMM_K_IND] = n
            gemm = Gemm(self.sys_cfg, (m, n, k), is_c_mat_used=False, wgrad_dims=wgrad_dims, dgrad_dims=dgrad_dims,
                        batched_gemm=True, num_cu_per_batch=num_cu_per_batch)
            qkv_dotprod_result = gemm.bprop(layer_params)
        qkv_dotprod_result.op_name = 'qkv GEMM'
        qkv_dotprod_result.num_cu_util *= self.sys_cfg.hw_cfg.num_se_per_cluster
        num_batches = math.ceil(mha_params.batch_sz / self.sys_cfg.hw_cfg.num_se_per_cluster)
        qkv_dotprod_result.cycles *= num_batches * mha_params.nhead
        qkv_dotprod_result.flop *= mha_params.nhead * mha_params.batch_sz
        qkv_dotprod_result.hbm_rd_bw *= mha_params.nhead * mha_params.batch_sz
        qkv_dotprod_result.hbm_wr_bw *= mha_params.nhead * mha_params.batch_sz
        return qkv_dotprod_result

    def compute_output_proj(self, mha_params, in_dims, out_dims, l2_hit_rate_act=0.0, l2_hit_rate_wt=0.0, direction='forward', tpu_partition_scheme=''):
        m = mha_params.q_seq_len * mha_params.batch_sz
        n = mha_params.dmodel
        k = mha_params.dv
        layer_params = LayerInputParam(in_dims, out_dims, l2_hit_rate_wt, l2_hit_rate_act, tpu_partition_scheme=tpu_partition_scheme)

        if direction == 'forward':
            gemm = Gemm(self.sys_cfg, (m, n, k), is_c_mat_used=False)
            q_out_proj_result = gemm.fprop(layer_params, not self.sys_cfg.sw_opt.training)
        else:
            wgrad_dims = [0] * 3
            dgrad_dims = [0] * 3
            wgrad_dims[GEMM_M_IND] = k
            wgrad_dims[GEMM_N_IND] = n
            wgrad_dims[GEMM_K_IND] = m
            dgrad_dims[GEMM_M_IND] = m
            dgrad_dims[GEMM_N_IND] = k
            dgrad_dims[GEMM_K_IND] = n
            gemm = Gemm(self.sys_cfg, (m, n, k), is_c_mat_used=False, wgrad_dims=wgrad_dims, dgrad_dims=dgrad_dims)
            q_out_proj_result = gemm.bprop(layer_params)
        q_out_proj_result.op_name = 'output projection GEMM'

        return q_out_proj_result

    def compute_attention(self, inputs, direction, inference=False):
        # For Attention the original operation struct 'op' is overloaded on top of input.in_dims
        op = copy.deepcopy(inputs.in_dims)
        in_dims = copy.deepcopy(op.inputs[IMG_IND].dims if direction == 'forward' else op.outputs[IMG_IND].dims)
        out_dims = copy.deepcopy(op.outputs[IMG_IND].dims if direction == 'forward' else op.inputs[IMG_IND].dims)
        data_size = get_dt_size(self.sys_cfg.sw_opt)
        dmodel = op.attributes['dmodel']
        dk = op.attributes['dk']
        dv = op.attributes['dv']
        nhead = op.attributes['nhead']
        is_self_attn = op.attributes['self_attention']
        batch_sz = op.outputs[0].dims[ATTN_BS_IND]
        q_seq_len = op.outputs[0].dims[ATTN_SEQ_LEN_IND]
        kv_seq_len = op.outputs[0].dims[ATTN_SEQ_LEN_IND]
        mha_params = MultiHeadAttnParams(batch_sz, dmodel, dk, dv, q_seq_len, kv_seq_len, nhead, is_self_attn)

        # Assumption of operation order for better performance/L2 reuse:
        # for one head at a time:
        #   1. Compute q and k linear projection gemm
        #   2. Compute Q*K gemm
        #   3. Softmax(Q*K/d)
        #   4. Compute v linear projection gemm
        #   5. Compute (Softmax(Q*K/d) * v) gemm
        # 6. Concat all heads
        # 7. Output projection gemm

        # Compute q, k, v linear projections
        l2_hit_rate_wt = 0.0  # unique 'nhead' weights fetched
        if op.index == 0:  # First layer
            l2_hit_rate_act, l3_hit_rate_act = 0.0, 0.0
        else:
            l2_hit_rate_act, l3_hit_rate_act = get_act_hit_rates(prod(in_dims) * data_size, 0, self.sys_cfg.hw_cfg, self.sys_cfg.sw_opt)
        q_in_proj_result = self.compute_input_proj(mha_params, in_dims, out_dims, l2_hit_rate_act, l2_hit_rate_wt, direction,
                                                   tpu_partition_scheme=inputs.tpu_partition_scheme)

        # Compute Scaled dot product attention
        # Compute Q*K dot product
        q_proj_size = batch_sz * q_seq_len * dk * data_size
        k_proj_size = batch_sz * q_seq_len * dk * data_size
        v_proj_size = batch_sz * q_seq_len * dk * data_size
        l2_hit_rate_act = min(self.sys_cfg.hw_cfg.l2_size / (q_proj_size + k_proj_size), 1.0)
        l2_hit_rate_wt = l2_hit_rate_act
        qk_dotprod_result = self.compute_qk_dotprod(mha_params, in_dims, out_dims, l2_hit_rate_act, l2_hit_rate_wt, direction,
                                                    tpu_partition_scheme='aggresive')

        # Compute softmax(Q*K)
        qk_softmax_act_size = batch_sz * q_seq_len * kv_seq_len * data_size
        softmax = Softmax(self.sys_cfg)
        sm_in_dims = [q_seq_len, kv_seq_len, 1, 1]
        sm_out_dims = [q_seq_len, kv_seq_len, 1, 1]
        l2_hit_rate_act = min(self.sys_cfg.hw_cfg.l2_size / qk_softmax_act_size, 1.0)
        layer_params = LayerInputParam(sm_in_dims, sm_out_dims, 0, l2_hit_rate_act)
        if direction == 'forward':
            qk_softmax_result = softmax.fprop(layer_params, inference)
        else:
            qk_softmax_result = softmax.bprop(layer_params)
        qk_softmax_result.op_name = 'qk softmax'
        qk_softmax_result.cycles *= nhead * batch_sz
        qk_softmax_result.flop *= nhead * batch_sz
        qk_softmax_result.hbm_rd_bw *= nhead * batch_sz
        qk_softmax_result.hbm_wr_bw *= nhead * batch_sz
        qk_softmax_act_size *= nhead * batch_sz

        # Compute QK*V dot product
        l2_hit_rate_act = min(self.sys_cfg.hw_cfg.l2_size / (q_proj_size + v_proj_size), 1.0)
        l2_hit_rate_wt = l2_hit_rate_act
        qkv_dotprod_result = self.compute_qkv_dotprod(mha_params, in_dims, out_dims, l2_hit_rate_act, l2_hit_rate_wt, direction,
                                                      tpu_partition_scheme='aggresive')

        # Concat results from all heads
        concat = Concat(self.sys_cfg)
        cc_in_dims = [batch_sz, nhead, q_seq_len, dv]
        cc_out_dims = [batch_sz, 1, q_seq_len, dv]
        if prod(cc_in_dims) * data_size <= self.sys_cfg.hw_cfg.l2_size:
            l2_hit_rate_act = 1.0
        else:
            l2_hit_rate_act = 0.0
        layer_params = LayerInputParam(cc_in_dims, cc_out_dims, l2_hit_rate_wt=0.0, l2_hit_rate_act=l2_hit_rate_act)
        nhead_concat_result = concat.fprop(layer_params, not self.sys_cfg.sw_opt.training)
        nhead_concat_result.op_name = 'nhead_concat'

        # Compute (QK*V) * Wo output projection
        q_out_proj_act_size = q_seq_len * batch_sz * dv * data_size
        q_out_proj_wt_size = dmodel * dv * data_size
        if q_out_proj_act_size < self.sys_cfg.hw_cfg.l2_size:
            l2_hit_rate_act = 1.0
        else:
            l2_hit_rate_act, l3_hit_rate_act = get_act_hit_rates(q_out_proj_act_size, q_out_proj_wt_size,
                                                                 self.sys_cfg.hw_cfg, self.sys_cfg.sw_opt)
        l2_hit_rate_wt = 0
        q_out_proj_result = self.compute_output_proj(mha_params, in_dims, out_dims, l2_hit_rate_act, l2_hit_rate_wt, direction,
                                                     tpu_partition_scheme=inputs.tpu_partition_scheme)

        cycles = q_in_proj_result.cycles + qk_dotprod_result.cycles + qk_softmax_result.cycles + \
                 nhead_concat_result.cycles + qkv_dotprod_result.cycles + q_out_proj_result.cycles

        flop = q_in_proj_result.flop + qk_dotprod_result.flop + qk_softmax_result.flop + \
               nhead_concat_result.flop + qkv_dotprod_result.flop + q_out_proj_result.flop

        hbm_rd_bw = q_in_proj_result.hbm_rd_bw + qk_dotprod_result.hbm_rd_bw + qk_softmax_result.hbm_rd_bw + \
               nhead_concat_result.hbm_rd_bw + qkv_dotprod_result.hbm_rd_bw + q_out_proj_result.hbm_rd_bw
        hbm_wr_bw = q_in_proj_result.hbm_wr_bw + qk_dotprod_result.hbm_wr_bw + qk_softmax_result.hbm_wr_bw + \
                    nhead_concat_result.hbm_wr_bw + qkv_dotprod_result.hbm_wr_bw + q_out_proj_result.hbm_wr_bw

        alu_util_factor = ((q_in_proj_result.cycles * q_in_proj_result.alu_util_factor +
                            qk_dotprod_result.cycles * qk_dotprod_result.alu_util_factor +
                            qk_softmax_result.cycles * qk_softmax_result.alu_util_factor +
                            qkv_dotprod_result.cycles * qkv_dotprod_result.alu_util_factor +
                            nhead_concat_result.cycles * nhead_concat_result.alu_util_factor +
                            q_out_proj_result.cycles * q_out_proj_result.alu_util_factor) / cycles)
        chip_util_factor = ((q_in_proj_result.cycles * q_in_proj_result.chip_util_factor +
                             qk_dotprod_result.cycles * qk_dotprod_result.chip_util_factor +
                             qk_softmax_result.cycles * qk_softmax_result.chip_util_factor +
                             qkv_dotprod_result.cycles * qkv_dotprod_result.chip_util_factor +
                             nhead_concat_result.cycles * nhead_concat_result.chip_util_factor +
                             q_out_proj_result.cycles * q_out_proj_result.chip_util_factor) / cycles)

        if self.sys_cfg.hw_cfg.tpu_en:
            q_in_proj_result.tpu_sub_res.vgpr_util_bytes_wt *= 3
            vgpr_util_bytes_wt = q_in_proj_result.tpu_sub_res.vgpr_util_bytes_wt + \
                                 q_out_proj_result.tpu_sub_res.vgpr_util_bytes_wt
            vgpr_util_bytes_res = max(q_in_proj_result.tpu_sub_res.vgpr_util_bytes_res,
                                      qk_dotprod_result.tpu_sub_res.vgpr_util_bytes_res,
                                      qkv_dotprod_result.tpu_sub_res.vgpr_util_bytes_res,
                                      q_out_proj_result.tpu_sub_res.vgpr_util_bytes_res)
            tpu_sub_res = TpuSubResults(vgpr_util_bytes_wt, vgpr_util_bytes_res)
        else:
            tpu_sub_res = TpuSubResults()
        speedup = 1  # TODO: Compute later
        layer_result = LayerResults(alu_util_factor, chip_util_factor, speedup, cycles, flop, hbm_rd_bw=hbm_rd_bw, hbm_wr_bw=hbm_wr_bw,
                                    op_name='multihead attention', tpu_sub_res=tpu_sub_res)
        layer_result.populate_sub_results(q_in_proj_result)
        layer_result.populate_sub_results(qk_dotprod_result)
        layer_result.populate_sub_results(qk_softmax_result)
        layer_result.populate_sub_results(qkv_dotprod_result)
        layer_result.populate_sub_results(nhead_concat_result)
        layer_result.populate_sub_results(q_out_proj_result)
        return layer_result

    def fprop(self, inputs, inference=False):
        return self.compute_attention(inputs, 'forward', inference)

    def bprop(self, inputs):
        return self.compute_attention(inputs, 'backward')
