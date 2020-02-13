'ir_version': 1
'producer_name': 'onnx-mims'
'graph': {
    'name': 'sparseNN'
    'nodes': [
		node {
				'op_type': 'Embedding'
				'attributes': {'op': 'pooling', 'parallel': 'model', 'ntables': 64, 'backward_update' : 'adagrad'}
				'inputs': [
					input {'name': 'gpu_0/indices', 'val_type': 1, 'dims': [1, 100], 'vals': []}
					input {'name': 'gpu_0/tables', 'val_type': 1, 'dims': [20000000, 128], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/emb_out_0', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Gemm'
				'attributes': {'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/dense', 'val_type': 1, 'dims': [1, 3200], 'vals': []}
					input {'name': 'gpu_0/btm_w_0', 'val_type': 1, 'dims': [2048, 3200], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/btm_fc_out_0', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/btm_fc_out_0', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/btm_act_out_0', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/btm_act_out_0', 'val_type': 1, 'dims': [], 'vals': []}
					input {'name': 'gpu_0/btm_w_1', 'val_type': 1, 'dims': [2048, 2048], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/btm_fc_out_1', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/btm_fc_out_1', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/btm_act_out_1', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/btm_act_out_1', 'val_type': 1, 'dims': [], 'vals': []}
					input {'name': 'gpu_0/btm_w_2', 'val_type': 1, 'dims': [2048, 2048], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/btm_fc_out_2', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/btm_fc_out_2', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/btm_act_out_2', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/btm_act_out_2', 'val_type': 1, 'dims': [], 'vals': []}
					input {'name': 'gpu_0/btm_w_3', 'val_type': 1, 'dims': [2048, 2048], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/btm_fc_out_3', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/btm_fc_out_3', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/btm_act_out_3', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/btm_act_out_3', 'val_type': 1, 'dims': [], 'vals': []}
					input {'name': 'gpu_0/btm_w_4', 'val_type': 1, 'dims': [128, 2048], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/btm_fc_out_4', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/btm_fc_out_4', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/btm_act_out_4', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
		node {
				'op_type': 'Concat'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/emb_out_0', 'val_type': 1, 'dims': [], 'vals': []}
					input {'name': 'gpu_0/btm_act_out_4', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/concat_out_0', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Communication'
				'attributes': {'op': 'allgather'}
				'inputs': [
					input {'name': 'gpu_0/concat_out_0', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/comm_out_0', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Interaction'
				'attributes': {'factor': 65}
				'inputs': [
					input {'name': 'gpu_0/comm_out_0', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/interaction_out_0', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Gemm'
				'attributes': {'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/interaction_out_0', 'val_type': 1, 'dims': [], 'vals': []}
					input {'name': 'gpu_0/top_w_0', 'val_type': 1, 'dims': [4096, 8320], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_fc_out_0', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/top_fc_out_0', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_act_out_0', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/top_act_out_0', 'val_type': 1, 'dims': [], 'vals': []}
					input {'name': 'gpu_0/top_w_1', 'val_type': 1, 'dims': [4096, 4096], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_fc_out_1', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/top_fc_out_1', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_act_out_1', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/top_act_out_1', 'val_type': 1, 'dims': [], 'vals': []}
					input {'name': 'gpu_0/top_w_2', 'val_type': 1, 'dims': [4096, 4096], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_fc_out_2', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/top_fc_out_2', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_act_out_2', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/top_act_out_2', 'val_type': 1, 'dims': [], 'vals': []}
					input {'name': 'gpu_0/top_w_3', 'val_type': 1, 'dims': [4096, 4096], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_fc_out_3', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/top_fc_out_3', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_act_out_3', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/top_act_out_3', 'val_type': 1, 'dims': [], 'vals': []}
					input {'name': 'gpu_0/top_w_4', 'val_type': 1, 'dims': [4096, 4096], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_fc_out_4', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/top_fc_out_4', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_act_out_4', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/top_act_out_4', 'val_type': 1, 'dims': [], 'vals': []}
					input {'name': 'gpu_0/top_w_5', 'val_type': 1, 'dims': [4096, 4096], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_fc_out_5', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/top_fc_out_5', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_act_out_5', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/top_act_out_5', 'val_type': 1, 'dims': [], 'vals': []}
					input {'name': 'gpu_0/top_w_6', 'val_type': 1, 'dims': [4096, 4096], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_fc_out_6', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/top_fc_out_6', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_act_out_6', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/top_act_out_6', 'val_type': 1, 'dims': [], 'vals': []}
					input {'name': 'gpu_0/top_w_7', 'val_type': 1, 'dims': [4096, 4096], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_fc_out_7', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/top_fc_out_7', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_act_out_7', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/top_act_out_7', 'val_type': 1, 'dims': [], 'vals': []}
					input {'name': 'gpu_0/top_w_8', 'val_type': 1, 'dims': [4096, 4096], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_fc_out_8', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/top_fc_out_8', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_act_out_8', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/top_act_out_8', 'val_type': 1, 'dims': [], 'vals': []}
					input {'name': 'gpu_0/top_w_9', 'val_type': 1, 'dims': [4096, 4096], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_fc_out_9', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/top_fc_out_9', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/top_act_out_9', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
		node {
				'op_type': 'Softmax'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/top_act_out_9', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/softmax_0', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
	
	]
}