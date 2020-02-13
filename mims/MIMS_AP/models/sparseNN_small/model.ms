'ir_version': 1
'producer_name': 'onnx-mims'
'graph': {
    'name': 'sparseNN'
    'nodes': [
		node {
				'op_type': 'Embedding'
				'attributes': {'op': 'pooling', 'parallel': 'model', 'ntables': 32, 'backward_update' : 'adagrad'}
				'inputs': [
					input {'name': 'gpu_0/indices', 'val_type': 1, 'dims': [1, 40], 'vals': []}
					input {'name': 'gpu_0/tables', 'val_type': 1, 'dims': [1000000, 32], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/emb_out_0', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
		node {
				'op_type': 'Gemm'
				'attributes': {'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/dense', 'val_type': 1, 'dims': [1, 512], 'vals': []}
					input {'name': 'gpu_0/btm_w_0', 'val_type': 1, 'dims': [512, 512], 'vals': []}
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
					input {'name': 'gpu_0/btm_w_1', 'val_type': 1, 'dims': [512, 512], 'vals': []}
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
					input {'name': 'gpu_0/btm_w_2', 'val_type': 1, 'dims': [32, 512], 'vals': []}
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
				'op_type': 'Concat'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/emb_out_0', 'val_type': 1, 'dims': [], 'vals': []}
					input {'name': 'gpu_0/btm_act_out_2', 'val_type': 1, 'dims': [], 'vals': []}
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
				'attributes': {'factor': 1}
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
					input {'name': 'gpu_0/top_w_0', 'val_type': 1, 'dims': [1024, 32], 'vals': []}
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
					input {'name': 'gpu_0/top_w_1', 'val_type': 1, 'dims': [1024, 1024], 'vals': []}
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
					input {'name': 'gpu_0/top_w_2', 'val_type': 1, 'dims': [1024, 1024], 'vals': []}
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
					input {'name': 'gpu_0/top_w_3', 'val_type': 1, 'dims': [1024, 1024], 'vals': []}
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
					input {'name': 'gpu_0/top_w_4', 'val_type': 1, 'dims': [1024, 1024], 'vals': []}
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
				'op_type': 'Softmax'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/top_act_out_4', 'val_type': 1, 'dims': [], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/softmax_0', 'val_type': 1, 'dims': [], 'vals': []}
				]
        }
	
	]
}