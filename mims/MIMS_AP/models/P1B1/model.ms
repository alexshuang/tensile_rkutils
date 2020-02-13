'ir_version': 1
'producer_name': 'onnx-mims'
'graph': {
    'name': 'candle-p1b1-ae'
    'nodes': [
		node {
				'op_type': 'Gemm'
				'attributes': {'broadcast': 1, 'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/data_0', 'val_type': 1, 'dims': [100, 60483], 'vals': []}
					input {'name': 'gpu_0/w_0', 'val_type': 1, 'dims': [2000, 60483], 'vals': []}
					input {'name': 'gpu_0/b_0', 'val_type': 1, 'dims': [2000], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/out_0', 'val_type': 1, 'dims': [100, 2000], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/out_0', 'val_type': 1, 'dims': [100, 2000, 1, 1], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/relu_0', 'val_type': 1, 'dims': [100, 2000, 1, 1], 'vals': []}
                ]
			}
		node {
				'op_type': 'Gemm'
				'attributes': {'broadcast': 1, 'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/relu_0', 'val_type': 1, 'dims': [100, 2000], 'vals': []}
					input {'name': 'gpu_0/w_1', 'val_type': 1, 'dims': [600, 2000], 'vals': []}
					input {'name': 'gpu_0/b_1', 'val_type': 1, 'dims': [600], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/out_1', 'val_type': 1, 'dims': [100, 600], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/out_1', 'val_type': 1, 'dims': [100, 600, 1, 1], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/relu_1', 'val_type': 1, 'dims': [100, 600, 1, 1], 'vals': []}
                ]
			}
		node {
				'op_type': 'Gemm'
				'attributes': {'broadcast': 1, 'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/relu_3', 'val_type': 1, 'dims': [100, 600], 'vals': []}
					input {'name': 'gpu_0/w_4', 'val_type': 1, 'dims': [2000, 600], 'vals': []}
					input {'name': 'gpu_0/b_4', 'val_type': 1, 'dims': [2000], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/out_4', 'val_type': 1, 'dims': [100, 2000], 'vals': []}
				]
			}
		node {
				'op_type': 'Relu'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/out_4', 'val_type': 1, 'dims': [100, 2000, 1, 1], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/relu_4', 'val_type': 1, 'dims': [100, 2000, 1, 1], 'vals': []}
                ]
			}
		node {
				'op_type': 'Gemm'
				'attributes': {'broadcast': 1, 'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/relu_4', 'val_type': 1, 'dims': [100, 2000], 'vals': []}
					input {'name': 'gpu_0/w_5', 'val_type': 1, 'dims': [60483, 2000], 'vals': []}
					input {'name': 'gpu_0/b_5', 'val_type': 1, 'dims': [2000], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/out_5', 'val_type': 1, 'dims': [100, 60483], 'vals': []}
				]
			}
		node {
				'op_type': 'Sigmoid'
				'attributes': {}
				'inputs': [
					input {'name': 'gpu_0/out_5', 'val_type': 1, 'dims': [100, 60483, 1, 1], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/sigmoid_5', 'val_type': 1, 'dims': [100, 60483, 1, 1], 'vals': []}
                ]
			}
	]
}