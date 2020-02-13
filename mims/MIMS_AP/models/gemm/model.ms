'ir_version': 1
'producer_name': 'onnx-mims'
'graph': {
    'name': 'gemm_test'
    'nodes': [
		node {
				'op_type': 'Gemm'
				'attributes': {'broadcast': 1, 'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/data_0', 'val_type': 1, 'dims': [1 , 8192], 'vals': []}
					input {'name': 'gpu_0/w_0', 'val_type': 1, 'dims': [8192 , 8192], 'vals': []}
					input {'name': 'gpu_0/b_0', 'val_type': 1, 'dims': [1024], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/out_0', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
	]
}