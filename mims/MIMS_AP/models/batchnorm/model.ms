'ir_version': 1
'producer_name': 'onnx-mims'
'graph': {
    'name': 'bn_test'
	'nodes': [
		node {
				'op_type': 'BatchNormalization'
				'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
				'inputs': [
					input {'name': 'conv2/3x3_reduce_1', 'val_type': 1, 'dims': [1, 64, 4, 4], 'vals': []}
					input {'name': 'conv2/3x3_reduce/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
					input {'name': 'conv2/3x3_reduce/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
					input {'name': 'conv2/3x3_reduce/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
					input {'name': 'conv2/3x3_reduce/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
				]
				'outputs': [
					output {'name': 'conv2/3x3_reduce/bn_1', 'val_type': 1, 'dims': [], 'vals': []}
				]
			}
	]
}