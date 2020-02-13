'ir_version': 1
'producer_name': 'onnx-mims'
'graph': {
    'name': 'amdlss'
    'nodes': [
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [2, 2], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/data_0', 'val_type': 1, 'dims': [1, 3, 1440, 2560], 'vals': []}
                input {'name': 'gpu_0/conv1_w_0', 'val_type': 1, 'dims': [48, 3, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/conv1_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Conv'
            'attributes': {'strides': [2, 2], 'pads': [2, 2, 2, 2], 'kernel_shape': [5, 5]}
            'inputs': [
                input {'name': 'gpu_0/data_0', 'val_type': 1, 'dims': [1, 3, 1440, 2560], 'vals': []}
                input {'name': 'gpu_0/conv2_w_0', 'val_type': 1, 'dims': [48, 3, 5, 5], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/conv2_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
        node {
            'op_type': 'Tanh'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/conv2_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/act2_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/act2_1', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/conv2_w_1', 'val_type': 1, 'dims': [48, 48, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/conv2_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
        node {
            'op_type': 'Tanh'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/conv2_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/act2_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/act2_2', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/conv2_w_2', 'val_type': 1, 'dims': [48, 48, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/conv2_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
        node {
            'op_type': 'Tanh'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/conv2_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/act2_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/act2_4', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/conv1_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/sum1_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'PixelShuffle'
            'attributes': {'shuffle_factor_height': 4, 'shuffle_factor_width': 4}
            'inputs': [
                input {'name': 'gpu_0/sum1_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/ps1_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		
    ]
}
