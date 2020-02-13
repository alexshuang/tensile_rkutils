'ir_version': 1
'producer_name': 'onnx-mims'
'graph': {
    'name': 'a3c'
    'nodes': [
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [4, 4], 'pads': [0, 0, 0, 0], 'kernel_shape': [8, 8]}
            'inputs': [
                input {'name': 'data_0', 'val_type': 1, 'dims': [1, 4, 84, 84], 'vals': []}
                input {'name': 'conv1_w_0', 'val_type': 1, 'dims': [16, 4, 8, 8], 'vals': []}
                input {'name': 'conv1_b_0', 'val_type': 1, 'dims': [16], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu0/conv_1', 'val_type': 1, 'dims': [1, 16, 20, 20], 'vals': []}
            ]
        }
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu0/conv_1', 'val_type': 1, 'dims': [1, 16, 20, 20], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu0/act_1', 'val_type': 1, 'dims': [1, 16, 20, 20], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [2, 2], 'pads': [0, 0, 0, 0], 'kernel_shape': [4, 4]}
            'inputs': [
                input {'name': 'gpu0/act_1', 'val_type': 1, 'dims': [1, 16, 20, 20], 'vals': []}
                input {'name': 'conv2_w_0', 'val_type': 1, 'dims': [32, 16, 4, 4], 'vals': []}
                input {'name': 'conv2_b_0', 'val_type': 1, 'dims': [32], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu0/conv_2', 'val_type': 1, 'dims': [1, 32, 9, 9], 'vals': []}
            ]
        }
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu0/conv_2', 'val_type': 1, 'dims': [1, 32, 9, 9], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/act_2', 'val_type': 1, 'dims': [1, 32, 9, 9], 'vals': []}
            ]
        }
		node {
            'op_type': 'Gemm'
            'attributes': {'transB': 1, 'transA': 0}
            'inputs': [
                input {'name': 'gpu_0/act_2', 'val_type': 1, 'dims': [1, 32, 9, 9], 'vals': []}
                input {'name': 'fc_w_0', 'val_type': 1, 'dims': [256, 2592], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/fc_3', 'val_type': 1, 'dims': [1, 256, 1, 1], 'vals': []}
            ]
        }
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu0/fc_3', 'val_type': 1, 'dims': [1, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/act_3', 'val_type': 1, 'dims': [1, 256, 1, 1], 'vals': []}
            ]
        }
		node {
            'op_type': 'Gemm'
            'attributes': {'transB': 1, 'transA': 0}
            'inputs': [
                input {'name': 'gpu_0/act_3', 'val_type': 1, 'dims': [1, 256, 1, 1], 'vals': []}
                input {'name': 'fc_w_0', 'val_type': 1, 'dims': [18, 256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/fc_4', 'val_type': 1, 'dims': [1, 18, 1, 1], 'vals': []}
            ]
        }
	]
}
		
		