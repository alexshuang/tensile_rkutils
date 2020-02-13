'ir_version': 1
'producer_name': 'onnx-mims'
'graph': {
    'name': 'facebook'
    'nodes': [
		node {
				'op_type': 'Gemm'
				'attributes': {'broadcast': 1, 'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/data_0', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
					input {'name': 'gpu_0/w_0', 'val_type': 1, 'dims': [1024, 1024], 'vals': []}
					input {'name': 'gpu_0/b_0', 'val_type': 1, 'dims': [1024], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/fc_out_0', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
				]
			}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/fc_out_0', 'val_type': 1, 'dims': [1, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/act_out_0', 'val_type': 1, 'dims': [1, 10008, 1, 1], 'vals': []}
            ]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'broadcast': 1, 'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/act_out_0', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
					input {'name': 'gpu_0/w_1', 'val_type': 1, 'dims': [1024, 1024], 'vals': []}
					input {'name': 'gpu_0/b_1', 'val_type': 1, 'dims': [1024], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/fc_out_1', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
				]
			}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/fc_out_1', 'val_type': 1, 'dims': [1, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/act_out_1', 'val_type': 1, 'dims': [1, 10008, 1, 1], 'vals': []}
            ]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'broadcast': 1, 'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/act_out_1', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
					input {'name': 'gpu_0/w_2', 'val_type': 1, 'dims': [1024, 1024], 'vals': []}
					input {'name': 'gpu_0/b_2', 'val_type': 1, 'dims': [1024], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/fc_out_2', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
				]
			}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/fc_out_2', 'val_type': 1, 'dims': [1, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/act_out_2', 'val_type': 1, 'dims': [1, 10008, 1, 1], 'vals': []}
            ]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'broadcast': 1, 'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/act_out_2', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
					input {'name': 'gpu_0/w_3', 'val_type': 1, 'dims': [1024, 1024], 'vals': []}
					input {'name': 'gpu_0/b_3', 'val_type': 1, 'dims': [1024], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/fc_out_3', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
				]
			}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/fc_out_3', 'val_type': 1, 'dims': [1, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/act_out_3', 'val_type': 1, 'dims': [1, 10008, 1, 1], 'vals': []}
            ]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'broadcast': 1, 'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/act_out_3', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
					input {'name': 'gpu_0/w_4', 'val_type': 1, 'dims': [1024, 1024], 'vals': []}
					input {'name': 'gpu_0/b_4', 'val_type': 1, 'dims': [1024], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/fc_out_4', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
				]
			}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/fc_out_4', 'val_type': 1, 'dims': [1, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/act_out_4', 'val_type': 1, 'dims': [1, 10008, 1, 1], 'vals': []}
            ]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'broadcast': 1, 'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/act_out_4', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
					input {'name': 'gpu_0/w_5', 'val_type': 1, 'dims': [1024, 1024], 'vals': []}
					input {'name': 'gpu_0/b_5', 'val_type': 1, 'dims': [1024], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/fc_out_5', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
				]
			}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/fc_out_5', 'val_type': 1, 'dims': [1, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/act_out_5', 'val_type': 1, 'dims': [1, 10008, 1, 1], 'vals': []}
            ]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'broadcast': 1, 'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/act_out_5', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
					input {'name': 'gpu_0/w_6', 'val_type': 1, 'dims': [1024, 1024], 'vals': []}
					input {'name': 'gpu_0/b_6', 'val_type': 1, 'dims': [1024], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/fc_out_6', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
				]
			}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/fc_out_6', 'val_type': 1, 'dims': [1, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/act_out_6', 'val_type': 1, 'dims': [1, 10008, 1, 1], 'vals': []}
            ]
        }
		node {
				'op_type': 'Gemm'
				'attributes': {'broadcast': 1, 'transB': 1}
				'inputs': [
					input {'name': 'gpu_0/act_out_6', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
					input {'name': 'gpu_0/w_7', 'val_type': 1, 'dims': [1024, 1024], 'vals': []}
					input {'name': 'gpu_0/b_7', 'val_type': 1, 'dims': [1024], 'vals': []}
				]
				'outputs': [
					output {'name': 'gpu_0/fc_out_7', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
				]
			}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/fc_out_7', 'val_type': 1, 'dims': [1, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/act_out_7', 'val_type': 1, 'dims': [1, 10008, 1, 1], 'vals': []}
            ]
        }
	]
}