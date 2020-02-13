'ir_version': 3
'producer_name': 'onnx-mims'
'graph': {
    'name': 'transformer'
    'nodes': [
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'False', 'self_attention': 1}
            'inputs': [
                input {'name': 'gpu_0/enc_emb_1', 'val_type': 1, 'dims': [1, 1, 1024, 50], 'vals': []}
				input {'name': 'gpu_0/enc_emb_1', 'val_type': 1, 'dims': [1, 1, 1024, 50], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_mha_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_emb_1', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/enc_mha_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_sum1_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/enc_sum1_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_bn1_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/enc_bn1_1', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_w0_1', 'val_type': 1, 'dims': [4096, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_fc1_1', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_fc1_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_act1_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/enc_act1_1', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_w1_1', 'val_type': 1, 'dims': [1024, 4096], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_fc2_1', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_bn1_1', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/enc_fc2_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_sum2_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/enc_sum2_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_bn2_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'False', 'self_attention': 1}
            'inputs': [
                input {'name': 'gpu_0/enc_bn2_1', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_bn2_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_mha_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_bn2_1', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/enc_mha_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_sum1_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/enc_sum1_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_bn1_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/enc_bn1_2', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_w0_2', 'val_type': 1, 'dims': [4096, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_fc1_2', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_fc1_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_act1_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/enc_act1_2', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_w1_2', 'val_type': 1, 'dims': [1024, 4096], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_fc2_2', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_bn1_2', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/enc_fc2_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_sum2_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/enc_sum2_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_bn2_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'False', 'self_attention': 1}
            'inputs': [
                input {'name': 'gpu_0/enc_bn2_2', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_bn2_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_mha_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_bn2_2', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/enc_mha_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_sum1_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/enc_sum1_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_bn1_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/enc_bn1_3', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_w0_3', 'val_type': 1, 'dims': [4096, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_fc1_3', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_fc1_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_act1_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/enc_act1_3', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_w1_3', 'val_type': 1, 'dims': [1024, 4096], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_fc2_3', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_bn1_3', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/enc_fc2_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_sum2_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/enc_sum2_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_bn2_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'False', 'self_attention': 1}
            'inputs': [
                input {'name': 'gpu_0/enc_bn2_3', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_bn2_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_mha_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_bn2_3', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/enc_mha_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_sum1_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/enc_sum1_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_bn1_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/enc_bn1_4', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_w0_4', 'val_type': 1, 'dims': [4096, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_fc1_4', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_fc1_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_act1_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/enc_act1_4', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_w1_4', 'val_type': 1, 'dims': [1024, 4096], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_fc2_4', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_bn1_4', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/enc_fc2_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_sum2_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/enc_sum2_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_bn2_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'False', 'self_attention': 1}
            'inputs': [
                input {'name': 'gpu_0/enc_bn2_4', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_bn2_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_mha_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_bn2_4', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/enc_mha_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_sum1_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/enc_sum1_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_bn1_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/enc_bn1_5', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_w0_5', 'val_type': 1, 'dims': [4096, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_fc1_5', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_fc1_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_act1_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/enc_act1_5', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_w1_5', 'val_type': 1, 'dims': [1024, 4096], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_fc2_5', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_bn1_5', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/enc_fc2_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_sum2_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/enc_sum2_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_bn2_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'False', 'self_attention': 1}
            'inputs': [
                input {'name': 'gpu_0/enc_bn2_5', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_bn2_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_mha_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_bn2_5', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/enc_mha_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_sum1_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/enc_sum1_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_bn1_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/enc_bn1_6', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_w0_6', 'val_type': 1, 'dims': [4096, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_fc1_6', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_fc1_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_act1_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/enc_act1_6', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_w1_6', 'val_type': 1, 'dims': [1024, 4096], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_fc2_6', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_bn1_6', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/enc_fc2_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_sum2_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/enc_sum2_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_bn2_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'True', 'self_attention': 1}
            'inputs': [
                input {'name': 'gpu_0/dec_emb_1', 'val_type': 1, 'dims': [1, 1, 1024, 50], 'vals': []}
				input {'name': 'gpu_0/dec_emb_1', 'val_type': 1, 'dims': [1, 1, 1024, 50], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_mha1_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_emb_1', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_mha1_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum1_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum1_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn1_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'True', 'self_attention': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_bn1_1', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_bn2_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_mha2_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_bn1_1', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_mha2_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum2_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum2_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn2_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/dec_bn2_1', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_w0_1', 'val_type': 1, 'dims': [4096, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_fc1_1', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_fc1_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_act1_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/dec_act1_1', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_w1_1', 'val_type': 1, 'dims': [1024, 4096], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_fc2_1', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_fc2_1', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_bn2_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum3_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum3_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn3_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'True', 'self_attention': 1}
            'inputs': [
                input {'name': 'gpu_0/dec_bn3_1', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_bn3_1', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_mha1_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_bn3_1', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_mha1_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum1_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum1_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn1_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'True', 'self_attention': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_bn1_2', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_bn2_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_mha2_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_bn1_2', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_mha2_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum2_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum2_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn2_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/dec_bn2_2', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_w0_2', 'val_type': 1, 'dims': [4096, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_fc1_2', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_fc1_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_act1_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/dec_act1_2', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_w1_2', 'val_type': 1, 'dims': [1024, 4096], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_fc2_2', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_fc2_2', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_bn2_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum3_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum3_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn3_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'True', 'self_attention': 1}
            'inputs': [
                input {'name': 'gpu_0/dec_bn3_2', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_bn3_2', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_mha1_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_bn3_2', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_mha1_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum1_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum1_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn1_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'True', 'self_attention': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_bn1_3', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_bn2_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_mha2_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_bn1_3', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_mha2_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum2_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum2_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn2_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/dec_bn2_3', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_w0_3', 'val_type': 1, 'dims': [4096, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_fc1_3', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_fc1_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_act1_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/dec_act1_3', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_w1_3', 'val_type': 1, 'dims': [1024, 4096], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_fc2_3', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_fc2_3', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_bn2_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum3_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum3_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn3_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'True', 'self_attention': 1}
            'inputs': [
                input {'name': 'gpu_0/dec_bn3_3', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_bn3_3', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_mha1_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_bn3_3', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_mha1_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum1_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum1_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn1_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'True', 'self_attention': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_bn1_4', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_bn2_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_mha2_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_bn1_4', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_mha2_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum2_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum2_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn2_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/dec_bn2_4', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_w0_4', 'val_type': 1, 'dims': [4096, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_fc1_4', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_fc1_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_act1_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/dec_act1_4', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_w1_4', 'val_type': 1, 'dims': [1024, 4096], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_fc2_4', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_fc2_4', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_bn2_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum3_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum3_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn3_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'True', 'self_attention': 1}
            'inputs': [
                input {'name': 'gpu_0/dec_bn3_4', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_bn3_4', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_mha1_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_bn3_4', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_mha1_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum1_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum1_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn1_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'True', 'self_attention': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_bn1_5', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_bn2_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_mha2_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_bn1_5', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_mha2_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum2_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum2_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn2_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/dec_bn2_5', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_w0_5', 'val_type': 1, 'dims': [4096, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_fc1_5', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_fc1_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_act1_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/dec_act1_5', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_w1_5', 'val_type': 1, 'dims': [1024, 4096], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_fc2_5', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_fc2_5', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_bn2_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum3_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum3_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn3_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'True', 'self_attention': 1}
            'inputs': [
                input {'name': 'gpu_0/dec_bn3_5', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_bn3_5', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_mha1_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_bn3_5', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_mha1_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum1_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum1_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn1_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Attention'
            'attributes': {'type': 'multihead', 'nhead': 16, 'dmodel': 1024, 'dk': 64, 'dv': 64, 'masked': 'True', 'self_attention': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_bn1_6', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/enc_bn2_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_mha2_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_bn1_6', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_mha2_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum2_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum2_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn2_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/dec_bn2_6', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_w0_6', 'val_type': 1, 'dims': [4096, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_fc1_6', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_fc1_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_act1_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
			'op_type': 'Gemm'
			'attributes': {'broadcast': 1, 'transB': 1}
			'inputs': [
				input {'name': 'gpu_0/dec_act1_6', 'val_type': 1, 'dims': [], 'vals': []}
				input {'name': 'gpu_0/dec_w1_6', 'val_type': 1, 'dims': [1024, 4096], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_fc2_6', 'val_type': 1, 'dims': [], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_fc2_6', 'val_type': 1, 'dims': [], 'vals': []}
                input {'name': 'gpu_0/dec_bn2_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_sum3_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1, 'is_test': 0}
            'inputs': [
                input {'name': 'gpu_0/dec_sum3_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_bn3_6', 'val_type': 1, 'dims': [], 'vals': []}
            ]
        }
	]
}