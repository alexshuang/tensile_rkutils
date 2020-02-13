'ir_version': 1
'producer_name': 'onnx-mims'
'graph': {
    'name': 'deepspeech2'
    'nodes': [
		node {
			'op_type': 'Conv'
			'attributes': {'strides': [1, 3], 'pads': [0, 0, 0, 5], 'kernel_shape': [13, 11]}
			'inputs': [
				input {'name': 'gpu_0/data_0', 'val_type': 1, 'dims': [1, 1, 13, 2998], 'vals': []}
				input {'name': 'gpu_0/conv1_w_0', 'val_type': 1, 'dims': [1152, 1, 13, 11], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/conv1_1', 'val_type': 1, 'dims': [1, 1152, 1, 1000], 'vals': []}
			]
		}
		node {
			'op_type': 'Rnn'
			'attributes': {'direction': 'bidirectional', 'activations': 'relu', 'hidden_size': 1152}
			'inputs': [
				input {'name': 'gpu_0/conv1_1', 'val_type': 1, 'dims': [1000, 1, 1152], 'vals': []}
				input {'name': 'gpu_0/layer1_input_weights', 'val_type': 1, 'dims': [2, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer1_recurrent_weights', 'val_type': 1, 'dims': [2, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer1_bias', 'val_type': 1, 'dims': [2, 2304], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/rnn1', 'val_type': 1, 'dims': [1000, 2, 1, 1152], 'vals': []}
			]
		}
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/rnn1', 'val_type': 1, 'dims': [1, 2304, 1000, 1], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_s_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_b_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_rm_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_riv_0', 'val_type': 1, 'dims': [2304], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/rnn_bn_1', 'val_type': 1, 'dims': [1, 2304, 1000, 1], 'vals': []}
            ]
        }
		node {
			'op_type': 'Rnn'
			'attributes': {'direction': 'bidirectional', 'activations': 'relu', 'hidden_size': 1152}
			'inputs': [
				input {'name': 'gpu_0/rnn_bn_1', 'val_type': 1, 'dims': [1000, 1, 1152], 'vals': []}
				input {'name': 'gpu_0/layer2_input_weights', 'val_type': 1, 'dims': [2, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer2_recurrent_weights', 'val_type': 1, 'dims': [2, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer2_bias', 'val_type': 1, 'dims': [2, 2304], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/rnn2', 'val_type': 1, 'dims': [1000, 2, 1, 1152], 'vals': []}
			]
		}
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/rnn2', 'val_type': 1, 'dims': [1, 2304, 1000, 1], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_s_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_b_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_rm_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_riv_0', 'val_type': 1, 'dims': [2304], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/rnn_bn_2', 'val_type': 1, 'dims': [1, 2304, 1000, 1], 'vals': []}
            ]
        }
		node {
			'op_type': 'Rnn'
			'attributes': {'direction': 'bidirectional', 'activations': 'relu', 'hidden_size': 1152}
			'inputs': [
				input {'name': 'gpu_0/rnn_bn_2', 'val_type': 1, 'dims': [1000, 1, 1152], 'vals': []}
				input {'name': 'gpu_0/layer3_input_weights', 'val_type': 1, 'dims': [2, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer3_recurrent_weights', 'val_type': 1, 'dims': [2, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer3_bias', 'val_type': 1, 'dims': [2, 2304], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/rnn3', 'val_type': 1, 'dims': [1000, 2, 1, 1152], 'vals': []}
			]
		}
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/rnn3', 'val_type': 1, 'dims': [1, 2304, 1000, 1], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_s_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_b_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_rm_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_riv_0', 'val_type': 1, 'dims': [2304], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/rnn_bn_3', 'val_type': 1, 'dims': [1, 2304, 1000, 1], 'vals': []}
            ]
        }
		node {
			'op_type': 'Rnn'
			'attributes': {'direction': 'bidirectional', 'activations': 'relu', 'hidden_size': 1152}
			'inputs': [
				input {'name': 'gpu_0/rnn_bn_3', 'val_type': 1, 'dims': [1000, 1, 1152], 'vals': []}
				input {'name': 'gpu_0/layer4_input_weights', 'val_type': 1, 'dims': [2, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer4_recurrent_weights', 'val_type': 1, 'dims': [2, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer4_bias', 'val_type': 1, 'dims': [2, 2304], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/rnn4', 'val_type': 1, 'dims': [1000, 2, 1, 1152], 'vals': []}
			]
		}
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/rnn4', 'val_type': 1, 'dims': [1, 2304, 1000, 1], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_s_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_b_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_rm_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_riv_0', 'val_type': 1, 'dims': [2304], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/rnn_bn_4', 'val_type': 1, 'dims': [1, 2304, 1000, 1], 'vals': []}
            ]
        }
		node {
			'op_type': 'Rnn'
			'attributes': {'direction': 'bidirectional', 'activations': 'relu', 'hidden_size': 1152}
			'inputs': [
				input {'name': 'gpu_0/rnn_bn_4', 'val_type': 1, 'dims': [1000, 1, 1152], 'vals': []}
				input {'name': 'gpu_0/layer5_input_weights', 'val_type': 1, 'dims': [2, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer5_recurrent_weights', 'val_type': 1, 'dims': [2, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer5_bias', 'val_type': 1, 'dims': [2, 2304], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/rnn5', 'val_type': 1, 'dims': [1000, 2, 1, 1152], 'vals': []}
			]
		}
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/rnn5', 'val_type': 1, 'dims': [1, 2304, 1000, 1], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_s_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_b_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_rm_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_riv_0', 'val_type': 1, 'dims': [2304], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/rnn_bn_5', 'val_type': 1, 'dims': [1, 2304, 1000, 1], 'vals': []}
            ]
        }
		node {
			'op_type': 'Rnn'
			'attributes': {'direction': 'bidirectional', 'activations': 'relu', 'hidden_size': 1152}
			'inputs': [
				input {'name': 'gpu_0/rnn_bn_5', 'val_type': 1, 'dims': [1000, 1, 1152], 'vals': []}
				input {'name': 'gpu_0/layer6_input_weights', 'val_type': 1, 'dims': [2, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer6_recurrent_weights', 'val_type': 1, 'dims': [2, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer6_bias', 'val_type': 1, 'dims': [2, 2304], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/rnn6', 'val_type': 1, 'dims': [1000, 2, 1, 1152], 'vals': []}
			]
		}
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/rnn6', 'val_type': 1, 'dims': [1, 2304, 1000, 1], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_s_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_b_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_rm_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_riv_0', 'val_type': 1, 'dims': [2304], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/rnn_bn_6', 'val_type': 1, 'dims': [1, 2304, 1000, 1], 'vals': []}
            ]
        }
		node {
			'op_type': 'Rnn'
			'attributes': {'direction': 'bidirectional', 'activations': 'relu', 'hidden_size': 1152}
			'inputs': [
				input {'name': 'gpu_0/rnn_bn_6', 'val_type': 1, 'dims': [1000, 1, 1152], 'vals': []}
				input {'name': 'gpu_0/layer7_input_weights', 'val_type': 1, 'dims': [2, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer7_recurrent_weights', 'val_type': 1, 'dims': [2, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer7_bias', 'val_type': 1, 'dims': [2, 2304], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/rnn7', 'val_type': 1, 'dims': [1000, 2, 1, 1152], 'vals': []}
			]
		}
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/rnn7', 'val_type': 1, 'dims': [1, 2304, 1000, 1], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_s_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_b_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_rm_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_riv_0', 'val_type': 1, 'dims': [2304], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/rnn_bn_7', 'val_type': 1, 'dims': [1, 2304, 1000, 1], 'vals': []}
            ]
        }
		node {
			'op_type': 'Rnn'
			'attributes': {'direction': 'bidirectional', 'activations': 'relu', 'hidden_size': 1152}
			'inputs': [
				input {'name': 'gpu_0/rnn_bn_7', 'val_type': 1, 'dims': [1000, 1, 1152], 'vals': []}
				input {'name': 'gpu_0/layer8_input_weights', 'val_type': 1, 'dims': [2, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer8_recurrent_weights', 'val_type': 1, 'dims': [2, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer8_bias', 'val_type': 1, 'dims': [2, 2304], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/rnn8', 'val_type': 1, 'dims': [1000, 2, 1, 1152], 'vals': []}
			]
		}
		node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/rnn8', 'val_type': 1, 'dims': [1, 2304, 1000, 1], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_s_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_b_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_rm_0', 'val_type': 1, 'dims': [2304], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_riv_0', 'val_type': 1, 'dims': [2304], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/rnn_bn_8', 'val_type': 1, 'dims': [1, 2304, 1000, 1], 'vals': []}
            ]
        }
		node {
			'op_type': 'Rnn'
			'attributes': {'direction': 'forward', 'activations': 'relu', 'hidden_size': 1152}
			'inputs': [
				input {'name': 'gpu_0/rnn_bn_8', 'val_type': 1, 'dims': [1000, 1, 1152], 'vals': []}
				input {'name': 'gpu_0/layer9_input_weights', 'val_type': 1, 'dims': [1, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer9_recurrent_weights', 'val_type': 1, 'dims': [1, 1152, 1152], 'vals': []}
				input {'name': 'gpu_0/layer9_bias', 'val_type': 1, 'dims': [2, 2304], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/rnn9', 'val_type': 1, 'dims': [1000, 2, 1, 1152], 'vals': []}
			]
		}
		node {
            'op_type': 'Gemm'
            'attributes': {'transB': 1, 'transA': 0}
            'inputs': [
                input {'name': 'gpu_0/rnn9', 'val_type': 1, 'dims': [32000, 1152], 'vals': []}
                input {'name': 'gpu_0/layer10_w_0', 'val_type': 1, 'dims': [1152, 1152], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/fc_0', 'val_type': 1, 'dims': [32000, 1152], 'vals': []}
            ]
        }
		node {
            'op_type': 'Gemm'
            'attributes': {'transB': 1, 'transA': 0}
            'inputs': [
                input {'name': 'gpu_0/fc_0', 'val_type': 1, 'dims': [1, 1152, 1000, 1], 'vals': []}
                input {'name': 'gpu_0/layer11_w_0', 'val_type': 1, 'dims': [29, 1152], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/fc_1', 'val_type': 1, 'dims': [1, 29, 1000, 1], 'vals': []}
            ]
        }
	]
}