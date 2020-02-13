'ir_version': 3
'producer_name': 'onnx-mims'
'graph': {
    'name': 'googlenmt'
    'nodes': [
		node {
			'op_type': 'Lstm'
			'attributes': {'direction': 'bidirectional', 'activations': ['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh', 'sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh'], 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/emb_x', 'val_type': 1, 'dims': [50, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer1_input_weights', 'val_type': 1, 'dims': [2, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer1_recurrent_weights', 'val_type': 1, 'dims': [2, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer1_bias', 'val_type': 1, 'dims': [2, 2048], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_rnn1', 'val_type': 1, 'dims': [50, 2, 1, 1024], 'vals': []}
			]
		}
		node {
			'op_type': 'Lstm'
			'attributes': {'direction': 'forward', 'activations': ['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh'], 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/enc_rnn1', 'val_type': 1, 'dims': [50, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer2_input_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer2_recurrent_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer2_bias', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_rnn2', 'val_type': 1, 'dims': [50, 1, 1, 1024], 'vals': []}
			]
		}
		node {
			'op_type': 'Lstm'
			'attributes': {'direction': 'forward', 'activations': ['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh'], 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/enc_rnn2', 'val_type': 1, 'dims': [50, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer3_input_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer3_recurrent_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer3_bias', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_rnn3', 'val_type': 1, 'dims': [50, 1, 1, 1024], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_rnn2', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
                input {'name': 'gpu_0/enc_rnn3', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_res3', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
        }
		node {
			'op_type': 'Lstm'
			'attributes': {'direction': 'forward', 'activations': ['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh'], 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/enc_res3', 'val_type': 1, 'dims': [50, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer4_input_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer4_recurrent_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer4_bias', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_rnn4', 'val_type': 1, 'dims': [50, 1, 1, 1024], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_res3', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
                input {'name': 'gpu_0/enc_rnn4', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_res4', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
        }
		node {
			'op_type': 'Lstm'
			'attributes': {'direction': 'forward', 'activations': ['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh'], 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/enc_res4', 'val_type': 1, 'dims': [50, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer5_input_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer5_recurrent_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer5_bias', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_rnn5', 'val_type': 1, 'dims': [50, 1, 1, 1024], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_res4', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
                input {'name': 'gpu_0/enc_rnn5', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_res5', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
        }
		node {
			'op_type': 'Lstm'
			'attributes': {'direction': 'forward', 'activations': ['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh'], 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/enc_res5', 'val_type': 1, 'dims': [50, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer6_input_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer6_recurrent_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer6_bias', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_rnn6', 'val_type': 1, 'dims': [50, 1, 1, 1024], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_res5', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
                input {'name': 'gpu_0/enc_rnn6', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_res6', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
        }
		node {
			'op_type': 'Lstm'
			'attributes': {'direction': 'forward', 'activations': ['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh'], 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/enc_res6', 'val_type': 1, 'dims': [50, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer7_input_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer7_recurrent_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer7_bias', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_rnn7', 'val_type': 1, 'dims': [50, 1, 1, 1024], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/enc_res6', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
                input {'name': 'gpu_0/enc_rnn7', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/enc_res7', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
        }
		node {
			'op_type': 'Lstm'
			'attributes': {'direction': 'forward', 'activations': ['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh'], 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/enc_res7', 'val_type': 1, 'dims': [50, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer8_input_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer8_recurrent_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/enc_layer8_bias', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/enc_rnn8', 'val_type': 1, 'dims': [50, 1, 1, 1024], 'vals': []}
			]
		}
		node {
			'op_type': 'Attention'
			'attributes': {'type': 'bahdanau', 'num_hidden_layers': 1, 'activations': ['tanh'], 'hidden_size': [1024]}
			'inputs': [
				input {'name': 'gpu_0/enc_rnn8', 'val_type': 1, 'dims': [1, 1024, 1, 50], 'vals': []}
				input {'name': 'gpu_0/dec_rnn1', 'val_type': 1, 'dims': [1, 1024, 1, 50], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/attn_1', 'val_type': 1, 'dims': [1, 1024, 1, 50], 'vals': []}
			]
		}
		node {
			'op_type': 'Lstm'
			'attributes': {'direction': 'forward', 'activations': ['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh'], 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/attn_1', 'val_type': 1, 'dims': [50, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer1_input_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer1_recurrent_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer1_bias', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_rnn1', 'val_type': 1, 'dims': [50, 1, 1, 1024], 'vals': []}
			]
		}
		node {
			'op_type': 'Lstm'
			'attributes': {'direction': 'forward', 'activations': ['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh'], 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/dec_rnn1', 'val_type': 1, 'dims': [50, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer2_input_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer2_recurrent_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer2_bias', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_rnn2', 'val_type': 1, 'dims': [50, 1, 1, 1024], 'vals': []}
			]
		}
		node {
			'op_type': 'Lstm'
			'attributes': {'direction': 'forward', 'activations': ['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh'], 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/dec_rnn2', 'val_type': 1, 'dims': [50, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer3_input_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer3_recurrent_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer3_bias', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_rnn3', 'val_type': 1, 'dims': [50, 1, 1, 1024], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_rnn2', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
                input {'name': 'gpu_0/dec_rnn3', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_res3', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
        }
		node {
			'op_type': 'Lstm'
			'attributes': {'direction': 'forward', 'activations': ['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh'], 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/dec_res3', 'val_type': 1, 'dims': [50, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer4_input_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer4_recurrent_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer4_bias', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_rnn4', 'val_type': 1, 'dims': [50, 1, 1, 1024], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_res3', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
                input {'name': 'gpu_0/dec_rnn4', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_res4', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
        }
		node {
			'op_type': 'Lstm'
			'attributes': {'direction': 'forward', 'activations': ['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh'], 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/dec_res4', 'val_type': 1, 'dims': [50, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer5_input_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer5_recurrent_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer5_bias', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_rnn5', 'val_type': 1, 'dims': [50, 1, 1, 1024], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_res4', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
                input {'name': 'gpu_0/dec_rnn5', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_res5', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
        }
		node {
			'op_type': 'Lstm'
			'attributes': {'direction': 'forward', 'activations': ['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh'], 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/dec_res5', 'val_type': 1, 'dims': [50, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer6_input_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer6_recurrent_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer6_bias', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_rnn6', 'val_type': 1, 'dims': [50, 1, 1, 1024], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_res5', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
                input {'name': 'gpu_0/dec_rnn6', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_res6', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
        }
		node {
			'op_type': 'Lstm'
			'attributes': {'direction': 'forward', 'activations': ['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh'], 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/dec_res6', 'val_type': 1, 'dims': [50, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer7_input_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer7_recurrent_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer7_bias', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_rnn7', 'val_type': 1, 'dims': [50, 1, 1, 1024], 'vals': []}
			]
		}
		node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_res6', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
                input {'name': 'gpu_0/dec_rnn7', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/dec_res7', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
        }
		node {
			'op_type': 'Lstm'
			'attributes': {'direction': 'forward', 'activations': ['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh'], 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/dec_res7', 'val_type': 1, 'dims': [50, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer8_input_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer8_recurrent_weights', 'val_type': 1, 'dims': [1, 4096, 1024], 'vals': []}
				input {'name': 'gpu_0/dec_layer8_bias', 'val_type': 1, 'dims': [1, 1024], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/dec_rnn8', 'val_type': 1, 'dims': [50, 1, 1, 1024], 'vals': []}
			]
		}
		node {
            'op_type': 'Softmax'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/dec_rnn8', 'val_type': 1, 'dims': [1, 50, 1, 1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/softmax_1', 'val_type': 1, 'dims': [1, 1, 1, 50], 'vals': []}
            ]
        }
 ]
}
