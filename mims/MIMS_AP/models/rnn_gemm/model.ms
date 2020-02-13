'ir_version': 1
'producer_name': 'onnx-mims'
'graph': {
    'name': 'deepspeech2'
    'nodes': [
		node {
			'op_type': 'Rnn'
			'attributes': {'direction': 'bidirectional', 'activations': 'relu', 'hidden_size': 1024}
			'inputs': [
				input {'name': 'gpu_0/conv1_1', 'val_type': 1, 'dims': [1000, 1, 1024], 'vals': []}
				input {'name': 'gpu_0/layer1_input_weights', 'val_type': 1, 'dims': [2, 1024, 1024], 'vals': []}
				input {'name': 'gpu_0/layer1_recurrent_weights', 'val_type': 1, 'dims': [2, 1024, 1024], 'vals': []}
				input {'name': 'gpu_0/layer1_bias', 'val_type': 1, 'dims': [2, 2048], 'vals': []}
			]
			'outputs': [
				output {'name': 'gpu_0/rnn1', 'val_type': 1, 'dims': [1000, 2, 1, 1024], 'vals': []}
			]
                }
	    ]
        }
