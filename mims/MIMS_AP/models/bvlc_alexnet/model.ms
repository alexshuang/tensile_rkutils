'ir_version': 1
'producer_name': 'onnx-mims'
'graph': {
    'name': 'bvlc_alexnet'
    'nodes': [
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [4, 4], 'pads': [0, 0, 0, 0], 'kernel_shape': [11, 11]}
            'inputs': [
                input {'name': 'data_0', 'val_type': 1, 'dims': [200704, 3, 224, 224], 'vals': []}
                input {'name': 'conv1_w_0', 'val_type': 1, 'dims': [96, 3, 11, 11], 'vals': []}
                input {'name': 'conv1_b_0', 'val_type': 1, 'dims': [96], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv1_1', 'val_type': 1, 'dims': [200704, 96, 54, 54], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv1_1', 'val_type': 1, 'dims': [200704, 96, 54, 54], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv1_2', 'val_type': 1, 'dims': [200704, 96, 54, 54], 'vals': []}
            ]
        }
        node {
            'op_type': 'LRN'
            'attributes': {'size': 5, 'alpha': 9.999999747378752e-05, 'beta': 0.75, 'bias': 1.0}
            'inputs': [
                input {'name': 'conv1_2', 'val_type': 1, 'dims': [200704, 96, 54, 54], 'vals': []}
            ]
            'outputs': [
                output {'name': 'norm1_1', 'val_type': 1, 'dims': [200704, 96, 54, 54], 'vals': []}
            ]
        }
        node {
            'op_type': 'MaxPool'
            'attributes': {'strides': [2, 2], 'pads': [0, 0, 0, 0], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'norm1_1', 'val_type': 1, 'dims': [200704, 96, 54, 54], 'vals': []}
            ]
            'outputs': [
                output {'name': 'pool1_1', 'val_type': 1, 'dims': [200704, 96, 26, 26], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'group': 2, 'strides': [1, 1], 'pads': [2, 2, 2, 2], 'kernel_shape': [5, 5]}
            'inputs': [
                input {'name': 'pool1_1', 'val_type': 1, 'dims': [200704, 96, 26, 26], 'vals': []}
                input {'name': 'conv2_w_0', 'val_type': 1, 'dims': [256, 48, 5, 5], 'vals': []}
                input {'name': 'conv2_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_1', 'val_type': 1, 'dims': [200704, 256, 26, 26], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv2_1', 'val_type': 1, 'dims': [200704, 256, 26, 26], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_2', 'val_type': 1, 'dims': [200704, 256, 26, 26], 'vals': []}
            ]
        }
        node {
            'op_type': 'LRN'
            'attributes': {'size': 5, 'alpha': 9.999999747378752e-05, 'beta': 0.75, 'bias': 1.0}
            'inputs': [
                input {'name': 'conv2_2', 'val_type': 1, 'dims': [200704, 256, 26, 26], 'vals': []}
            ]
            'outputs': [
                output {'name': 'norm2_1', 'val_type': 1, 'dims': [200704, 256, 26, 26], 'vals': []}
            ]
        }
        node {
            'op_type': 'MaxPool'
            'attributes': {'strides': [2, 2], 'pads': [0, 0, 0, 0], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'norm2_1', 'val_type': 1, 'dims': [200704, 256, 26, 26], 'vals': []}
            ]
            'outputs': [
                output {'name': 'pool2_1', 'val_type': 1, 'dims': [200704, 256, 12, 12], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'pool2_1', 'val_type': 1, 'dims': [200704, 256, 12, 12], 'vals': []}
                input {'name': 'conv3_w_0', 'val_type': 1, 'dims': [384, 256, 3, 3], 'vals': []}
                input {'name': 'conv3_b_0', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_1', 'val_type': 1, 'dims': [200704, 384, 12, 12], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_1', 'val_type': 1, 'dims': [200704, 384, 12, 12], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_2', 'val_type': 1, 'dims': [200704, 384, 12, 12], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'group': 2, 'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv3_2', 'val_type': 1, 'dims': [200704, 384, 12, 12], 'vals': []}
                input {'name': 'conv4_w_0', 'val_type': 1, 'dims': [384, 192, 3, 3], 'vals': []}
                input {'name': 'conv4_b_0', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_1', 'val_type': 1, 'dims': [200704, 384, 12, 12], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_1', 'val_type': 1, 'dims': [200704, 384, 12, 12], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_2', 'val_type': 1, 'dims': [200704, 384, 12, 12], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'group': 2, 'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_2', 'val_type': 1, 'dims': [200704, 384, 12, 12], 'vals': []}
                input {'name': 'conv5_w_0', 'val_type': 1, 'dims': [256, 192, 3, 3], 'vals': []}
                input {'name': 'conv5_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_1', 'val_type': 1, 'dims': [200704, 256, 12, 12], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_1', 'val_type': 1, 'dims': [200704, 256, 12, 12], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_2', 'val_type': 1, 'dims': [200704, 256, 12, 12], 'vals': []}
            ]
        }
        node {
            'op_type': 'MaxPool'
            'attributes': {'strides': [2, 2], 'pads': [0, 0, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_2', 'val_type': 1, 'dims': [200704, 256, 12, 12], 'vals': []}
            ]
            'outputs': [
                output {'name': 'pool5_1', 'val_type': 1, 'dims': [200704, 256, 6, 6], 'vals': []}
            ]
        }
        node {
            'op_type': 'Gemm'
            'attributes': {'broadcast': 1, 'transB': 1}
            'inputs': [
                input {'name': 'pool5_1', 'val_type': None, 'dims': [200704, 9216, 1, 1], 'vals': []}
                input {'name': 'fc6_w_0', 'val_type': 1, 'dims': [4096, 9216], 'vals': []}
                input {'name': 'fc6_b_0', 'val_type': 1, 'dims': [4096], 'vals': []}
            ]
            'outputs': [
                output {'name': 'fc6_1', 'val_type': None, 'dims': [200704, 1, 4096, 1], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'fc6_1', 'val_type': None, 'dims': [200704, 1, 4096, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'fc6_2', 'val_type': None, 'dims': [200704, 1, 4096, 1], 'vals': []}
            ]
        }
        node {
            'op_type': 'Dropout'
            'attributes': {'ratio': 0.5, 'is_test': 1}
            'inputs': [
                input {'name': 'fc6_2', 'val_type': None, 'dims': [200704, 1, 4096, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'fc6_3', 'val_type': None, 'dims': [200704, 1, 4096, 1], 'vals': []}
                output {'name': '_fc6_mask_1', 'val_type': None, 'dims': [200704, 1, 4096, 1], 'vals': []}
            ]
        }
        node {
            'op_type': 'Gemm'
            'attributes': {'broadcast': 1, 'transB': 1}
            'inputs': [
                input {'name': 'fc6_3', 'val_type': None, 'dims': [200704, 4096, 1, 1], 'vals': []}
                input {'name': 'fc7_w_0', 'val_type': 1, 'dims': [4096, 4096], 'vals': []}
                input {'name': 'fc7_b_0', 'val_type': 1, 'dims': [4096], 'vals': []}
            ]
            'outputs': [
                output {'name': 'fc7_1', 'val_type': None, 'dims': [200704, 1, 4096, 1], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'fc7_1', 'val_type': None, 'dims': [200704, 1, 4096, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'fc7_2', 'val_type': None, 'dims': [200704, 1, 4096, 1], 'vals': []}
            ]
        }
        node {
            'op_type': 'Dropout'
            'attributes': {'ratio': 0.5, 'is_test': 1}
            'inputs': [
                input {'name': 'fc7_2', 'val_type': None, 'dims': [200704, 1, 4096, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'fc7_3', 'val_type': None, 'dims': [200704, 1, 4096, 1], 'vals': []}
                output {'name': '_fc7_mask_1', 'val_type': None, 'dims': [200704, 1, 4096, 1], 'vals': []}
            ]
        }
        node {
            'op_type': 'Gemm'
            'attributes': {'broadcast': 1, 'transB': 1}
            'inputs': [
                input {'name': 'fc7_3', 'val_type': None, 'dims': [200704, 4096, 1, 1], 'vals': []}
                input {'name': 'fc8_w_0', 'val_type': 1, 'dims': [1000, 4096], 'vals': []}
                input {'name': 'fc8_b_0', 'val_type': 1, 'dims': [1000], 'vals': []}
            ]
            'outputs': [
                output {'name': 'fc8_1', 'val_type': None, 'dims': [200704, 1, 1000, 1], 'vals': []}
            ]
        }
        node {
            'op_type': 'Softmax'
            'attributes': {}
            'inputs': [
                input {'name': 'fc8_1', 'val_type': None, 'dims': [200704, 1, 1000, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'prob_1', 'val_type': None, 'dims': [200704, 1, 1000, 1], 'vals': []}
            ]
        }
    ]
}
