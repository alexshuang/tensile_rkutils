'ir_version': 3
'producer_name': 'onnx-mims'
'graph': {
    'name': 'mmdnn'
    'nodes': [
        node {
            'op_type': 'Transpose'
            'attributes': {'perm': [0, 3, 1, 2]}
            'inputs': [
                input {'name': 'data_orig', 'val_type': 1, 'dims': [32, 299, 299, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'data', 'val_type': 1, 'dims': [32, 3, 299, 299], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}
            'inputs': [
                input {'name': 'data', 'val_type': 1, 'dims': [32, 3, 299, 299], 'vals': []}
                input {'name': 'conv_conv2d_weight', 'val_type': 1, 'dims': [32, 3, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv_conv2d', 'val_type': 1, 'dims': [32, 32, 149, 149], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'conv_conv2d', 'val_type': 1, 'dims': [32, 32, 149, 149], 'vals': []}
                input {'name': 'conv_batchnorm_scale', 'val_type': 1, 'dims': [32], 'vals': []}
                input {'name': 'conv_batchnorm_bias', 'val_type': 1, 'dims': [32], 'vals': []}
                input {'name': 'conv_batchnorm_mean', 'val_type': 1, 'dims': [32], 'vals': []}
                input {'name': 'conv_batchnorm_var', 'val_type': 1, 'dims': [32], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv_batchnorm', 'val_type': 1, 'dims': [32, 32, 149, 149], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv_batchnorm', 'val_type': 1, 'dims': [32, 32, 149, 149], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv_relu', 'val_type': 1, 'dims': [32, 32, 149, 149], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'conv_relu', 'val_type': 1, 'dims': [32, 32, 149, 149], 'vals': []}
                input {'name': 'conv_1_conv2d_weight', 'val_type': 1, 'dims': [32, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv_1_conv2d', 'val_type': 1, 'dims': [32, 32, 147, 147], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'conv_1_conv2d', 'val_type': 1, 'dims': [32, 32, 147, 147], 'vals': []}
                input {'name': 'conv_1_batchnorm_scale', 'val_type': 1, 'dims': [32], 'vals': []}
                input {'name': 'conv_1_batchnorm_bias', 'val_type': 1, 'dims': [32], 'vals': []}
                input {'name': 'conv_1_batchnorm_mean', 'val_type': 1, 'dims': [32], 'vals': []}
                input {'name': 'conv_1_batchnorm_var', 'val_type': 1, 'dims': [32], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv_1_batchnorm', 'val_type': 1, 'dims': [32, 32, 147, 147], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv_1_batchnorm', 'val_type': 1, 'dims': [32, 32, 147, 147], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv_1_relu', 'val_type': 1, 'dims': [32, 32, 147, 147], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'conv_1_relu', 'val_type': 1, 'dims': [32, 32, 147, 147], 'vals': []}
                input {'name': 'conv_2_conv2d_weight', 'val_type': 1, 'dims': [64, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv_2_conv2d', 'val_type': 1, 'dims': [32, 64, 147, 147], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'conv_2_conv2d', 'val_type': 1, 'dims': [32, 64, 147, 147], 'vals': []}
                input {'name': 'conv_2_batchnorm_scale', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'conv_2_batchnorm_bias', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'conv_2_batchnorm_mean', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'conv_2_batchnorm_var', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv_2_batchnorm', 'val_type': 1, 'dims': [32, 64, 147, 147], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv_2_batchnorm', 'val_type': 1, 'dims': [32, 64, 147, 147], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv_2_relu', 'val_type': 1, 'dims': [32, 64, 147, 147], 'vals': []}
            ]
        }
        node {
            'op_type': 'MaxPool'
            'attributes': {'kernel_shape': [3, 3], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}
            'inputs': [
                input {'name': 'conv_2_relu', 'val_type': 1, 'dims': [32, 64, 147, 147], 'vals': []}
            ]
            'outputs': [
                output {'name': 'pool', 'val_type': 1, 'dims': [32, 64, 73, 73], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'pool', 'val_type': 1, 'dims': [32, 64, 73, 73], 'vals': []}
                input {'name': 'conv_3_conv2d_weight', 'val_type': 1, 'dims': [80, 64, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv_3_conv2d', 'val_type': 1, 'dims': [32, 80, 73, 73], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'conv_3_conv2d', 'val_type': 1, 'dims': [32, 80, 73, 73], 'vals': []}
                input {'name': 'conv_3_batchnorm_scale', 'val_type': 1, 'dims': [80], 'vals': []}
                input {'name': 'conv_3_batchnorm_bias', 'val_type': 1, 'dims': [80], 'vals': []}
                input {'name': 'conv_3_batchnorm_mean', 'val_type': 1, 'dims': [80], 'vals': []}
                input {'name': 'conv_3_batchnorm_var', 'val_type': 1, 'dims': [80], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv_3_batchnorm', 'val_type': 1, 'dims': [32, 80, 73, 73], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv_3_batchnorm', 'val_type': 1, 'dims': [32, 80, 73, 73], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv_3_relu', 'val_type': 1, 'dims': [32, 80, 73, 73], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'conv_3_relu', 'val_type': 1, 'dims': [32, 80, 73, 73], 'vals': []}
                input {'name': 'conv_4_conv2d_weight', 'val_type': 1, 'dims': [192, 80, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv_4_conv2d', 'val_type': 1, 'dims': [32, 192, 71, 71], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'conv_4_conv2d', 'val_type': 1, 'dims': [32, 192, 71, 71], 'vals': []}
                input {'name': 'conv_4_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'conv_4_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'conv_4_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'conv_4_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv_4_batchnorm', 'val_type': 1, 'dims': [32, 192, 71, 71], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv_4_batchnorm', 'val_type': 1, 'dims': [32, 192, 71, 71], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv_4_relu', 'val_type': 1, 'dims': [32, 192, 71, 71], 'vals': []}
            ]
        }
        node {
            'op_type': 'MaxPool'
            'attributes': {'kernel_shape': [3, 3], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}
            'inputs': [
                input {'name': 'conv_4_relu', 'val_type': 1, 'dims': [32, 192, 71, 71], 'vals': []}
            ]
            'outputs': [
                output {'name': 'pool1', 'val_type': 1, 'dims': [32, 192, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'pool1', 'val_type': 1, 'dims': [32, 192, 35, 35], 'vals': []}
                input {'name': 'mixed_conv_conv2d_weight', 'val_type': 1, 'dims': [64, 192, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_conv_conv2d', 'val_type': 1, 'dims': [32, 256, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'pool1', 'val_type': 1, 'dims': [32, 192, 35, 35], 'vals': []}
                input {'name': 'mixed_tower_conv_conv2d_weight', 'val_type': 1, 'dims': [48, 192, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'pool1', 'val_type': 1, 'dims': [32, 192, 35, 35], 'vals': []}
                input {'name': 'mixed_tower_1_conv_conv2d_weight', 'val_type': 1, 'dims': [64, 192, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'AveragePool'
            'attributes': {'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'pool1', 'val_type': 1, 'dims': [32, 192, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'avg_pool_mixed_pool', 'val_type': 1, 'dims': [32, 192, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_conv_conv2d', 'val_type': 1, 'dims': [32, 256, 35, 35], 'vals': []}
                input {'name': 'mixed_conv_batchnorm_scale', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_conv_batchnorm_bias', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_conv_batchnorm_mean', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_conv_batchnorm_var', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_conv_batchnorm', 'val_type': 1, 'dims': [32, 256, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
                input {'name': 'mixed_tower_conv_batchnorm_scale', 'val_type': 1, 'dims': [48], 'vals': []}
                input {'name': 'mixed_tower_conv_batchnorm_bias', 'val_type': 1, 'dims': [48], 'vals': []}
                input {'name': 'mixed_tower_conv_batchnorm_mean', 'val_type': 1, 'dims': [48], 'vals': []}
                input {'name': 'mixed_tower_conv_batchnorm_var', 'val_type': 1, 'dims': [48], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
                input {'name': 'mixed_tower_1_conv_batchnorm_scale', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_tower_1_conv_batchnorm_bias', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_tower_1_conv_batchnorm_mean', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_tower_1_conv_batchnorm_var', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'avg_pool_mixed_pool', 'val_type': 1, 'dims': [32, 192, 35, 35], 'vals': []}
                input {'name': 'mixed_tower_2_conv_conv2d_weight', 'val_type': 1, 'dims': [32, 192, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 32, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_conv_batchnorm', 'val_type': 1, 'dims': [32, 256, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_conv_relu', 'val_type': 1, 'dims': [32, 256, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_conv_relu', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 32, 35, 35], 'vals': []}
                input {'name': 'mixed_tower_2_conv_batchnorm_scale', 'val_type': 1, 'dims': [32], 'vals': []}
                input {'name': 'mixed_tower_2_conv_batchnorm_bias', 'val_type': 1, 'dims': [32], 'vals': []}
                input {'name': 'mixed_tower_2_conv_batchnorm_mean', 'val_type': 1, 'dims': [32], 'vals': []}
                input {'name': 'mixed_tower_2_conv_batchnorm_var', 'val_type': 1, 'dims': [32], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 32, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [5, 5], 'pads': [2, 2, 2, 2], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_tower_conv_relu', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
                input {'name': 'mixed_tower_conv_1_conv2d_weight', 'val_type': 1, 'dims': [64, 48, 5, 5], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
                input {'name': 'mixed_tower_1_conv_1_conv2d_weight', 'val_type': 1, 'dims': [96, 64, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 32, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 32, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
                input {'name': 'mixed_tower_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_tower_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_tower_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_tower_conv_1_batchnorm_var', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
                input {'name': 'mixed_tower_1_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_tower_1_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_tower_1_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_tower_1_conv_1_batchnorm_var', 'val_type': 1, 'dims': [96], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
                input {'name': 'mixed_tower_1_conv_2_conv2d_weight', 'val_type': 1, 'dims': [96, 96, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_1_conv_2_conv2d', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_tower_1_conv_2_conv2d', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
                input {'name': 'mixed_tower_1_conv_2_batchnorm_scale', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_tower_1_conv_2_batchnorm_bias', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_tower_1_conv_2_batchnorm_mean', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_tower_1_conv_2_batchnorm_var', 'val_type': 1, 'dims': [96], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_1_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_tower_1_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_tower_1_conv_2_relu', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'mixed_conv_relu', 'val_type': 1, 'dims': [32, 256, 35, 35], 'vals': []}
                input {'name': 'mixed_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
                input {'name': 'mixed_tower_1_conv_2_relu', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
                input {'name': 'mixed_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 32, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'ch_concat_mixed_chconcat', 'val_type': 1, 'dims': [32, 256, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_chconcat', 'val_type': 1, 'dims': [32, 256, 35, 35], 'vals': []}
                input {'name': 'mixed_1_conv_conv2d_weight', 'val_type': 1, 'dims': [64, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_conv_conv2d', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_chconcat', 'val_type': 1, 'dims': [32, 256, 35, 35], 'vals': []}
                input {'name': 'mixed_1_tower_conv_conv2d_weight', 'val_type': 1, 'dims': [48, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_chconcat', 'val_type': 1, 'dims': [32, 256, 35, 35], 'vals': []}
                input {'name': 'mixed_1_tower_1_conv_conv2d_weight', 'val_type': 1, 'dims': [64, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'AveragePool'
            'attributes': {'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_chconcat', 'val_type': 1, 'dims': [32, 256, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'avg_pool_mixed_1_pool', 'val_type': 1, 'dims': [32, 256, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_1_conv_conv2d', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
                input {'name': 'mixed_1_conv_batchnorm_scale', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_1_conv_batchnorm_bias', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_1_conv_batchnorm_mean', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_1_conv_batchnorm_var', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_1_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
                input {'name': 'mixed_1_tower_conv_batchnorm_scale', 'val_type': 1, 'dims': [48], 'vals': []}
                input {'name': 'mixed_1_tower_conv_batchnorm_bias', 'val_type': 1, 'dims': [48], 'vals': []}
                input {'name': 'mixed_1_tower_conv_batchnorm_mean', 'val_type': 1, 'dims': [48], 'vals': []}
                input {'name': 'mixed_1_tower_conv_batchnorm_var', 'val_type': 1, 'dims': [48], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_1_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
                input {'name': 'mixed_1_tower_1_conv_batchnorm_scale', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_1_tower_1_conv_batchnorm_bias', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_1_tower_1_conv_batchnorm_mean', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_1_tower_1_conv_batchnorm_var', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'avg_pool_mixed_1_pool', 'val_type': 1, 'dims': [32, 256, 35, 35], 'vals': []}
                input {'name': 'mixed_1_tower_2_conv_conv2d_weight', 'val_type': 1, 'dims': [64, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_conv_relu', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_1_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_conv_relu', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_1_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_1_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
                input {'name': 'mixed_1_tower_2_conv_batchnorm_scale', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_1_tower_2_conv_batchnorm_bias', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_1_tower_2_conv_batchnorm_mean', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_1_tower_2_conv_batchnorm_var', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [5, 5], 'pads': [2, 2, 2, 2], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_1_tower_conv_relu', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
                input {'name': 'mixed_1_tower_conv_1_conv2d_weight', 'val_type': 1, 'dims': [64, 48, 5, 5], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_1_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
                input {'name': 'mixed_1_tower_1_conv_1_conv2d_weight', 'val_type': 1, 'dims': [96, 64, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_1_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_1_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
                input {'name': 'mixed_1_tower_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_1_tower_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_1_tower_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_1_tower_conv_1_batchnorm_var', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_1_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
                input {'name': 'mixed_1_tower_1_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_1_tower_1_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_1_tower_1_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_1_tower_1_conv_1_batchnorm_var', 'val_type': 1, 'dims': [96], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_1_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_1_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_1_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
                input {'name': 'mixed_1_tower_1_conv_2_conv2d_weight', 'val_type': 1, 'dims': [96, 96, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_1_conv_2_conv2d', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_1_tower_1_conv_2_conv2d', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
                input {'name': 'mixed_1_tower_1_conv_2_batchnorm_scale', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_1_tower_1_conv_2_batchnorm_bias', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_1_tower_1_conv_2_batchnorm_mean', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_1_tower_1_conv_2_batchnorm_var', 'val_type': 1, 'dims': [96], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_1_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_1_tower_1_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_1_tower_1_conv_2_relu', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'mixed_1_conv_relu', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
                input {'name': 'mixed_1_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
                input {'name': 'mixed_1_tower_1_conv_2_relu', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
                input {'name': 'mixed_1_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'ch_concat_mixed_1_chconcat', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_1_chconcat', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
                input {'name': 'mixed_2_conv_conv2d_weight', 'val_type': 1, 'dims': [64, 288, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_conv_conv2d', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_1_chconcat', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
                input {'name': 'mixed_2_tower_conv_conv2d_weight', 'val_type': 1, 'dims': [48, 288, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_1_chconcat', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
                input {'name': 'mixed_2_tower_1_conv_conv2d_weight', 'val_type': 1, 'dims': [64, 288, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'AveragePool'
            'attributes': {'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_1_chconcat', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'avg_pool_mixed_2_pool', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_2_conv_conv2d', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
                input {'name': 'mixed_2_conv_batchnorm_scale', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_2_conv_batchnorm_bias', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_2_conv_batchnorm_mean', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_2_conv_batchnorm_var', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_2_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
                input {'name': 'mixed_2_tower_conv_batchnorm_scale', 'val_type': 1, 'dims': [48], 'vals': []}
                input {'name': 'mixed_2_tower_conv_batchnorm_bias', 'val_type': 1, 'dims': [48], 'vals': []}
                input {'name': 'mixed_2_tower_conv_batchnorm_mean', 'val_type': 1, 'dims': [48], 'vals': []}
                input {'name': 'mixed_2_tower_conv_batchnorm_var', 'val_type': 1, 'dims': [48], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_2_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
                input {'name': 'mixed_2_tower_1_conv_batchnorm_scale', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_2_tower_1_conv_batchnorm_bias', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_2_tower_1_conv_batchnorm_mean', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_2_tower_1_conv_batchnorm_var', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'avg_pool_mixed_2_pool', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
                input {'name': 'mixed_2_tower_2_conv_conv2d_weight', 'val_type': 1, 'dims': [64, 288, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_conv_relu', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_2_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_conv_relu', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_2_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_2_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
                input {'name': 'mixed_2_tower_2_conv_batchnorm_scale', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_2_tower_2_conv_batchnorm_bias', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_2_tower_2_conv_batchnorm_mean', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_2_tower_2_conv_batchnorm_var', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [5, 5], 'pads': [2, 2, 2, 2], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_2_tower_conv_relu', 'val_type': 1, 'dims': [32, 48, 35, 35], 'vals': []}
                input {'name': 'mixed_2_tower_conv_1_conv2d_weight', 'val_type': 1, 'dims': [64, 48, 5, 5], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_2_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
                input {'name': 'mixed_2_tower_1_conv_1_conv2d_weight', 'val_type': 1, 'dims': [96, 64, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_2_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_2_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
                input {'name': 'mixed_2_tower_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_2_tower_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_2_tower_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_2_tower_conv_1_batchnorm_var', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_2_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
                input {'name': 'mixed_2_tower_1_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_2_tower_1_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_2_tower_1_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_2_tower_1_conv_1_batchnorm_var', 'val_type': 1, 'dims': [96], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_2_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_2_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_2_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
                input {'name': 'mixed_2_tower_1_conv_2_conv2d_weight', 'val_type': 1, 'dims': [96, 96, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_1_conv_2_conv2d', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_2_tower_1_conv_2_conv2d', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
                input {'name': 'mixed_2_tower_1_conv_2_batchnorm_scale', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_2_tower_1_conv_2_batchnorm_bias', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_2_tower_1_conv_2_batchnorm_mean', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_2_tower_1_conv_2_batchnorm_var', 'val_type': 1, 'dims': [96], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_1_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_2_tower_1_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_2_tower_1_conv_2_relu', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'mixed_2_conv_relu', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
                input {'name': 'mixed_2_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
                input {'name': 'mixed_2_tower_1_conv_2_relu', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
                input {'name': 'mixed_2_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'ch_concat_mixed_2_chconcat', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}
            'inputs': [
                input {'name': 'ch_concat_mixed_2_chconcat', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
                input {'name': 'mixed_3_conv_conv2d_weight', 'val_type': 1, 'dims': [384, 288, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_3_conv_conv2d', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_2_chconcat', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
                input {'name': 'mixed_3_tower_conv_conv2d_weight', 'val_type': 1, 'dims': [64, 288, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_3_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'MaxPool'
            'attributes': {'kernel_shape': [3, 3], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}
            'inputs': [
                input {'name': 'ch_concat_mixed_2_chconcat', 'val_type': 1, 'dims': [32, 288, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'max_pool_mixed_3_pool', 'val_type': 1, 'dims': [32, 288, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_3_conv_conv2d', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_3_conv_batchnorm_scale', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_3_conv_batchnorm_bias', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_3_conv_batchnorm_mean', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_3_conv_batchnorm_var', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_3_conv_batchnorm', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_3_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
                input {'name': 'mixed_3_tower_conv_batchnorm_scale', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_3_tower_conv_batchnorm_bias', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_3_tower_conv_batchnorm_mean', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'mixed_3_tower_conv_batchnorm_var', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_3_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_3_conv_batchnorm', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_3_conv_relu', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_3_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_3_tower_conv_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_3_tower_conv_relu', 'val_type': 1, 'dims': [32, 64, 35, 35], 'vals': []}
                input {'name': 'mixed_3_tower_conv_1_conv2d_weight', 'val_type': 1, 'dims': [96, 64, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_3_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_3_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
                input {'name': 'mixed_3_tower_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_3_tower_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_3_tower_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_3_tower_conv_1_batchnorm_var', 'val_type': 1, 'dims': [96], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_3_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_3_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_3_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}
            'inputs': [
                input {'name': 'mixed_3_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 96, 35, 35], 'vals': []}
                input {'name': 'mixed_3_tower_conv_2_conv2d_weight', 'val_type': 1, 'dims': [96, 96, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_3_tower_conv_2_conv2d', 'val_type': 1, 'dims': [32, 96, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_3_tower_conv_2_conv2d', 'val_type': 1, 'dims': [32, 96, 17, 17], 'vals': []}
                input {'name': 'mixed_3_tower_conv_2_batchnorm_scale', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_3_tower_conv_2_batchnorm_bias', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_3_tower_conv_2_batchnorm_mean', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'mixed_3_tower_conv_2_batchnorm_var', 'val_type': 1, 'dims': [96], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_3_tower_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 96, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_3_tower_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 96, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_3_tower_conv_2_relu', 'val_type': 1, 'dims': [32, 96, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'mixed_3_conv_relu', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_3_tower_conv_2_relu', 'val_type': 1, 'dims': [32, 96, 17, 17], 'vals': []}
                input {'name': 'max_pool_mixed_3_pool', 'val_type': 1, 'dims': [32, 288, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'ch_concat_mixed_3_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_3_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_4_conv_conv2d_weight', 'val_type': 1, 'dims': [192, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_conv_conv2d', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_3_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_conv_conv2d_weight', 'val_type': 1, 'dims': [128, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_3_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_conv2d_weight', 'val_type': 1, 'dims': [128, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'AveragePool'
            'attributes': {'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_3_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'avg_pool_mixed_4_pool', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_4_conv_conv2d', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_4_conv_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_4_conv_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_4_conv_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_4_conv_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_conv_batchnorm', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_4_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_conv_batchnorm_scale', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_conv_batchnorm_bias', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_conv_batchnorm_mean', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_conv_batchnorm_var', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_4_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_batchnorm_scale', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_batchnorm_bias', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_batchnorm_mean', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_batchnorm_var', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'avg_pool_mixed_4_pool', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_2_conv_conv2d_weight', 'val_type': 1, 'dims': [192, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_4_conv_batchnorm', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_conv_relu', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_4_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_conv_relu', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_4_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_4_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_2_conv_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_4_tower_2_conv_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_4_tower_2_conv_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_4_tower_2_conv_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 7], 'pads': [0, 3, 0, 3], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_4_tower_conv_relu', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_conv_1_conv2d_weight', 'val_type': 1, 'dims': [128, 128, 1, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 1], 'pads': [3, 0, 3, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_4_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_1_conv2d_weight', 'val_type': 1, 'dims': [128, 128, 7, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_4_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_4_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_conv_1_batchnorm_var', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_4_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_1_batchnorm_var', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_4_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_4_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 1], 'pads': [3, 0, 3, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_4_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_conv_2_conv2d_weight', 'val_type': 1, 'dims': [192, 128, 7, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_conv_2_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 7], 'pads': [0, 3, 0, 3], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_4_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_2_conv2d_weight', 'val_type': 1, 'dims': [128, 128, 1, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_1_conv_2_conv2d', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_4_tower_conv_2_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_conv_2_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_4_tower_conv_2_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_4_tower_conv_2_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_4_tower_conv_2_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_4_tower_1_conv_2_conv2d', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_2_batchnorm_scale', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_2_batchnorm_bias', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_2_batchnorm_mean', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_2_batchnorm_var', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_1_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_4_tower_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_conv_2_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_4_tower_1_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_1_conv_2_relu', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 1], 'pads': [3, 0, 3, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_4_tower_1_conv_2_relu', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_3_conv2d_weight', 'val_type': 1, 'dims': [128, 128, 7, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_1_conv_3_conv2d', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_4_tower_1_conv_3_conv2d', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_3_batchnorm_scale', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_3_batchnorm_bias', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_3_batchnorm_mean', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_3_batchnorm_var', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_1_conv_3_batchnorm', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_4_tower_1_conv_3_batchnorm', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_1_conv_3_relu', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 7], 'pads': [0, 3, 0, 3], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_4_tower_1_conv_3_relu', 'val_type': 1, 'dims': [32, 128, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_4_conv2d_weight', 'val_type': 1, 'dims': [192, 128, 1, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_1_conv_4_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_4_tower_1_conv_4_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_4_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_4_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_4_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_4_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_1_conv_4_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_4_tower_1_conv_4_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_4_tower_1_conv_4_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'mixed_4_conv_relu', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_conv_2_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_1_conv_4_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_4_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'ch_concat_mixed_4_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_4_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_5_conv_conv2d_weight', 'val_type': 1, 'dims': [192, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_conv_conv2d', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_4_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_conv_conv2d_weight', 'val_type': 1, 'dims': [160, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_4_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_conv2d_weight', 'val_type': 1, 'dims': [160, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'AveragePool'
            'attributes': {'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_4_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'avg_pool_mixed_5_pool', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_5_conv_conv2d', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_5_conv_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_5_conv_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_5_conv_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_5_conv_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_conv_batchnorm', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_5_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_conv_batchnorm_scale', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_conv_batchnorm_bias', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_conv_batchnorm_mean', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_conv_batchnorm_var', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_5_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_batchnorm_scale', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_batchnorm_bias', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_batchnorm_mean', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_batchnorm_var', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'avg_pool_mixed_5_pool', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_2_conv_conv2d_weight', 'val_type': 1, 'dims': [192, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_5_conv_batchnorm', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_conv_relu', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_5_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_conv_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_5_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_5_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_2_conv_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_5_tower_2_conv_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_5_tower_2_conv_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_5_tower_2_conv_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 7], 'pads': [0, 3, 0, 3], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_5_tower_conv_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_conv_1_conv2d_weight', 'val_type': 1, 'dims': [160, 160, 1, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 1], 'pads': [3, 0, 3, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_5_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_1_conv2d_weight', 'val_type': 1, 'dims': [160, 160, 7, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_5_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_5_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_conv_1_batchnorm_var', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_5_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_1_batchnorm_var', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_5_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_5_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 1], 'pads': [3, 0, 3, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_5_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_conv_2_conv2d_weight', 'val_type': 1, 'dims': [192, 160, 7, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_conv_2_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 7], 'pads': [0, 3, 0, 3], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_5_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_2_conv2d_weight', 'val_type': 1, 'dims': [160, 160, 1, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_1_conv_2_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_5_tower_conv_2_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_conv_2_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_5_tower_conv_2_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_5_tower_conv_2_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_5_tower_conv_2_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_5_tower_1_conv_2_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_2_batchnorm_scale', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_2_batchnorm_bias', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_2_batchnorm_mean', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_2_batchnorm_var', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_1_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_5_tower_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_conv_2_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_5_tower_1_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_1_conv_2_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 1], 'pads': [3, 0, 3, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_5_tower_1_conv_2_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_3_conv2d_weight', 'val_type': 1, 'dims': [160, 160, 7, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_1_conv_3_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_5_tower_1_conv_3_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_3_batchnorm_scale', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_3_batchnorm_bias', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_3_batchnorm_mean', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_3_batchnorm_var', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_1_conv_3_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_5_tower_1_conv_3_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_1_conv_3_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 7], 'pads': [0, 3, 0, 3], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_5_tower_1_conv_3_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_4_conv2d_weight', 'val_type': 1, 'dims': [192, 160, 1, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_1_conv_4_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_5_tower_1_conv_4_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_4_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_4_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_4_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_4_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_1_conv_4_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_5_tower_1_conv_4_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_5_tower_1_conv_4_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'mixed_5_conv_relu', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_conv_2_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_1_conv_4_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_5_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'ch_concat_mixed_5_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_5_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_6_conv_conv2d_weight', 'val_type': 1, 'dims': [192, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_conv_conv2d', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_5_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_conv_conv2d_weight', 'val_type': 1, 'dims': [160, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_5_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_conv2d_weight', 'val_type': 1, 'dims': [160, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'AveragePool'
            'attributes': {'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_5_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'avg_pool_mixed_6_pool', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_6_conv_conv2d', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_6_conv_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_6_conv_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_6_conv_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_6_conv_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_conv_batchnorm', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_6_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_conv_batchnorm_scale', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_conv_batchnorm_bias', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_conv_batchnorm_mean', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_conv_batchnorm_var', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_6_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_batchnorm_scale', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_batchnorm_bias', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_batchnorm_mean', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_batchnorm_var', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'avg_pool_mixed_6_pool', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_2_conv_conv2d_weight', 'val_type': 1, 'dims': [192, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_6_conv_batchnorm', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_conv_relu', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_6_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_conv_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_6_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_6_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_2_conv_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_6_tower_2_conv_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_6_tower_2_conv_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_6_tower_2_conv_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 7], 'pads': [0, 3, 0, 3], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_6_tower_conv_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_conv_1_conv2d_weight', 'val_type': 1, 'dims': [160, 160, 1, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 1], 'pads': [3, 0, 3, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_6_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_1_conv2d_weight', 'val_type': 1, 'dims': [160, 160, 7, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_6_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_6_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_conv_1_batchnorm_var', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_6_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_1_batchnorm_var', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_6_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_6_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 1], 'pads': [3, 0, 3, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_6_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_conv_2_conv2d_weight', 'val_type': 1, 'dims': [192, 160, 7, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_conv_2_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 7], 'pads': [0, 3, 0, 3], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_6_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_2_conv2d_weight', 'val_type': 1, 'dims': [160, 160, 1, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_1_conv_2_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_6_tower_conv_2_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_conv_2_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_6_tower_conv_2_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_6_tower_conv_2_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_6_tower_conv_2_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_6_tower_1_conv_2_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_2_batchnorm_scale', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_2_batchnorm_bias', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_2_batchnorm_mean', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_2_batchnorm_var', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_1_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_6_tower_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_conv_2_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_6_tower_1_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_1_conv_2_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 1], 'pads': [3, 0, 3, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_6_tower_1_conv_2_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_3_conv2d_weight', 'val_type': 1, 'dims': [160, 160, 7, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_1_conv_3_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_6_tower_1_conv_3_conv2d', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_3_batchnorm_scale', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_3_batchnorm_bias', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_3_batchnorm_mean', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_3_batchnorm_var', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_1_conv_3_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_6_tower_1_conv_3_batchnorm', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_1_conv_3_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 7], 'pads': [0, 3, 0, 3], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_6_tower_1_conv_3_relu', 'val_type': 1, 'dims': [32, 160, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_4_conv2d_weight', 'val_type': 1, 'dims': [192, 160, 1, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_1_conv_4_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_6_tower_1_conv_4_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_4_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_4_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_4_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_4_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_1_conv_4_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_6_tower_1_conv_4_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_6_tower_1_conv_4_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'mixed_6_conv_relu', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_conv_2_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_1_conv_4_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_6_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'ch_concat_mixed_6_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_6_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_7_conv_conv2d_weight', 'val_type': 1, 'dims': [192, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_conv_conv2d', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_6_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_conv_conv2d_weight', 'val_type': 1, 'dims': [192, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_6_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_conv2d_weight', 'val_type': 1, 'dims': [192, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'AveragePool'
            'attributes': {'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_6_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'avg_pool_mixed_7_pool', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_7_conv_conv2d', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_7_conv_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_conv_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_conv_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_conv_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_conv_batchnorm', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_7_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_conv_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_conv_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_conv_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_conv_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_7_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'avg_pool_mixed_7_pool', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_2_conv_conv2d_weight', 'val_type': 1, 'dims': [192, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_7_conv_batchnorm', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_conv_relu', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_7_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_conv_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_7_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_7_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_2_conv_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_2_conv_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_2_conv_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_2_conv_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 7], 'pads': [0, 3, 0, 3], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_7_tower_conv_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_conv_1_conv2d_weight', 'val_type': 1, 'dims': [192, 192, 1, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 1], 'pads': [3, 0, 3, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_7_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_1_conv2d_weight', 'val_type': 1, 'dims': [192, 192, 7, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_7_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_7_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_conv_1_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_7_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_1_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_7_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_7_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 1], 'pads': [3, 0, 3, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_7_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_conv_2_conv2d_weight', 'val_type': 1, 'dims': [192, 192, 7, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_conv_2_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 7], 'pads': [0, 3, 0, 3], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_7_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_2_conv2d_weight', 'val_type': 1, 'dims': [192, 192, 1, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_1_conv_2_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_7_tower_conv_2_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_conv_2_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_conv_2_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_conv_2_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_conv_2_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_7_tower_1_conv_2_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_2_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_2_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_2_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_2_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_1_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_7_tower_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_conv_2_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_7_tower_1_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_1_conv_2_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 1], 'pads': [3, 0, 3, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_7_tower_1_conv_2_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_3_conv2d_weight', 'val_type': 1, 'dims': [192, 192, 7, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_1_conv_3_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_7_tower_1_conv_3_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_3_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_3_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_3_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_3_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_1_conv_3_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_7_tower_1_conv_3_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_1_conv_3_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 7], 'pads': [0, 3, 0, 3], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_7_tower_1_conv_3_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_4_conv2d_weight', 'val_type': 1, 'dims': [192, 192, 1, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_1_conv_4_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_7_tower_1_conv_4_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_4_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_4_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_4_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_4_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_1_conv_4_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_7_tower_1_conv_4_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_7_tower_1_conv_4_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'mixed_7_conv_relu', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_conv_2_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_1_conv_4_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_7_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'ch_concat_mixed_7_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_7_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_8_tower_conv_conv2d_weight', 'val_type': 1, 'dims': [192, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_7_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_conv2d_weight', 'val_type': 1, 'dims': [192, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'MaxPool'
            'attributes': {'kernel_shape': [3, 3], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}
            'inputs': [
                input {'name': 'ch_concat_mixed_7_chconcat', 'val_type': 1, 'dims': [32, 768, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'max_pool_mixed_8_pool', 'val_type': 1, 'dims': [32, 768, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_8_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_8_tower_conv_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_8_tower_conv_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_8_tower_conv_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_8_tower_conv_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_8_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_8_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_conv_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_8_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}
            'inputs': [
                input {'name': 'mixed_8_tower_conv_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_8_tower_conv_1_conv2d_weight', 'val_type': 1, 'dims': [320, 192, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 1280, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 7], 'pads': [0, 3, 0, 3], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_8_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_1_conv2d_weight', 'val_type': 1, 'dims': [192, 192, 1, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_8_tower_conv_1_conv2d', 'val_type': 1, 'dims': [32, 1280, 8, 8], 'vals': []}
                input {'name': 'mixed_8_tower_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [320], 'vals': []}
                input {'name': 'mixed_8_tower_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [320], 'vals': []}
                input {'name': 'mixed_8_tower_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [320], 'vals': []}
                input {'name': 'mixed_8_tower_conv_1_batchnorm_var', 'val_type': 1, 'dims': [320], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 1280, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_8_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_1_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_8_tower_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 1280, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 1280, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_8_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 1], 'pads': [3, 0, 3, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_8_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_2_conv2d_weight', 'val_type': 1, 'dims': [192, 192, 7, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_1_conv_2_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_8_tower_1_conv_2_conv2d', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_2_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_2_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_2_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_2_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_1_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_8_tower_1_conv_2_batchnorm', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_1_conv_2_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}
            'inputs': [
                input {'name': 'mixed_8_tower_1_conv_2_relu', 'val_type': 1, 'dims': [32, 192, 17, 17], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_3_conv2d_weight', 'val_type': 1, 'dims': [192, 192, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_1_conv_3_conv2d', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_8_tower_1_conv_3_conv2d', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_3_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_3_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_3_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_3_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_1_conv_3_batchnorm', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_8_tower_1_conv_3_batchnorm', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_8_tower_1_conv_3_relu', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'mixed_8_tower_conv_1_relu', 'val_type': 1, 'dims': [32, 1280, 8, 8], 'vals': []}
                input {'name': 'mixed_8_tower_1_conv_3_relu', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
                input {'name': 'max_pool_mixed_8_pool', 'val_type': 1, 'dims': [32, 768, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'ch_concat_mixed_8_chconcat', 'val_type': 1, 'dims': [32, 1280, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_8_chconcat', 'val_type': 1, 'dims': [32, 1280, 8, 8], 'vals': []}
                input {'name': 'mixed_9_conv_conv2d_weight', 'val_type': 1, 'dims': [320, 1280, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_conv_conv2d', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_8_chconcat', 'val_type': 1, 'dims': [32, 1280, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_conv_conv2d_weight', 'val_type': 1, 'dims': [384, 1280, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_8_chconcat', 'val_type': 1, 'dims': [32, 1280, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_1_conv_conv2d_weight', 'val_type': 1, 'dims': [448, 1280, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 448, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'AveragePool'
            'attributes': {'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_8_chconcat', 'val_type': 1, 'dims': [32, 1280, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'avg_pool_mixed_9_pool', 'val_type': 1, 'dims': [32, 1280, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_9_conv_conv2d', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
                input {'name': 'mixed_9_conv_batchnorm_scale', 'val_type': 1, 'dims': [320], 'vals': []}
                input {'name': 'mixed_9_conv_batchnorm_bias', 'val_type': 1, 'dims': [320], 'vals': []}
                input {'name': 'mixed_9_conv_batchnorm_mean', 'val_type': 1, 'dims': [320], 'vals': []}
                input {'name': 'mixed_9_conv_batchnorm_var', 'val_type': 1, 'dims': [320], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_conv_batchnorm', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_9_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_conv_batchnorm_scale', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_conv_batchnorm_bias', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_conv_batchnorm_mean', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_conv_batchnorm_var', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_9_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 448, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_1_conv_batchnorm_scale', 'val_type': 1, 'dims': [448], 'vals': []}
                input {'name': 'mixed_9_tower_1_conv_batchnorm_bias', 'val_type': 1, 'dims': [448], 'vals': []}
                input {'name': 'mixed_9_tower_1_conv_batchnorm_mean', 'val_type': 1, 'dims': [448], 'vals': []}
                input {'name': 'mixed_9_tower_1_conv_batchnorm_var', 'val_type': 1, 'dims': [448], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 448, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'avg_pool_mixed_9_pool', 'val_type': 1, 'dims': [32, 1280, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_2_conv_conv2d_weight', 'val_type': 1, 'dims': [192, 1280, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_9_conv_batchnorm', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_conv_relu', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_9_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_conv_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_9_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 448, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 448, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_9_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_2_conv_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_9_tower_2_conv_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_9_tower_2_conv_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_9_tower_2_conv_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 3], 'pads': [0, 1, 0, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_9_tower_conv_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_mixed_conv_conv2d_weight', 'val_type': 1, 'dims': [384, 384, 1, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_mixed_conv_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 1], 'pads': [1, 0, 1, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_9_tower_conv_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_mixed_conv_1_conv2d_weight', 'val_type': 1, 'dims': [384, 384, 3, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_mixed_conv_1_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_9_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 448, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_1_conv_1_conv2d_weight', 'val_type': 1, 'dims': [384, 448, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_9_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_9_tower_mixed_conv_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_mixed_conv_batchnorm_scale', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_mixed_conv_batchnorm_bias', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_mixed_conv_batchnorm_mean', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_mixed_conv_batchnorm_var', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_mixed_conv_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_9_tower_mixed_conv_1_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_mixed_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_mixed_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_mixed_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_mixed_conv_1_batchnorm_var', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_mixed_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_9_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_1_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_1_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_1_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_1_conv_1_batchnorm_var', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_9_tower_mixed_conv_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_mixed_conv_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_9_tower_mixed_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_mixed_conv_1_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_9_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 3], 'pads': [0, 1, 0, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_9_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_1_mixed_conv_conv2d_weight', 'val_type': 1, 'dims': [384, 384, 1, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_1_mixed_conv_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 1], 'pads': [1, 0, 1, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_9_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_1_mixed_conv_1_conv2d_weight', 'val_type': 1, 'dims': [384, 384, 3, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_1_mixed_conv_1_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_9_tower_1_mixed_conv_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_1_mixed_conv_batchnorm_scale', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_1_mixed_conv_batchnorm_bias', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_1_mixed_conv_batchnorm_mean', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_1_mixed_conv_batchnorm_var', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_1_mixed_conv_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_9_tower_1_mixed_conv_1_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_1_mixed_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_1_mixed_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_1_mixed_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_9_tower_1_mixed_conv_1_batchnorm_var', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_1_mixed_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_9_tower_1_mixed_conv_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_1_mixed_conv_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_9_tower_1_mixed_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_9_tower_1_mixed_conv_1_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'mixed_9_conv_relu', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_mixed_conv_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_mixed_conv_1_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_1_mixed_conv_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_1_mixed_conv_1_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_9_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'ch_concat_mixed_9_chconcat', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_9_chconcat', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
                input {'name': 'mixed_10_conv_conv2d_weight', 'val_type': 1, 'dims': [320, 2048, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_conv_conv2d', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_9_chconcat', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_conv_conv2d_weight', 'val_type': 1, 'dims': [384, 2048, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_9_chconcat', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_1_conv_conv2d_weight', 'val_type': 1, 'dims': [448, 2048, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 448, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'MaxPool'
            'attributes': {'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_9_chconcat', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'max_pool_mixed_10_pool', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_10_conv_conv2d', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
                input {'name': 'mixed_10_conv_batchnorm_scale', 'val_type': 1, 'dims': [320], 'vals': []}
                input {'name': 'mixed_10_conv_batchnorm_bias', 'val_type': 1, 'dims': [320], 'vals': []}
                input {'name': 'mixed_10_conv_batchnorm_mean', 'val_type': 1, 'dims': [320], 'vals': []}
                input {'name': 'mixed_10_conv_batchnorm_var', 'val_type': 1, 'dims': [320], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_conv_batchnorm', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_10_tower_conv_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_conv_batchnorm_scale', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_conv_batchnorm_bias', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_conv_batchnorm_mean', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_conv_batchnorm_var', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_10_tower_1_conv_conv2d', 'val_type': 1, 'dims': [32, 448, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_1_conv_batchnorm_scale', 'val_type': 1, 'dims': [448], 'vals': []}
                input {'name': 'mixed_10_tower_1_conv_batchnorm_bias', 'val_type': 1, 'dims': [448], 'vals': []}
                input {'name': 'mixed_10_tower_1_conv_batchnorm_mean', 'val_type': 1, 'dims': [448], 'vals': []}
                input {'name': 'mixed_10_tower_1_conv_batchnorm_var', 'val_type': 1, 'dims': [448], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 448, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'max_pool_mixed_10_pool', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_2_conv_conv2d_weight', 'val_type': 1, 'dims': [192, 2048, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_10_conv_batchnorm', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_conv_relu', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_10_tower_conv_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_conv_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_10_tower_1_conv_batchnorm', 'val_type': 1, 'dims': [32, 448, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 448, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_10_tower_2_conv_conv2d', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_2_conv_batchnorm_scale', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_10_tower_2_conv_batchnorm_bias', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_10_tower_2_conv_batchnorm_mean', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'mixed_10_tower_2_conv_batchnorm_var', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 3], 'pads': [0, 1, 0, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_10_tower_conv_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_mixed_conv_conv2d_weight', 'val_type': 1, 'dims': [384, 384, 1, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_mixed_conv_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 1], 'pads': [1, 0, 1, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_10_tower_conv_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_mixed_conv_1_conv2d_weight', 'val_type': 1, 'dims': [384, 384, 3, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_mixed_conv_1_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_10_tower_1_conv_relu', 'val_type': 1, 'dims': [32, 448, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_1_conv_1_conv2d_weight', 'val_type': 1, 'dims': [384, 448, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_10_tower_2_conv_batchnorm', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_10_tower_mixed_conv_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_mixed_conv_batchnorm_scale', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_mixed_conv_batchnorm_bias', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_mixed_conv_batchnorm_mean', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_mixed_conv_batchnorm_var', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_mixed_conv_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_10_tower_mixed_conv_1_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_mixed_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_mixed_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_mixed_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_mixed_conv_1_batchnorm_var', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_mixed_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_10_tower_1_conv_1_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_1_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_1_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_1_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_1_conv_1_batchnorm_var', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_10_tower_mixed_conv_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_mixed_conv_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_10_tower_mixed_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_mixed_conv_1_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_10_tower_1_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 3], 'pads': [0, 1, 0, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_10_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_1_mixed_conv_conv2d_weight', 'val_type': 1, 'dims': [384, 384, 1, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_1_mixed_conv_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 1], 'pads': [1, 0, 1, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'mixed_10_tower_1_conv_1_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_1_mixed_conv_1_conv2d_weight', 'val_type': 1, 'dims': [384, 384, 3, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_1_mixed_conv_1_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_10_tower_1_mixed_conv_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_1_mixed_conv_batchnorm_scale', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_1_mixed_conv_batchnorm_bias', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_1_mixed_conv_batchnorm_mean', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_1_mixed_conv_batchnorm_var', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_1_mixed_conv_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 0.0010000000474974513, 'is_test': 1}
            'inputs': [
                input {'name': 'mixed_10_tower_1_mixed_conv_1_conv2d', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_1_mixed_conv_1_batchnorm_scale', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_1_mixed_conv_1_batchnorm_bias', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_1_mixed_conv_1_batchnorm_mean', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'mixed_10_tower_1_mixed_conv_1_batchnorm_var', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_1_mixed_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_10_tower_1_mixed_conv_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_1_mixed_conv_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'mixed_10_tower_1_mixed_conv_1_batchnorm', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'mixed_10_tower_1_mixed_conv_1_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'mixed_10_conv_relu', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_mixed_conv_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_mixed_conv_1_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_1_mixed_conv_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_1_mixed_conv_1_relu', 'val_type': 1, 'dims': [32, 384, 8, 8], 'vals': []}
                input {'name': 'mixed_10_tower_2_conv_relu', 'val_type': 1, 'dims': [32, 192, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'ch_concat_mixed_10_chconcat', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
            ]
        }
        node {
            'op_type': 'AveragePool'
            'attributes': {'kernel_shape': [8, 8], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': 'ch_concat_mixed_10_chconcat', 'val_type': 1, 'dims': [32, 2048, 8, 8], 'vals': []}
            ]
            'outputs': [
                output {'name': 'global_pool', 'val_type': 1, 'dims': [32, 2048, 1, 1], 'vals': []}
            ]
        }
        node {
            'op_type': 'Flatten'
            'attributes': {}
            'inputs': [
                input {'name': 'global_pool', 'val_type': 1, 'dims': [32, 2048, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'flatten', 'val_type': 1, 'dims': [32, 2048, 1, 1], 'vals': []}
            ]
        }
        node {
            'op_type': 'Gemm'
            'attributes': {}
            'inputs': [
                input {'name': 'flatten', 'val_type': None, 'dims': [32, 2048, 1, 1], 'vals': []}
                input {'name': 'fc1_weight', 'val_type': 1, 'dims': [2048, 1008], 'vals': []}
                input {'name': 'fc1_bias', 'val_type': 1, 'dims': [1008], 'vals': []}
            ]
            'outputs': [
                output {'name': 'fc1', 'val_type': None, 'dims': [32, 1008, 1, 1], 'vals': []}
            ]
        }
        node {
            'op_type': 'Softmax'
            'attributes': {}
            'inputs': [
                input {'name': 'fc1', 'val_type': None, 'dims': [32, 1008, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'softmax', 'val_type': None, 'dims': [32, 1008, 1, 1], 'vals': []}
            ]
        }
    ]
}
