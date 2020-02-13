'ir_version': 3
'producer_name': 'onnx-mims'
'graph': {
    'name': 'densenet121'
    'nodes': [
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [2, 2], 'pads': [3, 3, 3, 3], 'kernel_shape': [7, 7]}
            'inputs': [
                input {'name': 'data_0', 'val_type': 1, 'dims': [200704, 3, 224, 224], 'vals': []}
                input {'name': 'conv1_w_0', 'val_type': 1, 'dims': [64, 3, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv1_1', 'val_type': 1, 'dims': [200704, 64, 112, 112], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv1_1', 'val_type': 1, 'dims': [200704, 64, 112, 112], 'vals': []}
                input {'name': 'conv1/bn_scale_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'conv1/bn_bias_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'conv1/bn_mean_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'conv1/bn_var_0', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv1/bn_1', 'val_type': 1, 'dims': [200704, 64, 112, 112], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv1/bn_1', 'val_type': 1, 'dims': [200704, 64, 112, 112], 'vals': []}
                input {'name': 'conv1/bn_w_0', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv1/bn_internal_1', 'val_type': 1, 'dims': [200704, 64, 112, 112], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv1/bn_internal_1', 'val_type': 1, 'dims': [200704, 64, 112, 112], 'vals': []}
                input {'name': 'conv1/bn_b_0', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv1/bn_2', 'val_type': 1, 'dims': [200704, 64, 112, 112], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv1/bn_2', 'val_type': 1, 'dims': [200704, 64, 112, 112], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv1/bn_3', 'val_type': 1, 'dims': [200704, 64, 112, 112], 'vals': []}
            ]
        }
        node {
            'op_type': 'MaxPool'
            'attributes': {'strides': [2, 2], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv1/bn_3', 'val_type': 1, 'dims': [200704, 64, 112, 112], 'vals': []}
            ]
            'outputs': [
                output {'name': 'pool1_1', 'val_type': 1, 'dims': [200704, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'pool1_1', 'val_type': 1, 'dims': [200704, 64, 56, 56], 'vals': []}
                input {'name': 'conv2_1/x1/bn_scale_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'conv2_1/x1/bn_bias_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'conv2_1/x1/bn_mean_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'conv2_1/x1/bn_var_0', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_1/x1/bn_1', 'val_type': 1, 'dims': [200704, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_1/x1/bn_1', 'val_type': 1, 'dims': [200704, 64, 56, 56], 'vals': []}
                input {'name': 'conv2_1/x1/bn_w_0', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_1/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_1/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 64, 56, 56], 'vals': []}
                input {'name': 'conv2_1/x1/bn_b_0', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_1/x1/bn_2', 'val_type': 1, 'dims': [200704, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv2_1/x1/bn_2', 'val_type': 1, 'dims': [200704, 64, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_1/x1/bn_3', 'val_type': 1, 'dims': [200704, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv2_1/x1/bn_3', 'val_type': 1, 'dims': [200704, 64, 56, 56], 'vals': []}
                input {'name': 'conv2_1/x1_w_0', 'val_type': 1, 'dims': [128, 64, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_1/x1_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv2_1/x1_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_1/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_1/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_1/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_1/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_1/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_1/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_1/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_1/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_1/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_1/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_1/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv2_1/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_1/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv2_1/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_1/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_1/x2_1', 'val_type': 1, 'dims': [200704, 32, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'pool1_1', 'val_type': 1, 'dims': [200704, 64, 56, 56], 'vals': []}
                input {'name': 'conv2_1/x2_1', 'val_type': 1, 'dims': [200704, 32, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_2_1_1', 'val_type': 1, 'dims': [200704, 96, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_2_1_1', 'val_type': 1, 'dims': [200704, 96, 56, 56], 'vals': []}
                input {'name': 'conv2_2/x1/bn_scale_0', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'conv2_2/x1/bn_bias_0', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'conv2_2/x1/bn_mean_0', 'val_type': 1, 'dims': [96], 'vals': []}
                input {'name': 'conv2_2/x1/bn_var_0', 'val_type': 1, 'dims': [96], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_2/x1/bn_1', 'val_type': 1, 'dims': [200704, 96, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_2/x1/bn_1', 'val_type': 1, 'dims': [200704, 96, 56, 56], 'vals': []}
                input {'name': 'conv2_2/x1/bn_w_0', 'val_type': 1, 'dims': [96], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_2/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 96, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_2/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 96, 56, 56], 'vals': []}
                input {'name': 'conv2_2/x1/bn_b_0', 'val_type': 1, 'dims': [96], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_2/x1/bn_2', 'val_type': 1, 'dims': [200704, 96, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv2_2/x1/bn_2', 'val_type': 1, 'dims': [200704, 96, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_2/x1/bn_3', 'val_type': 1, 'dims': [200704, 96, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv2_2/x1/bn_3', 'val_type': 1, 'dims': [200704, 96, 56, 56], 'vals': []}
                input {'name': 'conv2_2/x1_w_0', 'val_type': 1, 'dims': [128, 96, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_2/x1_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv2_2/x1_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_2/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_2/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_2/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_2/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_2/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_2/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_2/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_2/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_2/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_2/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_2/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv2_2/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_2/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv2_2/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_2/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_2/x2_1', 'val_type': 1, 'dims': [200704, 32, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_2_1_1', 'val_type': 1, 'dims': [200704, 96, 56, 56], 'vals': []}
                input {'name': 'conv2_2/x2_1', 'val_type': 1, 'dims': [200704, 32, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_2_2_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_2_2_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_3/x1/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_3/x1/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_3/x1/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_3/x1/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_3/x1/bn_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_3/x1/bn_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_3/x1/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_3/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_3/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_3/x1/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_3/x1/bn_2', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv2_3/x1/bn_2', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_3/x1/bn_3', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv2_3/x1/bn_3', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_3/x1_w_0', 'val_type': 1, 'dims': [128, 128, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_3/x1_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv2_3/x1_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_3/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_3/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_3/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_3/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_3/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_3/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_3/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_3/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_3/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_3/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_3/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv2_3/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_3/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv2_3/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_3/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_3/x2_1', 'val_type': 1, 'dims': [200704, 32, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_2_2_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_3/x2_1', 'val_type': 1, 'dims': [200704, 32, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_2_3_1', 'val_type': 1, 'dims': [200704, 160, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_2_3_1', 'val_type': 1, 'dims': [200704, 160, 56, 56], 'vals': []}
                input {'name': 'conv2_4/x1/bn_scale_0', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'conv2_4/x1/bn_bias_0', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'conv2_4/x1/bn_mean_0', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'conv2_4/x1/bn_var_0', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_4/x1/bn_1', 'val_type': 1, 'dims': [200704, 160, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_4/x1/bn_1', 'val_type': 1, 'dims': [200704, 160, 56, 56], 'vals': []}
                input {'name': 'conv2_4/x1/bn_w_0', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_4/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 160, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_4/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 160, 56, 56], 'vals': []}
                input {'name': 'conv2_4/x1/bn_b_0', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_4/x1/bn_2', 'val_type': 1, 'dims': [200704, 160, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv2_4/x1/bn_2', 'val_type': 1, 'dims': [200704, 160, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_4/x1/bn_3', 'val_type': 1, 'dims': [200704, 160, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv2_4/x1/bn_3', 'val_type': 1, 'dims': [200704, 160, 56, 56], 'vals': []}
                input {'name': 'conv2_4/x1_w_0', 'val_type': 1, 'dims': [128, 160, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_4/x1_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv2_4/x1_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_4/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_4/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_4/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_4/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_4/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_4/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_4/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_4/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_4/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_4/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_4/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv2_4/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_4/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv2_4/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_4/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_4/x2_1', 'val_type': 1, 'dims': [200704, 32, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_2_3_1', 'val_type': 1, 'dims': [200704, 160, 56, 56], 'vals': []}
                input {'name': 'conv2_4/x2_1', 'val_type': 1, 'dims': [200704, 32, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_2_4_1', 'val_type': 1, 'dims': [200704, 192, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_2_4_1', 'val_type': 1, 'dims': [200704, 192, 56, 56], 'vals': []}
                input {'name': 'conv2_5/x1/bn_scale_0', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'conv2_5/x1/bn_bias_0', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'conv2_5/x1/bn_mean_0', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'conv2_5/x1/bn_var_0', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_5/x1/bn_1', 'val_type': 1, 'dims': [200704, 192, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_5/x1/bn_1', 'val_type': 1, 'dims': [200704, 192, 56, 56], 'vals': []}
                input {'name': 'conv2_5/x1/bn_w_0', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_5/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 192, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_5/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 192, 56, 56], 'vals': []}
                input {'name': 'conv2_5/x1/bn_b_0', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_5/x1/bn_2', 'val_type': 1, 'dims': [200704, 192, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv2_5/x1/bn_2', 'val_type': 1, 'dims': [200704, 192, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_5/x1/bn_3', 'val_type': 1, 'dims': [200704, 192, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv2_5/x1/bn_3', 'val_type': 1, 'dims': [200704, 192, 56, 56], 'vals': []}
                input {'name': 'conv2_5/x1_w_0', 'val_type': 1, 'dims': [128, 192, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_5/x1_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv2_5/x1_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_5/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_5/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_5/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_5/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_5/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_5/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_5/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_5/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_5/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_5/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_5/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv2_5/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_5/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv2_5/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_5/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_5/x2_1', 'val_type': 1, 'dims': [200704, 32, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_2_4_1', 'val_type': 1, 'dims': [200704, 192, 56, 56], 'vals': []}
                input {'name': 'conv2_5/x2_1', 'val_type': 1, 'dims': [200704, 32, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_2_5_1', 'val_type': 1, 'dims': [200704, 224, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_2_5_1', 'val_type': 1, 'dims': [200704, 224, 56, 56], 'vals': []}
                input {'name': 'conv2_6/x1/bn_scale_0', 'val_type': 1, 'dims': [224], 'vals': []}
                input {'name': 'conv2_6/x1/bn_bias_0', 'val_type': 1, 'dims': [224], 'vals': []}
                input {'name': 'conv2_6/x1/bn_mean_0', 'val_type': 1, 'dims': [224], 'vals': []}
                input {'name': 'conv2_6/x1/bn_var_0', 'val_type': 1, 'dims': [224], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_6/x1/bn_1', 'val_type': 1, 'dims': [200704, 224, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_6/x1/bn_1', 'val_type': 1, 'dims': [200704, 224, 56, 56], 'vals': []}
                input {'name': 'conv2_6/x1/bn_w_0', 'val_type': 1, 'dims': [224], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_6/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 224, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_6/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 224, 56, 56], 'vals': []}
                input {'name': 'conv2_6/x1/bn_b_0', 'val_type': 1, 'dims': [224], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_6/x1/bn_2', 'val_type': 1, 'dims': [200704, 224, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv2_6/x1/bn_2', 'val_type': 1, 'dims': [200704, 224, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_6/x1/bn_3', 'val_type': 1, 'dims': [200704, 224, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv2_6/x1/bn_3', 'val_type': 1, 'dims': [200704, 224, 56, 56], 'vals': []}
                input {'name': 'conv2_6/x1_w_0', 'val_type': 1, 'dims': [128, 224, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_6/x1_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv2_6/x1_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_6/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_6/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_6/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv2_6/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_6/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_6/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_6/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_6/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_6/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_6/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_6/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv2_6/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_6/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv2_6/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
                input {'name': 'conv2_6/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_6/x2_1', 'val_type': 1, 'dims': [200704, 32, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_2_5_1', 'val_type': 1, 'dims': [200704, 224, 56, 56], 'vals': []}
                input {'name': 'conv2_6/x2_1', 'val_type': 1, 'dims': [200704, 32, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_2_6_1', 'val_type': 1, 'dims': [200704, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_2_6_1', 'val_type': 1, 'dims': [200704, 256, 56, 56], 'vals': []}
                input {'name': 'conv2_blk/bn_scale_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'conv2_blk/bn_bias_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'conv2_blk/bn_mean_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'conv2_blk/bn_var_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_blk/bn_1', 'val_type': 1, 'dims': [200704, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_blk/bn_1', 'val_type': 1, 'dims': [200704, 256, 56, 56], 'vals': []}
                input {'name': 'conv2_blk/bn_w_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_blk/bn_internal_1', 'val_type': 1, 'dims': [200704, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv2_blk/bn_internal_1', 'val_type': 1, 'dims': [200704, 256, 56, 56], 'vals': []}
                input {'name': 'conv2_blk/bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_blk/bn_2', 'val_type': 1, 'dims': [200704, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv2_blk/bn_2', 'val_type': 1, 'dims': [200704, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_blk/bn_3', 'val_type': 1, 'dims': [200704, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv2_blk/bn_3', 'val_type': 1, 'dims': [200704, 256, 56, 56], 'vals': []}
                input {'name': 'conv2_blk_w_0', 'val_type': 1, 'dims': [128, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv2_blk_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'AveragePool'
            'attributes': {'strides': [2, 2], 'pads': [0, 0, 0, 0], 'kernel_shape': [2, 2]}
            'inputs': [
                input {'name': 'conv2_blk_1', 'val_type': 1, 'dims': [200704, 128, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'pool2_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'pool2_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_1/x1/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_1/x1/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_1/x1/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_1/x1/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_1/x1/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_1/x1/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_1/x1/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_1/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_1/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_1/x1/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_1/x1/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_1/x1/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_1/x1/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv3_1/x1/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_1/x1_w_0', 'val_type': 1, 'dims': [128, 128, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_1/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv3_1/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_1/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_1/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_1/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_1/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_1/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_1/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_1/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_1/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_1/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_1/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_1/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_1/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_1/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv3_1/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_1/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_1/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'pool2_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_1/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_3_1_1', 'val_type': 1, 'dims': [200704, 160, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_3_1_1', 'val_type': 1, 'dims': [200704, 160, 28, 28], 'vals': []}
                input {'name': 'conv3_2/x1/bn_scale_0', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'conv3_2/x1/bn_bias_0', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'conv3_2/x1/bn_mean_0', 'val_type': 1, 'dims': [160], 'vals': []}
                input {'name': 'conv3_2/x1/bn_var_0', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_2/x1/bn_1', 'val_type': 1, 'dims': [200704, 160, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_2/x1/bn_1', 'val_type': 1, 'dims': [200704, 160, 28, 28], 'vals': []}
                input {'name': 'conv3_2/x1/bn_w_0', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_2/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 160, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_2/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 160, 28, 28], 'vals': []}
                input {'name': 'conv3_2/x1/bn_b_0', 'val_type': 1, 'dims': [160], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_2/x1/bn_2', 'val_type': 1, 'dims': [200704, 160, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_2/x1/bn_2', 'val_type': 1, 'dims': [200704, 160, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_2/x1/bn_3', 'val_type': 1, 'dims': [200704, 160, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv3_2/x1/bn_3', 'val_type': 1, 'dims': [200704, 160, 28, 28], 'vals': []}
                input {'name': 'conv3_2/x1_w_0', 'val_type': 1, 'dims': [128, 160, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_2/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv3_2/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_2/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_2/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_2/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_2/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_2/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_2/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_2/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_2/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_2/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_2/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_2/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_2/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_2/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv3_2/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_2/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_2/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_3_1_1', 'val_type': 1, 'dims': [200704, 160, 28, 28], 'vals': []}
                input {'name': 'conv3_2/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_3_2_1', 'val_type': 1, 'dims': [200704, 192, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_3_2_1', 'val_type': 1, 'dims': [200704, 192, 28, 28], 'vals': []}
                input {'name': 'conv3_3/x1/bn_scale_0', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'conv3_3/x1/bn_bias_0', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'conv3_3/x1/bn_mean_0', 'val_type': 1, 'dims': [192], 'vals': []}
                input {'name': 'conv3_3/x1/bn_var_0', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_3/x1/bn_1', 'val_type': 1, 'dims': [200704, 192, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_3/x1/bn_1', 'val_type': 1, 'dims': [200704, 192, 28, 28], 'vals': []}
                input {'name': 'conv3_3/x1/bn_w_0', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_3/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 192, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_3/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 192, 28, 28], 'vals': []}
                input {'name': 'conv3_3/x1/bn_b_0', 'val_type': 1, 'dims': [192], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_3/x1/bn_2', 'val_type': 1, 'dims': [200704, 192, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_3/x1/bn_2', 'val_type': 1, 'dims': [200704, 192, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_3/x1/bn_3', 'val_type': 1, 'dims': [200704, 192, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv3_3/x1/bn_3', 'val_type': 1, 'dims': [200704, 192, 28, 28], 'vals': []}
                input {'name': 'conv3_3/x1_w_0', 'val_type': 1, 'dims': [128, 192, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_3/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv3_3/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_3/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_3/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_3/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_3/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_3/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_3/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_3/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_3/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_3/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_3/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_3/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_3/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_3/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv3_3/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_3/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_3/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_3_2_1', 'val_type': 1, 'dims': [200704, 192, 28, 28], 'vals': []}
                input {'name': 'conv3_3/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_3_3_1', 'val_type': 1, 'dims': [200704, 224, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_3_3_1', 'val_type': 1, 'dims': [200704, 224, 28, 28], 'vals': []}
                input {'name': 'conv3_4/x1/bn_scale_0', 'val_type': 1, 'dims': [224], 'vals': []}
                input {'name': 'conv3_4/x1/bn_bias_0', 'val_type': 1, 'dims': [224], 'vals': []}
                input {'name': 'conv3_4/x1/bn_mean_0', 'val_type': 1, 'dims': [224], 'vals': []}
                input {'name': 'conv3_4/x1/bn_var_0', 'val_type': 1, 'dims': [224], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_4/x1/bn_1', 'val_type': 1, 'dims': [200704, 224, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_4/x1/bn_1', 'val_type': 1, 'dims': [200704, 224, 28, 28], 'vals': []}
                input {'name': 'conv3_4/x1/bn_w_0', 'val_type': 1, 'dims': [224], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_4/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 224, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_4/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 224, 28, 28], 'vals': []}
                input {'name': 'conv3_4/x1/bn_b_0', 'val_type': 1, 'dims': [224], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_4/x1/bn_2', 'val_type': 1, 'dims': [200704, 224, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_4/x1/bn_2', 'val_type': 1, 'dims': [200704, 224, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_4/x1/bn_3', 'val_type': 1, 'dims': [200704, 224, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv3_4/x1/bn_3', 'val_type': 1, 'dims': [200704, 224, 28, 28], 'vals': []}
                input {'name': 'conv3_4/x1_w_0', 'val_type': 1, 'dims': [128, 224, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_4/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv3_4/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_4/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_4/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_4/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_4/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_4/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_4/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_4/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_4/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_4/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_4/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_4/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_4/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_4/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv3_4/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_4/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_4/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_3_3_1', 'val_type': 1, 'dims': [200704, 224, 28, 28], 'vals': []}
                input {'name': 'conv3_4/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_3_4_1', 'val_type': 1, 'dims': [200704, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_3_4_1', 'val_type': 1, 'dims': [200704, 256, 28, 28], 'vals': []}
                input {'name': 'conv3_5/x1/bn_scale_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'conv3_5/x1/bn_bias_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'conv3_5/x1/bn_mean_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'conv3_5/x1/bn_var_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_5/x1/bn_1', 'val_type': 1, 'dims': [200704, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_5/x1/bn_1', 'val_type': 1, 'dims': [200704, 256, 28, 28], 'vals': []}
                input {'name': 'conv3_5/x1/bn_w_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_5/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_5/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 256, 28, 28], 'vals': []}
                input {'name': 'conv3_5/x1/bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_5/x1/bn_2', 'val_type': 1, 'dims': [200704, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_5/x1/bn_2', 'val_type': 1, 'dims': [200704, 256, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_5/x1/bn_3', 'val_type': 1, 'dims': [200704, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv3_5/x1/bn_3', 'val_type': 1, 'dims': [200704, 256, 28, 28], 'vals': []}
                input {'name': 'conv3_5/x1_w_0', 'val_type': 1, 'dims': [128, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_5/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv3_5/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_5/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_5/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_5/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_5/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_5/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_5/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_5/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_5/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_5/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_5/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_5/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_5/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_5/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv3_5/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_5/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_5/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_3_4_1', 'val_type': 1, 'dims': [200704, 256, 28, 28], 'vals': []}
                input {'name': 'conv3_5/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_3_5_1', 'val_type': 1, 'dims': [200704, 288, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_3_5_1', 'val_type': 1, 'dims': [200704, 288, 28, 28], 'vals': []}
                input {'name': 'conv3_6/x1/bn_scale_0', 'val_type': 1, 'dims': [288], 'vals': []}
                input {'name': 'conv3_6/x1/bn_bias_0', 'val_type': 1, 'dims': [288], 'vals': []}
                input {'name': 'conv3_6/x1/bn_mean_0', 'val_type': 1, 'dims': [288], 'vals': []}
                input {'name': 'conv3_6/x1/bn_var_0', 'val_type': 1, 'dims': [288], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_6/x1/bn_1', 'val_type': 1, 'dims': [200704, 288, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_6/x1/bn_1', 'val_type': 1, 'dims': [200704, 288, 28, 28], 'vals': []}
                input {'name': 'conv3_6/x1/bn_w_0', 'val_type': 1, 'dims': [288], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_6/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 288, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_6/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 288, 28, 28], 'vals': []}
                input {'name': 'conv3_6/x1/bn_b_0', 'val_type': 1, 'dims': [288], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_6/x1/bn_2', 'val_type': 1, 'dims': [200704, 288, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_6/x1/bn_2', 'val_type': 1, 'dims': [200704, 288, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_6/x1/bn_3', 'val_type': 1, 'dims': [200704, 288, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv3_6/x1/bn_3', 'val_type': 1, 'dims': [200704, 288, 28, 28], 'vals': []}
                input {'name': 'conv3_6/x1_w_0', 'val_type': 1, 'dims': [128, 288, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_6/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv3_6/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_6/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_6/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_6/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_6/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_6/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_6/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_6/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_6/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_6/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_6/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_6/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_6/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_6/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv3_6/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_6/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_6/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_3_5_1', 'val_type': 1, 'dims': [200704, 288, 28, 28], 'vals': []}
                input {'name': 'conv3_6/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_3_6_1', 'val_type': 1, 'dims': [200704, 320, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_3_6_1', 'val_type': 1, 'dims': [200704, 320, 28, 28], 'vals': []}
                input {'name': 'conv3_7/x1/bn_scale_0', 'val_type': 1, 'dims': [320], 'vals': []}
                input {'name': 'conv3_7/x1/bn_bias_0', 'val_type': 1, 'dims': [320], 'vals': []}
                input {'name': 'conv3_7/x1/bn_mean_0', 'val_type': 1, 'dims': [320], 'vals': []}
                input {'name': 'conv3_7/x1/bn_var_0', 'val_type': 1, 'dims': [320], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_7/x1/bn_1', 'val_type': 1, 'dims': [200704, 320, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_7/x1/bn_1', 'val_type': 1, 'dims': [200704, 320, 28, 28], 'vals': []}
                input {'name': 'conv3_7/x1/bn_w_0', 'val_type': 1, 'dims': [320], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_7/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 320, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_7/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 320, 28, 28], 'vals': []}
                input {'name': 'conv3_7/x1/bn_b_0', 'val_type': 1, 'dims': [320], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_7/x1/bn_2', 'val_type': 1, 'dims': [200704, 320, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_7/x1/bn_2', 'val_type': 1, 'dims': [200704, 320, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_7/x1/bn_3', 'val_type': 1, 'dims': [200704, 320, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv3_7/x1/bn_3', 'val_type': 1, 'dims': [200704, 320, 28, 28], 'vals': []}
                input {'name': 'conv3_7/x1_w_0', 'val_type': 1, 'dims': [128, 320, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_7/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv3_7/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_7/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_7/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_7/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_7/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_7/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_7/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_7/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_7/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_7/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_7/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_7/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_7/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_7/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv3_7/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_7/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_7/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_3_6_1', 'val_type': 1, 'dims': [200704, 320, 28, 28], 'vals': []}
                input {'name': 'conv3_7/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_3_7_1', 'val_type': 1, 'dims': [200704, 352, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_3_7_1', 'val_type': 1, 'dims': [200704, 352, 28, 28], 'vals': []}
                input {'name': 'conv3_8/x1/bn_scale_0', 'val_type': 1, 'dims': [352], 'vals': []}
                input {'name': 'conv3_8/x1/bn_bias_0', 'val_type': 1, 'dims': [352], 'vals': []}
                input {'name': 'conv3_8/x1/bn_mean_0', 'val_type': 1, 'dims': [352], 'vals': []}
                input {'name': 'conv3_8/x1/bn_var_0', 'val_type': 1, 'dims': [352], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_8/x1/bn_1', 'val_type': 1, 'dims': [200704, 352, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_8/x1/bn_1', 'val_type': 1, 'dims': [200704, 352, 28, 28], 'vals': []}
                input {'name': 'conv3_8/x1/bn_w_0', 'val_type': 1, 'dims': [352], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_8/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 352, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_8/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 352, 28, 28], 'vals': []}
                input {'name': 'conv3_8/x1/bn_b_0', 'val_type': 1, 'dims': [352], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_8/x1/bn_2', 'val_type': 1, 'dims': [200704, 352, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_8/x1/bn_2', 'val_type': 1, 'dims': [200704, 352, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_8/x1/bn_3', 'val_type': 1, 'dims': [200704, 352, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv3_8/x1/bn_3', 'val_type': 1, 'dims': [200704, 352, 28, 28], 'vals': []}
                input {'name': 'conv3_8/x1_w_0', 'val_type': 1, 'dims': [128, 352, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_8/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv3_8/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_8/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_8/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_8/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_8/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_8/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_8/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_8/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_8/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_8/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_8/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_8/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_8/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_8/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv3_8/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_8/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_8/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_3_7_1', 'val_type': 1, 'dims': [200704, 352, 28, 28], 'vals': []}
                input {'name': 'conv3_8/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_3_8_1', 'val_type': 1, 'dims': [200704, 384, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_3_8_1', 'val_type': 1, 'dims': [200704, 384, 28, 28], 'vals': []}
                input {'name': 'conv3_9/x1/bn_scale_0', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'conv3_9/x1/bn_bias_0', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'conv3_9/x1/bn_mean_0', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'conv3_9/x1/bn_var_0', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_9/x1/bn_1', 'val_type': 1, 'dims': [200704, 384, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_9/x1/bn_1', 'val_type': 1, 'dims': [200704, 384, 28, 28], 'vals': []}
                input {'name': 'conv3_9/x1/bn_w_0', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_9/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 384, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_9/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 384, 28, 28], 'vals': []}
                input {'name': 'conv3_9/x1/bn_b_0', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_9/x1/bn_2', 'val_type': 1, 'dims': [200704, 384, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_9/x1/bn_2', 'val_type': 1, 'dims': [200704, 384, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_9/x1/bn_3', 'val_type': 1, 'dims': [200704, 384, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv3_9/x1/bn_3', 'val_type': 1, 'dims': [200704, 384, 28, 28], 'vals': []}
                input {'name': 'conv3_9/x1_w_0', 'val_type': 1, 'dims': [128, 384, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_9/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv3_9/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_9/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_9/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_9/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_9/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_9/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_9/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_9/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_9/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_9/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_9/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_9/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_9/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_9/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv3_9/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_9/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_9/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_3_8_1', 'val_type': 1, 'dims': [200704, 384, 28, 28], 'vals': []}
                input {'name': 'conv3_9/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_3_9_1', 'val_type': 1, 'dims': [200704, 416, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_3_9_1', 'val_type': 1, 'dims': [200704, 416, 28, 28], 'vals': []}
                input {'name': 'conv3_10/x1/bn_scale_0', 'val_type': 1, 'dims': [416], 'vals': []}
                input {'name': 'conv3_10/x1/bn_bias_0', 'val_type': 1, 'dims': [416], 'vals': []}
                input {'name': 'conv3_10/x1/bn_mean_0', 'val_type': 1, 'dims': [416], 'vals': []}
                input {'name': 'conv3_10/x1/bn_var_0', 'val_type': 1, 'dims': [416], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_10/x1/bn_1', 'val_type': 1, 'dims': [200704, 416, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_10/x1/bn_1', 'val_type': 1, 'dims': [200704, 416, 28, 28], 'vals': []}
                input {'name': 'conv3_10/x1/bn_w_0', 'val_type': 1, 'dims': [416], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_10/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 416, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_10/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 416, 28, 28], 'vals': []}
                input {'name': 'conv3_10/x1/bn_b_0', 'val_type': 1, 'dims': [416], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_10/x1/bn_2', 'val_type': 1, 'dims': [200704, 416, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_10/x1/bn_2', 'val_type': 1, 'dims': [200704, 416, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_10/x1/bn_3', 'val_type': 1, 'dims': [200704, 416, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv3_10/x1/bn_3', 'val_type': 1, 'dims': [200704, 416, 28, 28], 'vals': []}
                input {'name': 'conv3_10/x1_w_0', 'val_type': 1, 'dims': [128, 416, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_10/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv3_10/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_10/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_10/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_10/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_10/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_10/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_10/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_10/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_10/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_10/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_10/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_10/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_10/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_10/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv3_10/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_10/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_10/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_3_9_1', 'val_type': 1, 'dims': [200704, 416, 28, 28], 'vals': []}
                input {'name': 'conv3_10/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_3_10_1', 'val_type': 1, 'dims': [200704, 448, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_3_10_1', 'val_type': 1, 'dims': [200704, 448, 28, 28], 'vals': []}
                input {'name': 'conv3_11/x1/bn_scale_0', 'val_type': 1, 'dims': [448], 'vals': []}
                input {'name': 'conv3_11/x1/bn_bias_0', 'val_type': 1, 'dims': [448], 'vals': []}
                input {'name': 'conv3_11/x1/bn_mean_0', 'val_type': 1, 'dims': [448], 'vals': []}
                input {'name': 'conv3_11/x1/bn_var_0', 'val_type': 1, 'dims': [448], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_11/x1/bn_1', 'val_type': 1, 'dims': [200704, 448, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_11/x1/bn_1', 'val_type': 1, 'dims': [200704, 448, 28, 28], 'vals': []}
                input {'name': 'conv3_11/x1/bn_w_0', 'val_type': 1, 'dims': [448], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_11/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 448, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_11/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 448, 28, 28], 'vals': []}
                input {'name': 'conv3_11/x1/bn_b_0', 'val_type': 1, 'dims': [448], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_11/x1/bn_2', 'val_type': 1, 'dims': [200704, 448, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_11/x1/bn_2', 'val_type': 1, 'dims': [200704, 448, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_11/x1/bn_3', 'val_type': 1, 'dims': [200704, 448, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv3_11/x1/bn_3', 'val_type': 1, 'dims': [200704, 448, 28, 28], 'vals': []}
                input {'name': 'conv3_11/x1_w_0', 'val_type': 1, 'dims': [128, 448, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_11/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv3_11/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_11/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_11/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_11/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_11/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_11/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_11/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_11/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_11/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_11/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_11/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_11/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_11/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_11/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv3_11/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_11/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_11/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_3_10_1', 'val_type': 1, 'dims': [200704, 448, 28, 28], 'vals': []}
                input {'name': 'conv3_11/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_3_11_1', 'val_type': 1, 'dims': [200704, 480, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_3_11_1', 'val_type': 1, 'dims': [200704, 480, 28, 28], 'vals': []}
                input {'name': 'conv3_12/x1/bn_scale_0', 'val_type': 1, 'dims': [480], 'vals': []}
                input {'name': 'conv3_12/x1/bn_bias_0', 'val_type': 1, 'dims': [480], 'vals': []}
                input {'name': 'conv3_12/x1/bn_mean_0', 'val_type': 1, 'dims': [480], 'vals': []}
                input {'name': 'conv3_12/x1/bn_var_0', 'val_type': 1, 'dims': [480], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_12/x1/bn_1', 'val_type': 1, 'dims': [200704, 480, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_12/x1/bn_1', 'val_type': 1, 'dims': [200704, 480, 28, 28], 'vals': []}
                input {'name': 'conv3_12/x1/bn_w_0', 'val_type': 1, 'dims': [480], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_12/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 480, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_12/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 480, 28, 28], 'vals': []}
                input {'name': 'conv3_12/x1/bn_b_0', 'val_type': 1, 'dims': [480], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_12/x1/bn_2', 'val_type': 1, 'dims': [200704, 480, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_12/x1/bn_2', 'val_type': 1, 'dims': [200704, 480, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_12/x1/bn_3', 'val_type': 1, 'dims': [200704, 480, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv3_12/x1/bn_3', 'val_type': 1, 'dims': [200704, 480, 28, 28], 'vals': []}
                input {'name': 'conv3_12/x1_w_0', 'val_type': 1, 'dims': [128, 480, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_12/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv3_12/x1_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_12/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_12/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_12/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv3_12/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_12/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_12/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_12/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_12/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_12/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_12/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_12/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_12/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_12/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv3_12/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 28, 28], 'vals': []}
                input {'name': 'conv3_12/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_12/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_3_11_1', 'val_type': 1, 'dims': [200704, 480, 28, 28], 'vals': []}
                input {'name': 'conv3_12/x2_1', 'val_type': 1, 'dims': [200704, 32, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_3_12_1', 'val_type': 1, 'dims': [200704, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_3_12_1', 'val_type': 1, 'dims': [200704, 512, 28, 28], 'vals': []}
                input {'name': 'conv3_blk/bn_scale_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'conv3_blk/bn_bias_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'conv3_blk/bn_mean_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'conv3_blk/bn_var_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_blk/bn_1', 'val_type': 1, 'dims': [200704, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_blk/bn_1', 'val_type': 1, 'dims': [200704, 512, 28, 28], 'vals': []}
                input {'name': 'conv3_blk/bn_w_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_blk/bn_internal_1', 'val_type': 1, 'dims': [200704, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv3_blk/bn_internal_1', 'val_type': 1, 'dims': [200704, 512, 28, 28], 'vals': []}
                input {'name': 'conv3_blk/bn_b_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_blk/bn_2', 'val_type': 1, 'dims': [200704, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv3_blk/bn_2', 'val_type': 1, 'dims': [200704, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_blk/bn_3', 'val_type': 1, 'dims': [200704, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv3_blk/bn_3', 'val_type': 1, 'dims': [200704, 512, 28, 28], 'vals': []}
                input {'name': 'conv3_blk_w_0', 'val_type': 1, 'dims': [256, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv3_blk_1', 'val_type': 1, 'dims': [200704, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'AveragePool'
            'attributes': {'strides': [2, 2], 'pads': [0, 0, 0, 0], 'kernel_shape': [2, 2]}
            'inputs': [
                input {'name': 'conv3_blk_1', 'val_type': 1, 'dims': [200704, 256, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'pool3_1', 'val_type': 1, 'dims': [200704, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'pool3_1', 'val_type': 1, 'dims': [200704, 256, 14, 14], 'vals': []}
                input {'name': 'conv4_1/x1/bn_scale_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'conv4_1/x1/bn_bias_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'conv4_1/x1/bn_mean_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'conv4_1/x1/bn_var_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_1/x1/bn_1', 'val_type': 1, 'dims': [200704, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_1/x1/bn_1', 'val_type': 1, 'dims': [200704, 256, 14, 14], 'vals': []}
                input {'name': 'conv4_1/x1/bn_w_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_1/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_1/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 256, 14, 14], 'vals': []}
                input {'name': 'conv4_1/x1/bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_1/x1/bn_2', 'val_type': 1, 'dims': [200704, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_1/x1/bn_2', 'val_type': 1, 'dims': [200704, 256, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_1/x1/bn_3', 'val_type': 1, 'dims': [200704, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_1/x1/bn_3', 'val_type': 1, 'dims': [200704, 256, 14, 14], 'vals': []}
                input {'name': 'conv4_1/x1_w_0', 'val_type': 1, 'dims': [128, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_1/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_1/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_1/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_1/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_1/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_1/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_1/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_1/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_1/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_1/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_1/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_1/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_1/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_1/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_1/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_1/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_1/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_1/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'pool3_1', 'val_type': 1, 'dims': [200704, 256, 14, 14], 'vals': []}
                input {'name': 'conv4_1/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_1_1', 'val_type': 1, 'dims': [200704, 288, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_1_1', 'val_type': 1, 'dims': [200704, 288, 14, 14], 'vals': []}
                input {'name': 'conv4_2/x1/bn_scale_0', 'val_type': 1, 'dims': [288], 'vals': []}
                input {'name': 'conv4_2/x1/bn_bias_0', 'val_type': 1, 'dims': [288], 'vals': []}
                input {'name': 'conv4_2/x1/bn_mean_0', 'val_type': 1, 'dims': [288], 'vals': []}
                input {'name': 'conv4_2/x1/bn_var_0', 'val_type': 1, 'dims': [288], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_2/x1/bn_1', 'val_type': 1, 'dims': [200704, 288, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_2/x1/bn_1', 'val_type': 1, 'dims': [200704, 288, 14, 14], 'vals': []}
                input {'name': 'conv4_2/x1/bn_w_0', 'val_type': 1, 'dims': [288], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_2/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 288, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_2/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 288, 14, 14], 'vals': []}
                input {'name': 'conv4_2/x1/bn_b_0', 'val_type': 1, 'dims': [288], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_2/x1/bn_2', 'val_type': 1, 'dims': [200704, 288, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_2/x1/bn_2', 'val_type': 1, 'dims': [200704, 288, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_2/x1/bn_3', 'val_type': 1, 'dims': [200704, 288, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_2/x1/bn_3', 'val_type': 1, 'dims': [200704, 288, 14, 14], 'vals': []}
                input {'name': 'conv4_2/x1_w_0', 'val_type': 1, 'dims': [128, 288, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_2/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_2/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_2/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_2/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_2/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_2/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_2/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_2/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_2/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_2/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_2/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_2/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_2/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_2/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_2/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_2/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_2/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_2/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_1_1', 'val_type': 1, 'dims': [200704, 288, 14, 14], 'vals': []}
                input {'name': 'conv4_2/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_2_1', 'val_type': 1, 'dims': [200704, 320, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_2_1', 'val_type': 1, 'dims': [200704, 320, 14, 14], 'vals': []}
                input {'name': 'conv4_3/x1/bn_scale_0', 'val_type': 1, 'dims': [320], 'vals': []}
                input {'name': 'conv4_3/x1/bn_bias_0', 'val_type': 1, 'dims': [320], 'vals': []}
                input {'name': 'conv4_3/x1/bn_mean_0', 'val_type': 1, 'dims': [320], 'vals': []}
                input {'name': 'conv4_3/x1/bn_var_0', 'val_type': 1, 'dims': [320], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_3/x1/bn_1', 'val_type': 1, 'dims': [200704, 320, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_3/x1/bn_1', 'val_type': 1, 'dims': [200704, 320, 14, 14], 'vals': []}
                input {'name': 'conv4_3/x1/bn_w_0', 'val_type': 1, 'dims': [320], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_3/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 320, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_3/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 320, 14, 14], 'vals': []}
                input {'name': 'conv4_3/x1/bn_b_0', 'val_type': 1, 'dims': [320], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_3/x1/bn_2', 'val_type': 1, 'dims': [200704, 320, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_3/x1/bn_2', 'val_type': 1, 'dims': [200704, 320, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_3/x1/bn_3', 'val_type': 1, 'dims': [200704, 320, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_3/x1/bn_3', 'val_type': 1, 'dims': [200704, 320, 14, 14], 'vals': []}
                input {'name': 'conv4_3/x1_w_0', 'val_type': 1, 'dims': [128, 320, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_3/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_3/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_3/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_3/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_3/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_3/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_3/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_3/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_3/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_3/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_3/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_3/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_3/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_3/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_3/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_3/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_3/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_3/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_2_1', 'val_type': 1, 'dims': [200704, 320, 14, 14], 'vals': []}
                input {'name': 'conv4_3/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_3_1', 'val_type': 1, 'dims': [200704, 352, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_3_1', 'val_type': 1, 'dims': [200704, 352, 14, 14], 'vals': []}
                input {'name': 'conv4_4/x1/bn_scale_0', 'val_type': 1, 'dims': [352], 'vals': []}
                input {'name': 'conv4_4/x1/bn_bias_0', 'val_type': 1, 'dims': [352], 'vals': []}
                input {'name': 'conv4_4/x1/bn_mean_0', 'val_type': 1, 'dims': [352], 'vals': []}
                input {'name': 'conv4_4/x1/bn_var_0', 'val_type': 1, 'dims': [352], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_4/x1/bn_1', 'val_type': 1, 'dims': [200704, 352, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_4/x1/bn_1', 'val_type': 1, 'dims': [200704, 352, 14, 14], 'vals': []}
                input {'name': 'conv4_4/x1/bn_w_0', 'val_type': 1, 'dims': [352], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_4/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 352, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_4/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 352, 14, 14], 'vals': []}
                input {'name': 'conv4_4/x1/bn_b_0', 'val_type': 1, 'dims': [352], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_4/x1/bn_2', 'val_type': 1, 'dims': [200704, 352, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_4/x1/bn_2', 'val_type': 1, 'dims': [200704, 352, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_4/x1/bn_3', 'val_type': 1, 'dims': [200704, 352, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_4/x1/bn_3', 'val_type': 1, 'dims': [200704, 352, 14, 14], 'vals': []}
                input {'name': 'conv4_4/x1_w_0', 'val_type': 1, 'dims': [128, 352, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_4/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_4/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_4/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_4/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_4/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_4/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_4/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_4/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_4/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_4/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_4/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_4/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_4/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_4/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_4/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_4/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_4/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_4/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_3_1', 'val_type': 1, 'dims': [200704, 352, 14, 14], 'vals': []}
                input {'name': 'conv4_4/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_4_1', 'val_type': 1, 'dims': [200704, 384, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_4_1', 'val_type': 1, 'dims': [200704, 384, 14, 14], 'vals': []}
                input {'name': 'conv4_5/x1/bn_scale_0', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'conv4_5/x1/bn_bias_0', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'conv4_5/x1/bn_mean_0', 'val_type': 1, 'dims': [384], 'vals': []}
                input {'name': 'conv4_5/x1/bn_var_0', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_5/x1/bn_1', 'val_type': 1, 'dims': [200704, 384, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_5/x1/bn_1', 'val_type': 1, 'dims': [200704, 384, 14, 14], 'vals': []}
                input {'name': 'conv4_5/x1/bn_w_0', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_5/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 384, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_5/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 384, 14, 14], 'vals': []}
                input {'name': 'conv4_5/x1/bn_b_0', 'val_type': 1, 'dims': [384], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_5/x1/bn_2', 'val_type': 1, 'dims': [200704, 384, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_5/x1/bn_2', 'val_type': 1, 'dims': [200704, 384, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_5/x1/bn_3', 'val_type': 1, 'dims': [200704, 384, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_5/x1/bn_3', 'val_type': 1, 'dims': [200704, 384, 14, 14], 'vals': []}
                input {'name': 'conv4_5/x1_w_0', 'val_type': 1, 'dims': [128, 384, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_5/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_5/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_5/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_5/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_5/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_5/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_5/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_5/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_5/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_5/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_5/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_5/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_5/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_5/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_5/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_5/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_5/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_5/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_4_1', 'val_type': 1, 'dims': [200704, 384, 14, 14], 'vals': []}
                input {'name': 'conv4_5/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_5_1', 'val_type': 1, 'dims': [200704, 416, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_5_1', 'val_type': 1, 'dims': [200704, 416, 14, 14], 'vals': []}
                input {'name': 'conv4_6/x1/bn_scale_0', 'val_type': 1, 'dims': [416], 'vals': []}
                input {'name': 'conv4_6/x1/bn_bias_0', 'val_type': 1, 'dims': [416], 'vals': []}
                input {'name': 'conv4_6/x1/bn_mean_0', 'val_type': 1, 'dims': [416], 'vals': []}
                input {'name': 'conv4_6/x1/bn_var_0', 'val_type': 1, 'dims': [416], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_6/x1/bn_1', 'val_type': 1, 'dims': [200704, 416, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_6/x1/bn_1', 'val_type': 1, 'dims': [200704, 416, 14, 14], 'vals': []}
                input {'name': 'conv4_6/x1/bn_w_0', 'val_type': 1, 'dims': [416], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_6/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 416, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_6/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 416, 14, 14], 'vals': []}
                input {'name': 'conv4_6/x1/bn_b_0', 'val_type': 1, 'dims': [416], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_6/x1/bn_2', 'val_type': 1, 'dims': [200704, 416, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_6/x1/bn_2', 'val_type': 1, 'dims': [200704, 416, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_6/x1/bn_3', 'val_type': 1, 'dims': [200704, 416, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_6/x1/bn_3', 'val_type': 1, 'dims': [200704, 416, 14, 14], 'vals': []}
                input {'name': 'conv4_6/x1_w_0', 'val_type': 1, 'dims': [128, 416, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_6/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_6/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_6/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_6/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_6/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_6/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_6/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_6/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_6/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_6/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_6/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_6/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_6/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_6/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_6/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_6/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_6/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_6/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_5_1', 'val_type': 1, 'dims': [200704, 416, 14, 14], 'vals': []}
                input {'name': 'conv4_6/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_6_1', 'val_type': 1, 'dims': [200704, 448, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_6_1', 'val_type': 1, 'dims': [200704, 448, 14, 14], 'vals': []}
                input {'name': 'conv4_7/x1/bn_scale_0', 'val_type': 1, 'dims': [448], 'vals': []}
                input {'name': 'conv4_7/x1/bn_bias_0', 'val_type': 1, 'dims': [448], 'vals': []}
                input {'name': 'conv4_7/x1/bn_mean_0', 'val_type': 1, 'dims': [448], 'vals': []}
                input {'name': 'conv4_7/x1/bn_var_0', 'val_type': 1, 'dims': [448], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_7/x1/bn_1', 'val_type': 1, 'dims': [200704, 448, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_7/x1/bn_1', 'val_type': 1, 'dims': [200704, 448, 14, 14], 'vals': []}
                input {'name': 'conv4_7/x1/bn_w_0', 'val_type': 1, 'dims': [448], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_7/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 448, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_7/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 448, 14, 14], 'vals': []}
                input {'name': 'conv4_7/x1/bn_b_0', 'val_type': 1, 'dims': [448], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_7/x1/bn_2', 'val_type': 1, 'dims': [200704, 448, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_7/x1/bn_2', 'val_type': 1, 'dims': [200704, 448, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_7/x1/bn_3', 'val_type': 1, 'dims': [200704, 448, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_7/x1/bn_3', 'val_type': 1, 'dims': [200704, 448, 14, 14], 'vals': []}
                input {'name': 'conv4_7/x1_w_0', 'val_type': 1, 'dims': [128, 448, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_7/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_7/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_7/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_7/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_7/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_7/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_7/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_7/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_7/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_7/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_7/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_7/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_7/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_7/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_7/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_7/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_7/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_7/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_6_1', 'val_type': 1, 'dims': [200704, 448, 14, 14], 'vals': []}
                input {'name': 'conv4_7/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_7_1', 'val_type': 1, 'dims': [200704, 480, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_7_1', 'val_type': 1, 'dims': [200704, 480, 14, 14], 'vals': []}
                input {'name': 'conv4_8/x1/bn_scale_0', 'val_type': 1, 'dims': [480], 'vals': []}
                input {'name': 'conv4_8/x1/bn_bias_0', 'val_type': 1, 'dims': [480], 'vals': []}
                input {'name': 'conv4_8/x1/bn_mean_0', 'val_type': 1, 'dims': [480], 'vals': []}
                input {'name': 'conv4_8/x1/bn_var_0', 'val_type': 1, 'dims': [480], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_8/x1/bn_1', 'val_type': 1, 'dims': [200704, 480, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_8/x1/bn_1', 'val_type': 1, 'dims': [200704, 480, 14, 14], 'vals': []}
                input {'name': 'conv4_8/x1/bn_w_0', 'val_type': 1, 'dims': [480], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_8/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 480, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_8/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 480, 14, 14], 'vals': []}
                input {'name': 'conv4_8/x1/bn_b_0', 'val_type': 1, 'dims': [480], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_8/x1/bn_2', 'val_type': 1, 'dims': [200704, 480, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_8/x1/bn_2', 'val_type': 1, 'dims': [200704, 480, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_8/x1/bn_3', 'val_type': 1, 'dims': [200704, 480, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_8/x1/bn_3', 'val_type': 1, 'dims': [200704, 480, 14, 14], 'vals': []}
                input {'name': 'conv4_8/x1_w_0', 'val_type': 1, 'dims': [128, 480, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_8/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_8/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_8/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_8/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_8/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_8/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_8/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_8/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_8/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_8/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_8/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_8/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_8/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_8/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_8/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_8/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_8/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_8/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_7_1', 'val_type': 1, 'dims': [200704, 480, 14, 14], 'vals': []}
                input {'name': 'conv4_8/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_8_1', 'val_type': 1, 'dims': [200704, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_8_1', 'val_type': 1, 'dims': [200704, 512, 14, 14], 'vals': []}
                input {'name': 'conv4_9/x1/bn_scale_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'conv4_9/x1/bn_bias_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'conv4_9/x1/bn_mean_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'conv4_9/x1/bn_var_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_9/x1/bn_1', 'val_type': 1, 'dims': [200704, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_9/x1/bn_1', 'val_type': 1, 'dims': [200704, 512, 14, 14], 'vals': []}
                input {'name': 'conv4_9/x1/bn_w_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_9/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_9/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 512, 14, 14], 'vals': []}
                input {'name': 'conv4_9/x1/bn_b_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_9/x1/bn_2', 'val_type': 1, 'dims': [200704, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_9/x1/bn_2', 'val_type': 1, 'dims': [200704, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_9/x1/bn_3', 'val_type': 1, 'dims': [200704, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_9/x1/bn_3', 'val_type': 1, 'dims': [200704, 512, 14, 14], 'vals': []}
                input {'name': 'conv4_9/x1_w_0', 'val_type': 1, 'dims': [128, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_9/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_9/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_9/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_9/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_9/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_9/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_9/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_9/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_9/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_9/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_9/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_9/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_9/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_9/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_9/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_9/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_9/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_9/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_8_1', 'val_type': 1, 'dims': [200704, 512, 14, 14], 'vals': []}
                input {'name': 'conv4_9/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_9_1', 'val_type': 1, 'dims': [200704, 544, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_9_1', 'val_type': 1, 'dims': [200704, 544, 14, 14], 'vals': []}
                input {'name': 'conv4_10/x1/bn_scale_0', 'val_type': 1, 'dims': [544], 'vals': []}
                input {'name': 'conv4_10/x1/bn_bias_0', 'val_type': 1, 'dims': [544], 'vals': []}
                input {'name': 'conv4_10/x1/bn_mean_0', 'val_type': 1, 'dims': [544], 'vals': []}
                input {'name': 'conv4_10/x1/bn_var_0', 'val_type': 1, 'dims': [544], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_10/x1/bn_1', 'val_type': 1, 'dims': [200704, 544, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_10/x1/bn_1', 'val_type': 1, 'dims': [200704, 544, 14, 14], 'vals': []}
                input {'name': 'conv4_10/x1/bn_w_0', 'val_type': 1, 'dims': [544], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_10/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 544, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_10/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 544, 14, 14], 'vals': []}
                input {'name': 'conv4_10/x1/bn_b_0', 'val_type': 1, 'dims': [544], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_10/x1/bn_2', 'val_type': 1, 'dims': [200704, 544, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_10/x1/bn_2', 'val_type': 1, 'dims': [200704, 544, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_10/x1/bn_3', 'val_type': 1, 'dims': [200704, 544, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_10/x1/bn_3', 'val_type': 1, 'dims': [200704, 544, 14, 14], 'vals': []}
                input {'name': 'conv4_10/x1_w_0', 'val_type': 1, 'dims': [128, 544, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_10/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_10/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_10/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_10/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_10/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_10/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_10/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_10/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_10/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_10/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_10/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_10/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_10/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_10/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_10/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_10/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_10/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_10/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_9_1', 'val_type': 1, 'dims': [200704, 544, 14, 14], 'vals': []}
                input {'name': 'conv4_10/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_10_1', 'val_type': 1, 'dims': [200704, 576, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_10_1', 'val_type': 1, 'dims': [200704, 576, 14, 14], 'vals': []}
                input {'name': 'conv4_11/x1/bn_scale_0', 'val_type': 1, 'dims': [576], 'vals': []}
                input {'name': 'conv4_11/x1/bn_bias_0', 'val_type': 1, 'dims': [576], 'vals': []}
                input {'name': 'conv4_11/x1/bn_mean_0', 'val_type': 1, 'dims': [576], 'vals': []}
                input {'name': 'conv4_11/x1/bn_var_0', 'val_type': 1, 'dims': [576], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_11/x1/bn_1', 'val_type': 1, 'dims': [200704, 576, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_11/x1/bn_1', 'val_type': 1, 'dims': [200704, 576, 14, 14], 'vals': []}
                input {'name': 'conv4_11/x1/bn_w_0', 'val_type': 1, 'dims': [576], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_11/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 576, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_11/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 576, 14, 14], 'vals': []}
                input {'name': 'conv4_11/x1/bn_b_0', 'val_type': 1, 'dims': [576], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_11/x1/bn_2', 'val_type': 1, 'dims': [200704, 576, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_11/x1/bn_2', 'val_type': 1, 'dims': [200704, 576, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_11/x1/bn_3', 'val_type': 1, 'dims': [200704, 576, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_11/x1/bn_3', 'val_type': 1, 'dims': [200704, 576, 14, 14], 'vals': []}
                input {'name': 'conv4_11/x1_w_0', 'val_type': 1, 'dims': [128, 576, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_11/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_11/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_11/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_11/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_11/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_11/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_11/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_11/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_11/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_11/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_11/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_11/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_11/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_11/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_11/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_11/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_11/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_11/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_10_1', 'val_type': 1, 'dims': [200704, 576, 14, 14], 'vals': []}
                input {'name': 'conv4_11/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_11_1', 'val_type': 1, 'dims': [200704, 608, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_11_1', 'val_type': 1, 'dims': [200704, 608, 14, 14], 'vals': []}
                input {'name': 'conv4_12/x1/bn_scale_0', 'val_type': 1, 'dims': [608], 'vals': []}
                input {'name': 'conv4_12/x1/bn_bias_0', 'val_type': 1, 'dims': [608], 'vals': []}
                input {'name': 'conv4_12/x1/bn_mean_0', 'val_type': 1, 'dims': [608], 'vals': []}
                input {'name': 'conv4_12/x1/bn_var_0', 'val_type': 1, 'dims': [608], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_12/x1/bn_1', 'val_type': 1, 'dims': [200704, 608, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_12/x1/bn_1', 'val_type': 1, 'dims': [200704, 608, 14, 14], 'vals': []}
                input {'name': 'conv4_12/x1/bn_w_0', 'val_type': 1, 'dims': [608], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_12/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 608, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_12/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 608, 14, 14], 'vals': []}
                input {'name': 'conv4_12/x1/bn_b_0', 'val_type': 1, 'dims': [608], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_12/x1/bn_2', 'val_type': 1, 'dims': [200704, 608, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_12/x1/bn_2', 'val_type': 1, 'dims': [200704, 608, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_12/x1/bn_3', 'val_type': 1, 'dims': [200704, 608, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_12/x1/bn_3', 'val_type': 1, 'dims': [200704, 608, 14, 14], 'vals': []}
                input {'name': 'conv4_12/x1_w_0', 'val_type': 1, 'dims': [128, 608, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_12/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_12/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_12/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_12/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_12/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_12/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_12/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_12/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_12/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_12/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_12/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_12/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_12/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_12/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_12/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_12/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_12/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_12/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_11_1', 'val_type': 1, 'dims': [200704, 608, 14, 14], 'vals': []}
                input {'name': 'conv4_12/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_12_1', 'val_type': 1, 'dims': [200704, 640, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_12_1', 'val_type': 1, 'dims': [200704, 640, 14, 14], 'vals': []}
                input {'name': 'conv4_13/x1/bn_scale_0', 'val_type': 1, 'dims': [640], 'vals': []}
                input {'name': 'conv4_13/x1/bn_bias_0', 'val_type': 1, 'dims': [640], 'vals': []}
                input {'name': 'conv4_13/x1/bn_mean_0', 'val_type': 1, 'dims': [640], 'vals': []}
                input {'name': 'conv4_13/x1/bn_var_0', 'val_type': 1, 'dims': [640], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_13/x1/bn_1', 'val_type': 1, 'dims': [200704, 640, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_13/x1/bn_1', 'val_type': 1, 'dims': [200704, 640, 14, 14], 'vals': []}
                input {'name': 'conv4_13/x1/bn_w_0', 'val_type': 1, 'dims': [640], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_13/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 640, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_13/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 640, 14, 14], 'vals': []}
                input {'name': 'conv4_13/x1/bn_b_0', 'val_type': 1, 'dims': [640], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_13/x1/bn_2', 'val_type': 1, 'dims': [200704, 640, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_13/x1/bn_2', 'val_type': 1, 'dims': [200704, 640, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_13/x1/bn_3', 'val_type': 1, 'dims': [200704, 640, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_13/x1/bn_3', 'val_type': 1, 'dims': [200704, 640, 14, 14], 'vals': []}
                input {'name': 'conv4_13/x1_w_0', 'val_type': 1, 'dims': [128, 640, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_13/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_13/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_13/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_13/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_13/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_13/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_13/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_13/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_13/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_13/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_13/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_13/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_13/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_13/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_13/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_13/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_13/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_13/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_12_1', 'val_type': 1, 'dims': [200704, 640, 14, 14], 'vals': []}
                input {'name': 'conv4_13/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_13_1', 'val_type': 1, 'dims': [200704, 672, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_13_1', 'val_type': 1, 'dims': [200704, 672, 14, 14], 'vals': []}
                input {'name': 'conv4_14/x1/bn_scale_0', 'val_type': 1, 'dims': [672], 'vals': []}
                input {'name': 'conv4_14/x1/bn_bias_0', 'val_type': 1, 'dims': [672], 'vals': []}
                input {'name': 'conv4_14/x1/bn_mean_0', 'val_type': 1, 'dims': [672], 'vals': []}
                input {'name': 'conv4_14/x1/bn_var_0', 'val_type': 1, 'dims': [672], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_14/x1/bn_1', 'val_type': 1, 'dims': [200704, 672, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_14/x1/bn_1', 'val_type': 1, 'dims': [200704, 672, 14, 14], 'vals': []}
                input {'name': 'conv4_14/x1/bn_w_0', 'val_type': 1, 'dims': [672], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_14/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 672, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_14/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 672, 14, 14], 'vals': []}
                input {'name': 'conv4_14/x1/bn_b_0', 'val_type': 1, 'dims': [672], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_14/x1/bn_2', 'val_type': 1, 'dims': [200704, 672, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_14/x1/bn_2', 'val_type': 1, 'dims': [200704, 672, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_14/x1/bn_3', 'val_type': 1, 'dims': [200704, 672, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_14/x1/bn_3', 'val_type': 1, 'dims': [200704, 672, 14, 14], 'vals': []}
                input {'name': 'conv4_14/x1_w_0', 'val_type': 1, 'dims': [128, 672, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_14/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_14/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_14/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_14/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_14/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_14/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_14/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_14/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_14/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_14/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_14/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_14/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_14/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_14/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_14/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_14/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_14/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_14/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_13_1', 'val_type': 1, 'dims': [200704, 672, 14, 14], 'vals': []}
                input {'name': 'conv4_14/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_14_1', 'val_type': 1, 'dims': [200704, 704, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_14_1', 'val_type': 1, 'dims': [200704, 704, 14, 14], 'vals': []}
                input {'name': 'conv4_15/x1/bn_scale_0', 'val_type': 1, 'dims': [704], 'vals': []}
                input {'name': 'conv4_15/x1/bn_bias_0', 'val_type': 1, 'dims': [704], 'vals': []}
                input {'name': 'conv4_15/x1/bn_mean_0', 'val_type': 1, 'dims': [704], 'vals': []}
                input {'name': 'conv4_15/x1/bn_var_0', 'val_type': 1, 'dims': [704], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_15/x1/bn_1', 'val_type': 1, 'dims': [200704, 704, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_15/x1/bn_1', 'val_type': 1, 'dims': [200704, 704, 14, 14], 'vals': []}
                input {'name': 'conv4_15/x1/bn_w_0', 'val_type': 1, 'dims': [704], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_15/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 704, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_15/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 704, 14, 14], 'vals': []}
                input {'name': 'conv4_15/x1/bn_b_0', 'val_type': 1, 'dims': [704], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_15/x1/bn_2', 'val_type': 1, 'dims': [200704, 704, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_15/x1/bn_2', 'val_type': 1, 'dims': [200704, 704, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_15/x1/bn_3', 'val_type': 1, 'dims': [200704, 704, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_15/x1/bn_3', 'val_type': 1, 'dims': [200704, 704, 14, 14], 'vals': []}
                input {'name': 'conv4_15/x1_w_0', 'val_type': 1, 'dims': [128, 704, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_15/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_15/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_15/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_15/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_15/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_15/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_15/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_15/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_15/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_15/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_15/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_15/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_15/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_15/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_15/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_15/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_15/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_15/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_14_1', 'val_type': 1, 'dims': [200704, 704, 14, 14], 'vals': []}
                input {'name': 'conv4_15/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_15_1', 'val_type': 1, 'dims': [200704, 736, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_15_1', 'val_type': 1, 'dims': [200704, 736, 14, 14], 'vals': []}
                input {'name': 'conv4_16/x1/bn_scale_0', 'val_type': 1, 'dims': [736], 'vals': []}
                input {'name': 'conv4_16/x1/bn_bias_0', 'val_type': 1, 'dims': [736], 'vals': []}
                input {'name': 'conv4_16/x1/bn_mean_0', 'val_type': 1, 'dims': [736], 'vals': []}
                input {'name': 'conv4_16/x1/bn_var_0', 'val_type': 1, 'dims': [736], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_16/x1/bn_1', 'val_type': 1, 'dims': [200704, 736, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_16/x1/bn_1', 'val_type': 1, 'dims': [200704, 736, 14, 14], 'vals': []}
                input {'name': 'conv4_16/x1/bn_w_0', 'val_type': 1, 'dims': [736], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_16/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 736, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_16/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 736, 14, 14], 'vals': []}
                input {'name': 'conv4_16/x1/bn_b_0', 'val_type': 1, 'dims': [736], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_16/x1/bn_2', 'val_type': 1, 'dims': [200704, 736, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_16/x1/bn_2', 'val_type': 1, 'dims': [200704, 736, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_16/x1/bn_3', 'val_type': 1, 'dims': [200704, 736, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_16/x1/bn_3', 'val_type': 1, 'dims': [200704, 736, 14, 14], 'vals': []}
                input {'name': 'conv4_16/x1_w_0', 'val_type': 1, 'dims': [128, 736, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_16/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_16/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_16/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_16/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_16/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_16/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_16/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_16/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_16/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_16/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_16/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_16/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_16/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_16/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_16/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_16/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_16/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_16/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_15_1', 'val_type': 1, 'dims': [200704, 736, 14, 14], 'vals': []}
                input {'name': 'conv4_16/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_16_1', 'val_type': 1, 'dims': [200704, 768, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_16_1', 'val_type': 1, 'dims': [200704, 768, 14, 14], 'vals': []}
                input {'name': 'conv4_17/x1/bn_scale_0', 'val_type': 1, 'dims': [768], 'vals': []}
                input {'name': 'conv4_17/x1/bn_bias_0', 'val_type': 1, 'dims': [768], 'vals': []}
                input {'name': 'conv4_17/x1/bn_mean_0', 'val_type': 1, 'dims': [768], 'vals': []}
                input {'name': 'conv4_17/x1/bn_var_0', 'val_type': 1, 'dims': [768], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_17/x1/bn_1', 'val_type': 1, 'dims': [200704, 768, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_17/x1/bn_1', 'val_type': 1, 'dims': [200704, 768, 14, 14], 'vals': []}
                input {'name': 'conv4_17/x1/bn_w_0', 'val_type': 1, 'dims': [768], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_17/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 768, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_17/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 768, 14, 14], 'vals': []}
                input {'name': 'conv4_17/x1/bn_b_0', 'val_type': 1, 'dims': [768], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_17/x1/bn_2', 'val_type': 1, 'dims': [200704, 768, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_17/x1/bn_2', 'val_type': 1, 'dims': [200704, 768, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_17/x1/bn_3', 'val_type': 1, 'dims': [200704, 768, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_17/x1/bn_3', 'val_type': 1, 'dims': [200704, 768, 14, 14], 'vals': []}
                input {'name': 'conv4_17/x1_w_0', 'val_type': 1, 'dims': [128, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_17/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_17/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_17/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_17/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_17/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_17/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_17/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_17/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_17/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_17/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_17/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_17/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_17/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_17/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_17/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_17/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_17/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_17/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_16_1', 'val_type': 1, 'dims': [200704, 768, 14, 14], 'vals': []}
                input {'name': 'conv4_17/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_17_1', 'val_type': 1, 'dims': [200704, 800, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_17_1', 'val_type': 1, 'dims': [200704, 800, 14, 14], 'vals': []}
                input {'name': 'conv4_18/x1/bn_scale_0', 'val_type': 1, 'dims': [800], 'vals': []}
                input {'name': 'conv4_18/x1/bn_bias_0', 'val_type': 1, 'dims': [800], 'vals': []}
                input {'name': 'conv4_18/x1/bn_mean_0', 'val_type': 1, 'dims': [800], 'vals': []}
                input {'name': 'conv4_18/x1/bn_var_0', 'val_type': 1, 'dims': [800], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_18/x1/bn_1', 'val_type': 1, 'dims': [200704, 800, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_18/x1/bn_1', 'val_type': 1, 'dims': [200704, 800, 14, 14], 'vals': []}
                input {'name': 'conv4_18/x1/bn_w_0', 'val_type': 1, 'dims': [800], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_18/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 800, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_18/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 800, 14, 14], 'vals': []}
                input {'name': 'conv4_18/x1/bn_b_0', 'val_type': 1, 'dims': [800], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_18/x1/bn_2', 'val_type': 1, 'dims': [200704, 800, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_18/x1/bn_2', 'val_type': 1, 'dims': [200704, 800, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_18/x1/bn_3', 'val_type': 1, 'dims': [200704, 800, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_18/x1/bn_3', 'val_type': 1, 'dims': [200704, 800, 14, 14], 'vals': []}
                input {'name': 'conv4_18/x1_w_0', 'val_type': 1, 'dims': [128, 800, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_18/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_18/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_18/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_18/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_18/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_18/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_18/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_18/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_18/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_18/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_18/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_18/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_18/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_18/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_18/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_18/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_18/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_18/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_17_1', 'val_type': 1, 'dims': [200704, 800, 14, 14], 'vals': []}
                input {'name': 'conv4_18/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_18_1', 'val_type': 1, 'dims': [200704, 832, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_18_1', 'val_type': 1, 'dims': [200704, 832, 14, 14], 'vals': []}
                input {'name': 'conv4_19/x1/bn_scale_0', 'val_type': 1, 'dims': [832], 'vals': []}
                input {'name': 'conv4_19/x1/bn_bias_0', 'val_type': 1, 'dims': [832], 'vals': []}
                input {'name': 'conv4_19/x1/bn_mean_0', 'val_type': 1, 'dims': [832], 'vals': []}
                input {'name': 'conv4_19/x1/bn_var_0', 'val_type': 1, 'dims': [832], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_19/x1/bn_1', 'val_type': 1, 'dims': [200704, 832, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_19/x1/bn_1', 'val_type': 1, 'dims': [200704, 832, 14, 14], 'vals': []}
                input {'name': 'conv4_19/x1/bn_w_0', 'val_type': 1, 'dims': [832], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_19/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 832, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_19/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 832, 14, 14], 'vals': []}
                input {'name': 'conv4_19/x1/bn_b_0', 'val_type': 1, 'dims': [832], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_19/x1/bn_2', 'val_type': 1, 'dims': [200704, 832, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_19/x1/bn_2', 'val_type': 1, 'dims': [200704, 832, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_19/x1/bn_3', 'val_type': 1, 'dims': [200704, 832, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_19/x1/bn_3', 'val_type': 1, 'dims': [200704, 832, 14, 14], 'vals': []}
                input {'name': 'conv4_19/x1_w_0', 'val_type': 1, 'dims': [128, 832, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_19/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_19/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_19/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_19/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_19/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_19/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_19/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_19/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_19/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_19/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_19/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_19/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_19/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_19/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_19/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_19/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_19/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_19/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_18_1', 'val_type': 1, 'dims': [200704, 832, 14, 14], 'vals': []}
                input {'name': 'conv4_19/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_19_1', 'val_type': 1, 'dims': [200704, 864, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_19_1', 'val_type': 1, 'dims': [200704, 864, 14, 14], 'vals': []}
                input {'name': 'conv4_20/x1/bn_scale_0', 'val_type': 1, 'dims': [864], 'vals': []}
                input {'name': 'conv4_20/x1/bn_bias_0', 'val_type': 1, 'dims': [864], 'vals': []}
                input {'name': 'conv4_20/x1/bn_mean_0', 'val_type': 1, 'dims': [864], 'vals': []}
                input {'name': 'conv4_20/x1/bn_var_0', 'val_type': 1, 'dims': [864], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_20/x1/bn_1', 'val_type': 1, 'dims': [200704, 864, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_20/x1/bn_1', 'val_type': 1, 'dims': [200704, 864, 14, 14], 'vals': []}
                input {'name': 'conv4_20/x1/bn_w_0', 'val_type': 1, 'dims': [864], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_20/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 864, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_20/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 864, 14, 14], 'vals': []}
                input {'name': 'conv4_20/x1/bn_b_0', 'val_type': 1, 'dims': [864], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_20/x1/bn_2', 'val_type': 1, 'dims': [200704, 864, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_20/x1/bn_2', 'val_type': 1, 'dims': [200704, 864, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_20/x1/bn_3', 'val_type': 1, 'dims': [200704, 864, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_20/x1/bn_3', 'val_type': 1, 'dims': [200704, 864, 14, 14], 'vals': []}
                input {'name': 'conv4_20/x1_w_0', 'val_type': 1, 'dims': [128, 864, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_20/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_20/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_20/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_20/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_20/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_20/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_20/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_20/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_20/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_20/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_20/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_20/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_20/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_20/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_20/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_20/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_20/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_20/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_19_1', 'val_type': 1, 'dims': [200704, 864, 14, 14], 'vals': []}
                input {'name': 'conv4_20/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_20_1', 'val_type': 1, 'dims': [200704, 896, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_20_1', 'val_type': 1, 'dims': [200704, 896, 14, 14], 'vals': []}
                input {'name': 'conv4_21/x1/bn_scale_0', 'val_type': 1, 'dims': [896], 'vals': []}
                input {'name': 'conv4_21/x1/bn_bias_0', 'val_type': 1, 'dims': [896], 'vals': []}
                input {'name': 'conv4_21/x1/bn_mean_0', 'val_type': 1, 'dims': [896], 'vals': []}
                input {'name': 'conv4_21/x1/bn_var_0', 'val_type': 1, 'dims': [896], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_21/x1/bn_1', 'val_type': 1, 'dims': [200704, 896, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_21/x1/bn_1', 'val_type': 1, 'dims': [200704, 896, 14, 14], 'vals': []}
                input {'name': 'conv4_21/x1/bn_w_0', 'val_type': 1, 'dims': [896], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_21/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 896, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_21/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 896, 14, 14], 'vals': []}
                input {'name': 'conv4_21/x1/bn_b_0', 'val_type': 1, 'dims': [896], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_21/x1/bn_2', 'val_type': 1, 'dims': [200704, 896, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_21/x1/bn_2', 'val_type': 1, 'dims': [200704, 896, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_21/x1/bn_3', 'val_type': 1, 'dims': [200704, 896, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_21/x1/bn_3', 'val_type': 1, 'dims': [200704, 896, 14, 14], 'vals': []}
                input {'name': 'conv4_21/x1_w_0', 'val_type': 1, 'dims': [128, 896, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_21/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_21/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_21/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_21/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_21/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_21/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_21/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_21/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_21/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_21/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_21/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_21/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_21/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_21/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_21/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_21/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_21/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_21/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_20_1', 'val_type': 1, 'dims': [200704, 896, 14, 14], 'vals': []}
                input {'name': 'conv4_21/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_21_1', 'val_type': 1, 'dims': [200704, 928, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_21_1', 'val_type': 1, 'dims': [200704, 928, 14, 14], 'vals': []}
                input {'name': 'conv4_22/x1/bn_scale_0', 'val_type': 1, 'dims': [928], 'vals': []}
                input {'name': 'conv4_22/x1/bn_bias_0', 'val_type': 1, 'dims': [928], 'vals': []}
                input {'name': 'conv4_22/x1/bn_mean_0', 'val_type': 1, 'dims': [928], 'vals': []}
                input {'name': 'conv4_22/x1/bn_var_0', 'val_type': 1, 'dims': [928], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_22/x1/bn_1', 'val_type': 1, 'dims': [200704, 928, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_22/x1/bn_1', 'val_type': 1, 'dims': [200704, 928, 14, 14], 'vals': []}
                input {'name': 'conv4_22/x1/bn_w_0', 'val_type': 1, 'dims': [928], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_22/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 928, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_22/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 928, 14, 14], 'vals': []}
                input {'name': 'conv4_22/x1/bn_b_0', 'val_type': 1, 'dims': [928], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_22/x1/bn_2', 'val_type': 1, 'dims': [200704, 928, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_22/x1/bn_2', 'val_type': 1, 'dims': [200704, 928, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_22/x1/bn_3', 'val_type': 1, 'dims': [200704, 928, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_22/x1/bn_3', 'val_type': 1, 'dims': [200704, 928, 14, 14], 'vals': []}
                input {'name': 'conv4_22/x1_w_0', 'val_type': 1, 'dims': [128, 928, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_22/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_22/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_22/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_22/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_22/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_22/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_22/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_22/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_22/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_22/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_22/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_22/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_22/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_22/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_22/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_22/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_22/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_22/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_21_1', 'val_type': 1, 'dims': [200704, 928, 14, 14], 'vals': []}
                input {'name': 'conv4_22/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_22_1', 'val_type': 1, 'dims': [200704, 960, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_22_1', 'val_type': 1, 'dims': [200704, 960, 14, 14], 'vals': []}
                input {'name': 'conv4_23/x1/bn_scale_0', 'val_type': 1, 'dims': [960], 'vals': []}
                input {'name': 'conv4_23/x1/bn_bias_0', 'val_type': 1, 'dims': [960], 'vals': []}
                input {'name': 'conv4_23/x1/bn_mean_0', 'val_type': 1, 'dims': [960], 'vals': []}
                input {'name': 'conv4_23/x1/bn_var_0', 'val_type': 1, 'dims': [960], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_23/x1/bn_1', 'val_type': 1, 'dims': [200704, 960, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_23/x1/bn_1', 'val_type': 1, 'dims': [200704, 960, 14, 14], 'vals': []}
                input {'name': 'conv4_23/x1/bn_w_0', 'val_type': 1, 'dims': [960], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_23/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 960, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_23/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 960, 14, 14], 'vals': []}
                input {'name': 'conv4_23/x1/bn_b_0', 'val_type': 1, 'dims': [960], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_23/x1/bn_2', 'val_type': 1, 'dims': [200704, 960, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_23/x1/bn_2', 'val_type': 1, 'dims': [200704, 960, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_23/x1/bn_3', 'val_type': 1, 'dims': [200704, 960, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_23/x1/bn_3', 'val_type': 1, 'dims': [200704, 960, 14, 14], 'vals': []}
                input {'name': 'conv4_23/x1_w_0', 'val_type': 1, 'dims': [128, 960, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_23/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_23/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_23/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_23/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_23/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_23/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_23/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_23/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_23/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_23/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_23/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_23/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_23/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_23/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_23/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_23/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_23/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_23/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_22_1', 'val_type': 1, 'dims': [200704, 960, 14, 14], 'vals': []}
                input {'name': 'conv4_23/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_23_1', 'val_type': 1, 'dims': [200704, 992, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_23_1', 'val_type': 1, 'dims': [200704, 992, 14, 14], 'vals': []}
                input {'name': 'conv4_24/x1/bn_scale_0', 'val_type': 1, 'dims': [992], 'vals': []}
                input {'name': 'conv4_24/x1/bn_bias_0', 'val_type': 1, 'dims': [992], 'vals': []}
                input {'name': 'conv4_24/x1/bn_mean_0', 'val_type': 1, 'dims': [992], 'vals': []}
                input {'name': 'conv4_24/x1/bn_var_0', 'val_type': 1, 'dims': [992], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_24/x1/bn_1', 'val_type': 1, 'dims': [200704, 992, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_24/x1/bn_1', 'val_type': 1, 'dims': [200704, 992, 14, 14], 'vals': []}
                input {'name': 'conv4_24/x1/bn_w_0', 'val_type': 1, 'dims': [992], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_24/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 992, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_24/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 992, 14, 14], 'vals': []}
                input {'name': 'conv4_24/x1/bn_b_0', 'val_type': 1, 'dims': [992], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_24/x1/bn_2', 'val_type': 1, 'dims': [200704, 992, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_24/x1/bn_2', 'val_type': 1, 'dims': [200704, 992, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_24/x1/bn_3', 'val_type': 1, 'dims': [200704, 992, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_24/x1/bn_3', 'val_type': 1, 'dims': [200704, 992, 14, 14], 'vals': []}
                input {'name': 'conv4_24/x1_w_0', 'val_type': 1, 'dims': [128, 992, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_24/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv4_24/x1_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_24/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_24/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_24/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv4_24/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_24/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_24/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_24/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_24/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_24/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_24/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_24/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_24/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_24/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv4_24/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 14, 14], 'vals': []}
                input {'name': 'conv4_24/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_24/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_4_23_1', 'val_type': 1, 'dims': [200704, 992, 14, 14], 'vals': []}
                input {'name': 'conv4_24/x2_1', 'val_type': 1, 'dims': [200704, 32, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_4_24_1', 'val_type': 1, 'dims': [200704, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_4_24_1', 'val_type': 1, 'dims': [200704, 1024, 14, 14], 'vals': []}
                input {'name': 'conv4_blk/bn_scale_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'conv4_blk/bn_bias_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'conv4_blk/bn_mean_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'conv4_blk/bn_var_0', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_blk/bn_1', 'val_type': 1, 'dims': [200704, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_blk/bn_1', 'val_type': 1, 'dims': [200704, 1024, 14, 14], 'vals': []}
                input {'name': 'conv4_blk/bn_w_0', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_blk/bn_internal_1', 'val_type': 1, 'dims': [200704, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv4_blk/bn_internal_1', 'val_type': 1, 'dims': [200704, 1024, 14, 14], 'vals': []}
                input {'name': 'conv4_blk/bn_b_0', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_blk/bn_2', 'val_type': 1, 'dims': [200704, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv4_blk/bn_2', 'val_type': 1, 'dims': [200704, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_blk/bn_3', 'val_type': 1, 'dims': [200704, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv4_blk/bn_3', 'val_type': 1, 'dims': [200704, 1024, 14, 14], 'vals': []}
                input {'name': 'conv4_blk_w_0', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv4_blk_1', 'val_type': 1, 'dims': [200704, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'AveragePool'
            'attributes': {'strides': [2, 2], 'pads': [0, 0, 0, 0], 'kernel_shape': [2, 2]}
            'inputs': [
                input {'name': 'conv4_blk_1', 'val_type': 1, 'dims': [200704, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'pool4_1', 'val_type': 1, 'dims': [200704, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'pool4_1', 'val_type': 1, 'dims': [200704, 512, 7, 7], 'vals': []}
                input {'name': 'conv5_1/x1/bn_scale_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'conv5_1/x1/bn_bias_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'conv5_1/x1/bn_mean_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'conv5_1/x1/bn_var_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_1/x1/bn_1', 'val_type': 1, 'dims': [200704, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_1/x1/bn_1', 'val_type': 1, 'dims': [200704, 512, 7, 7], 'vals': []}
                input {'name': 'conv5_1/x1/bn_w_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_1/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_1/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 512, 7, 7], 'vals': []}
                input {'name': 'conv5_1/x1/bn_b_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_1/x1/bn_2', 'val_type': 1, 'dims': [200704, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_1/x1/bn_2', 'val_type': 1, 'dims': [200704, 512, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_1/x1/bn_3', 'val_type': 1, 'dims': [200704, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv5_1/x1/bn_3', 'val_type': 1, 'dims': [200704, 512, 7, 7], 'vals': []}
                input {'name': 'conv5_1/x1_w_0', 'val_type': 1, 'dims': [128, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_1/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv5_1/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_1/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_1/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_1/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_1/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_1/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_1/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_1/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_1/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_1/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_1/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_1/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_1/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_1/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_1/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_1/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_1/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'pool4_1', 'val_type': 1, 'dims': [200704, 512, 7, 7], 'vals': []}
                input {'name': 'conv5_1/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_5_1_1', 'val_type': 1, 'dims': [200704, 544, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_5_1_1', 'val_type': 1, 'dims': [200704, 544, 7, 7], 'vals': []}
                input {'name': 'conv5_2/x1/bn_scale_0', 'val_type': 1, 'dims': [544], 'vals': []}
                input {'name': 'conv5_2/x1/bn_bias_0', 'val_type': 1, 'dims': [544], 'vals': []}
                input {'name': 'conv5_2/x1/bn_mean_0', 'val_type': 1, 'dims': [544], 'vals': []}
                input {'name': 'conv5_2/x1/bn_var_0', 'val_type': 1, 'dims': [544], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_2/x1/bn_1', 'val_type': 1, 'dims': [200704, 544, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_2/x1/bn_1', 'val_type': 1, 'dims': [200704, 544, 7, 7], 'vals': []}
                input {'name': 'conv5_2/x1/bn_w_0', 'val_type': 1, 'dims': [544], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_2/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 544, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_2/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 544, 7, 7], 'vals': []}
                input {'name': 'conv5_2/x1/bn_b_0', 'val_type': 1, 'dims': [544], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_2/x1/bn_2', 'val_type': 1, 'dims': [200704, 544, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_2/x1/bn_2', 'val_type': 1, 'dims': [200704, 544, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_2/x1/bn_3', 'val_type': 1, 'dims': [200704, 544, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv5_2/x1/bn_3', 'val_type': 1, 'dims': [200704, 544, 7, 7], 'vals': []}
                input {'name': 'conv5_2/x1_w_0', 'val_type': 1, 'dims': [128, 544, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_2/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv5_2/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_2/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_2/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_2/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_2/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_2/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_2/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_2/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_2/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_2/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_2/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_2/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_2/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_2/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_2/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_2/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_2/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_5_1_1', 'val_type': 1, 'dims': [200704, 544, 7, 7], 'vals': []}
                input {'name': 'conv5_2/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_5_2_1', 'val_type': 1, 'dims': [200704, 576, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_5_2_1', 'val_type': 1, 'dims': [200704, 576, 7, 7], 'vals': []}
                input {'name': 'conv5_3/x1/bn_scale_0', 'val_type': 1, 'dims': [576], 'vals': []}
                input {'name': 'conv5_3/x1/bn_bias_0', 'val_type': 1, 'dims': [576], 'vals': []}
                input {'name': 'conv5_3/x1/bn_mean_0', 'val_type': 1, 'dims': [576], 'vals': []}
                input {'name': 'conv5_3/x1/bn_var_0', 'val_type': 1, 'dims': [576], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_3/x1/bn_1', 'val_type': 1, 'dims': [200704, 576, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_3/x1/bn_1', 'val_type': 1, 'dims': [200704, 576, 7, 7], 'vals': []}
                input {'name': 'conv5_3/x1/bn_w_0', 'val_type': 1, 'dims': [576], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_3/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 576, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_3/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 576, 7, 7], 'vals': []}
                input {'name': 'conv5_3/x1/bn_b_0', 'val_type': 1, 'dims': [576], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_3/x1/bn_2', 'val_type': 1, 'dims': [200704, 576, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_3/x1/bn_2', 'val_type': 1, 'dims': [200704, 576, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_3/x1/bn_3', 'val_type': 1, 'dims': [200704, 576, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv5_3/x1/bn_3', 'val_type': 1, 'dims': [200704, 576, 7, 7], 'vals': []}
                input {'name': 'conv5_3/x1_w_0', 'val_type': 1, 'dims': [128, 576, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_3/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv5_3/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_3/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_3/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_3/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_3/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_3/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_3/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_3/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_3/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_3/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_3/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_3/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_3/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_3/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_3/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_3/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_3/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_5_2_1', 'val_type': 1, 'dims': [200704, 576, 7, 7], 'vals': []}
                input {'name': 'conv5_3/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_5_3_1', 'val_type': 1, 'dims': [200704, 608, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_5_3_1', 'val_type': 1, 'dims': [200704, 608, 7, 7], 'vals': []}
                input {'name': 'conv5_4/x1/bn_scale_0', 'val_type': 1, 'dims': [608], 'vals': []}
                input {'name': 'conv5_4/x1/bn_bias_0', 'val_type': 1, 'dims': [608], 'vals': []}
                input {'name': 'conv5_4/x1/bn_mean_0', 'val_type': 1, 'dims': [608], 'vals': []}
                input {'name': 'conv5_4/x1/bn_var_0', 'val_type': 1, 'dims': [608], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_4/x1/bn_1', 'val_type': 1, 'dims': [200704, 608, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_4/x1/bn_1', 'val_type': 1, 'dims': [200704, 608, 7, 7], 'vals': []}
                input {'name': 'conv5_4/x1/bn_w_0', 'val_type': 1, 'dims': [608], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_4/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 608, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_4/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 608, 7, 7], 'vals': []}
                input {'name': 'conv5_4/x1/bn_b_0', 'val_type': 1, 'dims': [608], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_4/x1/bn_2', 'val_type': 1, 'dims': [200704, 608, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_4/x1/bn_2', 'val_type': 1, 'dims': [200704, 608, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_4/x1/bn_3', 'val_type': 1, 'dims': [200704, 608, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv5_4/x1/bn_3', 'val_type': 1, 'dims': [200704, 608, 7, 7], 'vals': []}
                input {'name': 'conv5_4/x1_w_0', 'val_type': 1, 'dims': [128, 608, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_4/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv5_4/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_4/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_4/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_4/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_4/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_4/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_4/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_4/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_4/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_4/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_4/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_4/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_4/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_4/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_4/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_4/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_4/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_5_3_1', 'val_type': 1, 'dims': [200704, 608, 7, 7], 'vals': []}
                input {'name': 'conv5_4/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_5_4_1', 'val_type': 1, 'dims': [200704, 640, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_5_4_1', 'val_type': 1, 'dims': [200704, 640, 7, 7], 'vals': []}
                input {'name': 'conv5_5/x1/bn_scale_0', 'val_type': 1, 'dims': [640], 'vals': []}
                input {'name': 'conv5_5/x1/bn_bias_0', 'val_type': 1, 'dims': [640], 'vals': []}
                input {'name': 'conv5_5/x1/bn_mean_0', 'val_type': 1, 'dims': [640], 'vals': []}
                input {'name': 'conv5_5/x1/bn_var_0', 'val_type': 1, 'dims': [640], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_5/x1/bn_1', 'val_type': 1, 'dims': [200704, 640, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_5/x1/bn_1', 'val_type': 1, 'dims': [200704, 640, 7, 7], 'vals': []}
                input {'name': 'conv5_5/x1/bn_w_0', 'val_type': 1, 'dims': [640], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_5/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 640, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_5/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 640, 7, 7], 'vals': []}
                input {'name': 'conv5_5/x1/bn_b_0', 'val_type': 1, 'dims': [640], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_5/x1/bn_2', 'val_type': 1, 'dims': [200704, 640, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_5/x1/bn_2', 'val_type': 1, 'dims': [200704, 640, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_5/x1/bn_3', 'val_type': 1, 'dims': [200704, 640, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv5_5/x1/bn_3', 'val_type': 1, 'dims': [200704, 640, 7, 7], 'vals': []}
                input {'name': 'conv5_5/x1_w_0', 'val_type': 1, 'dims': [128, 640, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_5/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv5_5/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_5/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_5/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_5/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_5/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_5/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_5/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_5/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_5/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_5/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_5/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_5/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_5/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_5/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_5/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_5/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_5/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_5_4_1', 'val_type': 1, 'dims': [200704, 640, 7, 7], 'vals': []}
                input {'name': 'conv5_5/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_5_5_1', 'val_type': 1, 'dims': [200704, 672, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_5_5_1', 'val_type': 1, 'dims': [200704, 672, 7, 7], 'vals': []}
                input {'name': 'conv5_6/x1/bn_scale_0', 'val_type': 1, 'dims': [672], 'vals': []}
                input {'name': 'conv5_6/x1/bn_bias_0', 'val_type': 1, 'dims': [672], 'vals': []}
                input {'name': 'conv5_6/x1/bn_mean_0', 'val_type': 1, 'dims': [672], 'vals': []}
                input {'name': 'conv5_6/x1/bn_var_0', 'val_type': 1, 'dims': [672], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_6/x1/bn_1', 'val_type': 1, 'dims': [200704, 672, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_6/x1/bn_1', 'val_type': 1, 'dims': [200704, 672, 7, 7], 'vals': []}
                input {'name': 'conv5_6/x1/bn_w_0', 'val_type': 1, 'dims': [672], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_6/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 672, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_6/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 672, 7, 7], 'vals': []}
                input {'name': 'conv5_6/x1/bn_b_0', 'val_type': 1, 'dims': [672], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_6/x1/bn_2', 'val_type': 1, 'dims': [200704, 672, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_6/x1/bn_2', 'val_type': 1, 'dims': [200704, 672, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_6/x1/bn_3', 'val_type': 1, 'dims': [200704, 672, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv5_6/x1/bn_3', 'val_type': 1, 'dims': [200704, 672, 7, 7], 'vals': []}
                input {'name': 'conv5_6/x1_w_0', 'val_type': 1, 'dims': [128, 672, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_6/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv5_6/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_6/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_6/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_6/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_6/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_6/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_6/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_6/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_6/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_6/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_6/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_6/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_6/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_6/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_6/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_6/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_6/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_5_5_1', 'val_type': 1, 'dims': [200704, 672, 7, 7], 'vals': []}
                input {'name': 'conv5_6/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_5_6_1', 'val_type': 1, 'dims': [200704, 704, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_5_6_1', 'val_type': 1, 'dims': [200704, 704, 7, 7], 'vals': []}
                input {'name': 'conv5_7/x1/bn_scale_0', 'val_type': 1, 'dims': [704], 'vals': []}
                input {'name': 'conv5_7/x1/bn_bias_0', 'val_type': 1, 'dims': [704], 'vals': []}
                input {'name': 'conv5_7/x1/bn_mean_0', 'val_type': 1, 'dims': [704], 'vals': []}
                input {'name': 'conv5_7/x1/bn_var_0', 'val_type': 1, 'dims': [704], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_7/x1/bn_1', 'val_type': 1, 'dims': [200704, 704, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_7/x1/bn_1', 'val_type': 1, 'dims': [200704, 704, 7, 7], 'vals': []}
                input {'name': 'conv5_7/x1/bn_w_0', 'val_type': 1, 'dims': [704], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_7/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 704, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_7/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 704, 7, 7], 'vals': []}
                input {'name': 'conv5_7/x1/bn_b_0', 'val_type': 1, 'dims': [704], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_7/x1/bn_2', 'val_type': 1, 'dims': [200704, 704, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_7/x1/bn_2', 'val_type': 1, 'dims': [200704, 704, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_7/x1/bn_3', 'val_type': 1, 'dims': [200704, 704, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv5_7/x1/bn_3', 'val_type': 1, 'dims': [200704, 704, 7, 7], 'vals': []}
                input {'name': 'conv5_7/x1_w_0', 'val_type': 1, 'dims': [128, 704, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_7/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv5_7/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_7/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_7/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_7/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_7/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_7/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_7/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_7/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_7/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_7/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_7/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_7/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_7/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_7/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_7/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_7/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_7/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_5_6_1', 'val_type': 1, 'dims': [200704, 704, 7, 7], 'vals': []}
                input {'name': 'conv5_7/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_5_7_1', 'val_type': 1, 'dims': [200704, 736, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_5_7_1', 'val_type': 1, 'dims': [200704, 736, 7, 7], 'vals': []}
                input {'name': 'conv5_8/x1/bn_scale_0', 'val_type': 1, 'dims': [736], 'vals': []}
                input {'name': 'conv5_8/x1/bn_bias_0', 'val_type': 1, 'dims': [736], 'vals': []}
                input {'name': 'conv5_8/x1/bn_mean_0', 'val_type': 1, 'dims': [736], 'vals': []}
                input {'name': 'conv5_8/x1/bn_var_0', 'val_type': 1, 'dims': [736], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_8/x1/bn_1', 'val_type': 1, 'dims': [200704, 736, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_8/x1/bn_1', 'val_type': 1, 'dims': [200704, 736, 7, 7], 'vals': []}
                input {'name': 'conv5_8/x1/bn_w_0', 'val_type': 1, 'dims': [736], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_8/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 736, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_8/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 736, 7, 7], 'vals': []}
                input {'name': 'conv5_8/x1/bn_b_0', 'val_type': 1, 'dims': [736], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_8/x1/bn_2', 'val_type': 1, 'dims': [200704, 736, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_8/x1/bn_2', 'val_type': 1, 'dims': [200704, 736, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_8/x1/bn_3', 'val_type': 1, 'dims': [200704, 736, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv5_8/x1/bn_3', 'val_type': 1, 'dims': [200704, 736, 7, 7], 'vals': []}
                input {'name': 'conv5_8/x1_w_0', 'val_type': 1, 'dims': [128, 736, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_8/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv5_8/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_8/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_8/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_8/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_8/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_8/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_8/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_8/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_8/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_8/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_8/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_8/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_8/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_8/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_8/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_8/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_8/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_5_7_1', 'val_type': 1, 'dims': [200704, 736, 7, 7], 'vals': []}
                input {'name': 'conv5_8/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_5_8_1', 'val_type': 1, 'dims': [200704, 768, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_5_8_1', 'val_type': 1, 'dims': [200704, 768, 7, 7], 'vals': []}
                input {'name': 'conv5_9/x1/bn_scale_0', 'val_type': 1, 'dims': [768], 'vals': []}
                input {'name': 'conv5_9/x1/bn_bias_0', 'val_type': 1, 'dims': [768], 'vals': []}
                input {'name': 'conv5_9/x1/bn_mean_0', 'val_type': 1, 'dims': [768], 'vals': []}
                input {'name': 'conv5_9/x1/bn_var_0', 'val_type': 1, 'dims': [768], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_9/x1/bn_1', 'val_type': 1, 'dims': [200704, 768, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_9/x1/bn_1', 'val_type': 1, 'dims': [200704, 768, 7, 7], 'vals': []}
                input {'name': 'conv5_9/x1/bn_w_0', 'val_type': 1, 'dims': [768], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_9/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 768, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_9/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 768, 7, 7], 'vals': []}
                input {'name': 'conv5_9/x1/bn_b_0', 'val_type': 1, 'dims': [768], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_9/x1/bn_2', 'val_type': 1, 'dims': [200704, 768, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_9/x1/bn_2', 'val_type': 1, 'dims': [200704, 768, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_9/x1/bn_3', 'val_type': 1, 'dims': [200704, 768, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv5_9/x1/bn_3', 'val_type': 1, 'dims': [200704, 768, 7, 7], 'vals': []}
                input {'name': 'conv5_9/x1_w_0', 'val_type': 1, 'dims': [128, 768, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_9/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv5_9/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_9/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_9/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_9/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_9/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_9/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_9/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_9/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_9/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_9/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_9/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_9/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_9/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_9/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_9/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_9/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_9/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_5_8_1', 'val_type': 1, 'dims': [200704, 768, 7, 7], 'vals': []}
                input {'name': 'conv5_9/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_5_9_1', 'val_type': 1, 'dims': [200704, 800, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_5_9_1', 'val_type': 1, 'dims': [200704, 800, 7, 7], 'vals': []}
                input {'name': 'conv5_10/x1/bn_scale_0', 'val_type': 1, 'dims': [800], 'vals': []}
                input {'name': 'conv5_10/x1/bn_bias_0', 'val_type': 1, 'dims': [800], 'vals': []}
                input {'name': 'conv5_10/x1/bn_mean_0', 'val_type': 1, 'dims': [800], 'vals': []}
                input {'name': 'conv5_10/x1/bn_var_0', 'val_type': 1, 'dims': [800], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_10/x1/bn_1', 'val_type': 1, 'dims': [200704, 800, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_10/x1/bn_1', 'val_type': 1, 'dims': [200704, 800, 7, 7], 'vals': []}
                input {'name': 'conv5_10/x1/bn_w_0', 'val_type': 1, 'dims': [800], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_10/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 800, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_10/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 800, 7, 7], 'vals': []}
                input {'name': 'conv5_10/x1/bn_b_0', 'val_type': 1, 'dims': [800], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_10/x1/bn_2', 'val_type': 1, 'dims': [200704, 800, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_10/x1/bn_2', 'val_type': 1, 'dims': [200704, 800, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_10/x1/bn_3', 'val_type': 1, 'dims': [200704, 800, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv5_10/x1/bn_3', 'val_type': 1, 'dims': [200704, 800, 7, 7], 'vals': []}
                input {'name': 'conv5_10/x1_w_0', 'val_type': 1, 'dims': [128, 800, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_10/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv5_10/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_10/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_10/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_10/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_10/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_10/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_10/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_10/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_10/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_10/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_10/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_10/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_10/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_10/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_10/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_10/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_10/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_5_9_1', 'val_type': 1, 'dims': [200704, 800, 7, 7], 'vals': []}
                input {'name': 'conv5_10/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_5_10_1', 'val_type': 1, 'dims': [200704, 832, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_5_10_1', 'val_type': 1, 'dims': [200704, 832, 7, 7], 'vals': []}
                input {'name': 'conv5_11/x1/bn_scale_0', 'val_type': 1, 'dims': [832], 'vals': []}
                input {'name': 'conv5_11/x1/bn_bias_0', 'val_type': 1, 'dims': [832], 'vals': []}
                input {'name': 'conv5_11/x1/bn_mean_0', 'val_type': 1, 'dims': [832], 'vals': []}
                input {'name': 'conv5_11/x1/bn_var_0', 'val_type': 1, 'dims': [832], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_11/x1/bn_1', 'val_type': 1, 'dims': [200704, 832, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_11/x1/bn_1', 'val_type': 1, 'dims': [200704, 832, 7, 7], 'vals': []}
                input {'name': 'conv5_11/x1/bn_w_0', 'val_type': 1, 'dims': [832], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_11/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 832, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_11/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 832, 7, 7], 'vals': []}
                input {'name': 'conv5_11/x1/bn_b_0', 'val_type': 1, 'dims': [832], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_11/x1/bn_2', 'val_type': 1, 'dims': [200704, 832, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_11/x1/bn_2', 'val_type': 1, 'dims': [200704, 832, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_11/x1/bn_3', 'val_type': 1, 'dims': [200704, 832, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv5_11/x1/bn_3', 'val_type': 1, 'dims': [200704, 832, 7, 7], 'vals': []}
                input {'name': 'conv5_11/x1_w_0', 'val_type': 1, 'dims': [128, 832, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_11/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv5_11/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_11/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_11/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_11/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_11/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_11/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_11/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_11/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_11/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_11/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_11/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_11/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_11/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_11/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_11/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_11/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_11/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_5_10_1', 'val_type': 1, 'dims': [200704, 832, 7, 7], 'vals': []}
                input {'name': 'conv5_11/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_5_11_1', 'val_type': 1, 'dims': [200704, 864, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_5_11_1', 'val_type': 1, 'dims': [200704, 864, 7, 7], 'vals': []}
                input {'name': 'conv5_12/x1/bn_scale_0', 'val_type': 1, 'dims': [864], 'vals': []}
                input {'name': 'conv5_12/x1/bn_bias_0', 'val_type': 1, 'dims': [864], 'vals': []}
                input {'name': 'conv5_12/x1/bn_mean_0', 'val_type': 1, 'dims': [864], 'vals': []}
                input {'name': 'conv5_12/x1/bn_var_0', 'val_type': 1, 'dims': [864], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_12/x1/bn_1', 'val_type': 1, 'dims': [200704, 864, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_12/x1/bn_1', 'val_type': 1, 'dims': [200704, 864, 7, 7], 'vals': []}
                input {'name': 'conv5_12/x1/bn_w_0', 'val_type': 1, 'dims': [864], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_12/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 864, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_12/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 864, 7, 7], 'vals': []}
                input {'name': 'conv5_12/x1/bn_b_0', 'val_type': 1, 'dims': [864], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_12/x1/bn_2', 'val_type': 1, 'dims': [200704, 864, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_12/x1/bn_2', 'val_type': 1, 'dims': [200704, 864, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_12/x1/bn_3', 'val_type': 1, 'dims': [200704, 864, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv5_12/x1/bn_3', 'val_type': 1, 'dims': [200704, 864, 7, 7], 'vals': []}
                input {'name': 'conv5_12/x1_w_0', 'val_type': 1, 'dims': [128, 864, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_12/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv5_12/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_12/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_12/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_12/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_12/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_12/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_12/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_12/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_12/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_12/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_12/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_12/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_12/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_12/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_12/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_12/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_12/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_5_11_1', 'val_type': 1, 'dims': [200704, 864, 7, 7], 'vals': []}
                input {'name': 'conv5_12/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_5_12_1', 'val_type': 1, 'dims': [200704, 896, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_5_12_1', 'val_type': 1, 'dims': [200704, 896, 7, 7], 'vals': []}
                input {'name': 'conv5_13/x1/bn_scale_0', 'val_type': 1, 'dims': [896], 'vals': []}
                input {'name': 'conv5_13/x1/bn_bias_0', 'val_type': 1, 'dims': [896], 'vals': []}
                input {'name': 'conv5_13/x1/bn_mean_0', 'val_type': 1, 'dims': [896], 'vals': []}
                input {'name': 'conv5_13/x1/bn_var_0', 'val_type': 1, 'dims': [896], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_13/x1/bn_1', 'val_type': 1, 'dims': [200704, 896, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_13/x1/bn_1', 'val_type': 1, 'dims': [200704, 896, 7, 7], 'vals': []}
                input {'name': 'conv5_13/x1/bn_w_0', 'val_type': 1, 'dims': [896], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_13/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 896, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_13/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 896, 7, 7], 'vals': []}
                input {'name': 'conv5_13/x1/bn_b_0', 'val_type': 1, 'dims': [896], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_13/x1/bn_2', 'val_type': 1, 'dims': [200704, 896, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_13/x1/bn_2', 'val_type': 1, 'dims': [200704, 896, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_13/x1/bn_3', 'val_type': 1, 'dims': [200704, 896, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv5_13/x1/bn_3', 'val_type': 1, 'dims': [200704, 896, 7, 7], 'vals': []}
                input {'name': 'conv5_13/x1_w_0', 'val_type': 1, 'dims': [128, 896, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_13/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv5_13/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_13/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_13/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_13/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_13/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_13/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_13/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_13/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_13/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_13/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_13/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_13/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_13/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_13/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_13/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_13/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_13/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_5_12_1', 'val_type': 1, 'dims': [200704, 896, 7, 7], 'vals': []}
                input {'name': 'conv5_13/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_5_13_1', 'val_type': 1, 'dims': [200704, 928, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_5_13_1', 'val_type': 1, 'dims': [200704, 928, 7, 7], 'vals': []}
                input {'name': 'conv5_14/x1/bn_scale_0', 'val_type': 1, 'dims': [928], 'vals': []}
                input {'name': 'conv5_14/x1/bn_bias_0', 'val_type': 1, 'dims': [928], 'vals': []}
                input {'name': 'conv5_14/x1/bn_mean_0', 'val_type': 1, 'dims': [928], 'vals': []}
                input {'name': 'conv5_14/x1/bn_var_0', 'val_type': 1, 'dims': [928], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_14/x1/bn_1', 'val_type': 1, 'dims': [200704, 928, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_14/x1/bn_1', 'val_type': 1, 'dims': [200704, 928, 7, 7], 'vals': []}
                input {'name': 'conv5_14/x1/bn_w_0', 'val_type': 1, 'dims': [928], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_14/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 928, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_14/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 928, 7, 7], 'vals': []}
                input {'name': 'conv5_14/x1/bn_b_0', 'val_type': 1, 'dims': [928], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_14/x1/bn_2', 'val_type': 1, 'dims': [200704, 928, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_14/x1/bn_2', 'val_type': 1, 'dims': [200704, 928, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_14/x1/bn_3', 'val_type': 1, 'dims': [200704, 928, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv5_14/x1/bn_3', 'val_type': 1, 'dims': [200704, 928, 7, 7], 'vals': []}
                input {'name': 'conv5_14/x1_w_0', 'val_type': 1, 'dims': [128, 928, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_14/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv5_14/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_14/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_14/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_14/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_14/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_14/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_14/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_14/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_14/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_14/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_14/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_14/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_14/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_14/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_14/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_14/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_14/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_5_13_1', 'val_type': 1, 'dims': [200704, 928, 7, 7], 'vals': []}
                input {'name': 'conv5_14/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_5_14_1', 'val_type': 1, 'dims': [200704, 960, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_5_14_1', 'val_type': 1, 'dims': [200704, 960, 7, 7], 'vals': []}
                input {'name': 'conv5_15/x1/bn_scale_0', 'val_type': 1, 'dims': [960], 'vals': []}
                input {'name': 'conv5_15/x1/bn_bias_0', 'val_type': 1, 'dims': [960], 'vals': []}
                input {'name': 'conv5_15/x1/bn_mean_0', 'val_type': 1, 'dims': [960], 'vals': []}
                input {'name': 'conv5_15/x1/bn_var_0', 'val_type': 1, 'dims': [960], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_15/x1/bn_1', 'val_type': 1, 'dims': [200704, 960, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_15/x1/bn_1', 'val_type': 1, 'dims': [200704, 960, 7, 7], 'vals': []}
                input {'name': 'conv5_15/x1/bn_w_0', 'val_type': 1, 'dims': [960], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_15/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 960, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_15/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 960, 7, 7], 'vals': []}
                input {'name': 'conv5_15/x1/bn_b_0', 'val_type': 1, 'dims': [960], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_15/x1/bn_2', 'val_type': 1, 'dims': [200704, 960, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_15/x1/bn_2', 'val_type': 1, 'dims': [200704, 960, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_15/x1/bn_3', 'val_type': 1, 'dims': [200704, 960, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv5_15/x1/bn_3', 'val_type': 1, 'dims': [200704, 960, 7, 7], 'vals': []}
                input {'name': 'conv5_15/x1_w_0', 'val_type': 1, 'dims': [128, 960, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_15/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv5_15/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_15/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_15/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_15/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_15/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_15/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_15/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_15/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_15/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_15/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_15/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_15/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_15/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_15/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_15/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_15/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_15/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_5_14_1', 'val_type': 1, 'dims': [200704, 960, 7, 7], 'vals': []}
                input {'name': 'conv5_15/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_5_15_1', 'val_type': 1, 'dims': [200704, 992, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_5_15_1', 'val_type': 1, 'dims': [200704, 992, 7, 7], 'vals': []}
                input {'name': 'conv5_16/x1/bn_scale_0', 'val_type': 1, 'dims': [992], 'vals': []}
                input {'name': 'conv5_16/x1/bn_bias_0', 'val_type': 1, 'dims': [992], 'vals': []}
                input {'name': 'conv5_16/x1/bn_mean_0', 'val_type': 1, 'dims': [992], 'vals': []}
                input {'name': 'conv5_16/x1/bn_var_0', 'val_type': 1, 'dims': [992], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_16/x1/bn_1', 'val_type': 1, 'dims': [200704, 992, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_16/x1/bn_1', 'val_type': 1, 'dims': [200704, 992, 7, 7], 'vals': []}
                input {'name': 'conv5_16/x1/bn_w_0', 'val_type': 1, 'dims': [992], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_16/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 992, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_16/x1/bn_internal_1', 'val_type': 1, 'dims': [200704, 992, 7, 7], 'vals': []}
                input {'name': 'conv5_16/x1/bn_b_0', 'val_type': 1, 'dims': [992], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_16/x1/bn_2', 'val_type': 1, 'dims': [200704, 992, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_16/x1/bn_2', 'val_type': 1, 'dims': [200704, 992, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_16/x1/bn_3', 'val_type': 1, 'dims': [200704, 992, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'conv5_16/x1/bn_3', 'val_type': 1, 'dims': [200704, 992, 7, 7], 'vals': []}
                input {'name': 'conv5_16/x1_w_0', 'val_type': 1, 'dims': [128, 992, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_16/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'conv5_16/x1_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_16/x2/bn_scale_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_16/x2/bn_bias_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_16/x2/bn_mean_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'conv5_16/x2/bn_var_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_16/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_16/x2/bn_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_16/x2/bn_w_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_16/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_16/x2/bn_internal_1', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_16/x2/bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_16/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_16/x2/bn_2', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_16/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'conv5_16/x2/bn_3', 'val_type': 1, 'dims': [200704, 128, 7, 7], 'vals': []}
                input {'name': 'conv5_16/x2_w_0', 'val_type': 1, 'dims': [32, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_16/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Concat'
            'attributes': {'axis': 1}
            'inputs': [
                input {'name': 'concat_5_15_1', 'val_type': 1, 'dims': [200704, 992, 7, 7], 'vals': []}
                input {'name': 'conv5_16/x2_1', 'val_type': 1, 'dims': [200704, 32, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'concat_5_16_1', 'val_type': 1, 'dims': [200704, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'is_test': 1, 'epsilon': 9.999999747378752e-06, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'concat_5_16_1', 'val_type': 1, 'dims': [200704, 1024, 7, 7], 'vals': []}
                input {'name': 'conv5_blk/bn_scale_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'conv5_blk/bn_bias_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'conv5_blk/bn_mean_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'conv5_blk/bn_var_0', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_blk/bn_1', 'val_type': 1, 'dims': [200704, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Mul'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_blk/bn_1', 'val_type': 1, 'dims': [200704, 1024, 7, 7], 'vals': []}
                input {'name': 'conv5_blk/bn_w_0', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_blk/bn_internal_1', 'val_type': 1, 'dims': [200704, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {'axis': 1, 'broadcast': 1}
            'inputs': [
                input {'name': 'conv5_blk/bn_internal_1', 'val_type': 1, 'dims': [200704, 1024, 7, 7], 'vals': []}
                input {'name': 'conv5_blk/bn_b_0', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_blk/bn_2', 'val_type': 1, 'dims': [200704, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_blk/bn_2', 'val_type': 1, 'dims': [200704, 1024, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'conv5_blk/bn_3', 'val_type': 1, 'dims': [200704, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'GlobalAveragePool'
            'attributes': {}
            'inputs': [
                input {'name': 'conv5_blk/bn_3', 'val_type': 1, 'dims': [200704, 1024, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'pool5_1', 'val_type': 1, 'dims': [200704, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'pool5_1', 'val_type': 1, 'dims': [200704, 1024, 7, 7], 'vals': []}
                input {'name': 'fc6_w_0', 'val_type': 1, 'dims': [1000, 1024, 1, 1], 'vals': []}
                input {'name': 'fc6_b_0', 'val_type': 1, 'dims': [1000], 'vals': []}
            ]
            'outputs': [
                output {'name': 'fc6_1', 'val_type': 1, 'dims': [200704, 1000, 7, 7], 'vals': []}
            ]
        }
    ]
}
