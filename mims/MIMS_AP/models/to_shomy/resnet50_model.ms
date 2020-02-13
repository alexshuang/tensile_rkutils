'ir_version': 1
'producer_name': 'onnx-mims'
'graph': {
    'name': 'resnet50'
    'nodes': [
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [2, 2], 'pads': [3, 3, 3, 3], 'kernel_shape': [7, 7]}
            'inputs': [
                input {'name': 'gpu_0/data_0', 'val_type': 1, 'dims': [32, 3, 224, 224], 'vals': []}
                input {'name': 'gpu_0/conv1_w_0', 'val_type': 1, 'dims': [64, 3, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/conv1_1', 'val_type': 1, 'dims': [32, 64, 112, 112], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/conv1_1', 'val_type': 1, 'dims': [32, 64, 112, 112], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_s_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_b_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_rm_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res_conv1_bn_riv_0', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res_conv1_bn_1', 'val_type': 1, 'dims': [32, 64, 112, 112], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res_conv1_bn_1', 'val_type': 1, 'dims': [32, 64, 112, 112], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res_conv1_bn_2', 'val_type': 1, 'dims': [32, 64, 112, 112], 'vals': []}
            ]
        }
        node {
            'op_type': 'MaxPool'
            'attributes': {'strides': [2, 2], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res_conv1_bn_2', 'val_type': 1, 'dims': [32, 64, 112, 112], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/pool1_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/pool1_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch2a_w_0', 'val_type': 1, 'dims': [64, 64, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_0_branch2a_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_0_branch2a_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch2a_bn_s_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch2a_bn_b_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch2a_bn_rm_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch2a_bn_riv_0', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_0_branch2a_bn_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res2_0_branch2a_bn_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_0_branch2a_bn_2', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res2_0_branch2a_bn_2', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch2b_w_0', 'val_type': 1, 'dims': [64, 64, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_0_branch2b_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_0_branch2b_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch2b_bn_s_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch2b_bn_b_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch2b_bn_rm_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch2b_bn_riv_0', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_0_branch2b_bn_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res2_0_branch2b_bn_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_0_branch2b_bn_2', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_0_branch2b_bn_2', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch2c_w_0', 'val_type': 1, 'dims': [256, 64, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_0_branch2c_1', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_0_branch2c_1', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch2c_bn_s_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch2c_bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch2c_bn_rm_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch2c_bn_riv_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_0_branch2c_bn_1', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/pool1_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch1_w_0', 'val_type': 1, 'dims': [256, 64, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_0_branch1_1', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_0_branch1_1', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch1_bn_s_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch1_bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch1_bn_rm_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch1_bn_riv_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_0_branch1_bn_1', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res2_0_branch2c_bn_1', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch1_bn_1', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_0_branch2c_bn_2', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res2_0_branch2c_bn_2', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_0_branch2c_bn_3', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_0_branch2c_bn_3', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_1_branch2a_w_0', 'val_type': 1, 'dims': [64, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_1_branch2a_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_1_branch2a_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_1_branch2a_bn_s_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_1_branch2a_bn_b_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_1_branch2a_bn_rm_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_1_branch2a_bn_riv_0', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_1_branch2a_bn_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res2_1_branch2a_bn_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_1_branch2a_bn_2', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res2_1_branch2a_bn_2', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_1_branch2b_w_0', 'val_type': 1, 'dims': [64, 64, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_1_branch2b_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_1_branch2b_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_1_branch2b_bn_s_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_1_branch2b_bn_b_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_1_branch2b_bn_rm_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_1_branch2b_bn_riv_0', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_1_branch2b_bn_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res2_1_branch2b_bn_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_1_branch2b_bn_2', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_1_branch2b_bn_2', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_1_branch2c_w_0', 'val_type': 1, 'dims': [256, 64, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_1_branch2c_1', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_1_branch2c_1', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_1_branch2c_bn_s_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res2_1_branch2c_bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res2_1_branch2c_bn_rm_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res2_1_branch2c_bn_riv_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_1_branch2c_bn_1', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res2_1_branch2c_bn_1', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_0_branch2c_bn_3', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_1_branch2c_bn_2', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res2_1_branch2c_bn_2', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_1_branch2c_bn_3', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_1_branch2c_bn_3', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_2_branch2a_w_0', 'val_type': 1, 'dims': [64, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_2_branch2a_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_2_branch2a_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_2_branch2a_bn_s_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_2_branch2a_bn_b_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_2_branch2a_bn_rm_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_2_branch2a_bn_riv_0', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_2_branch2a_bn_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res2_2_branch2a_bn_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_2_branch2a_bn_2', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res2_2_branch2a_bn_2', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_2_branch2b_w_0', 'val_type': 1, 'dims': [64, 64, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_2_branch2b_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_2_branch2b_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_2_branch2b_bn_s_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_2_branch2b_bn_b_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_2_branch2b_bn_rm_0', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'gpu_0/res2_2_branch2b_bn_riv_0', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_2_branch2b_bn_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res2_2_branch2b_bn_1', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_2_branch2b_bn_2', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_2_branch2b_bn_2', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_2_branch2c_w_0', 'val_type': 1, 'dims': [256, 64, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_2_branch2c_1', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_2_branch2c_1', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_2_branch2c_bn_s_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res2_2_branch2c_bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res2_2_branch2c_bn_rm_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res2_2_branch2c_bn_riv_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_2_branch2c_bn_1', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res2_2_branch2c_bn_1', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res2_1_branch2c_bn_3', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_2_branch2c_bn_2', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res2_2_branch2c_bn_2', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res2_2_branch2c_bn_3', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_2_branch2c_bn_3', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch2a_w_0', 'val_type': 1, 'dims': [128, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_0_branch2a_1', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_0_branch2a_1', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch2a_bn_s_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch2a_bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch2a_bn_rm_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch2a_bn_riv_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_0_branch2a_bn_1', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res3_0_branch2a_bn_1', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_0_branch2a_bn_2', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [2, 2], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res3_0_branch2a_bn_2', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch2b_w_0', 'val_type': 1, 'dims': [128, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_0_branch2b_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_0_branch2b_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch2b_bn_s_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch2b_bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch2b_bn_rm_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch2b_bn_riv_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_0_branch2b_bn_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res3_0_branch2b_bn_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_0_branch2b_bn_2', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_0_branch2b_bn_2', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch2c_w_0', 'val_type': 1, 'dims': [512, 128, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_0_branch2c_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_0_branch2c_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch2c_bn_s_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch2c_bn_b_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch2c_bn_rm_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch2c_bn_riv_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_0_branch2c_bn_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [2, 2], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res2_2_branch2c_bn_3', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch1_w_0', 'val_type': 1, 'dims': [512, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_0_branch1_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_0_branch1_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch1_bn_s_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch1_bn_b_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch1_bn_rm_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch1_bn_riv_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_0_branch1_bn_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res3_0_branch2c_bn_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch1_bn_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_0_branch2c_bn_2', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res3_0_branch2c_bn_2', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_0_branch2c_bn_3', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_0_branch2c_bn_3', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_1_branch2a_w_0', 'val_type': 1, 'dims': [128, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_1_branch2a_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_1_branch2a_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_1_branch2a_bn_s_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_1_branch2a_bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_1_branch2a_bn_rm_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_1_branch2a_bn_riv_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_1_branch2a_bn_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res3_1_branch2a_bn_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_1_branch2a_bn_2', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res3_1_branch2a_bn_2', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_1_branch2b_w_0', 'val_type': 1, 'dims': [128, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_1_branch2b_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_1_branch2b_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_1_branch2b_bn_s_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_1_branch2b_bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_1_branch2b_bn_rm_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_1_branch2b_bn_riv_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_1_branch2b_bn_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res3_1_branch2b_bn_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_1_branch2b_bn_2', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_1_branch2b_bn_2', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_1_branch2c_w_0', 'val_type': 1, 'dims': [512, 128, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_1_branch2c_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_1_branch2c_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_1_branch2c_bn_s_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res3_1_branch2c_bn_b_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res3_1_branch2c_bn_rm_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res3_1_branch2c_bn_riv_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_1_branch2c_bn_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res3_1_branch2c_bn_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_0_branch2c_bn_3', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_1_branch2c_bn_2', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res3_1_branch2c_bn_2', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_1_branch2c_bn_3', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_1_branch2c_bn_3', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_2_branch2a_w_0', 'val_type': 1, 'dims': [128, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_2_branch2a_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_2_branch2a_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_2_branch2a_bn_s_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_2_branch2a_bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_2_branch2a_bn_rm_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_2_branch2a_bn_riv_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_2_branch2a_bn_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res3_2_branch2a_bn_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_2_branch2a_bn_2', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res3_2_branch2a_bn_2', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_2_branch2b_w_0', 'val_type': 1, 'dims': [128, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_2_branch2b_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_2_branch2b_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_2_branch2b_bn_s_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_2_branch2b_bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_2_branch2b_bn_rm_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_2_branch2b_bn_riv_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_2_branch2b_bn_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res3_2_branch2b_bn_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_2_branch2b_bn_2', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_2_branch2b_bn_2', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_2_branch2c_w_0', 'val_type': 1, 'dims': [512, 128, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_2_branch2c_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_2_branch2c_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_2_branch2c_bn_s_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res3_2_branch2c_bn_b_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res3_2_branch2c_bn_rm_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res3_2_branch2c_bn_riv_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_2_branch2c_bn_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res3_2_branch2c_bn_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_1_branch2c_bn_3', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_2_branch2c_bn_2', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res3_2_branch2c_bn_2', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_2_branch2c_bn_3', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_2_branch2c_bn_3', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_3_branch2a_w_0', 'val_type': 1, 'dims': [128, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_3_branch2a_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_3_branch2a_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_3_branch2a_bn_s_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_3_branch2a_bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_3_branch2a_bn_rm_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_3_branch2a_bn_riv_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_3_branch2a_bn_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res3_3_branch2a_bn_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_3_branch2a_bn_2', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res3_3_branch2a_bn_2', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_3_branch2b_w_0', 'val_type': 1, 'dims': [128, 128, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_3_branch2b_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_3_branch2b_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_3_branch2b_bn_s_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_3_branch2b_bn_b_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_3_branch2b_bn_rm_0', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'gpu_0/res3_3_branch2b_bn_riv_0', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_3_branch2b_bn_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res3_3_branch2b_bn_1', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_3_branch2b_bn_2', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_3_branch2b_bn_2', 'val_type': 1, 'dims': [32, 128, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_3_branch2c_w_0', 'val_type': 1, 'dims': [512, 128, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_3_branch2c_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_3_branch2c_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_3_branch2c_bn_s_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res3_3_branch2c_bn_b_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res3_3_branch2c_bn_rm_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res3_3_branch2c_bn_riv_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_3_branch2c_bn_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res3_3_branch2c_bn_1', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res3_2_branch2c_bn_3', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_3_branch2c_bn_2', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res3_3_branch2c_bn_2', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res3_3_branch2c_bn_3', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_3_branch2c_bn_3', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch2a_w_0', 'val_type': 1, 'dims': [256, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_0_branch2a_1', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_0_branch2a_1', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch2a_bn_s_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch2a_bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch2a_bn_rm_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch2a_bn_riv_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_0_branch2a_bn_1', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_0_branch2a_bn_1', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_0_branch2a_bn_2', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [2, 2], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res4_0_branch2a_bn_2', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch2b_w_0', 'val_type': 1, 'dims': [256, 256, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_0_branch2b_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_0_branch2b_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch2b_bn_s_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch2b_bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch2b_bn_rm_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch2b_bn_riv_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_0_branch2b_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_0_branch2b_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_0_branch2b_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_0_branch2b_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch2c_w_0', 'val_type': 1, 'dims': [1024, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_0_branch2c_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_0_branch2c_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch2c_bn_s_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch2c_bn_b_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch2c_bn_rm_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch2c_bn_riv_0', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_0_branch2c_bn_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [2, 2], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res3_3_branch2c_bn_3', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch1_w_0', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_0_branch1_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_0_branch1_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch1_bn_s_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch1_bn_b_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch1_bn_rm_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch1_bn_riv_0', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_0_branch1_bn_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_0_branch2c_bn_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch1_bn_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_0_branch2c_bn_2', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_0_branch2c_bn_2', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_0_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_0_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_1_branch2a_w_0', 'val_type': 1, 'dims': [256, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_1_branch2a_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_1_branch2a_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_1_branch2a_bn_s_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_1_branch2a_bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_1_branch2a_bn_rm_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_1_branch2a_bn_riv_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_1_branch2a_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_1_branch2a_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_1_branch2a_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res4_1_branch2a_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_1_branch2b_w_0', 'val_type': 1, 'dims': [256, 256, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_1_branch2b_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_1_branch2b_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_1_branch2b_bn_s_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_1_branch2b_bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_1_branch2b_bn_rm_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_1_branch2b_bn_riv_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_1_branch2b_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_1_branch2b_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_1_branch2b_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_1_branch2b_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_1_branch2c_w_0', 'val_type': 1, 'dims': [1024, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_1_branch2c_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_1_branch2c_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_1_branch2c_bn_s_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_1_branch2c_bn_b_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_1_branch2c_bn_rm_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_1_branch2c_bn_riv_0', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_1_branch2c_bn_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_1_branch2c_bn_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_0_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_1_branch2c_bn_2', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_1_branch2c_bn_2', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_1_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_1_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_2_branch2a_w_0', 'val_type': 1, 'dims': [256, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_2_branch2a_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_2_branch2a_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_2_branch2a_bn_s_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_2_branch2a_bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_2_branch2a_bn_rm_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_2_branch2a_bn_riv_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_2_branch2a_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_2_branch2a_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_2_branch2a_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res4_2_branch2a_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_2_branch2b_w_0', 'val_type': 1, 'dims': [256, 256, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_2_branch2b_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_2_branch2b_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_2_branch2b_bn_s_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_2_branch2b_bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_2_branch2b_bn_rm_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_2_branch2b_bn_riv_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_2_branch2b_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_2_branch2b_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_2_branch2b_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_2_branch2b_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_2_branch2c_w_0', 'val_type': 1, 'dims': [1024, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_2_branch2c_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_2_branch2c_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_2_branch2c_bn_s_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_2_branch2c_bn_b_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_2_branch2c_bn_rm_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_2_branch2c_bn_riv_0', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_2_branch2c_bn_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_2_branch2c_bn_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_1_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_2_branch2c_bn_2', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_2_branch2c_bn_2', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_2_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_2_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_3_branch2a_w_0', 'val_type': 1, 'dims': [256, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_3_branch2a_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_3_branch2a_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_3_branch2a_bn_s_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_3_branch2a_bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_3_branch2a_bn_rm_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_3_branch2a_bn_riv_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_3_branch2a_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_3_branch2a_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_3_branch2a_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res4_3_branch2a_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_3_branch2b_w_0', 'val_type': 1, 'dims': [256, 256, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_3_branch2b_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_3_branch2b_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_3_branch2b_bn_s_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_3_branch2b_bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_3_branch2b_bn_rm_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_3_branch2b_bn_riv_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_3_branch2b_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_3_branch2b_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_3_branch2b_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_3_branch2b_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_3_branch2c_w_0', 'val_type': 1, 'dims': [1024, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_3_branch2c_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_3_branch2c_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_3_branch2c_bn_s_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_3_branch2c_bn_b_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_3_branch2c_bn_rm_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_3_branch2c_bn_riv_0', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_3_branch2c_bn_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_3_branch2c_bn_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_2_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_3_branch2c_bn_2', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_3_branch2c_bn_2', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_3_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_3_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_4_branch2a_w_0', 'val_type': 1, 'dims': [256, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_4_branch2a_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_4_branch2a_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_4_branch2a_bn_s_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_4_branch2a_bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_4_branch2a_bn_rm_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_4_branch2a_bn_riv_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_4_branch2a_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_4_branch2a_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_4_branch2a_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res4_4_branch2a_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_4_branch2b_w_0', 'val_type': 1, 'dims': [256, 256, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_4_branch2b_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_4_branch2b_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_4_branch2b_bn_s_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_4_branch2b_bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_4_branch2b_bn_rm_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_4_branch2b_bn_riv_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_4_branch2b_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_4_branch2b_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_4_branch2b_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_4_branch2b_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_4_branch2c_w_0', 'val_type': 1, 'dims': [1024, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_4_branch2c_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_4_branch2c_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_4_branch2c_bn_s_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_4_branch2c_bn_b_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_4_branch2c_bn_rm_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_4_branch2c_bn_riv_0', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_4_branch2c_bn_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_4_branch2c_bn_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_3_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_4_branch2c_bn_2', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_4_branch2c_bn_2', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_4_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_4_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_5_branch2a_w_0', 'val_type': 1, 'dims': [256, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_5_branch2a_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_5_branch2a_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_5_branch2a_bn_s_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_5_branch2a_bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_5_branch2a_bn_rm_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_5_branch2a_bn_riv_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_5_branch2a_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_5_branch2a_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_5_branch2a_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res4_5_branch2a_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_5_branch2b_w_0', 'val_type': 1, 'dims': [256, 256, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_5_branch2b_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_5_branch2b_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_5_branch2b_bn_s_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_5_branch2b_bn_b_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_5_branch2b_bn_rm_0', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'gpu_0/res4_5_branch2b_bn_riv_0', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_5_branch2b_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_5_branch2b_bn_1', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_5_branch2b_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_5_branch2b_bn_2', 'val_type': 1, 'dims': [32, 256, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_5_branch2c_w_0', 'val_type': 1, 'dims': [1024, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_5_branch2c_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_5_branch2c_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_5_branch2c_bn_s_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_5_branch2c_bn_b_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_5_branch2c_bn_rm_0', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'gpu_0/res4_5_branch2c_bn_riv_0', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_5_branch2c_bn_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_5_branch2c_bn_1', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res4_4_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_5_branch2c_bn_2', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res4_5_branch2c_bn_2', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res4_5_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_5_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch2a_w_0', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_0_branch2a_1', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res5_0_branch2a_1', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch2a_bn_s_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch2a_bn_b_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch2a_bn_rm_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch2a_bn_riv_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_0_branch2a_bn_1', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res5_0_branch2a_bn_1', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_0_branch2a_bn_2', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [2, 2], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res5_0_branch2a_bn_2', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch2b_w_0', 'val_type': 1, 'dims': [512, 512, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_0_branch2b_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res5_0_branch2b_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch2b_bn_s_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch2b_bn_b_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch2b_bn_rm_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch2b_bn_riv_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_0_branch2b_bn_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res5_0_branch2b_bn_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_0_branch2b_bn_2', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res5_0_branch2b_bn_2', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch2c_w_0', 'val_type': 1, 'dims': [2048, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_0_branch2c_1', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res5_0_branch2c_1', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch2c_bn_s_0', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch2c_bn_b_0', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch2c_bn_rm_0', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch2c_bn_riv_0', 'val_type': 1, 'dims': [2048], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_0_branch2c_bn_1', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [2, 2], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res4_5_branch2c_bn_3', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch1_w_0', 'val_type': 1, 'dims': [2048, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_0_branch1_1', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res5_0_branch1_1', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch1_bn_s_0', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch1_bn_b_0', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch1_bn_rm_0', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch1_bn_riv_0', 'val_type': 1, 'dims': [2048], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_0_branch1_bn_1', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res5_0_branch2c_bn_1', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch1_bn_1', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_0_branch2c_bn_2', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res5_0_branch2c_bn_2', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_0_branch2c_bn_3', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res5_0_branch2c_bn_3', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_1_branch2a_w_0', 'val_type': 1, 'dims': [512, 2048, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_1_branch2a_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res5_1_branch2a_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_1_branch2a_bn_s_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_1_branch2a_bn_b_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_1_branch2a_bn_rm_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_1_branch2a_bn_riv_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_1_branch2a_bn_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res5_1_branch2a_bn_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_1_branch2a_bn_2', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res5_1_branch2a_bn_2', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_1_branch2b_w_0', 'val_type': 1, 'dims': [512, 512, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_1_branch2b_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res5_1_branch2b_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_1_branch2b_bn_s_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_1_branch2b_bn_b_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_1_branch2b_bn_rm_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_1_branch2b_bn_riv_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_1_branch2b_bn_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res5_1_branch2b_bn_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_1_branch2b_bn_2', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res5_1_branch2b_bn_2', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_1_branch2c_w_0', 'val_type': 1, 'dims': [2048, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_1_branch2c_1', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res5_1_branch2c_1', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_1_branch2c_bn_s_0', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'gpu_0/res5_1_branch2c_bn_b_0', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'gpu_0/res5_1_branch2c_bn_rm_0', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'gpu_0/res5_1_branch2c_bn_riv_0', 'val_type': 1, 'dims': [2048], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_1_branch2c_bn_1', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res5_1_branch2c_bn_1', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_0_branch2c_bn_3', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_1_branch2c_bn_2', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res5_1_branch2c_bn_2', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_1_branch2c_bn_3', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res5_1_branch2c_bn_3', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_2_branch2a_w_0', 'val_type': 1, 'dims': [512, 2048, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_2_branch2a_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res5_2_branch2a_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_2_branch2a_bn_s_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_2_branch2a_bn_b_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_2_branch2a_bn_rm_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_2_branch2a_bn_riv_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_2_branch2a_bn_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res5_2_branch2a_bn_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_2_branch2a_bn_2', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [1, 1, 1, 1], 'kernel_shape': [3, 3]}
            'inputs': [
                input {'name': 'gpu_0/res5_2_branch2a_bn_2', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_2_branch2b_w_0', 'val_type': 1, 'dims': [512, 512, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_2_branch2b_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res5_2_branch2b_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_2_branch2b_bn_s_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_2_branch2b_bn_b_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_2_branch2b_bn_rm_0', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'gpu_0/res5_2_branch2b_bn_riv_0', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_2_branch2b_bn_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res5_2_branch2b_bn_1', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_2_branch2b_bn_2', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'strides': [1, 1], 'pads': [0, 0, 0, 0], 'kernel_shape': [1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res5_2_branch2b_bn_2', 'val_type': 1, 'dims': [32, 512, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_2_branch2c_w_0', 'val_type': 1, 'dims': [2048, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_2_branch2c_1', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 1.0000000656873453e-05, 'is_test': 1, 'consumed_inputs': [0, 0, 0, 1, 1]}
            'inputs': [
                input {'name': 'gpu_0/res5_2_branch2c_1', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_2_branch2c_bn_s_0', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'gpu_0/res5_2_branch2c_bn_b_0', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'gpu_0/res5_2_branch2c_bn_rm_0', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'gpu_0/res5_2_branch2c_bn_riv_0', 'val_type': 1, 'dims': [2048], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_2_branch2c_bn_1', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Sum'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res5_2_branch2c_bn_1', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'gpu_0/res5_1_branch2c_bn_3', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_2_branch2c_bn_2', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/res5_2_branch2c_bn_2', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/res5_2_branch2c_bn_3', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'AveragePool'
            'attributes': {'strides': [1, 1], 'kernel_shape': [7, 7]}
            'inputs': [
                input {'name': 'gpu_0/res5_2_branch2c_bn_3', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/pool5_1', 'val_type': 1, 'dims': [32, 2048, 1, 1], 'vals': []}
            ]
        }
        node {
            'op_type': 'Gemm'
            'attributes': {'broadcast': 1, 'transB': 1}
            'inputs': [
                input {'name': 'gpu_0/pool5_1', 'val_type': 1, 'dims': [32, 2048], 'vals': []}
                input {'name': 'gpu_0/pred_w_0', 'val_type': 1, 'dims': [1000, 2048], 'vals': []}
                input {'name': 'gpu_0/pred_b_0', 'val_type': 1, 'dims': [1000], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/pred_1', 'val_type': 1, 'dims': [32, 1000], 'vals': []}
            ]
        }
        node {
            'op_type': 'Softmax'
            'attributes': {}
            'inputs': [
                input {'name': 'gpu_0/pred_1', 'val_type': 1, 'dims': [32, 1000, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'gpu_0/softmax_1', 'val_type': 1, 'dims': [32, 1000, 1, 1], 'vals': []}
            ]
        }
    ]
}
