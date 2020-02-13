'ir_version': 3
'producer_name': 'onnx-mims'
'graph': {
    'name': 'torch-jit-export'
    'nodes': [
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 7], 'pads': [3, 3, 3, 3], 'strides': [2, 2]}
            'inputs': [
                input {'name': 'actual_input_1', 'val_type': 1, 'dims': [32, 3, 224, 224], 'vals': []}
                input {'name': 'learned_0', 'val_type': 1, 'dims': [64, 3, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '627', 'val_type': 1, 'dims': [32, 64, 112, 112], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '627', 'val_type': 1, 'dims': [32, 64, 112, 112], 'vals': []}
                input {'name': 'learned_1', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'learned_2', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'learned_3', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'learned_4', 'val_type': 1, 'dims': [64], 'vals': []}
            ]
            'outputs': [
                output {'name': '628', 'val_type': 1, 'dims': [32, 64, 112, 112], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '628', 'val_type': 1, 'dims': [32, 64, 112, 112], 'vals': []}
            ]
            'outputs': [
                output {'name': '629', 'val_type': 1, 'dims': [32, 64, 112, 112], 'vals': []}
            ]
        }
        node {
            'op_type': 'MaxPool'
            'attributes': {'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [2, 2]}
            'inputs': [
                input {'name': '629', 'val_type': 1, 'dims': [32, 64, 112, 112], 'vals': []}
            ]
            'outputs': [
                output {'name': '630', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '630', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
                input {'name': 'learned_6', 'val_type': 1, 'dims': [128, 64, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '631', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '631', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
                input {'name': 'learned_7', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'learned_8', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'learned_9', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'learned_10', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': '632', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '632', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '633', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '633', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
                input {'name': 'learned_12', 'val_type': 1, 'dims': [128, 4, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '634', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '634', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
                input {'name': 'learned_13', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'learned_14', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': 'learned_15', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': '17', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': '635', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '635', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '636', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '636', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
                input {'name': '19', 'val_type': 1, 'dims': [256, 128, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '637', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '637', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': '20', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '21', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '22', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '23', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '638', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '630', 'val_type': 1, 'dims': [32, 64, 56, 56], 'vals': []}
                input {'name': '25', 'val_type': 1, 'dims': [256, 64, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '639', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '639', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': '26', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '27', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '28', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '29', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '640', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '638', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': '640', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '641', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '641', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '642', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '642', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': '31', 'val_type': 1, 'dims': [128, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '643', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '643', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
                input {'name': '32', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': '33', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': '34', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': '35', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': '644', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '644', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '645', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '645', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
                input {'name': '37', 'val_type': 1, 'dims': [128, 4, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '646', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '646', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
                input {'name': '38', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': '39', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': '40', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': '41', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': '647', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '647', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '648', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '648', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
                input {'name': '43', 'val_type': 1, 'dims': [256, 128, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '649', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '649', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': '44', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '45', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '46', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '47', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '650', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '650', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': '642', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '651', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '651', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '652', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '652', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': '49', 'val_type': 1, 'dims': [128, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '653', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '653', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
                input {'name': '50', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': '51', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': '52', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': '53', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': '654', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '654', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '655', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '655', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
                input {'name': '55', 'val_type': 1, 'dims': [128, 4, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '656', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '656', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
                input {'name': '56', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': '57', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': '58', 'val_type': 1, 'dims': [128], 'vals': []}
                input {'name': '59', 'val_type': 1, 'dims': [128], 'vals': []}
            ]
            'outputs': [
                output {'name': '657', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '657', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '658', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '658', 'val_type': 1, 'dims': [32, 128, 56, 56], 'vals': []}
                input {'name': '61', 'val_type': 1, 'dims': [256, 128, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '659', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '659', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': '62', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '63', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '64', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '65', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '660', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '660', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': '652', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '661', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '661', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '662', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '662', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': '67', 'val_type': 1, 'dims': [256, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '663', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '663', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': '68', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '69', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '70', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '71', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '664', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '664', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '665', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [2, 2]}
            'inputs': [
                input {'name': '665', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': '73', 'val_type': 1, 'dims': [256, 8, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '666', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '666', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
                input {'name': '74', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '75', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '76', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '77', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '667', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '667', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '668', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '668', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
                input {'name': '79', 'val_type': 1, 'dims': [512, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '669', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '669', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': '80', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '81', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '82', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '83', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '670', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}
            'inputs': [
                input {'name': '662', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': '85', 'val_type': 1, 'dims': [512, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '671', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '671', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': '86', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '87', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '88', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '89', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '672', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '670', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': '672', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '673', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '673', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '674', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '674', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': '91', 'val_type': 1, 'dims': [256, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '675', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '675', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
                input {'name': '92', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '93', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '94', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '95', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '676', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '676', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '677', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '677', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
                input {'name': '97', 'val_type': 1, 'dims': [256, 8, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '678', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '678', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
                input {'name': '98', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '99', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '100', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '101', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '679', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '679', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '680', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '680', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
                input {'name': '103', 'val_type': 1, 'dims': [512, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '681', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '681', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': '104', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '105', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '106', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '107', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '682', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '682', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': '674', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '683', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '683', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '684', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '684', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': '109', 'val_type': 1, 'dims': [256, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '685', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '685', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
                input {'name': '110', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '111', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '112', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '113', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '686', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '686', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '687', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '687', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
                input {'name': '115', 'val_type': 1, 'dims': [256, 8, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '688', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '688', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
                input {'name': '116', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '117', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '118', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '119', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '689', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '689', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '690', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '690', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
                input {'name': '121', 'val_type': 1, 'dims': [512, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '691', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '691', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': '122', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '123', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '124', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '125', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '692', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '692', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': '684', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '693', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '693', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '694', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '694', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': '127', 'val_type': 1, 'dims': [256, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '695', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '695', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
                input {'name': '128', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '129', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '130', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '131', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '696', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '696', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '697', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '697', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
                input {'name': '133', 'val_type': 1, 'dims': [256, 8, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '698', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '698', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
                input {'name': '134', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '135', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '136', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': '137', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '699', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '699', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '700', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '700', 'val_type': 1, 'dims': [32, 256, 28, 28], 'vals': []}
                input {'name': '139', 'val_type': 1, 'dims': [512, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '701', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '701', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': '140', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '141', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '142', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '143', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '702', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '702', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': '694', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '703', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '703', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '704', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '704', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': '145', 'val_type': 1, 'dims': [512, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '705', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '705', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': '146', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '147', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '148', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '149', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '706', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '706', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '707', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [2, 2]}
            'inputs': [
                input {'name': '707', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': '151', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '708', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '708', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '152', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '153', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '154', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '155', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '709', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '709', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '710', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '710', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '157', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '711', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '711', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '158', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '159', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '160', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '161', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '712', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}
            'inputs': [
                input {'name': '704', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': '163', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '713', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '713', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '164', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '165', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '166', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '167', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '714', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '712', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '714', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '715', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '715', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '716', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '716', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '169', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '717', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '717', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '170', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '171', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '172', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '173', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '718', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '718', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '719', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '719', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '175', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '720', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '720', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '176', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '177', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '178', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '179', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '721', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '721', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '722', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '722', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '181', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '723', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '723', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '182', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '183', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '184', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '185', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '724', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '724', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '716', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '725', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '725', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '726', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '726', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '187', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '727', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '727', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '188', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '189', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '190', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '191', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '728', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '728', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '729', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '729', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '193', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '730', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '730', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '194', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '195', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '196', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '197', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '731', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '731', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '732', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '732', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '199', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '733', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '733', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '200', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '201', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '202', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '203', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '734', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '734', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '726', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '735', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '735', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '736', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '736', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '205', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '737', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '737', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '206', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '207', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '208', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '209', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '738', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '738', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '739', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '739', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '211', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '740', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '740', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '212', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '213', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '214', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '215', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '741', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '741', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '742', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '742', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '217', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '743', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '743', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '218', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '219', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '220', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '221', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '744', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '744', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '736', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '745', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '745', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '746', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '746', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '223', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '747', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '747', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '224', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '225', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '226', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '227', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '748', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '748', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '749', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '749', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '229', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '750', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '750', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '230', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '231', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '232', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '233', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '751', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '751', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '752', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '752', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '235', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '753', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '753', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '236', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '237', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '238', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '239', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '754', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '754', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '746', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '755', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '755', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '756', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '756', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '241', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '757', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '757', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '242', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '243', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '244', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '245', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '758', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '758', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '759', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '759', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '247', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '760', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '760', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '248', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '249', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '250', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '251', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '761', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '761', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '762', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '762', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '253', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '763', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '763', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '254', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '255', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '256', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '257', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '764', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '764', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '756', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '765', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '765', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '766', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '766', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '259', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '767', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '767', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '260', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '261', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '262', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '263', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '768', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '768', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '769', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '769', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '265', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '770', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '770', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '266', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '267', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '268', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '269', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '771', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '771', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '772', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '772', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '271', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '773', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '773', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '272', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '273', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '274', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '275', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '774', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '774', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '766', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '775', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '775', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '776', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '776', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '277', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '777', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '777', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '278', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '279', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '280', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '281', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '778', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '778', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '779', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '779', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '283', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '780', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '780', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '284', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '285', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '286', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '287', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '781', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '781', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '782', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '782', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '289', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '783', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '783', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '290', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '291', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '292', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '293', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '784', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '784', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '776', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '785', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '785', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '786', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '786', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '295', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '787', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '787', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '296', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '297', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '298', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '299', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '788', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '788', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '789', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '789', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '301', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '790', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '790', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '302', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '303', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '304', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '305', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '791', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '791', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '792', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '792', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '307', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '793', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '793', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '308', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '309', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '310', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '311', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '794', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '794', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '786', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '795', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '795', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '796', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '796', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '313', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '797', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '797', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '314', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '315', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '316', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '317', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '798', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '798', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '799', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '799', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '319', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '800', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '800', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '320', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '321', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '322', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '323', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '801', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '801', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '802', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '802', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '325', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '803', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '803', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '326', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '327', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '328', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '329', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '804', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '804', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '796', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '805', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '805', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '806', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '806', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '331', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '807', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '807', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '332', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '333', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '334', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '335', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '808', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '808', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '809', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '809', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '337', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '810', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '810', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '338', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '339', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '340', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '341', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '811', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '811', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '812', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '812', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '343', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '813', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '813', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '344', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '345', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '346', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '347', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '814', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '814', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '806', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '815', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '815', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '816', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '816', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '349', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '817', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '817', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '350', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '351', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '352', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '353', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '818', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '818', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '819', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '819', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '355', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '820', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '820', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '356', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '357', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '358', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '359', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '821', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '821', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '822', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '822', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '361', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '823', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '823', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '362', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '363', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '364', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '365', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '824', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '824', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '816', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '825', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '825', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '826', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '826', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '367', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '827', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '827', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '368', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '369', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '370', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '371', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '828', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '828', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '829', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '829', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '373', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '830', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '830', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '374', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '375', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '376', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '377', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '831', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '831', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '832', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '832', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '379', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '833', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '833', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '380', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '381', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '382', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '383', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '834', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '834', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '826', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '835', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '835', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '836', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '836', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '385', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '837', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '837', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '386', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '387', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '388', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '389', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '838', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '838', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '839', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '839', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '391', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '840', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '840', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '392', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '393', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '394', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '395', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '841', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '841', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '842', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '842', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '397', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '843', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '843', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '398', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '399', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '400', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '401', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '844', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '844', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '836', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '845', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '845', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '846', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '846', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '403', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '847', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '847', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '404', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '405', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '406', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '407', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '848', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '848', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '849', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '849', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '409', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '850', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '850', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '410', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '411', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '412', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '413', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '851', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '851', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '852', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '852', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '415', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '853', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '853', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '416', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '417', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '418', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '419', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '854', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '854', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '846', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '855', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '855', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '856', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '856', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '421', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '857', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '857', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '422', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '423', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '424', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '425', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '858', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '858', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '859', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '859', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '427', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '860', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '860', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '428', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '429', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '430', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '431', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '861', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '861', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '862', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '862', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '433', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '863', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '863', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '434', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '435', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '436', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '437', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '864', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '864', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '856', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '865', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '865', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '866', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '866', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '439', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '867', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '867', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '440', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '441', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '442', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '443', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '868', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '868', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '869', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '869', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '445', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '870', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '870', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '446', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '447', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '448', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '449', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '871', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '871', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '872', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '872', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '451', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '873', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '873', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '452', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '453', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '454', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '455', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '874', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '874', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '866', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '875', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '875', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '876', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '876', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '457', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '877', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '877', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '458', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '459', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '460', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '461', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '878', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '878', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '879', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '879', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '463', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '880', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '880', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '464', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '465', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '466', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '467', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '881', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '881', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '882', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '882', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '469', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '883', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '883', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '470', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '471', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '472', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '473', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '884', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '884', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '876', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '885', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '885', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '886', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '886', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '475', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '887', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '887', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '476', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '477', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '478', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '479', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '888', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '888', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '889', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '889', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '481', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '890', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '890', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '482', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '483', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '484', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '485', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '891', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '891', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '892', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '892', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '487', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '893', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '893', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '488', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '489', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '490', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '491', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '894', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '894', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '886', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '895', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '895', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '896', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '896', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '493', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '897', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '897', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '494', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '495', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '496', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '497', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '898', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '898', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '899', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '899', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '499', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '900', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '900', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '500', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '501', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '502', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '503', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '901', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '901', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '902', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '902', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '505', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '903', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '903', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '506', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '507', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '508', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '509', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '904', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '904', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '896', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '905', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '905', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '906', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '906', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '511', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '907', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '907', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '512', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '513', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '514', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '515', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '908', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '908', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '909', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '909', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '517', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '910', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '910', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '518', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '519', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '520', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '521', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '911', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '911', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '912', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '912', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '523', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '913', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '913', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '524', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '525', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '526', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '527', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '914', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '914', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '906', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '915', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '915', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '916', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '916', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '529', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '917', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '917', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '530', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '531', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '532', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '533', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '918', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '918', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '919', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '919', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '535', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '920', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '920', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '536', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '537', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '538', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '539', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '921', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '921', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '922', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '922', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '541', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '923', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '923', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '542', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '543', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '544', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '545', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '924', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '924', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '916', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '925', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '925', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '926', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '926', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '547', 'val_type': 1, 'dims': [512, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '927', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '927', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '548', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '549', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '550', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '551', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '928', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '928', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '929', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '929', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '553', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '930', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '930', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '554', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '555', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '556', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': '557', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '931', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '931', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '932', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '932', 'val_type': 1, 'dims': [32, 512, 14, 14], 'vals': []}
                input {'name': '559', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '933', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '933', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '560', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '561', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '562', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '563', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '934', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '934', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '926', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '935', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '935', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '936', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '936', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '565', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '937', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '937', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '566', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '567', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '568', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '569', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '938', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '938', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '939', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [2, 2]}
            'inputs': [
                input {'name': '939', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '571', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '940', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '940', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
                input {'name': '572', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '573', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '574', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '575', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '941', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '941', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '942', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '942', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
                input {'name': '577', 'val_type': 1, 'dims': [2048, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '943', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '943', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': '578', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': '579', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': '580', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': '581', 'val_type': 1, 'dims': [2048], 'vals': []}
            ]
            'outputs': [
                output {'name': '944', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [2, 2]}
            'inputs': [
                input {'name': '936', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': '583', 'val_type': 1, 'dims': [2048, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '945', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '945', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': '584', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': '585', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': '586', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': '587', 'val_type': 1, 'dims': [2048], 'vals': []}
            ]
            'outputs': [
                output {'name': '946', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '944', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': '946', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '947', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '947', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '948', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '948', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': '589', 'val_type': 1, 'dims': [1024, 2048, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '949', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '949', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
                input {'name': '590', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '591', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '592', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '593', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '950', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '950', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '951', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '951', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
                input {'name': '595', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '952', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '952', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
                input {'name': '596', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '597', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '598', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '599', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '953', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '953', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '954', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '954', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
                input {'name': '601', 'val_type': 1, 'dims': [2048, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '955', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '955', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': '602', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': '603', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': '604', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': '605', 'val_type': 1, 'dims': [2048], 'vals': []}
            ]
            'outputs': [
                output {'name': '956', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '956', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': '948', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '957', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '957', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '958', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '958', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': '607', 'val_type': 1, 'dims': [1024, 2048, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '959', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '959', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
                input {'name': '608', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '609', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '610', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '611', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '960', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '960', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '961', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '961', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
                input {'name': '613', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '962', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '962', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
                input {'name': '614', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '615', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '616', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': '617', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '963', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '963', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '964', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '964', 'val_type': 1, 'dims': [32, 1024, 7, 7], 'vals': []}
                input {'name': '619', 'val_type': 1, 'dims': [2048, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '965', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'is_test': 1, 'momentum': 1.0}
            'inputs': [
                input {'name': '965', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': '620', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': '621', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': '622', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': '623', 'val_type': 1, 'dims': [2048], 'vals': []}
            ]
            'outputs': [
                output {'name': '966', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Add'
            'attributes': {}
            'inputs': [
                input {'name': '966', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': '958', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '967', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '967', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '968', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Pad'
            'attributes': {'mode': b'constant', 'pads': [0, 0, 0, 0, 0, 0, 0, 0], 'value': 0.0}
            'inputs': [
                input {'name': '968', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '969', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'AveragePool'
            'attributes': {'kernel_shape': [7, 7], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '969', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '970', 'val_type': 1, 'dims': [32, 2048, 1, 1], 'vals': []}
            ]
        }
        node {
            'op_type': 'Gemm'
            'attributes': {'alpha': 1.0, 'beta': 1.0, 'broadcast': 1, 'transB': 1}
            'inputs': [
                input {'name': '970', 'val_type': None, 'dims': [32, 2048], 'vals': []}
                input {'name': '625', 'val_type': 1, 'dims': [1000, 2048], 'vals': []}
                input {'name': '626', 'val_type': 1, 'dims': [1000], 'vals': []}
            ]
            'outputs': [
                output {'name': '971', 'val_type': None, 'dims': [], 'vals': []}
            ]
        }
		node {
            'op_type': 'Softmax'
            'attributes': {}
            'inputs': [
                input {'name': '971', 'val_type': 1, 'dims': [32, 1000, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'output1', 'val_type': 1, 'dims': [32, 1000, 1, 1], 'vals': []}
            ]
        }
    ]
}
