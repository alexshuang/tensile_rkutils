'ir_version': 4
'producer_name': 'onnx-mims'
'graph': {
    'name': 'torch-jit-export'
    'nodes': [
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 7], 'pads': [3, 3, 3, 3], 'strides': [2, 2]}
            'inputs': [
                input {'name': 'input', 'val_type': 1, 'dims': [32, 3, 224, 224], 'vals': []}
                input {'name': 'conv1.weight', 'val_type': 1, 'dims': [64, 3, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '627', 'val_type': 1, 'dims': [32, 64, 112, 112], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '627', 'val_type': 1, 'dims': [32, 64, 112, 112], 'vals': []}
                input {'name': 'bn1.weight', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'bn1.bias', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'bn1.running_mean', 'val_type': 1, 'dims': [64], 'vals': []}
                input {'name': 'bn1.running_var', 'val_type': 1, 'dims': [64], 'vals': []}
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
            'attributes': {'ceil_mode': 0, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [2, 2]}
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
                input {'name': 'layer1.0.conv1.weight', 'val_type': 1, 'dims': [256, 64, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '631', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '631', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'layer1.0.bn1.weight', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.0.bn1.bias', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.0.bn1.running_mean', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.0.bn1.running_var', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '632', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '632', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '633', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '633', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'layer1.0.conv2.weight', 'val_type': 1, 'dims': [256, 8, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '634', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '634', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'layer1.0.bn2.weight', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.0.bn2.bias', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.0.bn2.running_mean', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.0.bn2.running_var', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '635', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '635', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '636', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '636', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'layer1.0.conv3.weight', 'val_type': 1, 'dims': [256, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '637', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '637', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'layer1.0.bn3.weight', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.0.bn3.bias', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.0.bn3.running_mean', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.0.bn3.running_var', 'val_type': 1, 'dims': [256], 'vals': []}
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
                input {'name': 'layer1.0.downsample.0.weight', 'val_type': 1, 'dims': [256, 64, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '639', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '639', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'layer1.0.downsample.1.weight', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.0.downsample.1.bias', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.0.downsample.1.running_mean', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.0.downsample.1.running_var', 'val_type': 1, 'dims': [256], 'vals': []}
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
                input {'name': 'layer1.1.conv1.weight', 'val_type': 1, 'dims': [256, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '643', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '643', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'layer1.1.bn1.weight', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.1.bn1.bias', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.1.bn1.running_mean', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.1.bn1.running_var', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '644', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '644', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '645', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '645', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'layer1.1.conv2.weight', 'val_type': 1, 'dims': [256, 8, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '646', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '646', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'layer1.1.bn2.weight', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.1.bn2.bias', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.1.bn2.running_mean', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.1.bn2.running_var', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '647', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '647', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '648', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '648', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'layer1.1.conv3.weight', 'val_type': 1, 'dims': [256, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '649', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '649', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'layer1.1.bn3.weight', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.1.bn3.bias', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.1.bn3.running_mean', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.1.bn3.running_var', 'val_type': 1, 'dims': [256], 'vals': []}
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
                input {'name': 'layer1.2.conv1.weight', 'val_type': 1, 'dims': [256, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '653', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '653', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'layer1.2.bn1.weight', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.2.bn1.bias', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.2.bn1.running_mean', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.2.bn1.running_var', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '654', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '654', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '655', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '655', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'layer1.2.conv2.weight', 'val_type': 1, 'dims': [256, 8, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '656', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '656', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'layer1.2.bn2.weight', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.2.bn2.bias', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.2.bn2.running_mean', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.2.bn2.running_var', 'val_type': 1, 'dims': [256], 'vals': []}
            ]
            'outputs': [
                output {'name': '657', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '657', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '658', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '658', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'layer1.2.conv3.weight', 'val_type': 1, 'dims': [256, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '659', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '659', 'val_type': 1, 'dims': [32, 256, 56, 56], 'vals': []}
                input {'name': 'layer1.2.bn3.weight', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.2.bn3.bias', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.2.bn3.running_mean', 'val_type': 1, 'dims': [256], 'vals': []}
                input {'name': 'layer1.2.bn3.running_var', 'val_type': 1, 'dims': [256], 'vals': []}
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
                input {'name': 'layer2.0.conv1.weight', 'val_type': 1, 'dims': [512, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '663', 'val_type': 1, 'dims': [32, 512, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '663', 'val_type': 1, 'dims': [32, 512, 56, 56], 'vals': []}
                input {'name': 'layer2.0.bn1.weight', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.0.bn1.bias', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.0.bn1.running_mean', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.0.bn1.running_var', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '664', 'val_type': 1, 'dims': [32, 512, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '664', 'val_type': 1, 'dims': [32, 512, 56, 56], 'vals': []}
            ]
            'outputs': [
                output {'name': '665', 'val_type': 1, 'dims': [32, 512, 56, 56], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [2, 2]}
            'inputs': [
                input {'name': '665', 'val_type': 1, 'dims': [32, 512, 56, 56], 'vals': []}
                input {'name': 'layer2.0.conv2.weight', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '666', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '666', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.0.bn2.weight', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.0.bn2.bias', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.0.bn2.running_mean', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.0.bn2.running_var', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '667', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '667', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '668', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '668', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.0.conv3.weight', 'val_type': 1, 'dims': [512, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '669', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '669', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.0.bn3.weight', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.0.bn3.bias', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.0.bn3.running_mean', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.0.bn3.running_var', 'val_type': 1, 'dims': [512], 'vals': []}
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
                input {'name': 'layer2.0.downsample.0.weight', 'val_type': 1, 'dims': [512, 256, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '671', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '671', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.0.downsample.1.weight', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.0.downsample.1.bias', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.0.downsample.1.running_mean', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.0.downsample.1.running_var', 'val_type': 1, 'dims': [512], 'vals': []}
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
                input {'name': 'layer2.1.conv1.weight', 'val_type': 1, 'dims': [512, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '675', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '675', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.1.bn1.weight', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.1.bn1.bias', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.1.bn1.running_mean', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.1.bn1.running_var', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '676', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '676', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '677', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '677', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.1.conv2.weight', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '678', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '678', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.1.bn2.weight', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.1.bn2.bias', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.1.bn2.running_mean', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.1.bn2.running_var', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '679', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '679', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '680', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '680', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.1.conv3.weight', 'val_type': 1, 'dims': [512, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '681', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '681', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.1.bn3.weight', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.1.bn3.bias', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.1.bn3.running_mean', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.1.bn3.running_var', 'val_type': 1, 'dims': [512], 'vals': []}
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
                input {'name': 'layer2.2.conv1.weight', 'val_type': 1, 'dims': [512, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '685', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '685', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.2.bn1.weight', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.2.bn1.bias', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.2.bn1.running_mean', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.2.bn1.running_var', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '686', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '686', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '687', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '687', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.2.conv2.weight', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '688', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '688', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.2.bn2.weight', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.2.bn2.bias', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.2.bn2.running_mean', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.2.bn2.running_var', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '689', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '689', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '690', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '690', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.2.conv3.weight', 'val_type': 1, 'dims': [512, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '691', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '691', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.2.bn3.weight', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.2.bn3.bias', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.2.bn3.running_mean', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.2.bn3.running_var', 'val_type': 1, 'dims': [512], 'vals': []}
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
                input {'name': 'layer2.3.conv1.weight', 'val_type': 1, 'dims': [512, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '695', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '695', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.3.bn1.weight', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.3.bn1.bias', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.3.bn1.running_mean', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.3.bn1.running_var', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '696', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '696', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '697', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '697', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.3.conv2.weight', 'val_type': 1, 'dims': [512, 16, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '698', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '698', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.3.bn2.weight', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.3.bn2.bias', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.3.bn2.running_mean', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.3.bn2.running_var', 'val_type': 1, 'dims': [512], 'vals': []}
            ]
            'outputs': [
                output {'name': '699', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '699', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '700', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '700', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.3.conv3.weight', 'val_type': 1, 'dims': [512, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '701', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '701', 'val_type': 1, 'dims': [32, 512, 28, 28], 'vals': []}
                input {'name': 'layer2.3.bn3.weight', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.3.bn3.bias', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.3.bn3.running_mean', 'val_type': 1, 'dims': [512], 'vals': []}
                input {'name': 'layer2.3.bn3.running_var', 'val_type': 1, 'dims': [512], 'vals': []}
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
                input {'name': 'layer3.0.conv1.weight', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '705', 'val_type': 1, 'dims': [32, 1024, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '705', 'val_type': 1, 'dims': [32, 1024, 28, 28], 'vals': []}
                input {'name': 'layer3.0.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.0.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.0.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.0.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '706', 'val_type': 1, 'dims': [32, 1024, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '706', 'val_type': 1, 'dims': [32, 1024, 28, 28], 'vals': []}
            ]
            'outputs': [
                output {'name': '707', 'val_type': 1, 'dims': [32, 1024, 28, 28], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [2, 2]}
            'inputs': [
                input {'name': '707', 'val_type': 1, 'dims': [32, 1024, 28, 28], 'vals': []}
                input {'name': 'layer3.0.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '708', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '708', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.0.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.0.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.0.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.0.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '709', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '709', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '710', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '710', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.0.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '711', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '711', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.0.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.0.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.0.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.0.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.0.downsample.0.weight', 'val_type': 1, 'dims': [1024, 512, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '713', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '713', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.0.downsample.1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.0.downsample.1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.0.downsample.1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.0.downsample.1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.1.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '717', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '717', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.1.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.1.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.1.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.1.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '718', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '718', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '719', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '719', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.1.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '720', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '720', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.1.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.1.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.1.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.1.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '721', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '721', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '722', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '722', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.1.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '723', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '723', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.1.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.1.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.1.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.1.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.2.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '727', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '727', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.2.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.2.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.2.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.2.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '728', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '728', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '729', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '729', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.2.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '730', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '730', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.2.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.2.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.2.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.2.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '731', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '731', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '732', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '732', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.2.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '733', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '733', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.2.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.2.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.2.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.2.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.3.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '737', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '737', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.3.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.3.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.3.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.3.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '738', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '738', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '739', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '739', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.3.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '740', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '740', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.3.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.3.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.3.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.3.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '741', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '741', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '742', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '742', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.3.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '743', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '743', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.3.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.3.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.3.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.3.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.4.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '747', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '747', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.4.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.4.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.4.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.4.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '748', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '748', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '749', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '749', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.4.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '750', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '750', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.4.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.4.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.4.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.4.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '751', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '751', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '752', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '752', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.4.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '753', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '753', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.4.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.4.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.4.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.4.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.5.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '757', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '757', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.5.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.5.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.5.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.5.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '758', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '758', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '759', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '759', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.5.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '760', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '760', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.5.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.5.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.5.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.5.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '761', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '761', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '762', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '762', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.5.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '763', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '763', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.5.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.5.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.5.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.5.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.6.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '767', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '767', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.6.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.6.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.6.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.6.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '768', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '768', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '769', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '769', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.6.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '770', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '770', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.6.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.6.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.6.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.6.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '771', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '771', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '772', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '772', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.6.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '773', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '773', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.6.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.6.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.6.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.6.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.7.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '777', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '777', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.7.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.7.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.7.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.7.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '778', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '778', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '779', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '779', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.7.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '780', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '780', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.7.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.7.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.7.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.7.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '781', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '781', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '782', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '782', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.7.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '783', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '783', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.7.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.7.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.7.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.7.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.8.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '787', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '787', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.8.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.8.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.8.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.8.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '788', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '788', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '789', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '789', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.8.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '790', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '790', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.8.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.8.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.8.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.8.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '791', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '791', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '792', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '792', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.8.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '793', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '793', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.8.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.8.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.8.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.8.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.9.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '797', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '797', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.9.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.9.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.9.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.9.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '798', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '798', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '799', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '799', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.9.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '800', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '800', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.9.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.9.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.9.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.9.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '801', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '801', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '802', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '802', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.9.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '803', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '803', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.9.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.9.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.9.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.9.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.10.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '807', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '807', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.10.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.10.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.10.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.10.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '808', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '808', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '809', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '809', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.10.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '810', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '810', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.10.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.10.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.10.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.10.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '811', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '811', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '812', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '812', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.10.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '813', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '813', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.10.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.10.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.10.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.10.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.11.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '817', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '817', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.11.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.11.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.11.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.11.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '818', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '818', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '819', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '819', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.11.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '820', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '820', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.11.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.11.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.11.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.11.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '821', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '821', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '822', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '822', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.11.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '823', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '823', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.11.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.11.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.11.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.11.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.12.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '827', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '827', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.12.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.12.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.12.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.12.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '828', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '828', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '829', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '829', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.12.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '830', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '830', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.12.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.12.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.12.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.12.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '831', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '831', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '832', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '832', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.12.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '833', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '833', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.12.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.12.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.12.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.12.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.13.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '837', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '837', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.13.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.13.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.13.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.13.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '838', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '838', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '839', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '839', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.13.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '840', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '840', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.13.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.13.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.13.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.13.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '841', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '841', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '842', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '842', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.13.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '843', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '843', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.13.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.13.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.13.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.13.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.14.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '847', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '847', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.14.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.14.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.14.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.14.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '848', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '848', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '849', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '849', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.14.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '850', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '850', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.14.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.14.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.14.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.14.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '851', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '851', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '852', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '852', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.14.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '853', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '853', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.14.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.14.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.14.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.14.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.15.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '857', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '857', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.15.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.15.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.15.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.15.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '858', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '858', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '859', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '859', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.15.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '860', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '860', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.15.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.15.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.15.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.15.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '861', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '861', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '862', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '862', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.15.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '863', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '863', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.15.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.15.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.15.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.15.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.16.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '867', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '867', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.16.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.16.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.16.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.16.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '868', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '868', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '869', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '869', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.16.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '870', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '870', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.16.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.16.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.16.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.16.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '871', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '871', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '872', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '872', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.16.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '873', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '873', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.16.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.16.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.16.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.16.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.17.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '877', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '877', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.17.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.17.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.17.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.17.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '878', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '878', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '879', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '879', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.17.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '880', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '880', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.17.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.17.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.17.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.17.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '881', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '881', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '882', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '882', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.17.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '883', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '883', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.17.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.17.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.17.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.17.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.18.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '887', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '887', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.18.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.18.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.18.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.18.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '888', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '888', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '889', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '889', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.18.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '890', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '890', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.18.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.18.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.18.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.18.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '891', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '891', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '892', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '892', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.18.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '893', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '893', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.18.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.18.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.18.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.18.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.19.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '897', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '897', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.19.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.19.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.19.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.19.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '898', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '898', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '899', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '899', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.19.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '900', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '900', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.19.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.19.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.19.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.19.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '901', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '901', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '902', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '902', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.19.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '903', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '903', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.19.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.19.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.19.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.19.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.20.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '907', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '907', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.20.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.20.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.20.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.20.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '908', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '908', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '909', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '909', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.20.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '910', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '910', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.20.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.20.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.20.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.20.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '911', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '911', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '912', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '912', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.20.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '913', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '913', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.20.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.20.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.20.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.20.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.21.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '917', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '917', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.21.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.21.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.21.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.21.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '918', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '918', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '919', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '919', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.21.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '920', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '920', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.21.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.21.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.21.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.21.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '921', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '921', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '922', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '922', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.21.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '923', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '923', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.21.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.21.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.21.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.21.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer3.22.conv1.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '927', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '927', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.22.bn1.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.22.bn1.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.22.bn1.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.22.bn1.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '928', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '928', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '929', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '929', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.22.conv2.weight', 'val_type': 1, 'dims': [1024, 32, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '930', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '930', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.22.bn2.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.22.bn2.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.22.bn2.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.22.bn2.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
            ]
            'outputs': [
                output {'name': '931', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '931', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '932', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '932', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.22.conv3.weight', 'val_type': 1, 'dims': [1024, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '933', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '933', 'val_type': 1, 'dims': [32, 1024, 14, 14], 'vals': []}
                input {'name': 'layer3.22.bn3.weight', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.22.bn3.bias', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.22.bn3.running_mean', 'val_type': 1, 'dims': [1024], 'vals': []}
                input {'name': 'layer3.22.bn3.running_var', 'val_type': 1, 'dims': [1024], 'vals': []}
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
                input {'name': 'layer4.0.conv1.weight', 'val_type': 1, 'dims': [2048, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '937', 'val_type': 1, 'dims': [32, 2048, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '937', 'val_type': 1, 'dims': [32, 2048, 14, 14], 'vals': []}
                input {'name': 'layer4.0.bn1.weight', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.0.bn1.bias', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.0.bn1.running_mean', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.0.bn1.running_var', 'val_type': 1, 'dims': [2048], 'vals': []}
            ]
            'outputs': [
                output {'name': '938', 'val_type': 1, 'dims': [32, 2048, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '938', 'val_type': 1, 'dims': [32, 2048, 14, 14], 'vals': []}
            ]
            'outputs': [
                output {'name': '939', 'val_type': 1, 'dims': [32, 2048, 14, 14], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [2, 2]}
            'inputs': [
                input {'name': '939', 'val_type': 1, 'dims': [32, 2048, 14, 14], 'vals': []}
                input {'name': 'layer4.0.conv2.weight', 'val_type': 1, 'dims': [2048, 64, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '940', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '940', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'layer4.0.bn2.weight', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.0.bn2.bias', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.0.bn2.running_mean', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.0.bn2.running_var', 'val_type': 1, 'dims': [2048], 'vals': []}
            ]
            'outputs': [
                output {'name': '941', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '941', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '942', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '942', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'layer4.0.conv3.weight', 'val_type': 1, 'dims': [2048, 2048, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '943', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '943', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'layer4.0.bn3.weight', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.0.bn3.bias', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.0.bn3.running_mean', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.0.bn3.running_var', 'val_type': 1, 'dims': [2048], 'vals': []}
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
                input {'name': 'layer4.0.downsample.0.weight', 'val_type': 1, 'dims': [2048, 1024, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '945', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '945', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'layer4.0.downsample.1.weight', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.0.downsample.1.bias', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.0.downsample.1.running_mean', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.0.downsample.1.running_var', 'val_type': 1, 'dims': [2048], 'vals': []}
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
                input {'name': 'layer4.1.conv1.weight', 'val_type': 1, 'dims': [2048, 2048, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '949', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '949', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'layer4.1.bn1.weight', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.1.bn1.bias', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.1.bn1.running_mean', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.1.bn1.running_var', 'val_type': 1, 'dims': [2048], 'vals': []}
            ]
            'outputs': [
                output {'name': '950', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '950', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '951', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '951', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'layer4.1.conv2.weight', 'val_type': 1, 'dims': [2048, 64, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '952', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '952', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'layer4.1.bn2.weight', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.1.bn2.bias', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.1.bn2.running_mean', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.1.bn2.running_var', 'val_type': 1, 'dims': [2048], 'vals': []}
            ]
            'outputs': [
                output {'name': '953', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '953', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '954', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '954', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'layer4.1.conv3.weight', 'val_type': 1, 'dims': [2048, 2048, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '955', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '955', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'layer4.1.bn3.weight', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.1.bn3.bias', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.1.bn3.running_mean', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.1.bn3.running_var', 'val_type': 1, 'dims': [2048], 'vals': []}
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
                input {'name': 'layer4.2.conv1.weight', 'val_type': 1, 'dims': [2048, 2048, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '959', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '959', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'layer4.2.bn1.weight', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.2.bn1.bias', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.2.bn1.running_mean', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.2.bn1.running_var', 'val_type': 1, 'dims': [2048], 'vals': []}
            ]
            'outputs': [
                output {'name': '960', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '960', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '961', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 32, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
            'inputs': [
                input {'name': '961', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'layer4.2.conv2.weight', 'val_type': 1, 'dims': [2048, 64, 3, 3], 'vals': []}
            ]
            'outputs': [
                output {'name': '962', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '962', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'layer4.2.bn2.weight', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.2.bn2.bias', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.2.bn2.running_mean', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.2.bn2.running_var', 'val_type': 1, 'dims': [2048], 'vals': []}
            ]
            'outputs': [
                output {'name': '963', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Relu'
            'attributes': {}
            'inputs': [
                input {'name': '963', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
            'outputs': [
                output {'name': '964', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'Conv'
            'attributes': {'dilations': [1, 1], 'group': 1, 'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}
            'inputs': [
                input {'name': '964', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'layer4.2.conv3.weight', 'val_type': 1, 'dims': [2048, 2048, 1, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': '965', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
            ]
        }
        node {
            'op_type': 'BatchNormalization'
            'attributes': {'epsilon': 9.999999747378752e-06, 'momentum': 0.8999999761581421}
            'inputs': [
                input {'name': '965', 'val_type': 1, 'dims': [32, 2048, 7, 7], 'vals': []}
                input {'name': 'layer4.2.bn3.weight', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.2.bn3.bias', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.2.bn3.running_mean', 'val_type': 1, 'dims': [2048], 'vals': []}
                input {'name': 'layer4.2.bn3.running_var', 'val_type': 1, 'dims': [2048], 'vals': []}
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
            'attributes': {'alpha': 1.0, 'beta': 1.0, 'transB': 1}
            'inputs': [
                input {'name': '970', 'val_type': None, 'dims': [32, 100352, 1, 1], 'vals': []}
                input {'name': 'fc.weight', 'val_type': 1, 'dims': [1000, 2048], 'vals': []}
                input {'name': 'fc.bias', 'val_type': 1, 'dims': [1000], 'vals': []}
            ]
            'outputs': [
                output {'name': 'output', 'val_type': None, 'dims': [32, 1, 1000, 1], 'vals': []}
            ]
        }
		node {
            'op_type': 'Softmax'
            'attributes': {}
            'inputs': [
                input {'name': '971', 'val_type': None, 'dims': [32, 1, 1000, 1], 'vals': []}
            ]
            'outputs': [
                output {'name': 'output1', 'val_type': None, 'dims': [32, 1, 1000, 1], 'vals': []}
            ]
        }
    ]
}
