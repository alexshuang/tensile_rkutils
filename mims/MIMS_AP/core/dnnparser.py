from onnx import ModelProto, AttributeProto, helper, gen_proto
from core.protobuf_to_dict import protobuf_to_dict
from google.protobuf import json_format
from os import path
import ast
from math import floor
import copy
import numpy as np
from core.const import *
import math
from onnx import numpy_helper

first_rnn_layer = True
is_attn_based_nw = False


class MIMSModel:
    INDENT_LEN = 4
    PRODUCER = 'onnx-mims'

    def __init__(self, ir_version=None, mims_graph=None):
        self.ir_version = ir_version
        self.graph = mims_graph

    def from_py_dict(self, dict_model):
        self.__dict__.update(dict_model)
        for k, v in dict_model.items():
            if k == 'graph':
                self.__dict__[k] = MIMSGraph().from_py_dict(v)
        return self

    def to_onnx_model(self):
        onnx_graph = self.graph.to_onnx_graph()
        onnx_model = helper.make_model(onnx_graph, producer_name=self.PRODUCER, ir_version=self.ir_version)
        return onnx_model

    def from_onnx_model(self, onnx_model, batch_size):
        self.ir_version = onnx_model.ir_version
        self.graph = MIMSGraph().from_onnx_graph(onnx_model.graph, batch_size)
        return self

    def dump(self, file_path):  # dump model in MIMS format (closely follows the python syntax, a bit different)
        fid = open(file_path, 'w')

        def write_line(indent_level, line):
            fid.write('{}{}\n'.format(' ' * self.INDENT_LEN * indent_level, line))

        write_line(0, '\'ir_version\': {}'.format(self.ir_version))
        write_line(0, '\'producer_name\': \'{}\''.format(self.PRODUCER))
        write_line(0, '\'graph\': {')
        write_line(1, '\'name\': \'{}\''.format(self.graph.name))
        write_line(1, '\'nodes\': [')
        for node in self.graph.nodes:
            write_line(2, 'node {')
            write_line(3, '\'op_type\': \'{}\''.format(node.op_type))
            write_line(3, '\'attributes\': {}'.format(node.attributes))
            write_line(3, '\'inputs\': [')
            for inp in node.inputs:
                write_line(4, 'input {}'.format(vars(inp)))
            write_line(3, ']')
            write_line(3, '\'outputs\': [')
            for outp in node.outputs:
                write_line(4, 'output {}'.format(vars(outp)))
            write_line(3, ']')
            write_line(2, '}')
        write_line(1, ']')
        write_line(0, '}')
        fid.close()

    def load(self, file_path):
        with open(file_path, 'r') as mims_f:
            content = mims_f.readlines()
        # load data as a python expresson (ultimately, a nested dictionary) and convert to MIMSModel
        new_content = []
        for line in content:
            new_str = line.strip().replace(' ', '')
            if not new_str.startswith('\''):  # not standard synctax
                new_str = new_str.replace('node{', '{').replace('input{', '{').replace('output{', '{')
            if new_str and not new_str[-1] in '[{':  # add comma at the end if necessary
                new_str = '{},'.format(new_str)
            new_content.append(new_str)
        new_content = ''.join(new_content)
        content_dict = ast.literal_eval('{' + new_content + '}')  # extract object structure from string
        return self.from_py_dict(content_dict)

    def print_mims_graph(self):
        # onnx.helper.printable_graph(model.graph)
        for node in self.graph.nodes:
            print("------- " + node.op_type + " -------")
            children = [c.op_type for c in node.children]
            print("children " + str(children))
            parents = [c.op_type for c in node.parents]
            print("parents " + str(parents))
            print("attributes " + str(node.attributes))
            print("-------================-------")


class MIMSGraph:
    def __init__(self, name=None, mims_nodes=None):
        self.name = name
        self.nodes = mims_nodes
        #self.units

    def from_py_dict(self, dict_graph):
        self.__dict__.update(dict_graph)
        for k, v in dict_graph.items():
            if k == 'nodes':
                self.__dict__[k] = [MIMSNode().from_py_dict(dict_node) for dict_node in v]
        for i, mims_node in enumerate(self.nodes):  # derive index, parents, and children for each node
            mims_node.index = i
            mims_node.set_parents(self.nodes)
            mims_node.set_children(self.nodes)
        return self

    def from_onnx_graph(self, onnx_graph, batch_size):
        # outs = {outp.name: MIMSTensor().from_onnx_tensor_value_info(outp) for outp in self.model.graph.output}
        # inits = {init.name: MIMSTensor().from_onnx_tensor(init) for init in self.model.graph.initializer}
        self.name = onnx_graph.name
        self.nodes = [MIMSNode(index=i).from_onnx_node(onnx_node) for i, onnx_node in enumerate(onnx_graph.node) if onnx_node.op_type != 'Constant']

        const_nodes = [node for node in onnx_graph.node if node.op_type == 'Constant']
        graph_input_mims_tensors = {inp.name: MIMSTensor().from_onnx_tensor_value_info(inp) for inp in onnx_graph.input}  # ONNX V1 style, graph.input contains a list of all input value info protos with sizes
        if const_nodes:  # ONNX V3 style, graph.node contains 'Constant' tensor protos with sizes and data
            for const_node in const_nodes:
                for output_str in const_node.output:
                    input_tensor = copy.deepcopy(const_node.attribute[0].t)  # ATTN! assumes Const nodes have 1 attribute of type TensorProto
                    input_tensor.name = output_str
                    graph_input_mims_tensors.update({input_tensor.name: MIMSTensor().from_onnx_tensor(input_tensor)})

        for mims_node in self.nodes:
            mims_node.set_parents(self.nodes)
            mims_node.set_children(self.nodes)
            mims_node.set_input_mims_tensors(graph_input_mims_tensors)
        for mims_node in self.nodes:  # sets node-output/node-children-input dims; needs to happen after all children and parents have been set
            mims_node.set_unfilled_tensor_dims(batch_size)
        return self

    def to_onnx_graph(self):
        onnx_nodes = []
        graph_input = []
        graph_output = []
        graph_initializer = []
        for node in self.nodes:
            onnx_nodes.append(node.to_onnx_node())
            for i, inp in enumerate(node.inputs):
                if i not in range(len(node.parents)) and inp.val_type:  # only append tensors which are not passed as inputs/outputs; Netron breaks otherwise
                    graph_input.append(inp.to_onnx_tensor_value_info())
                    graph_initializer.append(inp.to_onnx_tensor())
        return helper.make_graph(onnx_nodes, self.name, graph_input, graph_output, graph_initializer)


class MIMSGraphUnit:
    def __init__(self, node_list=None):
        self.list_of_nodes = node_list if node_list else []

    def add_node_list(self, node_list=[]):
        self.list_of_nodes.append(node_list)


class MIMSNode:
    def __init__(self, name=None, op_type=None, attrs=None, inputs=None, outputs=None, parents=None, children=None, index=None):
        self.name = name
        self.op_type = op_type
        self.attributes = attrs if attrs else dict()
        self.inputs = inputs if inputs else []
        self.outputs = outputs if outputs else []
        self.parents = parents if parents else []  # list of 1 or more MIMSOpNode's
        self.children = children if children else []  # list of 1 or more MIMSOpNode's
        self.index = index  # index in serial order

    def from_py_dict(self, dict_node):
        self.__dict__.update(dict_node)
        for k, v in dict_node.items():
            if k == 'inputs':
                self.__dict__[k] = [MIMSTensor().from_py_dict(dict_tensor) for dict_tensor in v]
            elif k == 'outputs':
                self.__dict__[k] = [MIMSTensor().from_py_dict(dict_tensor) for dict_tensor in v]
        return self

    def from_onnx_node(self, onnx_node):
        self.name = onnx_node.name
        self.op_type = onnx_node.op_type
        self.inputs = [MIMSTensor(name=inp) for inp in onnx_node.input]
        self.outputs = [MIMSTensor(name=outp) for outp in onnx_node.output]
        self._set_attr_dict(onnx_node)  # e.g. dictionary { 'strides': [2,2], 'pads': [3,3,3,3] }
        return self

    def to_onnx_node(self):
        input_name_list = [mims_tensor.name for mims_tensor in self.inputs]
        output_name_list = [mims_tensor.name for mims_tensor in self.outputs]
        attrs = [helper.make_attribute(key, val) for key, val in self.attributes.items()]
        onnx_node = helper.make_node(self.op_type, input_name_list, output_name_list, self.name)
        onnx_node.attribute.extend(attrs)
        return onnx_node

    def set_input_mims_tensors(self, graph_input_mims_tensors):
        new_list = []
        for inp in self.inputs:
            if inp.name in graph_input_mims_tensors:
                new_list.append(graph_input_mims_tensors[inp.name])
            else:
                new_list.append(inp)
        self.inputs = new_list

    def set_parents(self, node_list):
        parents = []
        for inp in [mims_tensor.name for mims_tensor in self.inputs]:
            for prev_node in node_list:
                if inp in [mims_tensor.name for mims_tensor in prev_node.outputs]:
                    parents.append(prev_node)
        self.parents = parents

    def set_children(self, node_list):
        children = []
        for outp in [mims_tensor.name for mims_tensor in self.outputs]:
            for next_node in node_list:
                if outp in [mims_tensor.name for mims_tensor in next_node.inputs]:
                    children.append(next_node)
        self.children = children

    def set_unfilled_tensor_dims(self, gpu_cfg):
        dims = self._determine_output_dims(gpu_cfg)
        val_type = self.inputs[0].val_type
        a_trans = 0
        global first_rnn_layer
        global is_attn_based_nw
        if 'attention' in self.op_type.lower():
            is_attn_based_nw = True
        for o in self.outputs:  # set output sizes
            o.dims = dims
            o.val_type = val_type
        for child in self.children:  # set children input sizes
            i = child.parents.index(self)  # determine which child input to set based on whichever parent outputs it
            child.inputs[i].dims = [1] * 4
            if self.op_type.lower() not in ['rnn', 'lstm', 'gru'] and child.op_type.lower() in ['rnn', 'lstm', 'gru']:
                child.inputs[i].dims[RNN_IN_SEQ_LEN_IND] = dims[H_IND] * dims[W_IND]
                child.inputs[i].dims[RNN_IN_BS_IND] = dims[N_IND]
                if first_rnn_layer:
                    child.inputs[i].dims[RNN_IN_SZ_IND] = dims[C_IND]
                    first_rnn_layer = False
                else:
                    child.inputs[i].dims[RNN_IN_SZ_IND] = dims[C_IND] // child.inputs[RNN_IN_WT_IND].dims[RNN_IN_WT_NUM_DIR_IND]
            elif self.op_type.lower() in ['rnn', 'lstm', 'gru'] and child.op_type.lower() in ['rnn', 'lstm', 'gru']:
                child.inputs[i].dims[RNN_IN_SEQ_LEN_IND] = dims[RNN_OUT_SEQ_LEN_IND]
                child.inputs[i].dims[RNN_IN_BS_IND] = dims[RNN_OUT_BS_IND]
                child.inputs[i].dims[RNN_IN_SZ_IND] = dims[RNN_OUT_HIDDEN_SZ_IND] * dims[RNN_OUT_NUM_DIR_IND]
            elif self.op_type.lower() in ['rnn', 'lstm', 'gru'] and child.op_type.lower() not in ['rnn', 'lstm', 'gru', 'gemm']:
                child.inputs[i].dims[N_IND] = dims[RNN_OUT_BS_IND]
                child.inputs[i].dims[C_IND] = dims[RNN_OUT_NUM_DIR_IND] * dims[RNN_OUT_HIDDEN_SZ_IND]
                child.inputs[i].dims[H_IND] = dims[RNN_OUT_SEQ_LEN_IND]
                child.inputs[i].dims[W_IND] = 1
            elif self.op_type.lower() in ['attention'] and child.op_type.lower() not in ['rnn', 'lstm', 'gru', 'gemm', 'attention']:
                child.inputs[i].dims[N_IND] = dims[ATTN_BS_IND]
                child.inputs[i].dims[C_IND] = 1
                child.inputs[i].dims[H_IND] = dims[ATTN_HIDDEN_SZ_IND]
                child.inputs[i].dims[W_IND] = dims[ATTN_SEQ_LEN_IND]
            elif self.op_type.lower() not in ['rnn', 'lstm', 'gru', 'attention'] and 'gemm' in child.op_type.lower():
                if 'transA' in child.attributes:
                    a_trans = child.attributes['transA']
                if is_attn_based_nw:
                    child.inputs[i].dims[0] = dims[H_IND] if a_trans else dims[N_IND] * dims[W_IND]
                    child.inputs[i].dims[1] = dims[N_IND] * dims[W_IND] if a_trans else dims[H_IND]
                else:
                    child.inputs[i].dims[0] = dims[C_IND] * dims[H_IND] * dims[W_IND] if a_trans else dims[N_IND]
                    child.inputs[i].dims[1] = dims[N_IND] if a_trans else dims[C_IND] * dims[H_IND] * dims[W_IND]
            elif self.op_type.lower() in ['rnn', 'lstm', 'gru'] and 'gemm' in child.op_type.lower():
                if 'transA' in child.attributes:
                    a_trans = child.attributes['transA']
                child.inputs[i].dims[0] = dims[RNN_OUT_HIDDEN_SZ_IND] if a_trans else dims[RNN_OUT_BS_IND] * dims[RNN_OUT_SEQ_LEN_IND]
                child.inputs[i].dims[1] = dims[RNN_OUT_BS_IND] * dims[RNN_OUT_SEQ_LEN_IND] if a_trans else dims[RNN_OUT_HIDDEN_SZ_IND]
            else:
                child.inputs[i].dims = dims
                child.inputs[i].val_type = val_type

    def _determine_output_dims(self, gpu_cfg):
        dims = [1] * 4
        sw_opt = gpu_cfg.sw_opt
        num_cu_clusters = 2 if gpu_cfg.hw_cfg.chiplet_mode_en else 1  # Assumption: Dual die chiplet
        if 'parallel' in self.attributes:
            if self.attributes['parallel'] == 'model':
                batch_size = sw_opt.batch_size * sw_opt.mgpu_gpu_count * num_cu_clusters
            else:
                batch_size = sw_opt.batch_size
        else:
            batch_size = sw_opt.batch_size
        if batch_size and not self.parents:
            if self.op_type.lower() in ['rnn', 'lstm', 'gru']:
                self.inputs[0].dims[RNN_IN_BS_IND] = int(batch_size)
            else:
                if self.inputs[0].dims:
                    self.inputs[0].dims[N_IND] = int(batch_size)
                    if 'attention' in self.op_type.lower():
                        self.inputs[1].dims[N_IND] = int(batch_size)
        if self.op_type == 'Transpose':
            perm = self.attributes['perm'] if 'perm' in self.attributes else [0, 1, 2, 3]
            arr = np.array(self.inputs[0].dims)
            dims = list(arr[perm])
        elif self.op_type == 'Conv':
            strides = self.attributes['strides'] if 'strides' in self.attributes else [1, 1]
            pads = self.attributes['pads'] if 'pads' in self.attributes else [0, 0]
            pads = (pads[H_IND], pads[W_IND]) if len(pads) > 2 else pads
            dims[N_IND] = self.inputs[0].dims[N_IND]
            dims[C_IND] = self.inputs[1].dims[F_K_IND]
            dims[H_IND] = floor((abs(self.inputs[0].dims[H_IND] - self.inputs[1].dims[F_R_IND] + 2 * pads[0]) / strides[0]) + 1)
            dims[W_IND] = floor((abs(self.inputs[0].dims[W_IND] - self.inputs[1].dims[F_S_IND] + 2 * pads[1]) / strides[1]) + 1)
        elif 'Pool' in self.op_type:
            strides = self.attributes['strides'] if 'strides' in self.attributes else [1, 1]
            pads = self.attributes['pads'] if 'pads' in self.attributes else [0, 0]
            pads = (pads[H_IND], pads[W_IND]) if len(pads) > 2 else pads
            kernel = self.attributes['kernel_shape'] if 'kernel_shape' in self.attributes else [1, 1]
            dims[N_IND] = self.inputs[0].dims[N_IND]
            dims[C_IND] = self.inputs[0].dims[C_IND]
            dims[H_IND] = floor(((self.inputs[0].dims[H_IND] - kernel[0] + 2 * pads[0]) / strides[0]) + 1)
            dims[W_IND] = floor(((self.inputs[0].dims[W_IND] - kernel[1] + 2 * pads[1]) / strides[1]) + 1)
        elif 'Flatten' in self.op_type:
            dims = [self.inputs[0].dims[0], np.prod(self.inputs[0].dims[1:]), 1, 1]
        elif 'Concat' in self.op_type:
            assert (len(self.inputs) == 2)  # only support for 2 inputs for now
            # Concat involves adding the 'C' channels for two inputs
            dims_0 = copy.deepcopy(self.inputs[0].dims)
            dims_1 = copy.deepcopy(self.inputs[1].dims)
            dims = dims_0
            # Adjust C dimension of inputs if other dimensions dont match
            for ind in range(4):
                if ind == C_IND:
                    continue
                if dims_0[ind] != dims_1[ind]:
                    if dims_0[ind] > dims_1[ind]:
                        c_adjust_factor = dims_0[ind] // dims_1[ind]
                        dims_0[C_IND] *= c_adjust_factor
                        dims[ind] = dims_1[ind]
                    else:
                        c_adjust_factor = dims_1[ind] // dims_0[ind]
                        dims_1[C_IND] *= c_adjust_factor
                        dims[ind] = dims_0[ind]
            dims[C_IND] = dims_0[C_IND] + dims_1[C_IND]
        elif 'Gemm' in self.op_type:
            dims[N_IND] = batch_size if is_attn_based_nw else self.inputs[0].dims[N_IND]
            dims[C_IND] = 1
            dims[H_IND] = self.inputs[1].dims[0] if 'transB' in self.attributes and self.attributes['transB'] else self.inputs[1].dims[1]
            dims[W_IND] = int(self.inputs[0].dims[N_IND] / batch_size) if is_attn_based_nw else 1
        elif self.op_type.lower() in ['rnn', 'lstm', 'gru']:
            dims[RNN_OUT_SEQ_LEN_IND] = self.inputs[0].dims[RNN_IN_SEQ_LEN_IND]
            dims[RNN_OUT_NUM_DIR_IND] = self.inputs[1].dims[RNN_IN_WT_NUM_DIR_IND]
            dims[RNN_OUT_BS_IND] = self.inputs[0].dims[RNN_IN_BS_IND]
            if 'lstm' in self.op_type.lower():
                dims[RNN_OUT_HIDDEN_SZ_IND] = self.inputs[1].dims[RNN_IN_WT_HIDDEN_SZ_IND] // 4
            elif 'gru' in self.op_type.lower():
                dims[RNN_OUT_HIDDEN_SZ_IND] = self.inputs[1].dims[RNN_IN_WT_HIDDEN_SZ_IND] // 3
            else:
                dims[RNN_OUT_HIDDEN_SZ_IND] = self.inputs[1].dims[RNN_IN_WT_HIDDEN_SZ_IND]
        elif 'interaction' in self.op_type.lower():
            dims[N_IND] = self.inputs[0].dims[N_IND]
            dims[C_IND] = self.attributes['factor']
            dims[H_IND] = self.inputs[0].dims[H_IND]
            dims[W_IND] = self.inputs[0].dims[W_IND]
        elif 'embedding' in self.op_type.lower():
            if 'pooling' in self.attributes['op']:
                ntables = self.attributes['ntables']
                if sw_opt.multi_gpu and self.attributes['parallel'] == 'model':
                    ntables = math.ceil(self.attributes['ntables'] / (sw_opt.mgpu_gpu_count * num_cu_clusters))

                dims[N_IND] = batch_size
                dims[C_IND] = ntables
                dims[H_IND] = self.inputs[1].dims[EMB_DIM_IND]
                dims[W_IND] = 1
            else:
                raise NotImplementedError
        elif 'pixelshuffle' in self.op_type.lower():
            shuffle_factor_height = self.attributes['shuffle_factor_height']
            shuffle_factor_width = self.attributes['shuffle_factor_width']
            dims[N_IND] = self.inputs[0].dims[N_IND]
            dims[C_IND] = int(self.inputs[0].dims[C_IND] / (shuffle_factor_height * shuffle_factor_width))
            dims[H_IND] = int(self.inputs[0].dims[H_IND] * shuffle_factor_height)
            dims[W_IND] = int(self.inputs[0].dims[W_IND] * shuffle_factor_width)
        #elif 'attention' in self.op_type.lower():
        #    dims[0] = self.inputs[0].dims[3] * batch_size
        #    dims[1] = self.inputs[0].dims[1]
        #    dims[2] = 1
        #    dims[3] = 1
        else:
            dims = self.inputs[0].dims
        return dims

    def _set_attr_dict(self, node_proto):
        def get_attr_proto_val(attr):
            _map = {'FLOAT': 'f', 'INT': 'i', 'STRING': 's', 'TENSOR': 't', 'GRAPH': 'g', 'FLOATS': 'floats',
                    'INTS': 'ints', 'STRINGS': 'strings', 'TENSORS': 'tensors', 'GRAPHS': 'graphs'}
            # node_dict = json_format.MessageToDict(attr)
            attr_proto_dict = protobuf_to_dict(attr)
            for attr_type in list(_map.values()):
                if attr_type in attr_proto_dict:
                    return attr_proto_dict[attr_type]
            return None

        op_attr = dict()
        if hasattr(node_proto, 'attribute'):
            for attr_proto in node_proto.attribute:
                op_attr[attr_proto.name] = get_attr_proto_val(attr_proto)
        self.attributes = op_attr


class MIMSTensor:
    def __init__(self, name=None, val_type=None, dims=None, vals=None):
        self.name = name
        self.val_type = val_type
        self.dims = dims if dims else []
        self.vals = vals if vals else []

    def from_py_dict(self, dict_tensor):
        self.__dict__.update(dict_tensor)
        return self

    def from_onnx_tensor_value_info(self, tvi):
        self.name = tvi.name
        self.dims = [dim.dim_value for dim in tvi.type.tensor_type.shape.dim]
        self.val_type = tvi.type.tensor_type.elem_type
        return self

    def from_onnx_tensor(self, t):
        self.name = t.name
        self.dims = list(t.dims)
        self.val_type = t.data_type
        self.vals = [str(x) for x in numpy_helper.to_array(t).tolist()]
        return self

    def to_onnx_tensor_value_info(self):
        return helper.make_tensor_value_info(self.name, self.val_type, self.dims)

    def to_onnx_tensor(self):  # empty tensor
        return helper.make_tensor(self.name, self.val_type, self.dims, self.vals)


def load_mims(mims_path):
    return MIMSModel().load(mims_path)


def load_onnx(onnx_path):
    onnx_model = ModelProto()
    with open(onnx_path, 'rb') as model_f:
        content = model_f.read()
        onnx_model.ParseFromString(content)
    return onnx_model


def trim_onnx(onnx_model):   # IMPORTANT: deletes raw data from model
    # onnx proto spec: https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
    def clear_tensor(tensor_proto_obj):
        data_types = ['int32_data', 'int64_data', 'uint64_data', 'float_data', 'double_data', 'string_data', 'raw_data']
        for dt in data_types:
            if hasattr(tensor_proto_obj, dt) and getattr(tensor_proto_obj, dt) != []:
                if dt == 'raw_data':
                    tensor_proto_obj.raw_data = b''
                else:
                    del getattr(tensor_proto_obj, dt)[:]  # clears a repeated proto field (list)
                    return

    do_not_trim = []
    if hasattr(onnx_model.graph, 'node'):
        for node in onnx_model.graph.node:
            if node.op_type == 'Reshape':
                do_not_trim.append(node.input[1])  # second input is the shape array

    if hasattr(onnx_model.graph, 'initializer'):
        for initializer_tensor in onnx_model.graph.initializer:
            clear_tensor(initializer_tensor)
    if hasattr(onnx_model.graph, 'node'):
        # print(onnx_model.graph.node[1].attribute[0].t.float_data[0])
        for node in onnx_model.graph.node:
            if node.op_type == 'Constant':
                if node.output[0] in do_not_trim:  # assumes constant node only has one output
                    continue
                if hasattr(node, 'attribute'):
                    for attr in node.attribute:
                        if hasattr(attr, 't') and str(attr.t):
                            clear_tensor(attr.t)
                        elif hasattr(attr, 'tensors') and len(attr.tensors) > 0:
                            for tensor in attr.tensors:
                                clear_tensor(tensor)
                        else:
                            for repeated_attr in ['ints', 'floats', 'strings']:  # note 'g' and 'graphs' are not currently handled
                                if hasattr(attr, repeated_attr) and len(getattr(attr, repeated_attr)) > 0:
                                    del getattr(attr, repeated_attr)[:]  # clears a repeated proto field (list)
    return onnx_model


def load_onnx_to_mims(onnx_path, batch_size=None):
    trimmed_path_string = path.splitext(onnx_path)[0] + '.trimmed.onnx'
    if path.isfile(trimmed_path_string):    # already trimmed
        onnx_model = load_onnx(trimmed_path_string)
    else:
        onnx_model = trim_onnx(load_onnx(onnx_path))
        dump_onnx(onnx_model, trimmed_path_string)  # save trimmed model for consecutive loads
    return MIMSModel().from_onnx_model(onnx_model, batch_size)


def dump_onnx(onnx_model, onnx_path):
    # checker.check_model(onnx_model)
    with open(onnx_path, 'wb') as onnx_f:
        onnx_f.write(onnx_model.SerializeToString())


def load_onnx_from_json(json_path):
    with open(json_path, 'r') as json_f:
        # model = json_format.ParseDict(json_f.read(), ModelProto())
        return json_format.Parse(json_f.read(), ModelProto())


def dump_onnx_to_json(onnx_model, json_path):
    with open(json_path, 'w') as json_f:
        # dict = json_format.MessageToDict(self.model)
        json_f.write(json_format.MessageToJson(onnx_model))


# def test_func(self):
#     node_list = self.mims_model.graph.nodes
#     onnx_model = self.mims_model.to_onnx_model()
#     # checker.check_model(onnx_model)
#     with open('test_onnx_mims.pb', 'wb') as onnx_f:
#         onnx_f.write(onnx_model.SerializeToString())
#
#     self.mims_model.dump('test_mims.ms')
#     mims_model_from_txt = MIMSModel().load('test_mims.ms')
#     onnx_model_from_txt = mims_model_from_txt.to_onnx_model()
#     with open('test_onnx_mims_from_txt.pb', 'wb') as onnx_f2:
#         onnx_f2.write(onnx_model_from_txt.SerializeToString())
