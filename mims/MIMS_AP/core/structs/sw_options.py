from PyQt5.QtWidgets import *
import ast


class SWOptions:
    _gui_attr_map = {'cBoxModels': 'model',  # stores the mapping between gui element names and SWOptions attribute names
                     'chBoxFusion': 'kernel_fusion',
                     'chBoxTraining': 'training',
                     'cBoxDataFormat': 'data_format',
                     'txtBatchSize': 'batch_size',
                     'chBoxOptimizeGraph': 'optimize_graph',
                     'chBoxCalibrated': 'useCalibrated',
                     'chBoxRL': 'rl',
                     'txtTmax': 'rl_tmax',
                     'txtCpuThreads': 'rl_cpu_threads',
                     'chBoxMGPU': 'multi_gpu',
                     'txtGPUCount': 'mgpu_gpu_count',
                     'txtXGMILinksPerGPU': 'mgpu_xgmi_link_count',
                     'txtXGMI_BW': 'mgpu_xgmi_link_bw',
                     'txtXGMI_BWEfficiency': 'mgpu_xgmi_link_bw_eff',
                     'cBoxAlgorithm': 'mgpu_all_reduce_algo',
                     'cBoxTopology': 'mgpu_topology',
                     'txtStartupLatency': 'mgpu_startup_lat',
                     'txtNumGradientBits': 'mgpu_num_grad_bits',
                     'txtMultiNode': 'mgpu_multi_node',
                     'txtMultiNodeBW': 'mgpu_multi_node_link_bw',
                     'txtMultiNodeLinks': 'mgpu_multi_node_link_count',
                     'txtMNStartupLatency': 'mgpu_mn_startup_lat',
                     'chBoxSDMA_Xfer': 'mgpu_sdma_xfer',
                     'chBoxWeakScaling': 'mgpu_weak_scaling',
                     'chBoxDisableL2Opti': 'disable_l2_optimization',
                     'chBoxDisableTrailOpti': 'disable_trail_optimization',
                     'chBoxDisableMainKPart': 'disable_kpartitioning',
                     'txtPerfThreshold': 'PerfThreshold',
                     'chBoxDisableTrailKRepart': 'disable_trail_repart',
                     'chBoxUserDefMT': 'userDefMT',
                     'chBoxNoTrail': 'disable_trail',
                     'txtMainMT0': 'a_blocks',
                     'txtMainMT1': 'b_blocks',
                     'txtTrailMT0': 'trail_a_blocks',
                     'txtTrailMT1': 'trail_b_blocks',
                     'chBoxABinLDS': 'ABinLDS',
                     'chBoxTileGranularity': 'TileGranularity'}


    def __init__(self, sw_opt_widget, attr_default_val_map=None, prnn_opt_en=False):
        self.prnn_opt_en = prnn_opt_en
        self._attr_val_map = {**{self._gui_attr_map[ch.objectName()]: ch.isChecked() for ch in sw_opt_widget.findChildren(QCheckBox)},
                              **{self._gui_attr_map[ch.objectName()]: ast.literal_eval(ch.text()) for ch in sw_opt_widget.findChildren(QLineEdit)},
                              **{self._gui_attr_map[ch.objectName()]: ch.currentText() for ch in sw_opt_widget.findChildren(QComboBox)}}
        self._attr_default_val_map = attr_default_val_map if attr_default_val_map else self._attr_val_map.copy()  # if no default map, use attr_val_map
        for a, v in self._attr_val_map.items():
            setattr(self, a, v)  # create attribute

        setattr(self, 'fp16_inputs', getattr(self, 'data_format') == 'FP16')
        setattr(self, 'bf16_inputs', getattr(self, 'data_format') == "BF16") #Ashish added
        setattr(self, 'fp32_inputs', getattr(self, 'data_format') == 'FP32')
        setattr(self, 'fp64_inputs', getattr(self, 'data_format') == 'FP64')
        setattr(self, 'int8_inputs', getattr(self, 'data_format') == 'INT8')
        setattr(self, 'total_xgmi_bw', 0)
        setattr(self, 'total_mgpu_multi_node_bw', 0)
        setattr(self, 'mgpu_multi_node_mgpu_all_reduce_algo', 'default')

    def get_attr_default_val_map(self):
        return self._attr_default_val_map

    def reset_to_defaults(self):
        has_changed = False
        for attr_name in self._gui_attr_map.values():
            curr_val = getattr(self, attr_name)
            default_val = self._attr_default_val_map[attr_name]
            if curr_val is not default_val:
                has_changed = True
            setattr(self, attr_name, default_val)
        return has_changed

    def is_dirty(self):
        has_changed = False
        for attr_name in self._gui_attr_map.values():
            curr_val = getattr(self, attr_name)
            default_val = self._attr_default_val_map[attr_name]
            if curr_val is not default_val:
                has_changed = True
        # return has_changed
        return False
