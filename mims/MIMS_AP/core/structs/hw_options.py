from PyQt5.QtWidgets import *
import ast


class HWOptions:
    _gui_attr_map = {'chBoxL2Bcast': 'l2_bcast_en',    # stores the mapping between gui element names and SWOptions attribute names
                     'txtL2BcastCluster': 'l2_bcast_cluster_sz',
                     'chBoxXCu': 'cross_cu_share_en',
                     'chBoxStackedMem': 'stacked_mem_en',
                     'txCacheScale': 'cache_scale',
                     'txBWScale': 'BW_scale',
                     'chBoxGAtomics': 'global_atomics'}

    def __init__(self, hw_opt_widget, attr_default_val_map=None, worst_case_perf=False, l2_bcast_both_dims=True): #, global_atomics=True):
        self.worst_case_perf = worst_case_perf
        self.l2_bcast_both_dims = l2_bcast_both_dims
        #self.global_atomics = global_atomics
        self._attr_val_map = {**{self._gui_attr_map[ch.objectName()]: ch.isChecked() for ch in hw_opt_widget.findChildren(QCheckBox)},
                              **{self._gui_attr_map[ch.objectName()]: ast.literal_eval(ch.text()) for ch in hw_opt_widget.findChildren(QLineEdit)}}
        self._attr_default_val_map = attr_default_val_map if attr_default_val_map else self._attr_val_map.copy()  # if no default map, use attr_val_map
        for a, v in self._attr_val_map.items():
            setattr(self, a, v)  # create attribute

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
        return has_changed
