import ast
from math import ceil


class HWConfig:
    def __init__(self, q_table_view):
        table_model = q_table_view.model()
        if table_model:
            self.range_attr = None
            for r in range(table_model.rowCount()):
                attr = table_model.item(r, 0).text().lower()
                val_str = table_model.item(r, 1).text().lower()
                if ':' in val_str:  # process range attribute
                    if self.range_attr:  # range attribute already exists
                        print("Warning: Range iteration only allowed for one parameter. "
                              "Range for \"{}\" will not be iterated over.".format(self.range_attr))
                    self.range_attr = attr
                    val = [ast.literal_eval(x) for x in val_str.split(':')]
                elif 'x' in val_str and ';' not in val_str:
                    val = [int(x) for x in val_str.split('x')]
                elif 'x' in val_str and ';' in val_str:
                    val = []
                    for split_str in val_str.split(';'):
                        val.append([int(x) for x in split_str.split('x')])
                else:
                    val = ast.literal_eval(val_str)
                setattr(self, attr, val)
            self._set_derived_attr()

    def _set_derived_attr(self):
        setattr(self, 'num_cu_util', self.num_cu)

        setattr(self, 'ext_link_bw_curve', {})  # Define external links (GMI/XGMI) BW curve w.r.t. message size
        self.ext_link_bw_curve = {2 ** (22 - x): 1 - 0.05 * x for x in range(4)}  # Assumption: message size >= 4MB achieves peak rate. For message size < 4MB the BW starts dropping
        self.ext_link_bw_curve.update({2 ** (22 - x): 0.8 / (2 ** (x - 4)) for x in range(4, 8)})

        setattr(self, 'l2_read_bw', 0)
        if isinstance(self.l2_read_buses_per_se, list):
            self.l2_read_bw = [x * self.l2_read_bus_width * self.num_se_per_cluster for x in self.l2_read_buses_per_se]
        elif isinstance(self.l2_read_bus_width, list):
            self.l2_read_bw = [self.l2_read_buses_per_se * x * self.num_se_per_cluster for x in self.l2_read_bus_width]
        else:
            self.l2_read_bw = self.l2_read_buses_per_se * self.l2_read_bus_width * self.num_se_per_cluster

        setattr(self, 'l2_write_bw', 0)
        if isinstance(self.l2_write_buses_per_se, list):
            self.l2_write_bw = [x * self.l2_write_bus_width * self.num_se_per_cluster for x in self.l2_write_buses_per_se]
        elif isinstance(self.l2_write_bus_width, list):
            self.l2_write_bw = [self.l2_write_buses_per_se * x * self.num_se_per_cluster for x in self.l2_write_bus_width]
        else:
            self.l2_write_bw = self.l2_write_buses_per_se * self.l2_write_bus_width * self.num_se_per_cluster

        setattr(self, 'hbm_bw', 0)
        if isinstance(self.hbm_freq, list):
            self.hbm_bw = [x * (self.hbm_bus_width // 8) * self.hbm_efficiency / self.gpu_freq for x in self.hbm_freq]
        elif isinstance(self.hbm_bus_width, list):
            self.hbm_bw = [self.hbm_freq * (x // 8) * self.hbm_efficiency / self.gpu_freq for x in self.hbm_bus_width]
        elif isinstance(self.gpu_freq, list):
            self.hbm_bw = [self.hbm_freq * (self.hbm_bus_width // 8) * self.hbm_efficiency / x for x in self.gpu_freq]
        else:
            self.hbm_bw = self.hbm_freq * (self.hbm_bus_width // 8) * self.hbm_efficiency / self.gpu_freq

        setattr(self, 'total_gmi_bw', 0)
        if isinstance(self.num_gmi_links, list):
            self.total_gmi_bw = [x * self.gmi_link_bw / self.gpu_freq for x in self.num_gmi_links]
        elif isinstance(self.gmi_link_bw, list):
            self.total_gmi_bw = [self.num_gmi_links * x / self.gpu_freq for x in self.gmi_link_bw]
        elif isinstance(self.gpu_freq, list):
            self.total_gmi_bw = [self.num_gmi_links * self.gmi_link_bw / x for x in self.gpu_freq]
        else:
            self.total_gmi_bw = self.num_gmi_links * self.gmi_link_bw / self.gpu_freq

    def set_vals_from_range_attrs(self, idx, full_hw_cfg):  # selects value at idx from full_cfg's range_attr and sets present cfg's attrs
        if self.range_attr == 'num_gmi_links':
            setattr(self, 'total_gmi_bw', full_hw_cfg.total_gmi_bw[idx])
        elif self.range_attr == 'l2_read_bus_width':
            setattr(self, 'l2_read_bus_width', full_hw_cfg.l2_read_bus_width[idx])
        elif self.range_attr == 'fp16_dl_macs_per_cu':
            setattr(self, 'fp16_dl_macs_per_cu', full_hw_cfg.fp16_dl_macs_per_cu[idx])
        elif self.range_attr == 'gpu_freq':
            setattr(self, 'gpu_freq', full_hw_cfg.gpu_freq[idx])
            setattr(self, 'hbm_bw', full_hw_cfg.hbm_bw[idx])
            setattr(self, 'total_gmi_bw', full_hw_cfg.total_gmi_bw[idx])
            if idx > 0:
                latency = ceil(full_hw_cfg.l2_miss_latency * full_hw_cfg.gpu_freq[idx] / full_hw_cfg.gpu_freq[0])
                setattr(self, 'l2_miss_latency', latency)

