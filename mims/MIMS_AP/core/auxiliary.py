# Copyright (c) 2017-2018 Advanced Micro Devices, Inc.
# All Rights Reserved.
# Advanced Micro Devices Proprietary and Confidential.
from os import path, listdir, environ, lstat
from sys import platform
from math import ceil
import errno
from core.const import *
ERROR_INVALID_NAME = 123  # windows specific error code\


def print_node_info(op, op_id, num_nodes):
    if op.op_type == 'Conv':
        name = op.op_type + str(op.inputs[FILT_IND].dims[F_R_IND]) + 'x' + str(op.inputs[FILT_IND].dims[F_S_IND])
    else:
        name = op.op_type
    print(' node:', op_id, '/', num_nodes, ' - ', name)


def prod(array):
    val = 1
    for i in array:
        val *= i
    return val


# Get data type size in bytes
def get_dt_size(sw_opt):
    dt_size = 1
    if sw_opt.fp16_inputs:
        dt_size = 2
    elif sw_opt.fp32_inputs:
        dt_size = 4
    elif sw_opt.fp64_inputs:
        dt_size = 8
    elif sw_opt.bf16_inputs: #Ashish added
        dt_size = 2 #Ashish added
    return dt_size


def get_dl_macs_per_cu(hw_cfg, sw_opt):
    if sw_opt.fp16_inputs:
        dl_macs_per_cu = hw_cfg.fp16_dl_macs_per_cu
    elif sw_opt.fp32_inputs:
        dl_macs_per_cu = hw_cfg.fp32_dl_macs_per_cu
    elif sw_opt.fp64_inputs:
        dl_macs_per_cu = hw_cfg.fp64_dl_macs_per_cu
    elif sw_opt.bf16_inputs: #Ashish added
        dl_macs_per_cu = hw_cfg.bf16_dl_macs_per_cu #Ashish added
    else:  # INT8
        dl_macs_per_cu = hw_cfg.int8_dl_macs_per_cu
    return dl_macs_per_cu


def get_native_macs_per_cu(hw_cfg, sw_opt):
    if sw_opt.fp16_inputs:
        native_macs_per_cu = hw_cfg.legacy_fp16_macs_per_cu
    elif sw_opt.fp32_inputs:
        native_macs_per_cu = hw_cfg.legacy_fp32_macs_per_cu
    elif sw_opt.fp64_inputs:
        native_macs_per_cu = hw_cfg.legacy_fp64_macs_per_cu
    elif sw_opt.bf16_inputs: #Ashish added
        native_macs_per_cu = hw_cfg.legacy_fp16_macs_per_cu #Ashish added
    else:  # INT8
        native_macs_per_cu = hw_cfg.legacy_int8_macs_per_cu
    return native_macs_per_cu


def get_act_hit_rates(activation_size, weights_size, hw_config, sw_opt, cache_scale=0.8):
    l3_hit_rate_act = 0.0
    if hw_config.num_cu_clusters > 1 and not hw_config.l3_per_cluster:
        l2_hit_rate_act = 0.0
        if activation_size <= hw_config.l3_size:
            l3_hit_rate_act = HIGH_L3_HIT_RATE
        else:
            l3_hit_rate_act = (hw_config.l3_size * cache_scale - weights_size) / activation_size
    else:
        if activation_size <= hw_config.l2_size:
            l2_hit_rate_act = HIGH_L2_HIT_RATE
        else:
            if not sw_opt.disable_l2_optimization: #Ashish added
                l2_hit_rate_act = (hw_config.l2_size * cache_scale - weights_size) / activation_size
                overflow_act_size = activation_size - (hw_config.l2_size * cache_scale - weights_size)
                if hw_config.l3_size > 1 and hw_config.l3_per_cluster:
                    if overflow_act_size <= hw_config.l3_size:
                        l3_hit_rate_act = HIGH_L3_HIT_RATE
                    else:
                        if hw_config.l3_optimization_enabled:
                            l3_hit_rate_act = hw_config.l3_size * cache_scale / overflow_act_size
                        else:
                            l3_hit_rate_act = 0.0
            else: #Ashish added
                l2_hit_rate_act = 0.0 #Ashish added

    return l2_hit_rate_act, l3_hit_rate_act


def query_resources(target_dir_path, ext=None):
    return [f for f in listdir(target_dir_path)
            if (ext and path.isfile(path.join(target_dir_path, f)) and f.endswith(ext))
            or (not ext and path.isdir(path.join(target_dir_path, f)))]


def is_pathname_valid(pathname: str) -> bool:
    try:
        if not isinstance(pathname, str) or not pathname:
            return False
        _, pathname = path.splitdrive(pathname)
        root_dirname = environ.get('HOMEDRIVE', 'C:') if platform == 'win32' else path.sep
        assert path.isdir(root_dirname)
        root_dirname = root_dirname.rstrip(path.sep) + path.sep

        for pathname_part in pathname.split(path.sep):
            try:
                lstat(root_dirname + pathname_part)
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    except TypeError as exc:
        return False
    else:
        return True
