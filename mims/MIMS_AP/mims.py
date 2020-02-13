#!python

# Copyright (c) 2017-2018 Advanced Micro Devices, Inc.
# All Rights Reserved.
# Advanced Micro Devices Proprietary and Confidential.
import argparse
import glob
import inspect
import multiprocessing
from distutils.util import strtobool
from os import remove, rename, mkdir
from sys import argv, exit
from time import time

import netron
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import *  # QApplication, QWidget, QMainWindow, QSplitter

import core.dnnparser as dnnparser
from core.auxiliary import *
from core.backend import *
from core.results import *
from core.graph_optimizer import GraphResultOptimizer
from core.structs.full_config import FullConfig
from gui.guiMainWindow import Ui_MainWindow
from gui.guiDialog import Ui_Dialog
from ci_tests import diff
import traceback

SWAPNIL = True

class MyDialog(QDialog):
    def __init__(self, parent, win_title, desc_str):
        QDialog.__init__(self, flags=Qt.WindowCloseButtonHint | Qt.WindowTitleHint)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle(win_title)
        self.ui.lblDescription.setText(desc_str)

    def set_error(self, err_str):
        if err_str:
            self.ui.lblError.setText(err_str)

    def get_text(self):
        return self.ui.txtInput.text()


class MyMainWindow(QMainWindow):
    def __init__(self, **kwargs):
        QMainWindow.__init__(self, **kwargs)
        self.ui = Ui_MainWindow()
        self.webViewPreview = QWebEngineView()
        self.splash_path = 'file:///res/netron_splash.html'
        self.configs_dir, self.models_dir, self.presets_dir, self.out_dir = 'configs', 'models', 'presets', 'output'
        self.setup_gui()
        self.setup_signals_and_slots()

        self.server_proc = None
        self.work_thread = None
        self.base_conf_dict = None
        self.t0 = 0

        self.get_config_list()
        self.get_dnn_model_list()
        self.get_preset_list()

        self.default_cfg = FullConfig(self.ui.tblBaseConf, self.ui.gBoxHwOpt, self.ui.gBoxSWOpt, default_cfg=None)

        if SWAPNIL:  # TODO: REMOVE BEFORE RELEASE
            #print("======ATTN: Running in SWAPNIL mode======")
            self.ui.cBoxPresets.setCurrentIndex(1)

    def setup_gui(self):
        self.ui.setupUi(self)
        # self.ui.splitter.setSizes([500, 50])
        # self.ui.splitter.setSizes([30, 500])
        self.webViewPreview.load(QUrl(self.splash_path))
        self.ui.gBoxPreviewModel.layout().addWidget(self.webViewPreview)
        self.ui.wgtExtOpt.setCurrentIndex(0)
        self.ui.wgtExtOpt_row2.setCurrentIndex(0)
        self.ui.wgtExtOpt_row3.setCurrentIndex(0)

        if not path.isdir(self.presets_dir):
            mkdir(self.presets_dir)
        if not path.isdir(self.out_dir):
            mkdir(self.out_dir)

        self.ui.btnNewConf.setIcon(QIcon(QPixmap("res/icons8-file-48.png")))
        self.ui.btnSaveConf.setIcon(QIcon(QPixmap("res/icons8-save-50.png")))
        self.ui.btnSaveAsConf.setIcon(QIcon(QPixmap("res/icons8-save-as-50.png")))
        self.ui.btnDelConf.setIcon(QIcon(QPixmap("res/icons8-trash-can-40.png")))
        self.ui.btnSavePreset.setIcon(QIcon(QPixmap("res/icons8-save-50.png")))
        self.ui.btnSaveAsPreset.setIcon(QIcon(QPixmap("res/icons8-save-as-50.png")))
        self.ui.btnDelPreset.setIcon(QIcon(QPixmap("res/icons8-trash-can-40.png")))

    def setup_signals_and_slots(self):
        def show_btn_options(status):
            opt_dict = {"General": (0, self.ui.btnGeneral), "RL": (1, self.ui.btnRL), "MGPU": (2, self.ui.btnMGPU)}  # (wgt_page_idx, QButton)
            sender_btn = self.sender()
            for _, btn in opt_dict.values():
                font = btn.font()
                font.setBold(sender_btn.text() == btn.text())
                btn.setFont(font)
            self.ui.wgtExtOpt.setCurrentIndex(opt_dict[sender_btn.text()][0])
            self.ui.wgtExtOpt_row2.setCurrentIndex(opt_dict[sender_btn.text()][0])
            self.ui.wgtExtOpt_row3.setCurrentIndex(opt_dict[sender_btn.text()][0])
            #if sender_btn.text() == "Multi-GPU" or sender_btn.text() == "General":
            self.ui.wgtExtOpt_row2.show()
            self.ui.wgtExtOpt_row3.show()
            #    self.ui.wgtExtOpt_row2.setCurrentIndex(opt_dict[sender_btn.text()][0])
            #    self.ui.wgtExtOpt_row2.show()
            #else:
            #    self.ui.wgtExtOpt_row2.hide()

        self.ui.cBoxConfs.currentTextChanged.connect(self.load_base_conf_data)
        self.ui.cBoxModels.currentTextChanged.connect(self.load_dnn_preview)
        self.ui.cBoxPresets.currentTextChanged.connect(self.load_preset)
        self.ui.gBoxPreviewModel.toggled.connect(self.toggle_preview)
        self.ui.btnAddRow.clicked.connect(self.insert_row_action)
        self.ui.btnRemRow.clicked.connect(self.remove_row_action)
        self.ui.btnNewConf.clicked.connect(self.new_conf_action)
        self.ui.btnSaveConf.clicked.connect(self.save_conf_action)
        self.ui.btnSaveAsConf.clicked.connect(self.save_as_conf_action)
        self.ui.btnDelConf.clicked.connect(self.del_conf_action)
        self.ui.btnSavePreset.clicked.connect(self.save_preset_action)
        self.ui.btnSaveAsPreset.clicked.connect(self.save_as_preset_action)
        self.ui.btnDelPreset.clicked.connect(self.del_preset_action)
        self.ui.btnGeneral.clicked.connect(show_btn_options)
        self.ui.btnRL.clicked.connect(show_btn_options)
        self.ui.btnMGPU.clicked.connect(show_btn_options)
        self.ui.btnBrowse.clicked.connect(self.show_browse_dialog)
        self.ui.btnRun.clicked.connect(self.initiate_mims_run)
        #self.ui.btnRun.clicked.connect(self.run_mims)

    def derive_out_file_name(self, sel_model, sys_cfg, gpu_cfg, silent_mode=False):
        if self.ui.txtPath.text():
            self.out_dir = self.ui.txtPath.text()
            if not path.isdir(self.out_dir):
                mkdir(self.out_dir)
        s1 = path.join(self.out_dir, sel_model)
        s2 = path.splitext(self.ui.cBoxConfs.currentText())[0]
        s3 = '' if (silent_mode and sel_model == 'gemm') else '-bs{}'.format(sys_cfg.sw_opt.batch_size)
        s4 = '-ngpu{}'.format(gpu_cfg.sw_opt.mgpu_gpu_count*gpu_cfg.sw_opt.mgpu_multi_node) if gpu_cfg.sw_opt.multi_gpu else ''
        s5 = '-{}'.format(self.ui.cBoxTopology.currentText()) if gpu_cfg.sw_opt.multi_gpu else ''
        fname = '{}-results-{}{}{}{}{}.xlsx'.format(s1, s2, s3, s4, s5, self.ui.txtPostfix.text())
        if path.isfile(fname) and not SWAPNIL:
            try:
                outfile = open(fname, "r+")
                outfile.close()
                remove(fname)  # remove old results file
            except IOError:
                print("ERROR: Could not access \"{}\"! Please close Excel! \nAborting MIMS run...".format(fname))
                fname = None
        return fname

    def initiate_mims_run(self):
        self.ui.btnRun.setDisabled(True)
        #--- runner thread
        self.work_thread = MyWorkThread()  # ATTN: THREAD NEEDS TO BE INIT HERE AN NOT IN CONSTRUCTOR (otherwise it gets called repeatedly (bug))
        self.work_thread.doWork.connect(self.run_mims, Qt.DirectConnection)
        self.work_thread.finished.connect(lambda: self.ui.btnRun.setDisabled(False))
        self.work_thread.start()

    def run_mims(self, silent_mode=False):
        if self.ui.cBoxModels.currentIndex() == 0 and not silent_mode:
            return ''
        try:
            sys_cfg = FullConfig(self.ui.tblBaseConf, self.ui.gBoxHwOpt, self.ui.gBoxSWOpt, self.default_cfg, cfg_type='sys')
            gpu_cfg_orig = FullConfig(self.ui.tblBaseConf, self.ui.gBoxHwOpt, self.ui.gBoxSWOpt, self.default_cfg, cfg_type='gpu')
            sel_model = self.ui.cBoxModels.currentText()

            model_root_dir = path.join(self.models_dir, sel_model)
            model_files_found = list(set(glob.glob(path.join(model_root_dir, '*.onnx'))).difference(
                set(glob.glob(path.join(model_root_dir, '*.trimmed.onnx')))))
            model_file_no_ext = path.splitext(model_files_found[0])[0]  # pick first file in list

            if path.isfile(model_file_no_ext + '.ms'):
                mims_model = dnnparser.load_mims(model_file_no_ext + '.ms')
                for mims_node in mims_model.graph.nodes:  # sets node-output/node-children-input dims; needs to happen after all children and parents have been set
                    mims_node.set_unfilled_tensor_dims(gpu_cfg_orig)
            else:
                mims_model = dnnparser.load_onnx_to_mims(model_files_found[0], gpu_cfg_orig)
                mims_model.dump(model_file_no_ext + '.ms')  # so that next time load directly from .ms
            #create_graph_unit_list(mims_model.graph.nodes)
            # onnx.helper.printable_graph(model.graph)

            results_file_name = self.derive_out_file_name(sel_model, sys_cfg, gpu_cfg_orig, silent_mode)
            if not results_file_name:
                return
            result_writer = ResultWriter(results_file_name, sys_cfg)

            # Get results for all range values -- run mims for modified config
            # is_base_result = (not gpu_cfg_orig.is_dirty())
            is_base_result = True
            num_cfg_iter = len(sys_cfg.hw_cfg.range_attr) if sys_cfg.hw_cfg.range_attr else 1
            for cfg_i in range(num_cfg_iter):  # Run MIMS backend multiple times on parameters which have range
                gpu_cfg_orig.hw_cfg.set_vals_from_range_attrs(cfg_i, sys_cfg.hw_cfg)
                total_cycles, fwd_cycles, bwd_cycles, transfer_cycles_exposed, fwd_res, bwd_res = 0, 0, 0, 0, [], []
                num_allreduce_iter = 1 if not (gpu_cfg_orig.sw_opt.multi_gpu or gpu_cfg_orig.hw_cfg.chiplet_mode_en) else ALLREDUCE_ITER_LIMIT  # hardcoded number of iterations to try
                for allreduce_i in range(1):#num_allreduce_iter):  # Run MIMS backend for different num_cu for allReduce operation
                    gpu_cfg = copy.deepcopy(gpu_cfg_orig)  # a number of attributes get modified in BW pass for allreduce, so start fresh every iteration
                    allreduce_config = gpu_cfg.gen_allreduce_config(num_allreduce_cu=ALLREDUCE_NUM_CU_FACTOR*allreduce_i)
                    mgpu_backend = MGPUBackend(gpu_cfg, allreduce_config)
                    runner = MIMSBackend(mims_model.graph.nodes, gpu_cfg, allreduce_config, mgpu_backend)
                    if allreduce_i == 0:
                        fwd_res = runner.eval_graph()
                    bwd_res_i, bwd_cycles_i, transfer_cycles_exposed_i = [], 0, 0
                    if gpu_cfg.sw_opt.training:
                        bwd_res_i = runner.eval_graph(direction='backward')
                        graph_optimizer = GraphResultOptimizer(fwd_res, bwd_res_i, gpu_cfg, mgpu_backend)
                        fwd_cycles, bwd_cycles_i, transfer_cycles_exposed_i, fwd_res, bwd_res_i = graph_optimizer.get_total_nn_cycles()
                        cycles_i = fwd_cycles + bwd_cycles_i
                        if not total_cycles or total_cycles > cycles_i:  # set current as optimal results
                            total_cycles, fwd_cycles, bwd_cycles, transfer_cycles_exposed, bwd_res = \
                                cycles_i, fwd_cycles, bwd_cycles_i, transfer_cycles_exposed_i, bwd_res_i
                    else:
                        graph_optimizer = GraphResultOptimizer(fwd_res, bwd_res_i, gpu_cfg, mgpu_backend)
                        fwd_cycles, bwd_cycles_i, transfer_cycles_exposed, fwd_res, bwd_res_i = graph_optimizer.get_total_nn_cycles()
                    if (allreduce_config.hw_cfg.num_cu_util - ALLREDUCE_NUM_CU_FACTOR) == 0:
                        break
                # Write out results
                is_last_iter = (cfg_i == num_cfg_iter - 1 and is_base_result)
                result_writer.write(fwd_res, 'forward', is_base_result, cfg_i, is_last_iter and not gpu_cfg_orig.sw_opt.training, fwd_cycles)
                if gpu_cfg_orig.sw_opt.training:
                    result_writer.write(bwd_res, 'backward', is_base_result, cfg_i, is_last_iter, bwd_cycles, transfer_cycles_exposed)
            # Get base results -- run mims for default config as a baseline
            # if gpu_cfg.reset_to_defaults() and num_cfg_iter == 1:
            #     allreduce_config = gpu_cfg.gen_allreduce_config()  # TODO: IS THIS REALLY NEEDED??
            #     runner = MIMSBackend(mims_model.graph.nodes, gpu_cfg, allreduce_config)
            #     nw_base_results_fwd = runner.traverse_graph()
            #     result_writer.write(nw_base_results_fwd, is_last_iter=not gpu_cfg.sw_opt.training)
            #     if gpu_cfg.sw_opt.training:
            #         nw_base_results_bwd = runner.traverse_graph(direction='backward')
            #         result_writer.write(nw_base_results_bwd, direction='backward', is_last_iter=True)
        except:
            traceback.print_exc()

        if not SWAPNIL:  # TODO: REMOVE BEFORE RELEASE
            return results_file_name

    def get_config_list(self):
        cfgs = query_resources(self.configs_dir, 'csv')
        for cfg in cfgs:
            self.ui.cBoxConfs.addItem(cfg)
            index = self.ui.cBoxConfs.findText(cfg)
            self.ui.cBoxConfs.setItemData(index, cfg, Qt.ToolTipRole)

    def get_dnn_model_list(self):
        dnns = query_resources(self.models_dir)
        for dnn in dnns:
            self.ui.cBoxModels.addItem(dnn)

    def get_preset_list(self):
        presets = query_resources(self.presets_dir, 'ini')
        for preset in presets:
            self.ui.cBoxPresets.addItem(preset)
            index = self.ui.cBoxPresets.findText(preset)
            self.ui.cBoxPresets.setItemData(index, preset, Qt.ToolTipRole)

    def init_base_conf_table_model(self):
        base_conf_header = ['Name', 'Value']
        model = QStandardItemModel(self)
        model.setHorizontalHeaderLabels(base_conf_header)
        self.ui.tblBaseConf.setModel(model)
        header = self.ui.tblBaseConf.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        return model

    def insert_row_action(self):
        table_model = self.ui.tblBaseConf.model()
        if not table_model:
            table_model = self.init_base_conf_table_model()
        sel_model = self.ui.tblBaseConf.selectionModel()
        curr_idx = sel_model.currentIndex().row()
        at_idx = table_model.rowCount() if curr_idx == -1 else curr_idx + 1
        table_model.insertRow(at_idx, [QStandardItem('') for _ in range(table_model.columnCount())])

    def remove_row_action(self):
        table_model = self.ui.tblBaseConf.model()
        if table_model:
            sel_model = self.ui.tblBaseConf.selectionModel()
            curr_idx = sel_model.currentIndex().row()
            table_model.takeRow(curr_idx)

    def new_conf_action(self):
        self.ui.cBoxConfs.setCurrentIndex(0)
        self.init_base_conf_table_model()
        self.insert_row_action()

    def save_conf_action(self):
        if self.ui.tblBaseConf.model():
            if self.ui.cBoxConfs.currentIndex() == 0:  # saving new conf brings up Save As dialog
                self.save_as_conf_action()
            else:
                file_path = path.join(self.configs_dir, self.ui.cBoxConfs.currentText())
                self.save_conf(file_path)

    def save_as_conf_action(self):
        dialog = MyDialog(self, "Save As", "Name of new HW config:")
        dialog.setWindowIcon(QIcon(QPixmap('res/icons8-save-as-50.png')))

        def exec_rec(message):
            dialog.set_error(message)
            answer = dialog.exec()
            if answer == QDialog.Accepted:
                invalid_chars_present = '\\' in dialog.get_text() or '/' in dialog.get_text()
                path_to_check = path.join(self.configs_dir, dialog.get_text()) + '.csv'
                invalid_entry = not dialog.get_text() or not is_pathname_valid(path_to_check) \
                    or invalid_chars_present or path.isfile(path_to_check)
                if invalid_entry:
                    dialog.setWindowIcon(QIcon(QPixmap('res/icons8-save-50_red.png')))
                    exec_rec('Invalid or existing entry!')
                else:
                    self.save_conf(path_to_check)
        if self.ui.tblBaseConf.model():
            exec_rec(None)

    def save_conf(self, file_path):
        model = self.ui.tblBaseConf.model()
        conf_name = path.basename(file_path)
        with open(file_path, 'w') as f:
            f.write('\"sep=|\"\n')
            for r in range(model.rowCount()):
                line = ' | '.join([model.item(r, c).text() for c in range(model.columnCount()) if model.item(r, c)])
                last_item = model.item(r, model.columnCount() - 1)
                line = '{} | {}'.format(line, last_item.toolTip()) if last_item and last_item.toolTip() else line
                f.write(line + '\n')

        if self.ui.cBoxConfs.findText(conf_name) == -1:  # new configuration has been created (SAVE AS)
            self.ui.cBoxConfs.addItem(conf_name)
            self.ui.cBoxConfs.blockSignals(True)  # prevent load to be triggered when text change is signalled
            self.ui.cBoxConfs.setCurrentIndex(self.ui.cBoxConfs.findText(conf_name))
            self.ui.cBoxConfs.blockSignals(False)
    
    def load_base_conf_data(self, file_name=''):
        if self.ui.cBoxConfs.currentIndex() == 0:
            self.ui.tblBaseConf.setModel(None)
            return
        base_conf_data = []
        model = self.init_base_conf_table_model()
        file = open(path.join(self.configs_dir, file_name), 'r')
        sep = ','
        for line in file:
            if line.startswith("\"sep="):
                sep = line.replace('\"', '').replace('sep=', '').strip()
            else:
                row_entry = [s.strip() for s in line.split(sep, 3)]
                base_conf_data.append(row_entry)
        file.close()

        if base_conf_data:
            self.base_conf_dict = base_conf_data
            for r in range(len(base_conf_data)):
                for c in range(model.columnCount()):
                    item = QStandardItem(base_conf_data[r][c])
                    if c == 1 and len(base_conf_data[r]) == 3:  # set tooltip if available
                        item.setToolTip(base_conf_data[r][2])
                    model.setItem(r, c, item)

    def del_conf_action(self):
        if self.ui.cBoxConfs.currentIndex() == 0:
            return
        msg_box = QMessageBox()
        msg_box.setText('Delete {}'.format(self.ui.cBoxConfs.currentText()))
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowIcon(QIcon(QPixmap('res/icons8-trash-can-40.png')))
        msg_box.setWindowTitle('Are you sure?')
        answer = msg_box.exec()
        if answer == QMessageBox.Yes:
            file_path = path.join(self.configs_dir, self.ui.cBoxConfs.currentText())
            remove(file_path)  # remove selected conf from os
            self.ui.cBoxConfs.removeItem(self.ui.cBoxConfs.currentIndex())

    def save_preset_action(self):
        if self.ui.cBoxPresets.currentIndex() == 0:  # saving new conf brings up Save As dialog
            self.save_as_preset_action()
        else:
            file_path = path.join(self.presets_dir, self.ui.cBoxPresets.currentText())
            self.save_preset(file_path)

    def save_as_preset_action(self):
        dialog = MyDialog(self, "Save As", "Name of new MIMS preset:")
        dialog.setWindowIcon(QIcon(QPixmap('res/icons8-save-as-50.png')))

        def exec_rec(message):
            dialog.set_error(message)
            answer = dialog.exec()
            if answer == QDialog.Accepted:
                invalid_chars_present = '\\' in dialog.get_text() or '/' in dialog.get_text()
                path_to_check = path.join(self.presets_dir, dialog.get_text()) + '.ini'
                invalid_entry = not dialog.get_text() or not is_pathname_valid(path_to_check) \
                    or invalid_chars_present or path.isfile(path_to_check)
                if invalid_entry:
                    dialog.setWindowIcon(QIcon(QPixmap('res/icons8-save-50_red.png')))
                    exec_rec('Invalid or existing entry!')
                else:
                    self.save_preset(path_to_check)
        exec_rec(None)

    def save_preset(self, file_path):
        preset_name = path.basename(file_path)
        settings = QSettings(file_path, QSettings.IniFormat)
        settings.clear()
        for o_name, o in inspect.getmembers(self.ui):
            if isinstance(o, QComboBox):
                if o_name != 'cBoxPresets':
                    settings.setValue(o_name, o.itemText(o.currentIndex()))  # save combobox selection to registry
            elif isinstance(o, QLineEdit):
                settings.setValue(o_name, o.text())  # save ui values, so they can be restored next time
            elif isinstance(o, QCheckBox):
                settings.setValue(o_name, o.isChecked())

        if self.ui.cBoxPresets.findText(preset_name) == -1:  # new configuration has been created (SAVE AS)
            self.ui.cBoxPresets.addItem(preset_name)
            self.ui.cBoxPresets.blockSignals(True)  # prevent load to be triggered when text change is signalled
            self.ui.cBoxPresets.setCurrentIndex(self.ui.cBoxPresets.findText(preset_name))
            self.ui.cBoxPresets.blockSignals(False)

    def load_preset(self, preset_file='', silent_mode=False):
        if self.ui.cBoxPresets.currentIndex() == 0 and not silent_mode:
            return
        ini_file_path = preset_file if silent_mode else path.join(self.presets_dir, preset_file)
        settings = QSettings(ini_file_path, QSettings.IniFormat)
        self.ui.cBoxModels.blockSignals(silent_mode)  # block signal that would trigger starting a netron server (load_dnn_preview)
        for o_name, o in inspect.getmembers(self.ui):
            if isinstance(o, QComboBox):
                value = settings.value(o_name)  # get stored value
                if value:  # QComboBox has been found in .ini, so restore index
                    index = o.findText(value)
                    if index != -1:  # item is found in QComboBox
                        o.setCurrentIndex(index)  # select a combobox value by index
            elif isinstance(o, QLineEdit):
                value = settings.value(o_name)
                o.setText(value)  # restore lineEditFile
            elif isinstance(o, QCheckBox):
                value = settings.value(o_name)  # get stored value
                if value:
                    o.setChecked(strtobool(value))  # restore checkbox
        self.ui.cBoxModels.blockSignals(False)

    def del_preset_action(self):
        if self.ui.cBoxPresets.currentIndex() == 0:
            return
        msg_box = QMessageBox()
        msg_box.setText('Delete {}'.format(self.ui.cBoxPresets.currentText()))
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowIcon(QIcon(QPixmap('res/icons8-trash-can-40.png')))
        msg_box.setWindowTitle('Are you sure?')
        answer = msg_box.exec()
        if answer == QMessageBox.Yes:
            file_path = path.join(self.presets_dir, self.ui.cBoxPresets.currentText())
            remove(file_path)  # remove selected conf from os
            self.ui.cBoxPresets.removeItem(self.ui.cBoxPresets.currentIndex())

    def load_dnn_preview(self, model_name):
        if not self.ui.gBoxPreviewModel.isChecked() and model_name:
            return
        if self.server_proc and self.server_proc.is_alive():
            self.server_proc.terminate()
            while self.server_proc.is_alive():
                continue
            #print("Netron server terminated (after serving {} s)".format(time() - self.t0))
        if self.ui.cBoxModels.currentIndex() == 0 or not model_name:
            self.webViewPreview.load(QUrl(self.splash_path))
        else:
            target_onnx_file = ''
            model_root_dir = path.join(self.models_dir, model_name)
            onnx_files_found = list(set(glob.glob(path.join(model_root_dir, '*.onnx'))).difference(
                set(glob.glob(path.join(model_root_dir, '*.trimmed.onnx')))))
            ms_files_found = glob.glob(path.join(model_root_dir, '*.ms'))
            if ms_files_found:
                source_ms_file = ms_files_found[0]
                target_onnx_file = path.splitext(ms_files_found[0])[0] + '.onnx'
                target_onnx_exists = path.isfile(target_onnx_file)
                onnx_out_of_date = path.getmtime(target_onnx_file) < path.getmtime(source_ms_file) if target_onnx_exists else False
                if not target_onnx_exists or onnx_out_of_date:  # if no onnx file is found or onnx file is out of date, generate it from .ms
                    if onnx_out_of_date:   # if onnx exists, back it up before overwriting
                        bak_file = target_onnx_file + '.bak'
                        if path.isfile(bak_file):
                            remove(bak_file)
                        rename(target_onnx_file, bak_file)
                    mims_model = dnnparser.load_mims(source_ms_file)  # load first .ms file found
                    onnx_model = mims_model.to_onnx_model()
                    dnnparser.dump_onnx(onnx_model, target_onnx_file)
            elif onnx_files_found:  # if onnx files are found
                target_onnx_file = onnx_files_found[0]  # load first .onnx file found
                if path.isfile(path.splitext(target_onnx_file)[0] + '.trimmed.onnx'):
                    target_onnx_file = path.splitext(target_onnx_file)[0] + '.trimmed.onnx'

            if target_onnx_file:
                self.server_proc = multiprocessing.Process(target=netron.serve, args=(target_onnx_file, None, False, False, 1010, 'localhost'))
                self.server_proc.start()
                self.t0 = time()
                self.webViewPreview.load(QUrl("http://localhost:1010"))
            else:
                print("No loadable model files (*.ms; *.onnx) found for \"{}\".".format(model_name))

    def toggle_preview(self, is_enabled):
        model_name = self.ui.cBoxModels.currentText() if is_enabled else ''
        self.load_dnn_preview(model_name)

    def show_browse_dialog(self):
        browse_from = self.ui.txtPath.text()
        browse_from = "./" if browse_from == '' else browse_from
        dialog = QFileDialog()
        new_path = dialog.getExistingDirectory(None, "Select a directory...", browse_from, QFileDialog.ShowDirsOnly)
        if new_path != '':
            self.ui.txtPath.setText(new_path)
            self.out_dir = new_path

    def closeEvent(self, event):
        # terminate server
        if self.server_proc and self.server_proc.is_alive():
            self.server_proc.terminate()
            while self.server_proc.is_alive():
                continue
            #print("Netron server terminated (after serving {} s)".format(time() - self.t0))


# Function to determine if graph has multiple root nodes. If yes, function returns the second root node id
# property of root node: its inputs are no other node's outputs
def multiple_root_nodes(graph_nodes):
    prev_node_outputs = []
    [prev_node_outputs.append(x.name) for x in graph_nodes[0].outputs]
    total_nodes = len(graph_nodes)
    node_id = 0
    for node in graph_nodes:
        if node_id == 0:  # first node is always a root node
            node_id += 1
            continue
        matches = [match for match in node.inputs if match.name in prev_node_outputs]
        if len(matches) == 0:
            break
        else:
            [prev_node_outputs.append(x.name) for x in node.outputs]
            node_id += 1
    if node_id == total_nodes:
        return -1
    else:
        return node_id


def create_graph_unit_list(graph_nodes):
    graph_unit_list = []
    graph_unit = dnnparser.MIMSGraphUnit()
    node_list = []
    temp_node_name = ''
    second_root_node_id = multiple_root_nodes(graph_nodes)
    for node_id, node in enumerate(graph_nodes):
        node_list.append(node)
        if second_root_node_id > 0 and node_id == second_root_node_id - 1:  # one node previous to second_root_node_id is the last node in first parallel branch
            graph_unit.add_node_list(node_list)
            node_list = []
            temp_node_name = [input.name for input in node.children[0].inputs if input.name != node.outputs[0].name][0]  # get the name of other input of sink node which will be the last node in the second root node list
            continue
        if [match for match in node.outputs if match.name == temp_node_name]:  # This is end of node list of second root node
            graph_unit.add_node_list(node_list)
            graph_unit_list.append(graph_unit)
            node_list = []
            second_root_node_id = -1
            continue
        if len(node.children) > 1 and second_root_node_id == -1:
            graph_unit = dnnparser.MIMSGraphUnit(node_list)
            graph_unit_list.append(graph_unit)
            node_list = []

    return graph_unit_list


def change_hw_cfg_value(mims_window, kv_s):
    model = mims_window.ui.tblBaseConf.model()
    kv_s = kv_s if isinstance(kv_s[0], list) else [kv_s]  # in case only one kv pair is passed as cmd arg
    for key, value in kv_s:
        found = False
        for r in range(model.rowCount()):
            found = model.item(r, 0).text().lower() == key.lower()
            if found:
                model.item(r, 1).setText(value)
                break
        if not found:
            print("Warning: \"{}\" is not found in current HW configuration table.".format(key))


def change_sw_opt_value(mims_window, kv_s):
    kv_s = kv_s if isinstance(kv_s[0], list) else [kv_s]  # in case only one kv pair is passed as cmd arg

    for key, value in kv_s:
        found = False
        for ch in mims_window.ui.gBoxSWOpt.findChildren(QCheckBox):
            if 'chBox'+str(key) == ch.objectName():
                found = True
                if value == 'True':
                    ch.setChecked(True)
                else:
                    ch.setChecked(False)
        for ch in mims_window.ui.gBoxSWOpt.findChildren(QComboBox):
            if 'cBox'+str(key) == ch.objectName():
                found = True
                index = ch.findText(value)
                if index >= 0:
                    ch.setCurrentIndex(index)
        for ch in mims_window.ui.gBoxSWOpt.findChildren(QLineEdit):
            if 'txt'+str(key) == ch.objectName():
                found = True
                ch.setText(str(value))
        if not found:
            print("Warning: \"{}\" is not found in current SW Options.".format(key))


def change_hw_opt_value(mims_window, kv_s):
    kv_s = kv_s if isinstance(kv_s[0], list) else [kv_s]  # in case only one kv pair is passed as cmd arg

    for key, value in kv_s:
        found = False
        for ch in mims_window.ui.gBoxHwOpt.findChildren(QCheckBox):
            if 'chBox'+str(key) == ch.objectName():
                found = True
                if value == 'True':
                    ch.setChecked(True)
                else:
                    ch.setChecked(False)
        for ch in mims_window.ui.gBoxHwOpt.findChildren(QComboBox):
            if 'cBox'+str(key) == ch.objectName():
                found = True
                index = ch.findText(value)
                if index >= 0:
                    ch.setCurrentIndex(index)
        for ch in mims_window.ui.gBoxHwOpt.findChildren(QLineEdit):
            if 'txt'+str(key) == ch.objectName():
                found = True
                ch.setText(str(value))
        if not found:
            print("Warning: \"{}\" is not found in current SW Options.".format(key))


def change_gemm_dimms(mims_window, gemm_dimms):
    kv_s = gemm_dimms if isinstance(gemm_dimms[0], list) else gemm_dimms[0]
    if len(gemm_dimms) > 1:
        print("WARNING: len(gemm_dimms) > 1. MIMS will consider only 1st instance for this simulation")
    # open file with r+b (allow write and binary mode)
    f = open("./models/gemm/model.ms", 'r')
    # get array of lines
    f_content = f.readlines()
    f.close()
    # get middle line
    #update_line = len(f_content)/2
    # overwrite middle line
    f_content[9] = "\t\t\t\t\tinput {'name': 'gpu_0/data_0', 'val_type': 1, 'dims': [1 , "
    f_content[9] += str(kv_s[0][2])
    f_content[9] += "], 'vals': []}\n"
    f_content[10] = "\t\t\t\t\tinput {'name': 'gpu_0/w_0', 'val_type': 1, 'dims': ["
    f_content[10] += str(kv_s[0][1])
    f_content[10] += " , "
    f_content[10] += str(kv_s[0][2])
    f_content[10] += "], 'vals': []}\n"
    # return pointer to top of file so we can re-write the content with replaced string
    f = open("./models/gemm/model.ms", 'w')
    # clear file content
    #f.truncate()
    # re-write the content with the updated content
    f.writelines(f_content)
    # close file
    f.close()

    for ch in mims_window.ui.gBoxSWOpt.findChildren(QLineEdit):
        if ch.objectName() == 'txtBatchSize':
            ch.setText(str(kv_s[0][0]))


def change_hw_arch_or_DNN(mims_window, select_hw, select_DNN):
    if select_hw:
        select_hw = select_hw if isinstance(select_hw[0], list) else [select_hw] # in case only one kv pair is passed as cmd arg
        if len(select_hw) > 1:
            print("Warning: Only 1 instance of HW is expected. MIMS will only consider \"{}\"".format(select_hw[0][0]))
        for value in select_hw[0]:
            found = False
            for ch in mims_window.ui.gBoxHwOpt.findChildren(QComboBox):
                if ch.objectName() == 'cBoxConfs':
                    index = ch.findText(value)
                    if index >= 0:
                        found = True
                        ch.setCurrentIndex(index)
            if not found:
                print("Warning: \"{}\" is not found in current SW Options.".format(value))
    if select_DNN:
        select_DNN = select_DNN if isinstance(select_DNN[0], list) else [select_DNN]  # in case only one kv pair is passed as cmd arg
        if len(select_DNN) > 1:
            print("Warning: Only 1 instance of DNN is expected. MIMS will only consider \"{}\"".format(select_DNN[0][0]))
        for value in select_DNN[0]:
            print(value)
            found = False
            for ch in mims_window.ui.gBoxSWOpt.findChildren(QComboBox):
                if ch.objectName() == 'cBoxModels':
                    index = ch.findText(value)
                    if index >= 0:
                        found = True
                        ch.setCurrentIndex(index)
            if not found:
                print("Warning: \"{}\" is not found in DNN List.".format(value))


def silent_run(mims_window, preset_file, validate, postfix=None, hw_cfg_overwr=None,
               sw_opt_overwr=None, hw_opt_overwr=None, gemm_dimms=None, select_hw=None, select_DNN=None):  # used for cmd line execution
    mims_window.load_preset(preset_file, silent_mode=True)
    if postfix:
        mims_window.ui.txtPostfix.setText(postfix)
    if select_hw or select_DNN:
        change_hw_arch_or_DNN(mims_window, select_hw, select_DNN)
    if hw_cfg_overwr:
        change_hw_cfg_value(mims_window, hw_cfg_overwr)
    if sw_opt_overwr:
        change_sw_opt_value(mims_window, sw_opt_overwr)
    if hw_opt_overwr:
        change_hw_opt_value(mims_window, hw_opt_overwr)
    if gemm_dimms:
        change_gemm_dimms(mims_window, gemm_dimms)
    results_file = mims_window.run_mims(silent_mode=True)
    mims_window.close()
    if not results_file or validate and diff.compare(results_file):
        exit(1)  # validation not successful
    else:
        exit(0)


# -----MAIN -----#
if __name__ == '__main__':
    app = QApplication(argv)
    window = MyMainWindow()

    parser = argparse.ArgumentParser(description="-== MIMS: Machine Intelligence Model Simulator ==-")
    parser.add_argument("-preset", metavar="PRESET_FILE", help="configuration (preset) file (.ini)")
    parser.add_argument("-postfix", metavar="POSTFIX", help="Postfix for the results file name")
    parser.add_argument("-validate", help="Make sure the results match the reference \"golden\" ones", action="store_true")
    parser.add_argument("-select_hw", help="Replace current HW with selected HW (arg 1: HW_selected). Limit=1 per run", action='append', nargs=1) #Ashish added.
    parser.add_argument("-select_dnn", help="Replace current DNN with selected DNN (arg 1: DNN selected). Limit=1 per run", action='append', nargs=1) #Ashish added
    parser.add_argument("-hw_config", metavar=("KEY", "VALUE"), help="Overwrite the value of field (arg 1: key) in HW Config with (arg 2: value)", action='append', nargs=2) #Ashish added.. changed -overwr to -hw_config
    parser.add_argument("-sw_opt", metavar=("KEY" ,"VALUE"), help="overwrite value of field (arg 1: key) in SW Options with (arg 2: value)", action='append', nargs=2) #Ashish added
    parser.add_argument("-hw_opt", metavar=("KEY" , "VALUE"), help="Overwrite the value of field (arg 1: key) in HW Options with (arg 2: value)", action='append', nargs=2) #Ashish added
    parser.add_argument("-gemm_dimms", metavar=("M_dimm", "N_dimm", "K_dimm"), help="gemm dimension (arg 1: M, arg 2: N, arg 3: K).", action='append', nargs=3)
    args = parser.parse_args()

    if args.preset:  # cmd line mode
        file_in_curr_dir = args.preset if path.isfile(args.preset) else ''
        file_in_presets_dir = path.join(window.presets_dir, args.preset) if path.isfile(path.join(window.presets_dir, args.preset)) else ''
        if not file_in_curr_dir:
            if not file_in_presets_dir:
                print("Presets file \"{}\" not found. Exiting..".format(args.preset))
                exit(1)
            else:
                silent_run(window, file_in_presets_dir, args.validate, args.postfix, args.hw_config, args.sw_opt, args.hw_opt, args.gemm_dimms, args.selec_hw, args.select_dnn)

        else:
            silent_run(window, file_in_curr_dir, args.validate, args.postfix, args.hw_config, args.sw_opt, args.hw_opt, args.gemm_dimms, args.select_hw, args.select_dnn)
    else:  # gui mode
        window.show()
        exit(app.exec_())
