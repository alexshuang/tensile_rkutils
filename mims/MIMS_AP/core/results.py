import xlsxwriter
import re


def fill_percent_layer_time(nn_results, total_cycles):  # nn_results == neural network results
    total_layers_pc = 0
    for layr in nn_results:
        layr.percent_layer_time = (layr.cycles / total_cycles) * 100
        total_layers_pc += layr.percent_layer_time


class Summary:
    def __init__(self, param, batch_size, fwd_cycles, bwd_cycles, total_cycles, total_time_us, perf_gain=0):
        self.param = param
        self.batch_size = batch_size
        self.fwd_cycles = fwd_cycles
        self.bwd_cycles = bwd_cycles
        self.total_cycles = total_cycles
        self.total_time_us = total_time_us
        self.perf_gain = perf_gain


class Results:
    def __init__(self, source='', layer_type='CNN', batch_size=1, iChannels=0, iWidth=0, iHeight=0, fWidth=0, fHeight=0,
                 padW=0, padH=0, strideW=0, strideH=0, oChannels=0, oWidth=0, oHeight=0,
                 weight_size=0, activation_size=0, M=0, N=0, K=0, num_cu_utilized=0,
                 num_a_blocks=0, num_b_blocks=0, num_partitions=1, main_instr="0", threadTile="0", workGroup="0", alu_utilization=0, chip_utilization=0, cycles=0, gflops=0.0,
                 dram_rd_bw = 0, dram_wr_bw = 0,
                 delta_weight_transfer_cycles=0, flop=0, allreduce_num_cu=0, percent_layer_time=0,tpu_partition_scheme=None,
                 mem_bw_utilization=0, alu_cc=0, mem_cc=0, wr_cc=0, num_rounds=1, total_blocks=0, num_cu_util_trail=0,
                 num_a_blocks_trail=0, num_b_blocks_trail=0, num_partitions_trail=0, cycles_trail=0, unroll_factor = 1,
                 unroll_factor_trail=1, num_rounds_trail=1, alu_cc_trail=0, mem_cc_trail=0, wr_cc_trail=0):  # Ashish added
        self.source = source
        self.layer_type = layer_type
        self.batch_size = batch_size
        self.channels_or_rnn_seq_length = iChannels
        self.width_or_rnn_hidden_sz = iWidth
        self.height_or_rnn_input_sz = iHeight
        self.fWidth = fWidth
        self.fHeight = fHeight
        self.padW = padW
        self.padH = padH
        self.strideW = strideW
        self.strideH = strideH
        self.oChannels = oChannels
        self.oWidth = oWidth
        self.oHeight = oHeight
        self.weightSize = weight_size
        self.activationSize = activation_size
        self.M = M
        self.N = N
        self.K = K
        self.num_a_blocks = num_a_blocks
        self.num_b_blocks = num_b_blocks
        self.num_partitions = num_partitions
        self.num_cu_utilized = num_cu_utilized
        self.num_rounds = num_rounds #Ashish added
        self.main_instr = main_instr #Ashish added
        self.threadTile = threadTile #Ashish added
        self.workGroup = workGroup #Ashish added
        self.unroll_factor = unroll_factor #Ashish added
        self.alu_utilization = alu_utilization
        self.chip_utilization = chip_utilization
        self.cycles = cycles
        self.dram_rd_bw = dram_rd_bw
        self.dram_wr_bw = dram_wr_bw
        self.delta_weight_transfer_cycles = delta_weight_transfer_cycles
        self.transfer_cycles_exposed = 0
        self.flop = flop
        self.allreduce_num_cu = allreduce_num_cu
        self.percent_layer_time = percent_layer_time
        self.tpu_partition_scheme = tpu_partition_scheme
        self.mem_bw_utilization = mem_bw_utilization  # Ashish added
        self.alu_cc = alu_cc  # Ashish added
        self.mem_cc = mem_cc  # Ashish added
        self.wr_cc = wr_cc
        self.total_blocks = total_blocks  # Ashish added
        self.num_cu_util_trail = num_cu_util_trail #Ashish added
        self.num_a_blocks_trail = num_a_blocks_trail #Ashish added
        self.num_b_blocks_trail = num_b_blocks_trail #Ashish added
        self.num_partitions_trail = num_partitions_trail #Ashish added
        self.num_rounds_trail = num_rounds_trail #Ashish added
        self.unroll_factor_trail = unroll_factor_trail #Ashish added
        self.cycles_trail = cycles_trail #Ashish added
        self.alu_cc_trail = alu_cc_trail #Ashish added
        self.mem_cc_trail = mem_cc_trail #Ashish added
        self.wr_cc_trail = wr_cc_trail #Ashish added
        self.gflops = gflops #Ashish added


class ResultWriter:
    def __init__(self, filename, sys_cfg):
        self.sys_cfg = sys_cfg
        self.filename = filename
        self.base_cycles = 0
        self.base_conv_cycles = 0
        self.forward_cycles = 0
        self.forward_conv_cycles = 0
        self.delta_wt_transfer_cycles = 0
        self.transfer_cycles_exposed = 0
        self.result_book = xlsxwriter.Workbook(self.filename)
        self.bold_fmt = self.result_book.add_format({'bold': True})
        self.summary = []
        self.fwd_cycles = 0
        self.bwd_cycles = 0
        self.total_cycles = 0
        self.total_time = 0
        self.range_list = getattr(self.sys_cfg.hw_cfg, self.sys_cfg.hw_cfg.range_attr) if self.sys_cfg.hw_cfg.range_attr else []
        self.range_attr_str = ' '.join(w for w in re.split('_', self.sys_cfg.hw_cfg.range_attr)) if self.sys_cfg.hw_cfg.range_attr else ''

        if not len(self.range_list):
            base_sheet_fwd = self.result_book.add_worksheet('Base Results Fwd Pass')
            self.sheet_list = [base_sheet_fwd]
            if self.sys_cfg.sw_opt.training:
                base_sheet_bwd = self.result_book.add_worksheet('Base Results Bwd Pass')
                self.sheet_list.append(base_sheet_bwd)
        else:
            self.sheet_list = []
            for i in range(len(self.range_list)):
                string = self.range_attr_str + ' ' + str(self.range_list[i]) + ' ' + 'Fwd Pass'
                if len(string) > 31:
                    string = string.replace(" ", "")
                sheet = self.result_book.add_worksheet(string)
                self.sheet_list.append(sheet)
                if self.sys_cfg.sw_opt.training:
                    string = self.range_attr_str + ' ' + str(self.range_list[i]) + ' ' + 'Bwd Pass'
                    if len(string) > 31:
                        string = string.replace(" ", "")
                    sheet = self.result_book.add_worksheet(string)
                    self.sheet_list.append(sheet)

        #if self.sys_cfg.is_dirty() and not len(self.range_list):
        #    options_sheet_fwd = self.result_book.add_worksheet('Results with Options Fwd Pass')
        #    self.sheet_list.append(options_sheet_fwd)
        #    if self.sys_cfg.sw_opt.training:
        #        options_sheet_bwd = self.result_book.add_worksheet('Results with Options Bwd Pass')
        #        self.sheet_list.append(options_sheet_bwd)

        self.prepare_header()

    def get_tflops(self, flops, cycles, mai_accelerated=True):
        hw_cfg = self.sys_cfg.hw_cfg
        tflops = flops / (cycles / (hw_cfg.gpu_freq * 1e9)) / 1e12
        if mai_accelerated:
            macs = hw_cfg.fp16_dl_macs_per_cu if self.sys_cfg.sw_opt.fp16_inputs else \
                   (hw_cfg.fp32_dl_macs_per_cu if self.sys_cfg.sw_opt.fp32_inputs else hw_cfg.fp64_dl_macs_per_cu)
        else:
            macs = hw_cfg.legacy_fp16_macs_per_cu if self.sys_cfg.sw_opt.fp16_inputs or self.sys_cfg.sw_opt.bf16_inputs else \
                (hw_cfg.legacy_fp32_macs_per_cu if self.sys_cfg.sw_opt.fp32_inputs else hw_cfg.legacy_fp64_macs_per_cu)
        available_tflops = 2 * macs * hw_cfg.num_cu * hw_cfg.gpu_freq * 1e9 / 1e12
        tflops = min(tflops, available_tflops)
        return tflops

    def write(self, results, direction='forward', is_base_result=True, range_idx=0, is_last_iter=1, total_cycles=0,
              transfer_cycles_exposed=0):
        total_layers_pc = fill_percent_layer_time(results, total_cycles)
        row = 1
        conv_cycles = 0
        total_flop = 0
        total_act_size = 0
        total_wt_size = 0
        batch_size = 0
        cycles_ind = self.res_fields_ind['cycles']
        pc_layer_ind = self.res_fields_ind['percent_layer_time']
        layer_type_ind = self.res_fields_ind['layer_type']
        act_size_ind = self.res_fields_ind['activationSize']
        wt_size_ind = self.res_fields_ind['weightSize']
        num_cu_util_ind = self.res_fields_ind['num_cu_utilized']
        batch_size_ind = self.res_fields_ind['batch_size']
        alu_util_ind = self.res_fields_ind['alu_utilization']
        chip_util_ind = self.res_fields_ind['chip_utilization']
        flop_ind = self.res_fields_ind['flop']
        total_pc_layer_time = {}
        curr_layer = ''
        avg_gemm_alu_util = 0
        avg_gemm_chip_util = 0
        gemm_cycles = 0
        if self.sys_cfg.hw_cfg.tpu_en:
            gpu_freq = self.sys_cfg.hw_cfg.tpu_freq
        else:
            gpu_freq = self.sys_cfg.hw_cfg.gpu_freq[range_idx] if range_idx > 0 else self.sys_cfg.hw_cfg.gpu_freq
        if range_idx > 0:
            if self.sys_cfg.sw_opt.training:
                sheet = 2 * range_idx if direction == 'forward' else 2 * range_idx + 1
            else:
                sheet = range_idx
        else:
            if is_base_result:
                sheet = 0
                if self.sys_cfg.sw_opt.training:
                    sheet = 0 if direction == 'forward' else 1
            else:
                sheet = 1
                if self.sys_cfg.sw_opt.training:
                    sheet = 2 if direction == 'forward' else 3

        for layer_ind, res in enumerate(results):
            res_vals = vars(res)
            col = 0
            alu_util = 0
            chip_util = 0
            for res_ind, val in enumerate(res_vals.values()):
                if num_cu_util_ind == col or batch_size_ind == col and self.sys_cfg.hw_cfg.chiplet_mode_en and self.sys_cfg.sw_opt.training:
                    val *= self.sys_cfg.hw_cfg.num_cu_clusters
                if (act_size_ind == col or wt_size_ind == col) and 'sub' in res.source:
                    val = 0
                if layer_type_ind == col and 'sub' not in res.source:
                    self.sheet_list[sheet].write(row, col, val, self.bold_fmt)
                else:
                    self.sheet_list[sheet].write(row, col, val)
                if batch_size_ind == col:
                    batch_size = val
                if alu_util_ind == col:
                    alu_util = val
                if chip_util_ind == col:
                    chip_util = val
                if flop_ind == col:
                    total_flop += val
                if cycles_ind == col and 'sub' not in res.source:
                    if curr_layer == 'Conv':
                        conv_cycles += val
                    if curr_layer in ['Conv', 'Gemm', 'Rnn', 'Attention']:
                        gemm_cycles += val
                        avg_gemm_alu_util += alu_util * val
                        avg_gemm_chip_util += chip_util * val
                if layer_type_ind == col:
                    curr_layer = val
                    if val not in total_pc_layer_time.keys():
                        total_pc_layer_time.update({val:0})
                if pc_layer_ind == col:
                    total_pc_layer_time[curr_layer] += val
                if act_size_ind == col and 'sub' not in res.source:
                    total_act_size += val
                if wt_size_ind == col and 'sub' not in res.source:
                    total_wt_size += val
                col += 1
            row += 1

        if self.sys_cfg.sw_opt.rl and not self.sys_cfg.hw_cfg.chiplet_mode_en:
            total_cycles *= 2  # Assuming a dual actor-critic network
        total_time = (total_cycles / (gpu_freq * 1e9)) * 1e6
        tflops = self.get_tflops(total_flop, total_cycles)
        if is_base_result:
            self.base_cycles = total_cycles
            self.base_conv_cycles = conv_cycles
        if direction == 'forward':
            self.sheet_list[sheet].write(row, cycles_ind - 1, 'Forward Cycles', self.bold_fmt)
            self.forward_cycles = total_cycles
            self.forward_conv_cycles = conv_cycles
        else:
            self.sheet_list[sheet].write(row, cycles_ind - 1, 'Backward Cycles', self.bold_fmt)
            self.sheet_list[sheet].write(row, cycles_ind + 1, 'Transfer Cycles Exposed', self.bold_fmt)
            self.sheet_list[sheet].write(row, cycles_ind + 2, transfer_cycles_exposed)
        self.sheet_list[sheet].write(row, cycles_ind, total_cycles)
        if direction == 'forward':
            self.sheet_list[sheet].write(row + 1, cycles_ind - 1, 'Time in us', self.bold_fmt)
            self.sheet_list[sheet].write(row + 1, cycles_ind, total_time)
            print('Total time in us: {}'.format(total_time)) #Ashish remove later
        else:
            total_time = ((total_cycles + self.forward_cycles) / (gpu_freq * 1e9)) * 1e6
            self.sheet_list[sheet].write(row + 1, cycles_ind - 1, 'Total time in us', self.bold_fmt)
            self.sheet_list[sheet].write(row + 1, cycles_ind, total_time)
            print('Total time in us: {}'.format(total_time))
        if self.sys_cfg.sw_opt.training and direction == 'backward':
            self.sheet_list[sheet].write(row + 2, cycles_ind - 1, 'Total cycles', self.bold_fmt)
            self.sheet_list[sheet].write(row + 2, cycles_ind, self.forward_cycles + total_cycles)
            self.sheet_list[sheet].write(row + 3, cycles_ind - 1, 'Total Conv cycles', self.bold_fmt)
            self.sheet_list[sheet].write(row + 3, cycles_ind, self.forward_conv_cycles + conv_cycles)
            self.sheet_list[sheet].write(row + 4, cycles_ind - 1, 'Average GEMM ALU Utilization%', self.bold_fmt)
            self.sheet_list[sheet].write(row + 4, cycles_ind, avg_gemm_alu_util / gemm_cycles)
            self.sheet_list[sheet].write(row + 5, cycles_ind - 1, 'Average GEMM Chip Utilization%', self.bold_fmt)
            self.sheet_list[sheet].write(row + 5, cycles_ind, avg_gemm_chip_util / gemm_cycles)
        if not self.sys_cfg.sw_opt.training:
            self.sheet_list[sheet].write(row + 2, cycles_ind - 1, 'Average GEMM ALU Utilization%', self.bold_fmt)
            self.sheet_list[sheet].write(row + 2, cycles_ind, avg_gemm_alu_util / gemm_cycles)
            self.sheet_list[sheet].write(row + 3, cycles_ind - 1, 'Average GEMM Chip Utilization%', self.bold_fmt)
            self.sheet_list[sheet].write(row + 3, cycles_ind, avg_gemm_chip_util / gemm_cycles)
            self.sheet_list[sheet].write(row + 4, cycles_ind - 1, 'Throughput (Samples/s)', self.bold_fmt)
            self.sheet_list[sheet].write(row + 4, cycles_ind, (self.sys_cfg.sw_opt.batch_size / total_time) * 1e6)
            self.sheet_list[sheet].write(row + 5, cycles_ind - 1, 'freq(GHz)', self.bold_fmt)
            self.sheet_list[sheet].write(row + 5, cycles_ind, gpu_freq)
            #print('Inferences/s: {}'.format((self.sys_cfg.sw_opt.batch_size / total_time) * 1e6)) #Ashish remove later
        elif direction == 'backward':
            self.sheet_list[sheet].write(row + 6, cycles_ind - 1, 'Throughput (Samples/s)', self.bold_fmt)
            self.sheet_list[sheet].write(row + 6, cycles_ind, (self.sys_cfg.sw_opt.batch_size / total_time) * 1e6)
            self.sheet_list[sheet].write(row + 7, cycles_ind - 1, 'freq(GHz)', self.bold_fmt)
            self.sheet_list[sheet].write(row + 7, cycles_ind, gpu_freq)
            print('Training Throughput (Samples/s): {}'.format((self.sys_cfg.sw_opt.batch_size / total_time) * 1e6))
        if not is_base_result:
            self.sheet_list[sheet].write(row + 3, cycles_ind - 1, '% Perf gain', self.bold_fmt)
            self.sheet_list[sheet].write(row + 3, cycles_ind, ((self.base_cycles - total_cycles) / self.base_cycles)*100)
            self.sheet_list[sheet].write(row + 4, cycles_ind - 1, '% Conv Perf gain', self.bold_fmt)
            self.sheet_list[sheet].write(row + 4, cycles_ind, ((self.base_conv_cycles - conv_cycles) / self.base_conv_cycles) * 100)
        ind = 0
        for layer, pc_time in total_pc_layer_time.items():
            self.sheet_list[sheet].write(0, len(self.res_fields_ind) + ind, '% time on ' + layer, self.bold_fmt)
            self.sheet_list[sheet].write(1, len(self.res_fields_ind) + ind, pc_time)
            ind += 1
        if self.sys_cfg.sw_opt.training and direction == 'backward':
            self.sheet_list[sheet].write(0, len(self.res_fields_ind) + ind, '% time on exposed transfer cycles', self.bold_fmt)
            self.sheet_list[sheet].write(1, len(self.res_fields_ind) + ind, (transfer_cycles_exposed/total_cycles)*100)

        self.sheet_list[sheet].write(row, wt_size_ind - 1, 'Total Size (in MB)', self.bold_fmt)
        self.sheet_list[sheet].write(row, wt_size_ind, total_wt_size / 1e6)
        self.sheet_list[sheet].write(row, act_size_ind, total_act_size / 1e6)

        if range_idx > 0:
            param = self.range_attr_str + ' ' + str(self.range_list[range_idx])
            if direction == 'forward':
                self.fwd_cycles = total_cycles
                if not self.sys_cfg.sw_opt.training:
                    summary = Summary(param, batch_size, self.fwd_cycles, 0, self.fwd_cycles, total_time)
                    self.summary.append(summary)
            else:
                self.bwd_cycles = total_cycles
                self.total_cycles = self.fwd_cycles + self.bwd_cycles
                self.total_time = total_time
                summary = Summary(param, batch_size, self.fwd_cycles, self.bwd_cycles, self.total_cycles, total_time)
                self.summary.append(summary)

        if is_last_iter:
            if range_idx > 0:
                self.write_summary()
            self.result_book.close()

    def prepare_header(self):
        res = Results()
        res_fields = vars(res)
        keys = list(res_fields)
        self.res_fields_ind = {keys[i]: i for i in range(len(keys))}
        for sheet in self.sheet_list:
            col = 0
            row = 0
            for field in res_fields.keys():
                sheet.write(row, col, field, self.bold_fmt)
                col += 1

    def write_summary(self):
        sheet = self.result_book.add_worksheet('Summary')
        summary_fields = vars(self.summary[0])

        col = 0
        row = 0
        for field in summary_fields:
            sheet.write(row, col, field, self.bold_fmt)
            col += 1
        col = 0
        row = 1
        prev_time = getattr(self.summary[0], 'total_time_us')
        for ind, summary in enumerate(self.summary):
            curr_time = getattr(summary, 'total_time_us')
            perf_gain = ((prev_time - curr_time) / prev_time) * 100
            prev_time = curr_time
            for field in summary_fields:
                if 'perf_gain' in field:
                    sheet.write(row, col, perf_gain)
                else:
                    sheet.write(row, col, getattr(summary, field))
                col += 1
            row += 1
            col = 0

        # Graph plots
        clm_chart = self.result_book.add_chart({'type': 'column'})
        cat_range = '='+sheet.name+'!'+'A2:A'+str(len(self.summary)+1)
        val_range = '='+sheet.name+'!'+'F2:F'+str(len(self.summary)+1)
        clm_chart.add_series({'categories': cat_range,
                              'values': val_range})

        line_chart = self.result_book.add_chart({'type': 'line'})
        val_range = '=' + sheet.name + '!' + 'G2:G' + str(len(self.summary) + 1)
        line_chart.add_series({'categories': cat_range,
                              'values': val_range,
                              'marker': {'type': 'square', 'fill': {'color': 'red'}},
                              'y2_axis': True,
                               })
        line_chart.set_y2_axis({'name': 'Perf Gain %',
                               })
        clm_chart.combine(line_chart)
        clm_chart.set_title({'name': 'Effect of ' + self.range_attr_str + ' on total time'})
        clm_chart.set_y_axis({'name': 'Total Execution Time in us',
                              'major_gridlines': {'visible': False},
                              })
        clm_chart.set_legend({'position': 'none'})
        clm_chart.set_style(35)
        sheet.insert_chart('F11', clm_chart)
