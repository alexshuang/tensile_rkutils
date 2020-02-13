from PyQt5.QtCore import QThread, pyqtSignal


class FilterParam:
    def __init__(self, dims, pads, strides, group=1):
        self.dims = dims
        self.pads = pads
        self.strides = strides
        self.group = group


class LayerInputParam:
    def __init__(self, in_dims, out_dims, l2_hit_rate_wt=0, l2_hit_rate_act=0, l3_hit_rate_act=0, data=0, last_node=False, tpu_partition_scheme=''):
        self.in_dims = in_dims  # In NCHW format
        self.out_dims = out_dims  # In NCHW format
        self.data = data
        self.l2_hit_rate_wt = l2_hit_rate_wt
        self.l2_hit_rate_act = l2_hit_rate_act
        self.l3_hit_rate_act = l3_hit_rate_act
        self.last_node = last_node
        self.tpu_partition_scheme = tpu_partition_scheme


class LayerResults:
    def __init__(self, alu_util_factor=0.0, chip_util_factor=0.0, speedup=0, cycles=0.0, flop=0.0, num_cu_util=0, m=0, n=0, k=0,
                 num_a_blocks=0, num_b_blocks=0, num_partitions=1, hbm_rd_bw=0, hbm_wr_bw=0, op_name='', tpu_sub_res=None,
                 alu_cc=0, mem_cc=0,gflops=0.0, #Ashish added
                 num_rounds=1, total_blocks=0, num_cu_util_trail=0, num_a_blocks_trail=0, num_b_blocks_trail=0,
                 num_partitions_trail=0, cycles_trail=0, wr_cc=0, main_instr="0", threadTile="0", workGroup="0", unroll_factor=1,
                 unroll_factor_trail=1, num_rounds_trail=1, alu_cc_trail=0, mem_cc_trail=0, wr_cc_trail=0):  # Ashish added
        self.m = m
        self.n = n
        self.k = k
        self.alu_util_factor = alu_util_factor
        self.chip_util_factor = chip_util_factor
        self.speedup = speedup
        self.cycles = cycles
        self.flop = flop
        self.num_cu_util = num_cu_util
        self.num_a_blocks = num_a_blocks
        self.num_b_blocks = num_b_blocks
        self.num_partitions = num_partitions
        self.hbm_rd_bw = hbm_rd_bw
        self.hbm_wr_bw = hbm_wr_bw
        self.op_name = op_name
        self.tpu_sub_res = tpu_sub_res
        self.alu_cc = alu_cc  # Ashish added
        self.mem_cc = mem_cc  # Ashish added
        self.gflops=gflops #Ashish added
        self.num_rounds = num_rounds  # Ashish added
        self.total_blocks = total_blocks  # Ashish added
        self.num_cu_util_trail = num_cu_util_trail #Ashish added
        self.num_a_blocks_trail = num_a_blocks_trail #Ashish added
        self.num_b_blocks_trail = num_b_blocks_trail #Ashish added
        self.num_partitions_trail = num_partitions_trail #Ashish added
        self.cycles_trail = cycles_trail #Ashish added
        self.wr_cc = wr_cc
        self.main_instr = main_instr
        self.threadTile = threadTile
        self.workGroup = workGroup
        self.unroll_factor = unroll_factor
        self.unroll_factor_trail = unroll_factor_trail
        self.num_rounds_trail = num_rounds_trail
        self.alu_cc_trail = alu_cc_trail
        self.mem_cc_trail = mem_cc_trail
        self.wr_cc_trail = wr_cc_trail
        self.sub_results = []

    def populate_sub_results(self, sub_res):
        self.sub_results.append(sub_res)


class TpuSubResults:
    def __init__(self, vgpr_util_bytes_wt=0, vgpr_util_bytes_res=0):
        self.vgpr_util_bytes_wt = vgpr_util_bytes_wt
        self.vgpr_util_bytes_res = vgpr_util_bytes_res


class MyWorkThread(QThread):
    doWork = pyqtSignal()
    stopWork = pyqtSignal()

    def __init__(self, parent=None, target=None):
        QThread.__init__(self)
        self.target = target

    def stop(self):
        if self.isRunning():
            self.stopWork.emit()
            print('terminating..')

    def run(self):
        if self.target:
            self.target()
        else:
            self.doWork.emit()
            # self.emit(QtCore.SIGNAL("doWork"))
