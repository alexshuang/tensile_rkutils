import math
import traceback
import heapq
import copy
from core.const import *
from core.structs.auxiliary import TpuSubResults
from core.auxiliary import get_dt_size, get_dl_macs_per_cu

#TENSOR_BLOCK_WIDTH16 = 16
#TENSOR_BLOCK_HEIGHT16 = 16
#TENSOR_BLOCK_WIDTH4 = 4
#TENSOR_BLOCK_HEIGHT64 = 64
#TENSOR_BLOCK_WIDTH64 = 64
#TENSOR_BLOCK_HEIGHT4 = 4
#TENSOR_BLOCK_SIZE16x16 = TENSOR_BLOCK_WIDTH16 * TENSOR_BLOCK_HEIGHT16
#TENSOR_BLOCK_SIZE64x4 = TENSOR_BLOCK_WIDTH4 * TENSOR_BLOCK_HEIGHT64
#TENSOR_BLOCK_SIZE4x4 = TENSOR_BLOCK_WIDTH4 * TENSOR_BLOCK_HEIGHT4
MAX_CU = 128
PERF_THRESH = 5  # %
f_k_ind = 0
f_c_ind = 1
f_r_ind = 2
f_s_ind = 3

IMPLICIT_GEMM_ADDR_CC = 180
IMPLICIT_GEMM_LDS_ADDR_CC = 30
NUM_WAVES_PER_WG = 4


def swap(a, b):
    a ^= b
    b ^= a
    a ^= b
    return a, b


def is_prime(num):
    if num == 1:
        return False
    prime = True
    for i in range(2, int(num)):
        if num % i == 0:
            prime = False
            break
    return prime


def get_factors(num):
    factors_list = []
    for i in range(1, num+1):
        if num % i == 0:
            factors = [i, int(num // i)]
            factors_list.append(factors)
    return factors_list


def find_nxt_multiple(multiple, num):
    multiples = [i for i in range(multiple, num+1) if num%i == 0]
    return multiples[0]


class GemmRes:
    def __init__(self, alu_util_factor=0.0, chip_util_factor=0.0, speedup=0.0, cycles=1<<31, num_cu_util=0,
                 num_a_blocks=0, num_b_blocks=0, num_partitions=0, tpu_sub_res=None,
                 alu_cc=0, mem_cc=0, num_rounds=1, total_blocks=0, num_cu_util_trail=0, #Ashish added
                 num_a_block_trail=0, num_b_block_trail=0, num_partitions_trail=0, cycles_trail=0, wr_cc=0, main_instr="0",
                 threadTile="0", workGroup="0", unroll_factor=1, unroll_factor_trail=1, num_rounds_trail=1, alu_cc_trail=0,
                 mem_cc_trail=0, wr_cc_trail=0):  # Ashish added
        self.alu_util_factor = alu_util_factor
        self.chip_util_factor = chip_util_factor
        self.speedup = speedup
        self.cycles = cycles
        self.num_cu_util = num_cu_util
        self.num_a_blocks = num_a_blocks
        self.num_b_blocks = num_b_blocks
        self.num_partitions = num_partitions
        self.tpu_sub_res = tpu_sub_res
        self.alu_cc = alu_cc  # Ashish added
        self.mem_cc = mem_cc  # Ashish added
        self.num_rounds = num_rounds  # Ashish added
        self.total_blocks = total_blocks  # Ashish added
        self.num_cu_util_trail = num_cu_util_trail #Ashish added
        self.num_a_blocks_trail = num_a_block_trail #Ashish added
        self.num_b_blocks_trail = num_b_block_trail #Ashish added
        self.num_partitions_trail = num_partitions_trail #Ashish added
        self.cycles_trail = cycles_trail #Ashish added
        self.wr_cc = wr_cc #Ashish added
        self.main_instr = main_instr
        self.threadTile = threadTile
        self.workGroup = workGroup
        self.unroll_factor = unroll_factor
        self.unroll_factor_trail = unroll_factor_trail
        self.num_rounds_trail = num_rounds_trail
        self.alu_cc_trail = alu_cc_trail
        self.mem_cc_trail = mem_cc_trail
        self.wr_cc_trail = wr_cc_trail
        #self.cycles_trail = cycles_trail #Ashish added


class GemmDims:
    def __init__(self, M, N, K):
        self.M = M
        self.N = N
        self.K = K


class ImplicitGemm:
    def __init__(self, enable=True, in_dims=[], out_dims=[], filt_dims=[], strides=[], pads=[], format='nchw'):
        self.enable = enable
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.filt_dims = filt_dims
        self.strides = strides
        self.pads = pads
        self.format = format

    def get_filt_indices(self, x, c_ind):
        c = self.in_dims[c_ind]
        s = self.filt_dims[f_s_ind]
        for ihat in range(self.filt_dims[f_r_ind]):
            for jhat in range(self.filt_dims[f_s_ind]):
                for chat in range(self.filt_dims[f_c_ind]):
                    x_computed = ihat * s * c + jhat * c + chat
                    if x == x_computed:
                        return ihat, jhat, chat
        assert 0

    def get_num_unique_pixels(self, x1, y1, x0=0, y0=0, num_partitions=1):
        num_unique_pixels = 0
        if self.format == 'nchw':
            h_ind = 2
            w_ind = 3
            c_ind = 1
        elif self.format == 'nhwc':
            h_ind = 1
            w_ind = 2
            c_ind = 3
        elif self.format == 'chwn':
            h_ind = 1
            w_ind = 2
            c_ind = 0
        elif self.format == 'cnhw':
            h_ind = 2
            w_ind = 3
            c_ind = 0
        else:
            assert 0

        P = self.out_dims[h_ind]
        Q = self.out_dims[w_ind]
        H = self.in_dims[h_ind]
        W = self.in_dims[w_ind]
        C = self.in_dims[c_ind]
        u = self.strides[1]
        v = self.strides[2]
        padW = self.pads[1]
        padH = self.pads[2]
        R = self.filt_dims[f_r_ind]
        S = self.filt_dims[f_s_ind]

        if self.format == 'nhwc':
            n0 = int(y0 / (P * Q))
            c0 = x0 % C
            h0 = int(((y0 % (P * Q)) / Q) * v + x0 / (S * C))
            w0 = int((((y0 % Q) * u * C) + (x0 % (S * C))) / C)

            alternateFirstTerm = False
            extraPixelsLeft = 0
            if (w0 < padW or w0 >= (W + padW) or h0 < padH or h0 >= (H + padH)):
                if w0 < padW:
                    if h0 > padW and h0 < (H + padH):
                        h0 = h0 - padH
                        if h0 < 0:
                            h0 = 0

                if w0 > (W + padW):
                    h0 += u
                    if h0 > padW and h0 < (H + padH):
                        h0 = h0 - padH
                        if h0 < 0:
                            h0 = 0

                if h0 < padH:
                    alternateFirstTerm = True
                    if (h0 + u) > padH + 1:
                        if w0 > padW:
                            w0 = w0 - padW
                            if w0 < 0:
                                w0 = 0
                        extraPixelsLeft = ((h0 + u) - padH) * w0 * C
                    else:
                        extraPixelsLeft = 0
                    h0 = 0

                if h0 >= (H + padH):
                    h0 = 0
                    n0 += 1
                w0 = 0  # happens every time
            else:  # we aren't dynamically inserting a 0, but we need to shift the h,w coordinate systems by the padded amount
                h0 = h0 - padH
                w0 = w0 - padW

            n1 = int(y1 // (P * Q))
            c1 = x1 % C
            h1 = int(((y1 % (P * Q)) // Q) * v + x1 // (S * C))
            w1 = int((((y1 % Q) * u * C) + (x1 % (S * C))) // C)

            if w1 < padW or w1 >= (W + padW) or h1 < padH or h1 > (H + padH):
                if w1 < padW:
                    h1 = h1 - u
                    if h1 > padW and h1 < (H + padH):
                        h1 = h1 - padH

                if w1 >= (W + padW):
                    if h1 > padW and h1 < (H + padH):
                        h1 = h1 - padH

                if h1 < padH:
                    h1 = H - 1
                    n1 -= 1

                if h1 >= (H + padH):
                    h1 = H - 1

                w1 = W - 1
            else:  # we aren't dynamically inserting a 0, but we need to shift the h,w coordinate systems by the padded amount
                h1 = h1 - padH
                w1 = w1 - padW

            h0Star = h0 + n0 * H
            h1Star = h1 + n1 * H

            if not alternateFirstTerm:
                if h0 + R < padH + H:
                    extraPixelsLeft = u * w0 * C
                else:
                    extraPixelsLeft = ((padH + H - 1) - h0) * w0 * C

            if h0Star + R - 1 == h1Star:
                num_unique_pixels = (w1 - w0 + 1) * (h1Star - h0Star + 1) * C
            else:
                additionalRemovalChunk = 0
                if (h0Star + R) > h1Star and (w1 + S) < w0:
                    additionalRemovalChunk = ((R - u) * (w0 - w1) * C)
                if h1 < u:
                    extraPixelsRight = ((h1+1) * (W - w1 - 1) * C)
                else:
                    extraPixelsRight = (u * (W - w1 - 1) * C)
                num_unique_pixels = (((h1Star - h0Star) + 1) * W * C) - (extraPixelsLeft + extraPixelsRight) - additionalRemovalChunk

            if num_partitions == 1:
                # non-overlapping tiles case
                if u > R:
                    num_unique_pixels = int(num_unique_pixels * (1 - ((u - R) / u)))
                if v > S:
                    num_unique_pixels = int(num_unique_pixels * (1 - ((v - S) / v)))
            else:
                # we have a kSplit happening... we know that numLines and numPixels will always result in more area being deducted than R and S
                # these, like the logic above for non-overlapping strides are just approximations not exact unique pixel counts, but they should get close
                num_pixels = R * S / num_partitions
                num_lines = math.ceil(num_pixels / S)

                if num_pixels < v:
                    num_unique_pixels = int(num_unique_pixels * (1 - num_pixels / v))
                if num_lines < u:
                    num_unique_pixels = int(num_unique_pixels * (1 - num_lines / u))

            return num_unique_pixels, C * v
        elif self.format == 'cnhw':
            # num_unique_pixels are calculated for only one channel
            num_channels = math.ceil((x1+1) / (R*S))
            x1 = x0 + R*S - 1
            n0 = int(y0 // (P * Q))
            c0 = int(x0 // (R * S))
            h0 = int((((y0 % (P * Q)) // Q) * v) + ((x0 % (R * S)) // S))
            w0 = int(((y0 % Q) * u) + (x0 % S))

            if w0 < padW or w0 >= W + padW or h0 < padH or h0 >= H + padH:
                # Padded region corner cases
                # -------------------------
                # |  0  |     1     |  2  |
                # |-----------------------|
                # |     |           |     |
                # |     |           |     |
                # |  3  |           |  4  |
                # |     |           |     |
                # |     |           |     |
                # |-----------------------|
                # |  5  |     6     |  7  |
                # -------------------------

                if h0 < padH and w0 < padW:  # Region 0
                    if u <= R and v <= S:  # overlapping strides in both dimensions
                        h0 = 0
                        w0 = 0

                if h0 < padH and w0 >= padW and w0 < padW + W:  # Region 1
                    if u <= R and v <= S:  # overlapping strides in both dimensions
                        h0 = 0
                        w0 = 0

                if h0 < padH and w0 >= padW + W:  # Region 2
                    if u <= R and v <= S:  # overlapping strides in both dimensions
                        h0 = 0
                        w0 = 0

                if h0 >= padH and h0 < padH + H and w0 < padW:  # Region 3
                    if u <= R and v <= S:  # overlapping strides in both dimensions
                        h0 = 0
                        w0 = 0

                if h0 >= padH and h0 < padH + H and w0 >= padW + W:  # Region 4
                    if u <= R and v <= S:  # overlapping strides in both dimensions
                        if h0 + u >= padH + H:  # corner case where by shifting to the next row the stride carries us into a completely padded region
                            h0 = 0
                            w0 = 0
                            n0 += 1
                        else:
                            h0 = h0 + u
                            w0 = 0

                if h0 >= padH + H and w0 < padW:  # Region 5
                    if u <= R and v <= S:  # overlapping strides in both dimensions
                        h0 = 0
                        w0 = 0
                        n0 += 1

                if h0 >= padH + H and w0 >= padW and w0 < padW + W:  # Region 6
                    if u <= R and v <= S:  # overlapping strides in both dimensions
                        h0 = 0
                        w0 = 0
                        n0 += 1

                if h0 >= padH + H and w0 >= padW + W:  # Region 7
                    if u <= R and v <= S:  # overlapping strides in both dimensions
                        h0 = 0
                        w0 = 0
                        n0 += 1
            else:
                # we aren't dynamically inserting a 0, but we need to shift the h,w coordinate systems by the padded amount
                h0 = h0 - padH
                w0 = w0 - padW

            # if we ever hit this corner case, that means we are in the last CU and the
            # entire image space that is owned by this CU is padded 0 values.
            #if n0 == N:
            #    return 0

            n1 = int(y1 // (P * Q))
            c1 = int(x1 // (R * S))
            h1 = int((((y1 % (P * Q)) // Q) * v) + ((x1 % (R * S)) // S))
            w1 = int(((y1 % Q) * u) + (x1 % S))

            if w1 < padW or w1 >= W + padW or h1 < padH or h1 >= H + padH:
                # Padded region corner cases
                # -------------------------
                # |  0  |     1     |  2  |
                # |-----------------------|
                # |     |           |     |
                # |     |           |     |
                # |  3  |           |  4  |
                # |     |           |     |
                # |     |           |     |
                # |-----------------------|
                # |  5  |     6     |  7  |
                # -------------------------

                if h1 < padH and w1 < padW:  # Region 0
                    if u <= R and v <= S:  # overlapping strides in both dimensions
                        h1 = H - 1
                        w1 = W - 1
                        n1 -= 1

                if h1 < padH and w1 >= padW and w1 < padW + W:  # Region 1
                    if u <= R and v <= S:  # overlapping strides in both dimensions
                        h1 = H - 1
                        w1 = W - 1
                        n1 -= 1

                if h1 < padH and w1 >= padW + W:  # Region 2
                    if u <= R and v <= S:  # overlapping strides in both dimensions
                        h1 = H - 1
                        w1 = W - 1
                        n1 -= 1

                if h1 >= padH and h1 < padH + H and w1 < padW:  # Region 3
                    if u <= R and v <= S:  # overlapping strides in both dimensions
                        if h1 - u < padH:
                            h1 = H - 1
                            w1 = W - 1
                            n1 -= 1
                        else:
                            h1 = h1 - u
                            w1 = W - 1

                if h1 >= padH and h1 < padH + H and w1 >= padW + W:  # Region 4
                    if u <= R and v <= S:  # overlapping strides in both dimensions
                        # h1 = h1
                        w1 = W - 1

                if h1 >= padH + H and w1 < padW:  # Region 5
                    if u <= R and v <= S:  # overlapping strides in both dimensions
                        h1 = H - 1
                        w1 = W - 1

                if h1 >= padH + H and w1 >= padW and w1 < padW + W:  # Region 6
                    if u <= R and v <= S:  # overlapping strides in both dimensions
                        h1 = H - 1
                        w1 = W - 1

                if h1 >= padH + H and w1 >= padW + W:  # Region 7
                    if u <= R and v <= S:  # overlapping strides in both dimensions
                        h1 = H - 1
                        w1 = W - 1
            else:
                # we aren't dynamically inserting a 0, but we need to shift the h,w coordinate systems by the padded amount
                h1 = h1 - padH
                w1 = w1 - padW

            if (n1 < 0):
                return 0, 0  # the entire region for this CU is 0 padded

            if (n1 < n0):
                return 0, 0  # the entire region for this CU is 0 padded, the first filter got shifted to the next n, and the last filter got shifted to the previous n, which will only happen if the entire region is padded

            h0Star = h0 + n0 * H
            h1Star = h1 + n1 * H

            extraPixelsLeft = w0
            extraPixelsRight = W - (w1 + 1)

            num_unique_pixels = (((h1Star - h0Star) + 1) * W) - extraPixelsLeft - extraPixelsRight

            if (num_partitions > C):
                num_unique_pixels = num_unique_pixels  # don't adjust for now, leave unefficient

            return num_unique_pixels * num_channels, C * v
        else:
            assert "Unsupported implicit gemm format"


    def get_num_unique_pixels_(self, gemm_rows_per_cu, gemm_cols_per_cu):
        if self.format == 'nchw':
            h_ind = 2
            w_ind = 3
            stride_ind = 3
        elif self.format == 'nhwc':
            h_ind = 1
            w_ind = 2
            stride_ind = 3
        elif self.format == 'chwn':
            h_ind = 1
            w_ind = 2
            stride_ind = 3

        p = self.out_dims[h_ind]
        q = self.out_dims[w_ind]
        batches_per_cu = math.ceil(gemm_rows_per_cu / (p * q))
        x = gemm_cols_per_cu - 1
        y = gemm_rows_per_cu - 1
        n = batches_per_cu - 1
        h = self.in_dims[h_ind] + 2 * self.pads[h_ind]
        w = self.in_dims[w_ind] + 2 * self.pads[w_ind]
        u = self.strides[h_ind]
        v = self.strides[w_ind]
        stride = self.in_dims[stride_ind]
        r = self.filt_dims[f_r_ind]
        s = self.filt_dims[f_s_ind]

        if gemm_cols_per_cu < stride * r * s:  # Split K case
            partial_pixels = 0
            img_col_ind = 0
            if gemm_cols_per_cu > stride * s and q > 1:
                img_col_ind = (y % q) * v * stride + x % (s * stride)
                img_pixels_per_cu = math.floor(gemm_rows_per_cu / q) * q * stride
                residual_pixels = w * stride / v - (s * stride - (gemm_cols_per_cu % (s * stride)))
                img_col_ind = residual_pixels - 1 if residual_pixels > img_col_ind else img_col_ind
            else:
                if gemm_cols_per_cu > stride * v and q > 1:  # overlap between tiles
                    img_pixels_per_cu = math.floor(gemm_rows_per_cu / q) * q * stride
                    if gemm_rows_per_cu % q:
                        img_col_ind = (y % q) * v * stride + x % (s * stride) + 1
                else:  # No overlap between adjacent tiles. 3x3 kernels with stride 2 will possibly have no duplicate data in L1
                    img_pixels_per_cu = math.floor(gemm_rows_per_cu / q) * q * gemm_cols_per_cu
                    if gemm_rows_per_cu % q:
                        img_col_ind = (y % q) * gemm_cols_per_cu + x % (s * stride) + 1
            img_pixels_per_cu += (partial_pixels + img_col_ind)
        else:
            img_ch_ind = x % stride
            img_row_ind = ((y % (p * q)) // q) * v + x // (s * stride)
            img_col_ind = (((y % q) * u * stride) + (x % (s * stride))) // stride
            img_pixels_per_cu = n * h * w * stride + img_row_ind * w * stride + img_col_ind * stride + img_ch_ind + 1

        return img_pixels_per_cu, stride * v


class Winograd:
    def __init__(self, hw_config, sw_options, enable=True, m=[0, 0], r=[0, 0], opt_wino=True, split_tile=False):
        self.hw_cfg = hw_config
        self.sw_opt = sw_options
        self.enable = enable
        self.m = m
        self.r = r
        self.split_tile = split_tile
        self.opt_wino = opt_wino
        self.num_cu_per_tile = 1
        self.tile_size = (m[0] + r[0] - 1) * (m[1] + r[1] - 1)

    def use_winograd(self, M, N):
        use_wg = True
        total_blocks = math.ceil((M * N) / (self.hw_cfg.dl_instr_large_block[INSTR_M_IND]*self.hw_cfg.dl_instr_large_block[INSTR_N_IND]))
        blocks_per_cu = max(total_blocks // self.hw_cfg.num_cu_util, 1)
        out_size = 8 if self.sw_opt.fp64_inputs else 4
        tiles_per_cu = (self.tile_size / math.sqrt(self.tile_size)) if self.opt_wino else self.tile_size
        total_size = blocks_per_cu * self.hw_cfg.dl_instr_large_block[INSTR_M_IND] * self.hw_cfg.dl_instr_large_block[INSTR_M_IND] * out_size * tiles_per_cu
        if total_size > self.hw_cfg.accum_vgpr_size_per_cu:
            use_wg = False
        return use_wg


class GEMM:
    def __init__(self, hw_config, hw_options, sw_options, l2_hit_rate_act=0.0, l2_hit_rate_wt=0.0,
                 l3_hit_rate_act=0.0, act='A', winograd=None, implicit_gemm=None, prnn_opt_en=False,
                 batched_gemm=False, num_cu_per_batch=1, tpu_partition_scheme='', stashed_weights=0):
        self.hw_cfg = hw_config
        self.hw_opt = hw_options
        self.sw_opt = sw_options
        self.act = act
        self.l2_hit_rate_act = l2_hit_rate_act
        self.l2_hit_rate_wt = l2_hit_rate_wt
        self.l3_hit_rate_act = l3_hit_rate_act
        self.batched_gemm = batched_gemm
        self.prnn_opt_en = prnn_opt_en
        self.num_cu_per_batch = num_cu_per_batch
        if winograd:
            self.winograd = winograd
        else:
            self.winograd = Winograd(0, 0, False)
        if implicit_gemm:
            self.implicit_gemm = implicit_gemm
        else:
            self.implicit_gemm = ImplicitGemm(enable=False)
        self.tpu_partition_scheme = tpu_partition_scheme
        self.num_cu_width = 1
        self.num_cu_height = 1
        self.tpu_inference = self.hw_cfg.tpu_en and not self.sw_opt.training
        self.stashed_weights = stashed_weights

    def set_tensor_params(self, M, N, K, num_partitions=1, trail=0):
        global MAX_CU
        MAX_CU = self.hw_cfg.num_cu_util
        self.M = M
        self.N = N
        self.K = K
        self.in_size = get_dt_size(self.sw_opt)
        self.out_size = get_dt_size(self.sw_opt)
        dl_instr_large_block_m = self.hw_cfg.dl_instr_large_block[INSTR_M_IND]
        dl_instr_large_block_n = self.hw_cfg.dl_instr_large_block[INSTR_N_IND]
        dl_instr_large_block_k = self.hw_cfg.dl_instr_large_block[INSTR_K_IND] #if self.sw_opt.fp16_inputs or self.sw_opt.int8_inputs \
            #else self.hw_cfg.dl_instr_large_block[INSTR_K_IND]/self.in_size
        dl_instr_large_block_num_blocks = self.hw_cfg.dl_instr_large_block[INSTR_NUM_BLOCKS_IND]
        dl_instr_small_block_m = self.hw_cfg.dl_instr_small_block[INSTR_M_IND] if M >= N else self.hw_cfg.dl_instr_small_block[INSTR_N_IND]
        dl_instr_small_block_n = self.hw_cfg.dl_instr_small_block[INSTR_N_IND] if M >= N else self.hw_cfg.dl_instr_small_block[INSTR_M_IND]
        dl_instr_small_block_k = self.hw_cfg.dl_instr_small_block[INSTR_K_IND] #if self.sw_opt.fp16_inputs or self.sw_opt.int8_inputs \
            #else self.hw_cfg.dl_instr_small_block[INSTR_K_IND]/self.in_size
        dl_instr_small_block_num_blocks = self.hw_cfg.dl_instr_small_block[INSTR_NUM_BLOCKS_IND]
        if (M >= dl_instr_large_block_m) and (N >= dl_instr_large_block_n):
            K_partitioned = math.ceil(K / num_partitions)
            # Use large block DL instruction; If K_partitioned not divisible by dl_instr_large_block_k and number of K blocks < 10 then look into other instructions if available
            if (K_partitioned > dl_instr_large_block_k and
                    (not K_partitioned % dl_instr_large_block_k or K_partitioned // dl_instr_large_block_k > 10 and self.hw_cfg.dl_other_instr_block)):
                # assert(dl_instr_large_block_num_blocks == 1)
                self.TENSOR_BLOCK_WIDTH = dl_instr_large_block_n if M >= N else dl_instr_large_block_n * dl_instr_large_block_num_blocks
                self.TENSOR_BLOCK_HEIGHT = dl_instr_large_block_m * dl_instr_large_block_num_blocks if M >= N else dl_instr_large_block_m
                self.TENSOR_BLOCK_K = dl_instr_large_block_k
                inst_m = dl_instr_large_block_m
                inst_n = dl_instr_large_block_n
                inst_k = dl_instr_large_block_k
                inst_nblock = dl_instr_large_block_num_blocks
            else: # Look in other DL instructions
                instr = self.hw_cfg.dl_other_instr_block
                #if len(self.hw_cfg.dl_other_instr_block) > 1:
                #    for instr_temp in self.hw_cfg.dl_other_instr_block:
                #        if K_partitioned >= instr_temp[INSTR_K_IND]:
                #            instr = instr_temp
                #        else:
                #            continue
                #instr = [instr for instr in self.hw_cfg.dl_other_instr_block if K_partitioned >= instr[INSTR_K_IND]]
                #instr = instr[0]
                self.TENSOR_BLOCK_WIDTH = instr[INSTR_N_IND] if M >= N else instr[INSTR_N_IND] * instr[INSTR_NUM_BLOCKS_IND]
                self.TENSOR_BLOCK_HEIGHT = instr[INSTR_M_IND] * instr[INSTR_NUM_BLOCKS_IND] if M >= N else instr[INSTR_M_IND]
                self.TENSOR_BLOCK_K = instr[INSTR_K_IND] # if self.sw_opt.fp16_inputs or self.sw_opt.int8_inputs \
                    #else instr[INSTR_K_IND]/self.in_size
                #self.TENSOR_BLOCK_SIZE = instr[INSTR_M_IND] * instr[INSTR_N_IND] * instr[INSTR_NUM_BLOCKS_IND]
                #self.TENSOR_A_BLOCK_SIZE = self.in_size * instr[INSTR_M_IND] * instr[INSTR_NUM_BLOCKS_IND] * instr[INSTR_K_IND] if M >= N else \
                #    self.in_size * self.TENSOR_BLOCK_K * instr[INSTR_N_IND]
                #self.TENSOR_B_BLOCK_SIZE = self.in_size * instr[INSTR_K_IND] * instr[INSTR_N_IND] if M >= N else \
                #    self.in_size * instr[INSTR_K_IND] * instr[INSTR_N_IND] * instr[INSTR_NUM_BLOCKS_IND]
                #self.TENSOR_OUT_BLOCK_SIZE = self.out_size * instr[INSTR_M_IND] * instr[INSTR_N_IND] * instr[INSTR_NUM_BLOCKS_IND]
                #total_macs = instr[INSTR_M_IND] * instr[INSTR_N_IND] * instr[INSTR_K_IND] * instr[INSTR_NUM_BLOCKS_IND]
                inst_m = instr[INSTR_M_IND]
                inst_n = instr[INSTR_N_IND]
                inst_k = instr[INSTR_K_IND]
                inst_nblock = instr[INSTR_NUM_BLOCKS_IND]
        else:  # Use small block DL instruction
            self.TENSOR_BLOCK_WIDTH = dl_instr_small_block_n if M >= N else dl_instr_small_block_n * dl_instr_small_block_num_blocks
            self.TENSOR_BLOCK_HEIGHT = dl_instr_small_block_m * dl_instr_small_block_num_blocks if M >= N else dl_instr_small_block_m
            self.TENSOR_BLOCK_K = dl_instr_small_block_k
            #self.TENSOR_BLOCK_SIZE = dl_instr_small_block_m * dl_instr_small_block_n * dl_instr_small_block_num_blocks
            #self.TENSOR_A_BLOCK_SIZE = self.in_size * dl_instr_small_block_m * dl_instr_small_block_num_blocks * dl_instr_small_block_k if M >= N else \
            #                            self.in_size * dl_instr_small_block_m * dl_instr_small_block_k
            #self.TENSOR_B_BLOCK_SIZE = self.in_size * dl_instr_small_block_k * dl_instr_small_block_n if M >= N else \
            #                            self.in_size * dl_instr_small_block_k * dl_instr_small_block_n * dl_instr_small_block_num_blocks
            #self.TENSOR_OUT_BLOCK_SIZE = self.out_size * dl_instr_small_block_m * dl_instr_small_block_n * dl_instr_small_block_num_blocks
            #total_macs = dl_instr_small_block_m * dl_instr_small_block_n * dl_instr_small_block_k * dl_instr_small_block_num_blocks
            inst_m = dl_instr_small_block_m
            inst_n = dl_instr_small_block_n
            inst_k = dl_instr_small_block_k
            inst_nblock = dl_instr_small_block_num_blocks

        self.TENSOR_BLOCK_SIZE = self.TENSOR_BLOCK_WIDTH * self.TENSOR_BLOCK_HEIGHT
        self.TENSOR_A_BLOCK_SIZE = self.in_size * self.TENSOR_BLOCK_HEIGHT * self.TENSOR_BLOCK_K
        self.TENSOR_B_BLOCK_SIZE = self.in_size * self.TENSOR_BLOCK_WIDTH * self.TENSOR_BLOCK_K
        self.TENSOR_OUT_BLOCK_SIZE = self.out_size * self.TENSOR_BLOCK_HEIGHT * self.TENSOR_BLOCK_WIDTH
        total_macs = self.TENSOR_BLOCK_HEIGHT * self.TENSOR_BLOCK_WIDTH * self.TENSOR_BLOCK_K
        self.M_BLOCKS = math.ceil(M / self.TENSOR_BLOCK_HEIGHT) if trail else int(max(M // self.TENSOR_BLOCK_HEIGHT, 1))
        self.N_BLOCKS = math.ceil(N / self.TENSOR_BLOCK_WIDTH) if trail else int(max(N // self.TENSOR_BLOCK_WIDTH, 1))
        self.K_BLOCKS = int(math.ceil(K / self.TENSOR_BLOCK_K))
        self.result_vgpr_size = self.hw_cfg.accum_vgpr_size_per_cu if self.hw_cfg.accum_vgpr_size_per_cu > 0 else \
            (self.hw_cfg.arch_vgpr_size_per_cu if self.sw_opt.ABinLDS else self.hw_cfg.arch_vgpr_size_per_cu - 65536) #allow some space for A/B in vgprs Ashish added
        self.MAX_BLOCKS = max(int(self.result_vgpr_size / (2 * self.TENSOR_OUT_BLOCK_SIZE if self.sw_opt.fp16_inputs or
                                                                                    self.sw_opt.int8_inputs or
                                                                                    self.sw_opt.bf16_inputs else
                                                      self.TENSOR_OUT_BLOCK_SIZE)), 1)
        #print(self.MAX_BLOCKS, " ", self.hw_cfg.dl_instr_large_block[0], " ", self.hw_cfg.dl_instr_large_block[1], " ", self.hw_cfg.dl_instr_large_block[2])
        dl_macs_per_cu = get_dl_macs_per_cu(self.hw_cfg, self.sw_opt)
        self.TENSOR_LATENCY = math.ceil(total_macs / dl_macs_per_cu)
        return inst_m, inst_n, inst_k, inst_nblock

    # This method gives consideration to factors being multiple of M/N dimension
    def find_block_multiples_2(self, num_blocks):
        factors = [[i, int(num_blocks//i)] for i in range(1, int(math.sqrt(num_blocks))+1) if num_blocks % i == 0]
        final_ratio = num_blocks
        final_factor = [1, num_blocks]
        for i in range(len(factors)):
            factor_1 = factors[i][0]
            factor_2 = factors[i][1]
            ratio = factor_2 / factor_1
            if ((self.M_BLOCKS > self.N_BLOCKS and (self.N_BLOCKS % factor_1 == 0 or self.N_BLOCKS % factor_2 == 0)) or
                (self.M_BLOCKS < self.N_BLOCKS and (self.M_BLOCKS % factor_1 == 0 or self.M_BLOCKS % factor_2 == 0))) and \
                ratio < final_ratio:
                final_factor = factors[i]
                final_ratio = ratio
        return final_factor[0], final_factor[1]

    # This method finds optimum squarish factors
    def find_block_multiples_1(self, num):
        a = math.floor(math.sqrt(num))
        b = max(a - 1, 1)
        while a * b != num:
            b += 1
            if a * b > num:
                a = max(a - 1, 1)
                b = max(a - 1, 1)
        return a, b

    def get_trail_cu_multiples(self, num_cu, num_blocks, is_partial=False):
        cu_multiples = []
        if num_blocks < num_cu:
            cu_multiple = num_blocks
            cu_multiples.append(cu_multiple)
            iter = 2
            if is_partial:
                while 1:
                    cu_multiple = num_blocks * iter
                    iter += 1
                    if cu_multiple < self.hw_cfg.num_cu_per_cluster:
                        cu_multiples.append(cu_multiple)
                    else:
                        break
        else:
            for i in range(num_cu, 1, -1):
                if num_blocks % i == 0:
                    cu_multiples.append(i)
        return cu_multiples

    def get_tpu_ramp_down_latency(self, gemm_dims, num_cu, num_a_blocks, num_b_blocks, num_rounds, num_partitions=1):
        if self.hw_cfg.tpu_en and not self.sw_opt.training:  # weights are assumed to be pinned
            l2_rd_bw_per_se = self.hw_cfg.l2_read_buses_per_se * self.hw_cfg.l2_read_bus_width
            num_cu_per_se, num_se_util = self.get_cu_se_util(num_cu)
            bcast_a, bcast_b = self.get_bcast_factors(num_a_blocks, num_b_blocks, num_cu_per_se, num_se_util, num_partitions)
            bcast_factor = bcast_a if self.act == 'A' else bcast_b
            bcast_cluster_sz = num_cu_per_se // bcast_factor
            l2_repeater_latency = self.hw_cfg.l2_repeater_latency
            if self.implicit_gemm.enable:
                num_pixels_per_block = 0
                iter = 0
                x1 = self.TENSOR_BLOCK_WIDTH - 1
                # For implicit GEMM assumption is unique pixels are fetched for first BLOCK_HEIGHT X K
                while not num_pixels_per_block:
                    iter += 1
                    y1 = iter*self.TENSOR_BLOCK_HEIGHT - 1
                    num_pixels_per_block, _ = self.implicit_gemm.get_num_unique_pixels(x1, y1, x0=0, y0=(iter-1)*self.TENSOR_BLOCK_HEIGHT,
                                                                                       num_partitions=num_partitions)
                startup_latency = num_pixels_per_block * self.in_size / l2_rd_bw_per_se
            else:
                startup_latency = math.ceil((self.TENSOR_A_BLOCK_SIZE if self.act == 'A' else self.TENSOR_B_BLOCK_SIZE) / l2_rd_bw_per_se)  # latency to push single instruction block
            ramp_up_latency = (startup_latency * num_cu_per_se / bcast_factor) + l2_repeater_latency * (bcast_cluster_sz - 1) + \
                                l2_repeater_latency * (math.ceil(bcast_factor / SLICE_SPLIT) - 1) * bcast_cluster_sz
            # Write latency = latency of last XPE in stack to write out results
            if num_partitions < num_cu_per_se:
                write_latency = num_cu_per_se * l2_repeater_latency / SLICE_SPLIT
            else:  # num_partitions == num_cu_per_se is a special case where the XPE physically closest to TL2 writes out data
                write_latency = l2_repeater_latency
            ramp_down_latency = ramp_up_latency + write_latency
        else:
            assert 0  # Only for inference case for now
        return ramp_down_latency * num_rounds

    def get_vgpr_utils(self, gemm_dims, num_a_blocks, num_b_blocks, num_partitions):
        data_size = get_dt_size(self.sw_opt)
        num_wt_blocks = num_b_blocks if self.act == 'A' else num_a_blocks
        num_act_blocks = num_a_blocks if self.act == 'A' else num_b_blocks
        vgpr_util_bytes_wt = num_wt_blocks * self.TENSOR_BLOCK_WIDTH * gemm_dims.K / num_partitions * data_size
        vgpr_util_bytes_res = num_a_blocks * num_b_blocks * \
                              (self.TENSOR_OUT_BLOCK_SIZE * 2 if self.sw_opt.fp16_inputs or self.sw_opt.int8_inputs or self.sw_opt.bf16_inputs else self.TENSOR_OUT_BLOCK_SIZE)
        vgpr_util_bytes_act = self.get_vgpr_act_util_bytes(gemm_dims, num_act_blocks, num_partitions)
        # Adding activation bytes to result since similar to result area, activation area in SRAM will be reused
        vgpr_util_bytes_res += vgpr_util_bytes_act
        return vgpr_util_bytes_wt, vgpr_util_bytes_res

    def get_cu_se_util(self, num_cu):
        if self.hw_opt.l2_bcast_en:  # In case of L2 bcast and small num_cu all CUs are assigned to a single SE as much as possible to increase bcast capability
            num_cu_per_se = min(num_cu, math.ceil(self.hw_cfg.num_cu_per_cluster / self.hw_cfg.num_se_per_cluster))
            num_se_util = int((num_cu-1) // (self.hw_cfg.num_cu_per_cluster / self.hw_cfg.num_se_per_cluster)) + 1
        else:  # Assumption is CUs are spread across SEs as much as possible
            num_cu_per_se = math.ceil(num_cu / self.hw_cfg.num_se_per_cluster)
            num_se_util = min(self.hw_cfg.num_se_per_cluster, num_cu)
        return int(num_cu_per_se), int(num_se_util)

    def dimension_check(self, M, N, num_a_blocks, num_b_blocks):
        blocks_per_cu = num_a_blocks * num_b_blocks
        M_blocks_pred = num_a_blocks * self.TENSOR_BLOCK_HEIGHT
        N_blocks_pred = num_b_blocks * self.TENSOR_BLOCK_WIDTH
        if N < N_blocks_pred:
            num_b_blocks = N / self.TENSOR_BLOCK_WIDTH
            num_a_blocks = math.floor(blocks_per_cu / num_b_blocks)
        if M < M_blocks_pred:
            num_a_blocks = M / self.TENSOR_BLOCK_HEIGHT
            num_b_blocks = math.floor(blocks_per_cu / num_a_blocks)

        return num_a_blocks, num_b_blocks

    def winograd_check(self, num_a_blocks, num_b_blocks, blocks_per_cu):
        num_rounds = 1
        # This assumes we perform outer product on At*tile_res and inner product on (At*tile_res)*A
        # This reduces VGPR memory requirement by 1/4th in case of F(2,3)
        out_size = 8 if self.sw_opt.fp64_inputs else 4
        tiles_per_cu = (self.winograd.tile_size / math.sqrt(self.winograd.tile_size)) if self.winograd.opt_wino else \
                        self.winograd.tile_size
        total_size = blocks_per_cu * self.TENSOR_BLOCK_SIZE * out_size * tiles_per_cu
        if total_size > self.hw_cfg.accum_vgpr_size_per_cu:
            new_blocks_per_cu = self.MAX_BLOCKS / tiles_per_cu
            if self.winograd.split_tile:
                raise NotImplementedError  # For now
                #cu_factor = math.ceil(num_a_blocks * num_b_blocks / blocks_per_cu)
                #while self.winograd.tile_size % cu_factor:
                #    cu_factor += 1
                #self.winograd.num_cu_per_tile = cu_factor
                #assert (total_size / self.winograd.num_cu_per_tile <= self.hw_config.accum_vgpr_size_per_cu)
            else:
                num_a_blocks, num_b_blocks = self.find_block_multiples_1(new_blocks_per_cu)
                num_rounds = math.ceil(blocks_per_cu / new_blocks_per_cu)
        return num_a_blocks, num_b_blocks, num_rounds

    def get_cu_partitions(self, K, num_cu, trail=0, num_partitions=1):
        num_cu_per_partition = []
        num_partition_list = []
        if self.hw_cfg.tpu_en:
            num_cu_per_se, _ = self.get_cu_se_util(num_cu)
            max_partitions = num_cu_per_se
        else:
            max_partitions = 8

        if self.sw_opt.disable_trail_optimization and trail:
            #if not num_cu%num_partitions and not K%num_partitions:
            num_partition_list.append(num_partitions)
            num_cu_per_partition.append(math.ceil(num_cu/num_partitions))
            #else:
            #    num_partition_list.append(1)
            #    num_cu_per_partition.append(num_partitions)
        else:
            for x in range(1, max_partitions+1):
                if not num_cu % x and not K % x:
                    if self.hw_cfg.tpu_en:
                        # For ML Chiplet make sure K partitions is divisible by PEs used per slice
                        if num_cu_per_se % x:
                            continue
                    num_cu_per_partition.append(int(num_cu / x))
                    num_partition_list.append(x)
        return list(reversed(num_cu_per_partition)), list(reversed(num_partition_list))

    def get_cu_HxW(self, gemm_dims, num_a_blocks, num_b_blocks, num_cu, total_blocks):
        if gemm_dims.M >= gemm_dims.N:
            num_cu_width = self.N_BLOCKS / num_b_blocks
            if num_cu_width <= num_cu:
                num_cu_height = min(num_cu, total_blocks) // num_cu_width
            else:
                num_cu_height = self.M_BLOCKS / num_a_blocks
                num_cu_width = min(num_cu, total_blocks) // num_cu_height
        else:
            num_cu_height = self.M_BLOCKS / num_a_blocks
            if num_cu_height <= num_cu:
                num_cu_width = min(num_cu, total_blocks) // num_cu_height
            else:
                num_cu_width = self.N_BLOCKS / num_b_blocks
                num_cu_height = min(num_cu, total_blocks) // num_cu_width
        return num_cu_width, num_cu_height

    def get_vgpr_act_util_bytes(self, gemm_dims, num_act_blocks, num_partitions):
        # For implicit GEMM assumption is that the unique pixels are pulled into the SRAM for a given layer and
        # pieces from this memory would be sent to MAC unit based on implicit GEMM addressor
        # For 1x1 conv filter there is no reuse, so no need to pull in unique pixels in SRAM upfront
        if self.implicit_gemm.enable and not (self.implicit_gemm.filt_dims[f_r_ind] == 1 and self.implicit_gemm.filt_dims[f_s_ind] == 1):
            x1 = self.TENSOR_BLOCK_WIDTH - 1
            y1 = int(num_act_blocks * self.TENSOR_BLOCK_HEIGHT) - 1
            num_unique_pixels, _ = self.implicit_gemm.get_num_unique_pixels(x1, y1, num_partitions=num_partitions)
            vgpr_util_bytes_act = num_unique_pixels * self.in_size * 2  # Assuming double buffering
        else:

            vgpr_util_bytes_act = min(num_act_blocks, PIPELINE_ACTIVATION_BLOCKS) * self.TENSOR_A_BLOCK_SIZE
        return vgpr_util_bytes_act

    def get_num_rounds(self, gemm_dims, num_a_blocks, num_b_blocks, num_partitions):
        num_rounds = 1
        num_act_blocks_per_cu = num_a_blocks if self.act == 'A' else num_b_blocks
        num_wt_blocks_per_cu = num_b_blocks if self.act == 'A' else num_a_blocks
        vgpr_util_bytes_act = self.get_vgpr_act_util_bytes(gemm_dims, num_act_blocks_per_cu, num_partitions)
        vgpr_util_bytes_res = num_act_blocks_per_cu * num_wt_blocks_per_cu * \
                              (self.TENSOR_OUT_BLOCK_SIZE * 2 if self.sw_opt.fp16_inputs or self.sw_opt.int8_inputs or self.sw_opt.bf16_inputs else self.TENSOR_OUT_BLOCK_SIZE)
        # If activation blocks occupies more than quarter of SRAM then perform heavy partitioning for activations
        assert 'moderate' in self.tpu_partition_scheme
        sram_thresh = MODERATE_LV1_SRAM_THRESH if self.tpu_partition_scheme == 'moderate_lv1' else MODERATE_LV2_SRAM_THRESH
        if (vgpr_util_bytes_act + vgpr_util_bytes_res) >= self.hw_cfg.arch_vgpr_size_per_cu / sram_thresh:
            while (vgpr_util_bytes_act + vgpr_util_bytes_res) / num_rounds >= self.hw_cfg.arch_vgpr_size_per_cu / sram_thresh and \
                    num_act_blocks_per_cu // num_rounds > 1:
                num_rounds += 1
        else:  # Apply light partitioning for activations
            num_rounds = 2 if num_act_blocks_per_cu // 2 else 1

        num_a_blocks = int(num_a_blocks // num_rounds) if self.act == 'A' else num_a_blocks
        num_b_blocks = num_b_blocks if self.act == 'A' else int(num_b_blocks // num_rounds)

        # Create most squarish result area from each round
        #factors_list = get_factors(num_rounds)
        #diff = 1 << 31
        #num_rounds_a = num_rounds
        #num_rounds_b = 1
        #for factors in factors_list:
        #    squarish_factor = (num_a_blocks // factors[0]) / (num_b_blocks // factors[1])
        #    squarish_factor = 1 / squarish_factor if squarish_factor < 1 else squarish_factor
        #    curr_diff = abs(1 - squarish_factor)  # difference from ideal squarish factor of 1
        #    if curr_diff < diff:
        #        diff = curr_diff
        #        num_rounds_a = factors[0]
        #        num_rounds_b = factors[1]
        #num_a_blocks //= num_rounds_a
        #num_b_blocks //= num_rounds_b
        return num_a_blocks, num_b_blocks, num_rounds

    def get_partition(self, num_cu_left, wr_left, wr_right, algo='balance_all'):
        if algo == 'balance_all': #balances A+B fetch as well as result writes
            numer = ((self.hw_cfg.num_cu_util * num_cu_left - num_cu_left ** 2) * (wr_right - wr_left + self.N_BLOCKS * self.K_BLOCKS)) + \
                    (num_cu_left * self.M_BLOCKS * self.K_BLOCKS)
            denom = (self.hw_cfg.num_cu_util * self.M_BLOCKS) + (2 * self.hw_cfg.num_cu_util * num_cu_left * self.N_BLOCKS) - \
                    (2 * num_cu_left ** 2 * self.N_BLOCKS)
        elif algo == 'balance_A+B_fetch':
            numer = (self.hw_cfg.num_cu_util * num_cu_left * self.N_BLOCKS * self.K_BLOCKS) + (num_cu_left * self.M_BLOCKS * self.K_BLOCKS) - \
                    (num_cu_left ** 2 * self.N_BLOCKS * self.K_BLOCKS)
            denom = (self.hw_cfg.num_cu_util * self.M_BLOCKS) + (2 * self.hw_cfg.num_cu_util * num_cu_left * self.N_BLOCKS) - \
                    (2 * num_cu_left ** 2 * self.N_BLOCKS)
        elif algo == 'balance_A_fetch':
            numer = (num_cu_left * self.K_BLOCKS)
            denom = self.hw_cfg.num_cu_util
        else:
            raise NotImplementedError

        partition = math.floor(numer / denom)
        if partition == 0:
            partition = math.ceil(numer / denom)

        return partition

    def get_speedup_factor(self, num_cu_util, chip_util_factor):
        cu_speedup_factor = (num_cu_util / self.hw_cfg.num_cu_util) * self.hw_cfg.cu_speedup
        alu_speedup_factor = (chip_util_factor / 100) * self.hw_cfg.alu_speedup_per_cu
        total_speedup_factor = cu_speedup_factor * alu_speedup_factor
        return total_speedup_factor

    def persistent_rnn_check(self, m_blocks, n_blocks, k_blocks):
        if self.prnn_opt_en:
            if self.act == 'A':
                weight_size_per_cu = n_blocks * self.TENSOR_BLOCK_WIDTH * k_blocks * self.TENSOR_BLOCK_HEIGHT * self.in_size
            else:
                weight_size_per_cu = m_blocks * self.TENSOR_BLOCK_HEIGHT * k_blocks * self.TENSOR_BLOCK_WIDTH * self.in_size
            if weight_size_per_cu > self.hw_cfg.arch_vgpr_size_per_cu:  # disable prnn if weights cannot fit into VGPRs
                return False
            else:
                return True

    def get_effective_read_bw(self, num_a_blocks, num_b_blocks, num_cu, num_cu_per_se, num_partitions=1, bcast_bw_redx_a=1, bcast_bw_redx_b=1):
        l2_bw_per_se = self.hw_cfg.l2_read_buses_per_se * self.hw_cfg.l2_read_bus_width
        l2_rd_bw_per_cu = l2_bw_per_se / (self.hw_cfg.num_cu_per_cluster / self.hw_cfg.num_se_per_cluster) if self.hw_cfg.tpu_en else \
                           min(l2_bw_per_se / num_cu_per_se, self.hw_cfg.l1_bw)
        l3_rd_bw_per_cu = self.hw_cfg.l3_bw / self.hw_cfg.num_cu_per_cluster
        hbm_rd_bw_per_cu = self.hw_cfg.hbm_bw / self.hw_cfg.num_cu_per_cluster

        num_cu_gemm = num_cu - (self.hw_cfg.num_cu - self.hw_cfg.num_cu_util)
        # Compute approximate l2 hit rates for A and B matrices
        # Logic: Out of total 'num_a_blocks' one CU needs to approx. fetch 'num_a_blocks/num_cu_width'
        # unique blocks per iteration while others are hit in L2
        l2_hit_rate_a = (num_a_blocks - math.ceil(num_a_blocks / self.num_cu_width)) / num_a_blocks
        l2_hit_rate_b = (num_b_blocks - math.ceil(num_b_blocks / self.num_cu_height)) / num_b_blocks
        # In case of broadcast effective L2 hit rates are decreased as broadcast delivers duplicate data to multiple CUs in a SE in one shot
        if bcast_bw_redx_a > 1:
            l2_hit_rate_a = 0 if self.num_cu_width <= self.hw_opt.l2_bcast_cluster_sz else l2_hit_rate_a / bcast_bw_redx_a
        if bcast_bw_redx_b > 1:
            l2_hit_rate_b = 0 if self.num_cu_height <= self.hw_opt.l2_bcast_cluster_sz else l2_hit_rate_b / bcast_bw_redx_b

        l2_hit_rate_a = max(self.l2_hit_rate_act, l2_hit_rate_a) if self.act == 'A' else \
                        max(self.l2_hit_rate_wt, l2_hit_rate_a)
        l2_hit_rate_b = max(self.l2_hit_rate_wt, l2_hit_rate_b) if self.act == 'A' else \
                        max(self.l2_hit_rate_act, l2_hit_rate_b)

        l3_hit_rate_a = self.l3_hit_rate_act if self.act == 'A' else 0
        l3_hit_rate_b = self.l3_hit_rate_act if self.act == 'B' else 0
        effective_rd_bw_a = l2_rd_bw_per_cu * l2_hit_rate_a + \
                            l3_rd_bw_per_cu * (1 - l2_hit_rate_a) * l3_hit_rate_a + \
                            hbm_rd_bw_per_cu * (1 - l2_hit_rate_a) * (1 - l3_hit_rate_a)
        effective_rd_bw_b = l2_rd_bw_per_cu * l2_hit_rate_b + \
                            l3_rd_bw_per_cu * (1 - l2_hit_rate_b) * l3_hit_rate_b + \
                            hbm_rd_bw_per_cu * (1 - l2_hit_rate_b) * (1 - l3_hit_rate_b)

        return effective_rd_bw_a, effective_rd_bw_b, l2_hit_rate_a, l2_hit_rate_b

    def get_aux_alu_cc(self, gemm_dims, num_a_blocks, num_b_blocks, is_partial=False, num_partitions=1):
        aux_alu_cc_per_res = 0
        b_partition_height = math.floor(gemm_dims.K / num_partitions)
        cu_partition_width = b_partition_height if is_partial else gemm_dims.K
        num_iter = max(math.ceil(cu_partition_width / self.TENSOR_BLOCK_K),1)
        if self.implicit_gemm.enable and (self.implicit_gemm.filt_dims[F_R_IND] > 1 or self.implicit_gemm.filt_dims[F_S_IND] > 1) \
            and not self.hw_cfg.tpu_en:
            addr_alu_cc = IMPLICIT_GEMM_ADDR_CC * math.ceil(num_a_blocks / NUM_WAVES_PER_WG)
            lds_addr_cc = IMPLICIT_GEMM_LDS_ADDR_CC * num_b_blocks  # assuming B blocks shared through LDS
            aux_alu_cc_per_res = ((addr_alu_cc + lds_addr_cc) * num_iter)
        return aux_alu_cc_per_res

    def get_alu_cycles(self, gemm_dims, num_a_blocks, num_b_blocks, is_partial=False, num_partitions=1,
                       is_nonuniform_partial=False, partition_blocks_per_cu=1):
        num_a_X_b = num_a_blocks * num_b_blocks
        if is_nonuniform_partial:
            alu_cc_per_res = self.TENSOR_LATENCY * partition_blocks_per_cu * (num_b_blocks if self.M >= self.N else num_a_blocks)
        else:
            # compute ALU cc per result area partition
            alu_cc_per_iter = num_a_X_b * self.TENSOR_LATENCY
            b_partition_height = math.floor(gemm_dims.K / num_partitions)
            cu_partition_width = b_partition_height if is_partial else gemm_dims.K
            num_iter = max(math.ceil(cu_partition_width / self.TENSOR_BLOCK_K), 1)
            alu_cc_per_res = alu_cc_per_iter * num_iter
            if self.winograd.enable:
                alu_cc_per_res *= math.ceil(self.winograd.tile_size / self.winograd.num_cu_per_tile)
        return alu_cc_per_res

    def get_memory_cycles(self, gemm_dims, num_a_blocks, num_b_blocks, is_c_fetched,
                          num_cu=MAX_CU, is_partial=False, num_partitions=1,
                          is_nonuniform_partial=False, partition_blocks_per_cu=1,
                          stash_weights=False):
        bcast_bw_redx_a, bcast_bw_redx_b = 1, 1
        b_partition_height = math.floor(gemm_dims.K / (1 if is_nonuniform_partial else num_partitions))
        cu_partition_width = b_partition_height if is_partial else gemm_dims.K
        num_iter = max(math.ceil(cu_partition_width / self.TENSOR_BLOCK_K), 1)
        num_a_x_b = num_a_blocks * num_b_blocks
        # In multi-GPU setup some CUs are performing allReduce operation which makes num_cu_util < num_cu but
        # when computing BW/CU need to consider all CUs as entire machine may be occupied at a given time
        num_cu_util = num_cu + (self.hw_cfg.num_cu - self.hw_cfg.num_cu_util)
        num_cu_per_se, num_se_util = self.get_cu_se_util((num_cu * self.hw_cfg.num_se_per_cluster) if self.batched_gemm else num_cu)

        if (self.sw_opt.fp16_inputs or self.sw_opt.int8_inputs or self.sw_opt.bf16_inputs) and num_partitions > 1:
            # In case of TPU, CUs/XPEs which merge partial results are assigned to same SE/Slice.
            # So if XPEs across different partitions fit within a slice/SE then final result write out is still FP16
            if self.hw_cfg.tpu_en and math.ceil(num_partitions / num_cu_per_se) == 1:
                out_size = self.TENSOR_OUT_BLOCK_SIZE
            else:
                out_size = 3 * self.TENSOR_OUT_BLOCK_SIZE  # Penalize next layer for reading FP32
        else:
            out_size = self.TENSOR_OUT_BLOCK_SIZE
        num_a_x_b_block_size = num_a_x_b * out_size
        if self.winograd.enable:
            num_a_x_b_block_size *= (self.winograd.m[0] * self.winograd.m[1])

        if self.hw_opt.stacked_mem_en:
            l2_wr_bw = self.hw_cfg.stacked_mem_bw
        else:
            l2_bw_per_se = self.hw_cfg.l2_write_buses_per_se * self.hw_cfg.l2_write_bus_width
            l2_wr_bw = min(l2_bw_per_se, min(l2_bw_per_se / num_cu_per_se, self.hw_cfg.l1_bw) * num_cu_per_se) * num_se_util if self.hw_cfg.l1_bw else \
                        l2_bw_per_se * num_se_util

        wr_bytes_per_cu = 0
        if is_nonuniform_partial:
            if self.hw_opt.global_atomics:
                max_num_partitions = num_partitions
                atomics_redx_factor = 2
                if self.hw_cfg.tpu_en:  # For TPU Chiplet assumption is we have write accumulate busses within SEs/Slices and CU/XPEs which merge partial results are assigned to same SE/Slice
                    max_num_partitions = math.ceil(num_partitions / num_cu_per_se)
                    atomics_redx_factor = 2 if max_num_partitions > 1 else 1
                wr_bytes_per_cu = min(num_partitions, max_num_partitions) * num_a_x_b_block_size / atomics_redx_factor if not self.sw_opt.training or not is_c_fetched else \
                           (min(num_partitions, max_num_partitions) + 1) * num_a_x_b_block_size / atomics_redx_factor
                K_blocks = gemm_dims.K / self.TENSOR_BLOCK_K
                if partition_blocks_per_cu % K_blocks:
                    remainder_blocks = math.ceil(gemm_dims.N / self.TENSOR_BLOCK_WIDTH) if self.M >= self.N else \
                                        math.ceil(gemm_dims.M / self.TENSOR_BLOCK_HEIGHT)
                    wr_bytes_per_cu += (remainder_blocks * out_size)
            else:
                assert 0  # assert for now
        elif is_partial:
            if self.hw_opt.global_atomics:
                max_num_partitions = num_partitions
                atomics_redx_factor = 2
                if self.hw_cfg.tpu_en:  # For TPU Chiplet assumption is we have write accumulate busses within SEs/Slices and CU/XPEs which merge partial results are assigned to same SE/Slice
                    max_num_partitions = math.ceil(num_partitions / num_cu_per_se)
                    atomics_redx_factor = 2 if max_num_partitions > 1 else 1
                wr_bytes_per_cu = min(num_partitions, max_num_partitions) * num_a_x_b_block_size / atomics_redx_factor if not self.sw_opt.training or not is_c_fetched else \
                           (min(num_partitions, max_num_partitions) + 1) * num_a_x_b_block_size / atomics_redx_factor
            elif self.hw_opt.cross_cu_share_en:
                wr_bytes_per_cu = (num_partitions - 1) * num_a_x_b_block_size if not self.sw_opt.training or not is_c_fetched else \
                            num_partitions * num_a_x_b_block_size
            else:
                wr_bytes_per_cu = 2 * num_partitions * num_a_x_b_block_size if is_c_fetched else \
                           (2 * num_partitions - 1) * num_a_x_b_block_size
        else:
            if is_c_fetched:
                if not self.sw_opt.training and self.hw_opt.global_atomics:
                    wr_bytes_per_cu = num_a_x_b_block_size
                else:
                    wr_bytes_per_cu = 2 * num_a_x_b_block_size
            else:
                wr_bytes_per_cu = num_a_x_b_block_size

        if self.winograd.enable and self.winograd.split_tile:
            wr_bytes_per_cu += num_a_x_b_block_size * self.winograd.num_cu_per_tile * \
                        (self.winograd.tile_size / self.winograd.num_cu_per_tile)

        if self.hw_cfg.tpu_en:
            total_wr_bytes = wr_bytes_per_cu * math.ceil(num_cu_per_se / num_partitions) * num_se_util
        else:
            total_wr_bytes = wr_bytes_per_cu * num_cu / num_partitions
        if self.hw_cfg.num_cu_clusters > 1:
            l2_wr_hit_rate = 0.0
            l3_wr_hit_rate = min(self.hw_cfg.l3_size / total_wr_bytes, 1)
        else:
            l2_wr_hit_rate = min(self.hw_cfg.l2_size / total_wr_bytes, 1)
            l3_wr_hit_rate = 0.0
            if l2_wr_hit_rate < 1 and self.hw_cfg.l3_size > 0 and self.hw_cfg.l3_per_cluster:
                l3_wr_hit_rate = min(self.hw_cfg.l3_size / ((wr_bytes_per_cu * num_cu) - self.hw_cfg.l2_size), 1)

        effective_wr_bw = l2_wr_hit_rate * l2_wr_bw + (1 - l2_wr_hit_rate) * l3_wr_hit_rate * self.hw_cfg.l3_bw + \
                          (1 - l2_wr_hit_rate) * (1 - l3_wr_hit_rate) * self.hw_cfg.hbm_bw
        effective_wr_bw = effective_wr_bw * self.hw_opt.BW_scale
        l2_wr_cc = total_wr_bytes / effective_wr_bw
        # For TPU/ML Chiplet calculate cycles from result accumulation in case of K space partitioning
        #if self.hw_cfg.tpu_en and num_partitions > 1:
        #    l2_accum_bw_per_se = self.hw_cfg.l2_accumulate_buses_per_se * self.hw_cfg.l2_accumulate_bus_width
        #    aux_factor = 2 if self.sw_opt.fp16_inputs else 1  # Accumulation data is transferred in FP32
        #    accum_cc = num_a_x_b_block_size * aux_factor * (num_partitions - 1) / l2_accum_bw_per_se
        #    l2_wr_cc += accum_cc

        #print("l2_wr_cc_per_res:", l2_wr_cc_per_res)

        if self.hw_opt.stacked_mem_en:
            effective_rd_bw_a = self.hw_cfg.stacked_mem_bw
            effective_rd_bw_b = self.hw_cfg.stacked_mem_bw
            l2_hit_rate_a = 1
            l2_hit_rate_b = 1
        else:
            _num_cu = self.num_cu_per_batch if self.batched_gemm else num_cu_util
            if self.hw_opt.l2_bcast_en:
                bcast_bw_redx_a, bcast_bw_redx_b = self.get_bcast_factors(num_a_blocks, num_b_blocks, num_cu_per_se, num_se_util, num_partitions)
            effective_rd_bw_a, effective_rd_bw_b, l2_hit_rate_a, l2_hit_rate_b = \
                self.get_effective_read_bw(num_a_blocks, num_b_blocks, _num_cu, num_cu_per_se, num_partitions, bcast_bw_redx_a, bcast_bw_redx_b)
            effective_rd_bw_a = effective_rd_bw_a * self.hw_opt.BW_scale
            effective_rd_bw_b = effective_rd_bw_b * self.hw_opt.BW_scale

        if self.winograd.enable:
            tile_x = self.winograd.m[0] + self.winograd.r[0] - 1
            tile_y = self.winograd.m[1] + self.winograd.r[1] - 1
            overlap_ratio = (self.winograd.r[0] - 1) * tile_y / (tile_x * tile_y)
            tile_size = math.ceil(self.winograd.tile_size / self.winograd.num_cu_per_tile) * self.in_size
            mem_rd_a_blocks = num_a_blocks * self.TENSOR_BLOCK_HEIGHT * tile_size * self.TENSOR_BLOCK_WIDTH * num_iter
            # Assumes overlapped pixels between tiles are cached in L1
            if self.hw_cfg.l1_size:
                mem_rd_cc_a_blocks = (mem_rd_a_blocks * (1 - overlap_ratio) / bcast_bw_redx_a) / effective_rd_bw_a + \
                                     (mem_rd_a_blocks * overlap_ratio) / self.hw_cfg.l1_bw
            else:
                mem_rd_cc_a_blocks = (mem_rd_a_blocks / bcast_bw_redx_a) / effective_rd_bw_a
            filt_size = self.winograd.r[0] * self.winograd.r[1] * self.in_size
            mem_rd_b_blocks = num_b_blocks * self.TENSOR_BLOCK_WIDTH * filt_size * self.TENSOR_BLOCK_HEIGHT * num_iter
            mem_rd_cc_b_blocks = (mem_rd_b_blocks / bcast_bw_redx_b) / effective_rd_bw_b
        if self.implicit_gemm.enable:
            if self.hw_cfg.l1_size:  # If L1$ present duplicate data comes at L1$ BW
                img_area_per_cu, stride = self.implicit_gemm.get_num_unique_pixels(cu_partition_width - 1, num_a_blocks * self.TENSOR_BLOCK_HEIGHT - 1,
                                                                                   num_partitions=num_partitions)

                overlap_data = num_a_blocks * self.TENSOR_BLOCK_HEIGHT * stride * self.in_size
                duplicate_data = num_a_blocks * self.TENSOR_A_BLOCK_SIZE * num_iter - img_area_per_cu * self.in_size
                duplicate_data = 0 if duplicate_data < 0 else duplicate_data  # In some cases need to read more data than actually used specially when stride > 1
                #if not (self.implicit_gemm.strides[1] == 2 and self.implicit_gemm.strides[2] == 2):  # Currently known issues with strides = 2, so ignore checks
                #    assert (duplicate_data == 0 or self.implicit_gemm.filt_dims[2] > 1 or self.implicit_gemm.filt_dims[3] > 1)
                if overlap_data > self.hw_cfg.l1_size:  # Data completely overwritten in L1$
                    mem_rd_cc_a_blocks = ((duplicate_data + img_area_per_cu * self.in_size) / bcast_bw_redx_a) / effective_rd_bw_a
                else:
                    mem_rd_cc_a_blocks = (img_area_per_cu * self.in_size / bcast_bw_redx_a) / effective_rd_bw_a + \
                                         duplicate_data / self.hw_cfg.l1_bw
            else:
                mem_rd_cc_a_blocks = (num_a_blocks * self.TENSOR_A_BLOCK_SIZE * num_iter / bcast_bw_redx_a) / effective_rd_bw_a
            mem_rd_cc_b_blocks = (num_b_blocks * self.TENSOR_B_BLOCK_SIZE * num_iter / bcast_bw_redx_b) / effective_rd_bw_b

        else:
            if is_nonuniform_partial:
                mem_rd_cc_a_blocks = (partition_blocks_per_cu * self.TENSOR_A_BLOCK_SIZE / bcast_bw_redx_a) / effective_rd_bw_a if gemm_dims.M >= gemm_dims.N else \
                                      (num_a_blocks * self.TENSOR_A_BLOCK_SIZE * num_iter / bcast_bw_redx_a) / effective_rd_bw_a
                mem_rd_cc_b_blocks = (partition_blocks_per_cu * self.TENSOR_B_BLOCK_SIZE / bcast_bw_redx_b) / effective_rd_bw_b if gemm_dims.N > gemm_dims.M else \
                                      (num_b_blocks * self.TENSOR_B_BLOCK_SIZE * num_iter / bcast_bw_redx_b) / effective_rd_bw_b
            else:
                mem_rd_cc_a_blocks = (num_a_blocks * self.TENSOR_A_BLOCK_SIZE * num_iter / bcast_bw_redx_a) / effective_rd_bw_a
                mem_rd_cc_b_blocks = (num_b_blocks * self.TENSOR_B_BLOCK_SIZE * num_iter / bcast_bw_redx_b) / effective_rd_bw_b

        # Weights are assumed to be stashed in VGPRs for inference/persistent RNNs
        if (self.hw_cfg.tpu_en and not self.sw_opt.training) or (self.prnn_opt_en and self.persistent_rnn_check(num_a_blocks, num_b_blocks, num_iter)):
            if self.act == 'A':
                mem_rd_cc_b_blocks = 0
                l2_hit_rate_b = 1.0
            else:
                mem_rd_cc_a_blocks = 0
                l2_hit_rate_a = 1.0

        # Compute residual cycles from latency exposed during A/B block fetches
        total_mem_rd_cc = 0
        l2_hit_rate = min(l2_hit_rate_a, l2_hit_rate_b)
        if self.tpu_inference:
            bcast_factor = bcast_bw_redx_a if self.act == 'A' else bcast_bw_redx_b
            bcast_cluster_sz = num_cu_per_se // bcast_factor
            l2_hit_latency = self.hw_cfg.l2_repeater_latency * (bcast_cluster_sz - 1) + \
                          self.hw_cfg.l2_repeater_latency * (math.ceil(bcast_factor / SLICE_SPLIT) - 1) * bcast_cluster_sz
        else:
            l2_hit_latency = self.hw_cfg.l2_hit_latency

        average_latency = l2_hit_rate * l2_hit_latency + (1 - l2_hit_rate) * self.hw_cfg.l2_miss_latency
        mem_rd_cc_per_iter = (mem_rd_cc_a_blocks + mem_rd_cc_b_blocks) / num_iter
        alu_cc_per_iter = self.get_alu_cycles(gemm_dims, num_a_blocks, num_b_blocks, is_partial, num_partitions,
                                              is_nonuniform_partial, partition_blocks_per_cu) / num_iter
        aux_alu_cc_per_iter = self.get_aux_alu_cc(gemm_dims, num_a_blocks, num_b_blocks, is_partial, num_partitions) / num_iter
        alu_cc_per_iter += aux_alu_cc_per_iter
        k_loop_unroll_factor = [i for i in range(1, self.K_BLOCKS+1)]

        for i in range(len(k_loop_unroll_factor)):
            mem_rd_cc_per_unroll_iter = mem_rd_cc_per_iter * k_loop_unroll_factor[i]
            alu_cc_per_unroll_iter = alu_cc_per_iter * k_loop_unroll_factor[i]
            total_a_mem = num_a_blocks * self.TENSOR_A_BLOCK_SIZE * k_loop_unroll_factor[i]
            total_b_mem = num_b_blocks * self.TENSOR_B_BLOCK_SIZE * k_loop_unroll_factor[i]
            if self.hw_cfg.lds_size:
                if self.sw_opt.ABinLDS:
                    inputs_fit_in_onchip = (total_a_mem + total_b_mem) < self.hw_cfg.lds_size
                else:
                    inputs_fit_in_onchip = total_a_mem < (self.hw_cfg.arch_vgpr_size_per_cu if self.hw_cfg.accum_vgpr_size_per_cu != 0
                                                          else self.hw_cfg.arch_vgpr_size_per_cu-self.result_vgpr_size)\
                                           and total_b_mem < self.hw_cfg.lds_size
            else:
                inputs_fit_in_onchip = (total_a_mem + total_b_mem) < self.hw_cfg.arch_vgpr_size_per_cu
            if i > 1 and (num_iter / k_loop_unroll_factor[i] < 10 or not inputs_fit_in_onchip):  # If True, then stop unrolling K
                unroll_factor = k_loop_unroll_factor[i]
                break
            if alu_cc_per_unroll_iter < (mem_rd_cc_per_unroll_iter + average_latency) and average_latency:
                total_mem_rd_cc = (mem_rd_cc_per_unroll_iter + aux_alu_cc_per_iter + average_latency) * num_iter / k_loop_unroll_factor[i]
                unroll_factor = k_loop_unroll_factor[i]
            else:
                total_mem_rd_cc = (mem_rd_cc_per_unroll_iter + aux_alu_cc_per_iter) * num_iter / k_loop_unroll_factor[i]
                unroll_factor = k_loop_unroll_factor[i]
                break

        if stash_weights:
            mem_rd_cc_a_blocks = 0 if self.M >= self.N else mem_rd_cc_a_blocks
            mem_rd_cc_b_blocks = 0 if self.N > self.M else mem_rd_cc_b_blocks

        # print("mem_rd_cc_per_res:", mem_rd_cc_a_blocks + mem_rd_cc_b_blocks)
        total_mem_cc = total_mem_rd_cc + l2_wr_cc
        return total_mem_cc, l2_wr_cc, unroll_factor

    def find_non_uniform_partition(self, M, N, K, is_c_mat_used):
        if M >= N:
            total_blocks = math.ceil(M / self.TENSOR_BLOCK_HEIGHT)
            _num_b_blocks = math.ceil(N / self.TENSOR_BLOCK_WIDTH)
        else:
            total_blocks = math.ceil(N / self.TENSOR_BLOCK_WIDTH)
            _num_a_blocks = math.ceil(M / self.TENSOR_BLOCK_HEIGHT)

        num_cu_left_part = 1
        num_cu_right_part = 1
        chip_util_factor_list = []
        speedup_factor_list = []
        cycles_list = []
        num_cu_util_list = []
        num_a_blocks_list = []
        num_b_blocks_list = []
        num_blocks = self.M_BLOCKS if M >= N else self.N_BLOCKS
        if is_prime(num_blocks):
            num_blocks -= 1
        factor = 1
        left_cu_thresh = 10
        while 1:
            if num_blocks % factor or (num_blocks / factor) > self.hw_cfg.num_cu_util:
                factor += 1
                continue
            if (num_blocks / factor) < left_cu_thresh: # Stop at some CU threshold
                break
            num_cu_left_part = num_blocks / factor
            num_cu_right_part = self.hw_cfg.num_cu_util - num_cu_left_part

            num_a_blocks_left = math.ceil(self.M_BLOCKS / num_cu_left_part) if M >= N else _num_a_blocks
            num_b_blocks_left = math.ceil(self.N_BLOCKS / num_cu_left_part) if N > M else _num_b_blocks
            num_a_blocks_right = math.ceil(self.M_BLOCKS / num_cu_right_part) if M >= N else _num_a_blocks
            num_b_blocks_right = math.ceil(self.N_BLOCKS / num_cu_right_part) if N > M else _num_b_blocks
            write_blocks_left = num_a_blocks_left if M >= N else num_b_blocks_left
            write_blocks_right = num_a_blocks_right if M >= N else num_b_blocks_right

            K_blocks_left_part = self.get_partition(num_cu_left_part, write_blocks_left, write_blocks_right, 'balance_A+B_fetch')
            total_blocks_right_part = (self.K_BLOCKS - K_blocks_left_part) * num_blocks

            while total_blocks_right_part % num_cu_right_part:
                num_cu_right_part -= 1
            num_a_blocks_right = math.ceil(self.M_BLOCKS / num_cu_right_part) if M >= N else _num_a_blocks
            num_b_blocks_right = math.ceil(self.N_BLOCKS / num_cu_right_part) if N > M else _num_b_blocks

            num_cu_util1 = num_cu_left_part + num_cu_right_part
            total_blocks_left_part = K_blocks_left_part * num_blocks
            curr_gemm_dims = GemmDims(M, N, K_blocks_left_part * self.TENSOR_BLOCK_K)

            alu_cc_left = self.get_alu_cycles(curr_gemm_dims, num_a_blocks_left, num_b_blocks_left,
                                              is_partial=True, num_partitions=2, is_nonuniform_partial=True,
                                              partition_blocks_per_cu=total_blocks_left_part // num_cu_left_part)
            mem_cc_left = self.get_memory_cycles(curr_gemm_dims, num_a_blocks_left, num_b_blocks_left,
                                                 is_c_mat_used, num_cu=num_cu_util1,
                                                 is_partial=True, num_partitions=2, is_nonuniform_partial=True,
                                                 partition_blocks_per_cu=total_blocks_left_part // num_cu_left_part)

            cycles_left = max(alu_cc_left, mem_cc_left)

            curr_gemm_dims = GemmDims(M, N, (self.K_BLOCKS - K_blocks_left_part) * self.TENSOR_BLOCK_K)
            alu_cc_right = self.get_alu_cycles(curr_gemm_dims, num_a_blocks_right, num_b_blocks_right,
                                               is_partial=True, num_partitions=2, is_nonuniform_partial=True,
                                               partition_blocks_per_cu=total_blocks_right_part // num_cu_right_part)
            mem_cc_right = self.get_memory_cycles(curr_gemm_dims, num_a_blocks_right, num_b_blocks_right,
                                                  is_c_mat_used, num_cu=num_cu_util1,
                                                  is_partial=True, num_partitions=2, is_nonuniform_partial=True,
                                                  partition_blocks_per_cu=total_blocks_right_part // num_cu_right_part)

            cycles_right = max(alu_cc_right, mem_cc_right)

            cycles1 = max(cycles_left, cycles_right)
            chip_util_factor1 = ((alu_cc_left * num_cu_left_part + alu_cc_right * num_cu_right_part) / (cycles1 * num_cu_util1)) * 100

            cycles2 = 1 << 30
            if (total_blocks_right_part // num_cu_right_part) % (self.K_BLOCKS - K_blocks_left_part):
                # Try with uniform partition along M for right partition
                num_a_blocks_left = math.ceil(self.M_BLOCKS / num_cu_left_part) if M >= N else _num_a_blocks
                num_b_blocks_left = math.ceil(self.N_BLOCKS / num_cu_left_part) if N > M else _num_b_blocks
                num_a_blocks_right = 2 * num_a_blocks_left if M >= N else _num_a_blocks
                num_b_blocks_right = 2 * num_b_blocks_left if N > M else _num_b_blocks
                num_cu_right_part = self.M_BLOCKS // num_a_blocks_right if M >= N else self.N_BLOCKS // num_b_blocks_right
                num_cu_util2 = num_cu_left_part + num_cu_right_part
                total_blocks_left_part = K_blocks_left_part * num_blocks
                if num_cu_util2 < self.hw_cfg.num_cu_util:
                    curr_gemm_dims = GemmDims(M, N, K_blocks_left_part * self.TENSOR_BLOCK_K)
                    alu_cc_left = self.get_alu_cycles(curr_gemm_dims, num_a_blocks_left, num_b_blocks_left,
                                                      is_partial=True, num_partitions=2, is_nonuniform_partial=True,
                                                      partition_blocks_per_cu=total_blocks_left_part // num_cu_left_part)
                    mem_cc_left = self.get_memory_cycles(curr_gemm_dims, num_a_blocks_left, num_b_blocks_left,
                                                         is_c_mat_used, num_cu=num_cu_util2,
                                                         is_partial=True, num_partitions=2, is_nonuniform_partial=True,
                                                         partition_blocks_per_cu=total_blocks_left_part // num_cu_left_part)

                    cycles_left = max(alu_cc_left, mem_cc_left)

                    curr_gemm_dims = GemmDims(M, N, (self.K_BLOCKS - K_blocks_left_part) * self.TENSOR_BLOCK_K)
                    alu_cc_right = self.get_alu_cycles(curr_gemm_dims, num_a_blocks_right, num_b_blocks_right,
                                                       is_partial=True, num_partitions=2, is_nonuniform_partial=True,
                                                       partition_blocks_per_cu=total_blocks_right_part // num_cu_right_part)
                    mem_cc_right = self.get_memory_cycles(curr_gemm_dims, num_a_blocks_right, num_b_blocks_right,
                                                          is_c_mat_used, num_cu=num_cu_util2,
                                                          is_partial=True, num_partitions=2, is_nonuniform_partial=True,
                                                          partition_blocks_per_cu=total_blocks_right_part // num_cu_right_part)

                    cycles_right = max(alu_cc_right, mem_cc_right)

                    cycles2 = max(cycles_left, cycles_right)
                    chip_util_factor2 = ((alu_cc_left * num_cu_left_part + alu_cc_right * num_cu_right_part) /
                                        (cycles2 * num_cu_util2)) * 100

            if cycles1 < cycles2:
                chip_util_factor_list.append(chip_util_factor1)
                num_cu_util_list.append(num_cu_util1)
                cycles_list.append(cycles1)
                speedup_factor = self.get_speedup_factor(num_cu_util1, chip_util_factor1)
                speedup_factor_list.append(speedup_factor)
            else:
                chip_util_factor_list.append(chip_util_factor2)
                num_cu_util_list.append(num_cu_util2)
                cycles_list.append(cycles2)
                speedup_factor = self.get_speedup_factor(num_cu_util2, chip_util_factor2)
                speedup_factor_list.append(speedup_factor)
            factor += 1

        if len(cycles_list) == 0:
            return 0, 1e30, 0
        min_vals = heapq.nsmallest(len(cycles_list), cycles_list)
        min_idx = cycles_list.index(min_vals[0])
        for i in range(len(min_vals) - 1):
            perf_diff = ((min_vals[i + 1] - min_vals[0]) / min_vals[0]) * 100
            curr_idx = cycles_list.index(min_vals[i + 1])
            if perf_diff < 5 and num_cu_util_list[curr_idx] > num_cu_util_list[min_idx]:
                min_idx = curr_idx

        return chip_util_factor_list[min_idx], cycles_list[min_idx], num_cu_util_list[min_idx]

    def find_opt_partitions(self, gemm_dims, num_cu, is_c_mat_used, avail_cu, trail=0, num_partitions_trail_nokpart=1):
        num_cu_per_partition, num_partitions_list = self.get_cu_partitions(gemm_dims.K, num_cu, trail, num_partitions_trail_nokpart)
        num_rounds = 1
        chip_util_factor_list = []
        alu_util_factor_list = []
        cycles_list = []
        alu_cycles_list = []
        mem_cycles_list = []
        wr_cycles_list = []
        num_cu_util_list = []
        num_a_blocks_list = []
        num_b_blocks_list = []
        num_rounds_list = []
        vgpr_util_bytes_wt_list = []
        vgpr_util_bytes_res_list = []
        num_a_blocks_trail_list = []
        num_b_blocks_trail_list = []
        num_partitions_trail_list = []
        num_cu_util_trail_list = []
        unroll_factor_list = []
        unroll_factor_trail_list = []
        unroll_factor_trail = 1
        num_rounds_trail_list = []
        alu_cc_trail_list = []
        mem_cc_trail_list = []
        wr_cc_trail_list = []

        data_size = get_dt_size(self.sw_opt)

        if self.hw_cfg.tpu_en and not self.sw_opt.training and 'conservative' in self.tpu_partition_scheme:
            total_weight_blocks = self.N_BLOCKS if self.act == 'A' else self.M_BLOCKS
            max_partitions = max(int(math.floor(num_cu / total_weight_blocks)), 1)
        else:
            max_partitions = math.ceil(gemm_dims.K / self.TENSOR_BLOCK_K)
        for i in range(len(num_partitions_list)):
            if num_partitions_list[i] > max_partitions:
                cycles_list.append(1 << 31)
                alu_cycles_list.append(1 << 31)
                mem_cycles_list.append(1 << 31)
                wr_cycles_list.append(1 << 31)
                chip_util_factor_list.append(0)
                alu_util_factor_list.append(0)
                num_cu_util_list.append(0)
                num_a_blocks_list.append(0)
                num_b_blocks_list.append(0)
                num_a_blocks_trail_list.append(0)
                num_b_blocks_trail_list.append(0)
                num_partitions_trail_list.append(0)
                num_cu_util_trail_list.append(0)
                unroll_factor_list.append(1)
                unroll_factor_trail_list.append(1)
                num_rounds_trail_list.append(1)
                alu_cc_trail_list.append(0)
                mem_cc_trail_list.append(0)
                wr_cc_trail_list.append(0)
                continue
            is_partial = True if num_partitions_list[i] > 1 else False
            inst_m, inst_n,inst_k, inst_nblock = self.set_tensor_params(gemm_dims.M, gemm_dims.N, gemm_dims.K, num_partitions=num_partitions_list[i], trail=trail)
            curr_gemm_dims = copy.deepcopy(gemm_dims)
            num_a_blocks, num_b_blocks, num_rounds, total_blocks, trailing_blocks, num_cu_util, reject = \
                self.get_num_blocks(curr_gemm_dims, num_cu_per_partition[i], num_partitions=num_partitions_list[i], curr_trail=trail)
            if num_cu_util < 1 or ((trail or self.sw_opt.disable_trail)and trailing_blocks) or reject:
                cycles_list.append(1 << 31)
                alu_cycles_list.append(1 << 31)
                mem_cycles_list.append(1 << 31)
                wr_cycles_list.append(1 << 31)
                chip_util_factor_list.append(0)
                alu_util_factor_list.append(0)
                vgpr_util_bytes_wt_list.append(1 << 31)
                vgpr_util_bytes_res_list.append(1 << 31)
                num_cu_util_list.append(0)
                num_a_blocks_list.append(0)
                num_b_blocks_list.append(0)
                num_rounds_list.append(0)
                num_a_blocks_trail_list.append(0)
                num_b_blocks_trail_list.append(0)
                num_partitions_trail_list.append(0)
                num_cu_util_trail_list.append(0)
                unroll_factor_list.append(1)
                unroll_factor_trail_list.append(1)
                num_rounds_trail_list.append(1)
                alu_cc_trail_list.append(0)
                mem_cc_trail_list.append(0)
                wr_cc_trail_list.append(0)
                continue
            alu_cc = self.get_alu_cycles(curr_gemm_dims, num_a_blocks, num_b_blocks, is_partial=is_partial,
                                         num_partitions=num_partitions_list[i])
            mem_cc, wr_cc, unroll_factor = self.get_memory_cycles(curr_gemm_dims, num_a_blocks, num_b_blocks, is_c_mat_used, num_cu=num_cu_util,
                                                   is_partial=is_partial, num_partitions=num_partitions_list[i])
            if alu_cc > mem_cc:
                cycles = num_rounds * alu_cc + wr_cc
            else:
                cycles = num_rounds * mem_cc
            if not trail:
                cycles += (self.get_tpu_ramp_down_latency(curr_gemm_dims, num_cu_util, num_a_blocks, num_b_blocks, num_rounds, num_partitions_list[i])
                           if self.hw_cfg.tpu_en else self.hw_cfg.inst_fetch_bubble)
            else:
                cycles += (self.get_tpu_ramp_down_latency(curr_gemm_dims, num_cu_util, num_a_blocks, num_b_blocks, num_rounds, num_partitions_list[i])
                           if self.hw_cfg.tpu_en else 0)
            alu_util_factor = (num_rounds * alu_cc / cycles) * 100
            chip_util_factor = alu_util_factor * (num_cu_util / avail_cu)

            vgpr_util_bytes_wt, vgpr_util_bytes_res = self.get_vgpr_utils(curr_gemm_dims, num_a_blocks, num_b_blocks, num_partitions_list[i])

            num_a_blocks_trail = 0
            num_b_blocks_trail = 0
            num_partitions_trail = 1
            num_cu_util_trail = 0
            num_rounds_trail = 1
            mem_cc_trail = 0
            wr_cc_trail = 0
            alu_cc_trail = 0

            # Separate kernel to handle trailing blocks
            if trailing_blocks and not trail:
                trail_gemm_dims = []
                if (gemm_dims.M - curr_gemm_dims.M) > 0:
                    M_trail = gemm_dims.M - curr_gemm_dims.M
                    N_trail = gemm_dims.N
                    trail_gemm_dims = GemmDims(M_trail, N_trail, gemm_dims.K)
                elif (gemm_dims.N - curr_gemm_dims.N) > 0:
                    N_trail = gemm_dims.N - curr_gemm_dims.N
                    M_trail = gemm_dims.M
                    trail_gemm_dims = GemmDims(M_trail, N_trail, gemm_dims.K)
                else:
                    assert(0)
                cycles_trail, alu_cc_trail, num_cu_util_trail, num_a_blocks_trail, num_b_blocks_trail, num_partitions_trail, \
                unroll_factor_trail, num_rounds_trail, mem_cc_trail, wr_cc_trail = \
                    self.process_trail_blocks(trail_gemm_dims, num_cu, is_c_mat_used, avail_cu, is_partial=is_partial, num_partitions_nokpart=num_partitions_list[i])
                total_cycles = cycles + cycles_trail
                chip_util_factor = ((num_rounds * alu_cc * (num_cu_util / avail_cu) + alu_cc_trail * (num_cu_util_trail / avail_cu)) / total_cycles) * 100
                cycles = total_cycles

            chip_util_factor_list.append(chip_util_factor)
            alu_util_factor_list.append(alu_util_factor)
            cycles_list.append(cycles)
            alu_cycles_list.append(alu_cc * num_rounds)
            mem_cycles_list.append(mem_cc * num_rounds)
            wr_cycles_list.append(wr_cc)
            num_cu_util_list.append(num_cu_util)
            num_a_blocks_list.append(num_a_blocks)
            num_b_blocks_list.append(num_b_blocks)
            num_rounds_list.append(num_rounds)
            vgpr_util_bytes_wt_list.append(vgpr_util_bytes_wt)
            vgpr_util_bytes_res_list.append(vgpr_util_bytes_res)
            num_a_blocks_trail_list.append(num_a_blocks_trail)
            num_b_blocks_trail_list.append(num_b_blocks_trail)
            num_partitions_trail_list.append(num_partitions_trail)
            num_cu_util_trail_list.append(num_cu_util_trail)
            unroll_factor_list.append(unroll_factor)
            unroll_factor_trail_list.append(unroll_factor_trail)
            num_rounds_trail_list.append(num_rounds_trail)
            alu_cc_trail_list.append(alu_cc_trail) #multiplied with num_rounds_trail in process_rail_kernel
            mem_cc_trail_list.append(mem_cc_trail) #multiplied with num_rounds_trail in process_rail_kernel
            wr_cc_trail_list.append(wr_cc_trail)
            #if trail:
            #    print('num_cu ', num_cu, 'num_a_blocks ', num_a_blocks, 'num_b_blocks ', num_b_blocks, 'num_part ',
            #         num_partitions_list[i], 'alu_util', alu_util_factor, 'chip_util ', chip_util_factor, 'cycles', cycles)

        if self.hw_cfg.tpu_en and not self.sw_opt.training and "moderate" in self.tpu_partition_scheme and not trail:
            if "moderate" in self.tpu_partition_scheme:
                min_vals = heapq.nsmallest(len(vgpr_util_bytes_wt_list), vgpr_util_bytes_wt_list)
                min_idx = vgpr_util_bytes_wt_list.index(min_vals[0])
        else:
            min_vals = heapq.nsmallest(len(cycles_list), cycles_list)
            min_idx = cycles_list.index(min_vals[0])
            for i in range(len(min_vals) - 1):
                perf_diff = ((min_vals[i + 1] - min_vals[0]) / min_vals[0]) * 100
                curr_idx = cycles_list.index(min_vals[i + 1])
                if perf_diff < self.sw_opt.PerfThreshold and num_cu_util_list[curr_idx] > num_cu_util_list[min_idx]:
                    min_idx = curr_idx

        cycles = cycles_list[min_idx]
        alu_cc = alu_cycles_list[min_idx]
        mem_cc = mem_cycles_list[min_idx]
        wr_cc = wr_cycles_list[min_idx]
        num_partitions = num_partitions_list[min_idx]
        chip_util_factor = chip_util_factor_list[min_idx]
        alu_util_factor = alu_util_factor_list[min_idx]
        num_cu_util = num_cu_util_list[min_idx]
        num_a_blocks = num_a_blocks_list[min_idx]
        num_b_blocks = num_b_blocks_list[min_idx]
        num_a_blocks_trail = num_a_blocks_trail_list[min_idx]
        num_b_blocks_trail = num_b_blocks_trail_list[min_idx]
        num_partitions_trail = num_partitions_trail_list[min_idx]
        num_cu_util_trail = num_cu_util_trail_list[min_idx]
        unroll_factor = unroll_factor_list[min_idx]
        unroll_factor_trail = unroll_factor_trail_list[min_idx]
        num_rounds_trail = num_rounds_trail_list[min_idx]
        mem_cc_trail = mem_cc_trail_list[min_idx]
        alu_cc_trail = alu_cc_trail_list[min_idx]
        wr_cc_trail = wr_cc_trail_list[min_idx]
        return num_a_blocks, num_b_blocks, num_partitions, chip_util_factor, alu_util_factor, num_cu_util, cycles, alu_cc, \
               mem_cc, wr_cc, num_a_blocks_trail, num_b_blocks_trail, num_partitions_trail, num_cu_util_trail, unroll_factor,\
               unroll_factor_trail, num_rounds_trail, alu_cc_trail, mem_cc_trail, wr_cc_trail #Ashish added

    def get_bcast_factors(self, num_a_blocks, num_b_blocks, num_cu_per_se, num_se, num_partitions):
        # For TPU/ML Chiplet the CUs which needs to merge partial results are assigned to a single SE as much as possible
        # because of the separate accumulate bus in a SE/Slice; For non-TPU/GPU case CUs which needs to merge partial results
        # are assigned to separate SEs as much as possible to enhance broadcast for CUs working on a particular partition
        if self.hw_cfg.tpu_en:
            if self.M >= self.N:
                bcast_cluster_width = min(self.N_BLOCKS // num_b_blocks, math.ceil(num_cu_per_se / num_partitions))
                bcast_cluster_height = math.ceil(math.ceil(num_cu_per_se / num_partitions) / bcast_cluster_width)
            else:
                bcast_cluster_height = min(self.M_BLOCKS // num_a_blocks, math.ceil(num_cu_per_se / num_partitions))
                bcast_cluster_width = math.ceil(math.ceil(num_cu_per_se / num_partitions) / bcast_cluster_height)
        else:
            if self.M >= self.N:
                bcast_cluster_width = self.N_BLOCKS // num_b_blocks
                # For K-space partition, the broadcast is limited to CUs working on particular partition
                bcast_cluster_height = min(math.ceil(num_cu_per_se / bcast_cluster_width), num_cu_per_se * num_se // num_partitions)
            else:
                bcast_cluster_height = self.M_BLOCKS // num_a_blocks
                # For K-space partition, the broadcast is limited to CUs working on particular partition
                bcast_cluster_width = min(math.ceil(num_cu_per_se / bcast_cluster_height), num_cu_per_se * num_se // num_partitions)

        bcast_bw_redx_a = min(bcast_cluster_width, self.hw_opt.l2_bcast_cluster_sz)
        bcast_bw_redx_b = min(bcast_cluster_height, self.hw_opt.l2_bcast_cluster_sz)
        return bcast_bw_redx_a, bcast_bw_redx_b

    def process_trail_blocks(self, gemm_dims, num_cu, is_c_mat_used, avail_cu, is_partial=False, tpu_inf_flag=False, num_partitions_nokpart=1):
        inst_m, inst_n, inst_k, inst_nblock = self.set_tensor_params(gemm_dims.M, gemm_dims.N, gemm_dims.K, trail=1)
        trail_blocks = self.M_BLOCKS * self.N_BLOCKS
        # For trailing blocks try to employ num_cu as multiple of total trailing blocks
        num_cu_multiples = self.get_trail_cu_multiples(int(avail_cu), trail_blocks, is_partial)
        all_cu_list = [i for i in range(avail_cu, avail_cu // 16, -1)]
        num_cu_lists = []
        num_cu_lists.append(num_cu_multiples)
        num_cu_lists.append(all_cu_list)
        cycles_trail_list = []
        alu_cc_trail_list = []
        num_cu_util_trail_list = []
        num_a_blocks_trail_list = [] #Ashish added
        num_b_blocks_trail_list = [] #Ashish added
        num_partitions_trail_list = [] #Ashish added
        unroll_factor_trail_list = [] #Ashish added
        num_rounds_trail_list = [] #Ashish added
        mem_cc_trail_list = [] #Ashish added
        wr_cc_trail_list = [] #Ashish added

        for cu_list in num_cu_lists:
            for num_cu_trail in cu_list:
                if num_cu_trail <= avail_cu // 16:  # ignore very small num_cu_trail
                    continue
                curr_gemm_dims = copy.deepcopy(gemm_dims)
                num_rounds_trail = 1
                if tpu_inf_flag:
                    num_a_blocks_trail, num_b_blocks_trail, _, trailing_blocks, num_cu_util_trail, num_partitions_trail, reject = \
                        self.get_num_blocks_inference(curr_gemm_dims, num_cu=num_cu_trail, trail=1)
                else:
                    num_partitions_trail = 1
                    num_a_blocks_trail, num_b_blocks_trail, num_rounds_trail, _, trailing_blocks, num_cu_util_trail, reject = \
                        self.get_num_blocks(curr_gemm_dims, num_cu=num_cu_trail, curr_trail=1)
                # Ignore partition if it creates further trailing blocks
                if trailing_blocks or reject or not num_cu_util_trail:
                    continue
                alu_cc_trail = self.get_alu_cycles(gemm_dims, num_a_blocks_trail, num_b_blocks_trail)
                mem_cc_trail, wr_cc_trail, unroll_factor_trail = self.get_memory_cycles(gemm_dims, num_a_blocks_trail, num_b_blocks_trail,
                                                                   is_c_mat_used, num_cu=num_cu_util_trail)
                if alu_cc_trail > mem_cc_trail:
                    cycles_trail = num_rounds_trail * alu_cc_trail + wr_cc_trail
                else:
                    cycles_trail = num_rounds_trail * mem_cc_trail
                alu_util_factor_trail = (alu_cc_trail / cycles_trail) * 100

                # Try K-space partition on trailing blocks only if main kernel is K partitioned
                if not self.sw_opt.disable_trail_repart and alu_util_factor_trail < 100 and (gemm_dims.K >= gemm_dims.M or gemm_dims.K >= gemm_dims.N) and not self.batched_gemm \
                        and not tpu_inf_flag:
                    num_a_blocks_trail, num_b_blocks_trail, num_partitions_trail, _, _, num_cu_util_trail, cycles_trail, alu_cc_trail, \
                    mem_cc_trail, wr_cc_trail, _, _, _, _, unroll_factor_trail, _, _, _, _, _ = \
                        self.find_opt_partitions(GemmDims(gemm_dims.M, gemm_dims.N, gemm_dims.K), num_cu_trail,
                                                 is_c_mat_used, avail_cu, trail=1, num_partitions_trail_nokpart=num_partitions_nokpart)
                cycles_trail_list.append(cycles_trail) #Ashish added
                alu_cc_trail_list.append(alu_cc_trail * num_rounds_trail) #Ashish added
                num_cu_util_trail_list.append(num_cu_util_trail) #Ashish added
                num_a_blocks_trail_list.append(num_a_blocks_trail) #Ashish added
                num_b_blocks_trail_list.append(num_b_blocks_trail) #Ashish added
                num_partitions_trail_list.append(num_partitions_trail) #Ashish added
                unroll_factor_trail_list.append(unroll_factor_trail)
                num_rounds_trail_list.append(num_rounds_trail)
                mem_cc_trail_list.append(mem_cc_trail * num_rounds_trail)
                wr_cc_trail_list.append(wr_cc_trail)
            if len(cycles_trail_list):  # if no cu partition found from the CU multiple list then try the exhaustive list
                break

        min_vals = heapq.nsmallest(len(cycles_trail_list), cycles_trail_list)
        if min_vals == []:  # If no suitable partition found (creating no further trail blocks) then populate with large cycle count which will lead to this config getting rejected down the line
            cycles_trail_list.append(1 << 31)
            alu_cc_trail_list.append(0)
            num_cu_util_trail_list.append(0)
            num_a_blocks_trail_list.append(0)
            num_b_blocks_trail_list.append(0)
            num_partitions_trail_list.append(1)
            unroll_factor_trail_list.append(1)
            num_rounds_trail_list.append(1)
            mem_cc_trail_list.append(0)
            wr_cc_trail_list.append(0)
            min_idx = 0
        else:
            min_idx = cycles_trail_list.index(min_vals[0])
            for i in range(len(min_vals) - 1):
                perf_diff = ((min_vals[i + 1] - min_vals[0]) / min_vals[0]) * 100
                curr_idx = cycles_trail_list.index(min_vals[i + 1])
                if perf_diff < 3 and num_cu_util_trail_list[curr_idx] > num_cu_util_trail_list[min_idx]:
                    min_idx = curr_idx

        return cycles_trail_list[min_idx], alu_cc_trail_list[min_idx], num_cu_util_trail_list[min_idx], \
               num_a_blocks_trail_list[min_idx], num_b_blocks_trail_list[min_idx], num_partitions_trail_list[min_idx], \
               unroll_factor_trail_list[min_idx], num_rounds_trail_list[min_idx], mem_cc_trail_list[min_idx], \
               wr_cc_trail_list[min_idx] #Ashish added

    def get_num_blocks_inference(self, gemm_dims, num_cu, trail=0):
        reject_config = False
        M_nxt_mult = self.M_BLOCKS * self.TENSOR_BLOCK_HEIGHT
        N_nxt_mult = self.N_BLOCKS * self.TENSOR_BLOCK_WIDTH
        # Aim is to equally divide weight matrix among all XPEs which most probably creates K -space partitions
        total_blocks = math.ceil((M_nxt_mult * N_nxt_mult) / self.TENSOR_BLOCK_SIZE)
        total_weight_blocks = self.N_BLOCKS if self.act == 'A' else self.M_BLOCKS
        # Try to contain partitions within a slice
        max_partitions, _ = self.get_cu_se_util(num_cu)
        num_partitions = min(int(math.ceil(num_cu / total_weight_blocks)), max_partitions)
        while num_cu % num_partitions:  # num_partitions have to be divisible by num_cu
            num_partitions -= 1

        # Reset block sizes and other tensor params since there is a chance K/num_partitions < instr_block_K
        inst_m, inst_n, inst_k, inst_nblock = self.set_tensor_params(gemm_dims.M, gemm_dims.N, gemm_dims.K, num_partitions=num_partitions, trail=trail)
        num_weight_blocks_per_cu = max(int(total_weight_blocks * num_partitions // num_cu), 1)
        blocks_per_cu = max(total_blocks // (num_cu / num_partitions), 1)
        num_act_blocks_per_cu = max(blocks_per_cu // num_weight_blocks_per_cu, 1)

        if self.act == 'A':
            num_a_blocks = num_act_blocks_per_cu
            num_b_blocks = num_weight_blocks_per_cu
            #self.num_cu_width = int(self.N_BLOCKS // num_b_blocks)
            #self.num_cu_height = int(min(num_cu // num_partitions, total_blocks) // self.num_cu_width)
        else:
            num_b_blocks = num_act_blocks_per_cu
            num_a_blocks = num_weight_blocks_per_cu
            #self.num_cu_height = int(self.M_BLOCKS // num_a_blocks)
            #self.num_cu_width = int(min(num_cu // num_partitions, total_blocks) // self.num_cu_height)
        # Adjust num_a_blocks/num_b_blocks if it exceeds M/N dimensions
        num_a_blocks, num_b_blocks = self.dimension_check(M_nxt_mult, N_nxt_mult, num_a_blocks, num_b_blocks)
        self.num_cu_width, self.num_cu_height = self.get_cu_HxW(gemm_dims, num_a_blocks, num_b_blocks, num_cu//num_partitions, total_blocks)

        M_curr = self.num_cu_height * num_a_blocks * self.TENSOR_BLOCK_HEIGHT
        N_curr = self.num_cu_width * num_b_blocks * self.TENSOR_BLOCK_WIDTH
        num_cu_util = self.num_cu_height * self.num_cu_width * num_partitions
        trailing_blocks = 1 if M_curr < gemm_dims.M or N_curr < gemm_dims.N else 0
        reject_config = True if M_curr < M_nxt_mult and N_curr < N_nxt_mult else False

        gemm_dims.M = M_curr
        gemm_dims.N = N_curr

        return num_a_blocks, num_b_blocks, total_blocks, trailing_blocks, num_cu_util, num_partitions, reject_config

    def get_num_blocks(self, gemm_dims, num_cu, num_partitions=1, curr_trail=0):
        num_a_blocks = 1
        num_b_blocks = 1
        reject_config = False
        num_rounds = 1  # number of kernels needed to launch
        trailing_blocks = 0
        M_nxt_mult = self.M_BLOCKS * self.TENSOR_BLOCK_HEIGHT
        N_nxt_mult = self.N_BLOCKS * self.TENSOR_BLOCK_WIDTH

        total_blocks = math.ceil((M_nxt_mult * N_nxt_mult) / self.TENSOR_BLOCK_SIZE)
        blocks_per_cu = max(total_blocks // num_cu, 1)
        if blocks_per_cu > self.MAX_BLOCKS: # Depends on accumulation register file size
            if self.MAX_BLOCKS == 0:
                reject_config = True
                return 1, 1, 1, 1, 1, 1, reject_config
            else:
                blocks_per_cu = self.MAX_BLOCKS
                num_rounds = total_blocks // (self.MAX_BLOCKS * num_cu)

        if is_prime(blocks_per_cu) and blocks_per_cu > 8:
            blocks_per_cu -= 1

        if self.sw_opt.userDefMT:
            if curr_trail:
                if self.sw_opt.trail_a_blocks and self.sw_opt.trail_b_blocks:
                    num_a_blocks = self.sw_opt.trail_a_blocks // self.TENSOR_BLOCK_HEIGHT
                    num_b_blocks = self.sw_opt.trail_b_blocks // self.TENSOR_BLOCK_WIDTH
                else:
                    num_a_blocks, num_b_blocks = self.find_block_multiples_1(blocks_per_cu)
            else:
                if self.sw_opt.a_blocks and self.sw_opt.b_blocks:
                    num_a_blocks = self.sw_opt.a_blocks // self.TENSOR_BLOCK_HEIGHT
                    num_b_blocks = self.sw_opt.b_blocks // self.TENSOR_BLOCK_WIDTH
                else:
                    num_a_blocks, num_b_blocks = self.find_block_multiples_1(blocks_per_cu)
            if num_a_blocks * num_b_blocks > self.MAX_BLOCKS:
                reject_config = True
        else:
            num_a_blocks, num_b_blocks = self.find_block_multiples_1(blocks_per_cu)
        #self.TileGranularity0 = (gemm_dims.M/())

        # Attempt to make the result area more squarish
        if num_a_blocks > num_b_blocks and num_a_blocks / num_b_blocks > 2 and not curr_trail:
            skew_factor = num_a_blocks // num_b_blocks
            num_b_blocks *= (skew_factor // 2)
            num_a_blocks = blocks_per_cu // num_b_blocks
        elif num_b_blocks > num_a_blocks and num_b_blocks / num_a_blocks > 2 and not curr_trail:
            skew_factor = num_b_blocks // num_a_blocks
            num_a_blocks *= (skew_factor // 2)
            num_b_blocks = blocks_per_cu // num_a_blocks

        # Swap num_a_blocks, num_b_blocks if the swapped combination results in more CU util
        num_cu_width1, num_cu_height1 = self.get_cu_HxW(gemm_dims, num_a_blocks, num_b_blocks, num_cu, total_blocks)
        num_cu_width2, num_cu_height2 = self.get_cu_HxW(gemm_dims, num_b_blocks, num_a_blocks, num_cu, total_blocks)
        if num_cu_width1 * num_cu_height1 < num_cu_width2 * num_cu_height2:
            num_a_blocks, num_b_blocks = swap(num_a_blocks, num_b_blocks)

        #if gemm_dims.M >= gemm_dims.N and num_a_blocks < num_b_blocks:
        #    num_a_blocks, num_b_blocks = swap(num_a_blocks, num_b_blocks)
        #elif gemm_dims.N > gemm_dims.M and num_b_blocks < num_a_blocks:
        #    num_a_blocks, num_b_blocks = swap(num_a_blocks, num_b_blocks)

        if self.winograd.enable:
            num_a_blocks, num_b_blocks, num_rounds_winograd = self.winograd_check(num_a_blocks, num_b_blocks,
                                                                                  blocks_per_cu)
            num_rounds *= num_rounds_winograd
            blocks_per_cu = num_a_blocks * num_b_blocks

        if gemm_dims.M >= gemm_dims.N and self.N_BLOCKS % num_b_blocks and not (self.N_BLOCKS % num_a_blocks) and num_partitions == 1:
            num_a_blocks, num_b_blocks = swap(num_a_blocks, num_b_blocks)
        elif gemm_dims.N > gemm_dims.M and self.M_BLOCKS % num_a_blocks and not (self.M_BLOCKS % num_b_blocks) and num_partitions == 1:
            num_a_blocks, num_b_blocks = swap(num_a_blocks, num_b_blocks)

        if self.hw_opt.l2_bcast_en and self.hw_opt.l2_bcast_both_dims and num_partitions == 1:
            # Try to get at least 4 CUs in shorter dimension if possible to
            # enhance broadcast in that dimension
            if gemm_dims.M >= gemm_dims.N and num_b_blocks > 1 and math.ceil(self.N_BLOCKS / num_b_blocks) < 4: # and num_a_blocks * num_b_blocks <= 16:
                num_a_blocks *= num_b_blocks
                num_b_blocks = 1
            elif gemm_dims.N > gemm_dims.M and num_a_blocks > 1 and math.ceil(self.M_BLOCKS / num_a_blocks) < 4: # and num_a_blocks * num_b_blocks <= 16:
                num_b_blocks *= num_a_blocks
                num_a_blocks = 1

        # Make sure num_a_blocks/num_blocks is multiple of M_BLOCKS/N_BLOCKS whichever is the smaller dimension
        if gemm_dims.M >= gemm_dims.N and self.N_BLOCKS % num_b_blocks and num_b_blocks < self.N_BLOCKS:
            num_b_blocks = find_nxt_multiple(num_b_blocks, self.N_BLOCKS)
            num_b_blocks = 1 if num_b_blocks > blocks_per_cu else num_b_blocks
            num_a_blocks = blocks_per_cu // num_b_blocks
        elif gemm_dims.M < gemm_dims.N and self.M_BLOCKS % num_a_blocks and num_a_blocks < self.M_BLOCKS:
            num_a_blocks = find_nxt_multiple(num_a_blocks, self.M_BLOCKS)
            num_a_blocks = 1 if num_a_blocks > blocks_per_cu else num_a_blocks
            num_b_blocks = blocks_per_cu // num_a_blocks

        # Adjust num_a_blocks/num_b_blocks if it exceeds M/N dimensions
        num_a_blocks, num_b_blocks = self.dimension_check(M_nxt_mult, N_nxt_mult, num_a_blocks, num_b_blocks)

        self.num_cu_width, self.num_cu_height = self.get_cu_HxW(gemm_dims, num_a_blocks, num_b_blocks, num_cu, total_blocks)
        # if num_cu_width(num_cu_height) not a mulitple of num_b_blocks(num_a_blocks) then reject partition scheme
        if (gemm_dims.M >= gemm_dims.N and (self.N_BLOCKS/num_b_blocks) > self.num_cu_width) or \
            (gemm_dims.N > gemm_dims.M and (self.M_BLOCKS / num_a_blocks) > self.num_cu_height):
            reject_config = True
        if not reject_config:
            num_rounds = total_blocks // (num_a_blocks * num_b_blocks * self.num_cu_width * self.num_cu_height)


        if not curr_trail and self.tpu_inference and self.tpu_partition_scheme in ['moderate_lv1', 'moderate_lv2']:  # Serialze result generation into multiple rounds
            num_a_blocks, num_b_blocks, num_rounds = self.get_num_rounds(gemm_dims, num_a_blocks, num_b_blocks, num_partitions)
            M_curr = self.num_cu_height * num_a_blocks * self.TENSOR_BLOCK_HEIGHT * (num_rounds if self.act == 'A' else 1)
            N_curr = self.num_cu_width * num_b_blocks * self.TENSOR_BLOCK_WIDTH * (1 if self.act == 'A' else num_rounds)
        else:
            M_curr = self.num_cu_height * num_a_blocks * self.TENSOR_BLOCK_HEIGHT * (num_rounds if gemm_dims.M >= gemm_dims.N else 1)
            N_curr = self.num_cu_width * num_b_blocks * self.TENSOR_BLOCK_WIDTH * (num_rounds if gemm_dims.N > gemm_dims.M else 1)

        if M_curr > M_nxt_mult:
            num_cu_height = gemm_dims.M // (num_a_blocks * self.TENSOR_BLOCK_HEIGHT * (num_rounds if gemm_dims.M >= gemm_dims.N else 1))
            M_curr = num_cu_height * num_a_blocks * self.TENSOR_BLOCK_HEIGHT * (num_rounds if gemm_dims.M >= gemm_dims.N else 1)
            assert(M_curr <= gemm_dims.M)
        if N_curr > N_nxt_mult:
            num_cu_width = gemm_dims.N // (num_b_blocks * self.TENSOR_BLOCK_WIDTH * (num_rounds if gemm_dims.N > gemm_dims.M else 1))
            N_curr = num_cu_width * num_b_blocks * self.TENSOR_BLOCK_WIDTH * (num_rounds if gemm_dims.N > gemm_dims.M else 1)
            assert (N_curr <= gemm_dims.N)

        if self.batched_gemm and self.num_cu_per_batch == 1:
            num_cu_util = self.hw_cfg.num_cu_util
        else:
            num_cu_util = self.num_cu_height * self.num_cu_width * num_partitions
        trailing_blocks = 1 if M_curr < gemm_dims.M or N_curr < gemm_dims.N else 0
        gemm_dims.M = M_curr
        gemm_dims.N = N_curr
        assert num_a_blocks >= 1 and num_b_blocks >= 1
        return num_a_blocks, num_b_blocks, num_rounds, total_blocks, trailing_blocks, num_cu_util, reject_config

    def perform_gemm(self, M, N, K, is_c_mat_used, avail_cu):
        inst_m, inst_n, inst_k, inst_nblock = self.set_tensor_params(M, N, K)
        curr_gemm_dims = GemmDims(M, N, K)
        num_cu = int(self.num_cu_per_batch if self.batched_gemm else self.hw_cfg.num_cu_util)
        num_rounds = 1
        num_act_rounds = 1
        num_partitions = 1
        trailing_blocks = 0
        tpu_inf_flag = self.hw_cfg.tpu_en and not self.sw_opt.training and self.tpu_partition_scheme in ['conservative']

        if tpu_inf_flag:
            num_a_blocks, num_b_blocks, total_blocks, trailing_blocks, num_cu_util, num_partitions, reject_cfg = \
                self.get_num_blocks_inference(curr_gemm_dims, num_cu=num_cu)
        else:
            num_a_blocks, num_b_blocks, num_rounds, total_blocks, trailing_blocks, num_cu_util, reject_cfg = \
                    self.get_num_blocks(curr_gemm_dims, num_cu=num_cu)
        if reject_cfg:
            res = GemmRes()
            return res

        if self.sw_opt.disable_trail and trailing_blocks and not self.sw_opt.TileGranularity: #Ashish added
            res = GemmRes()
            return res

        is_partial = num_partitions > 1
        alu_cc = self.get_alu_cycles(curr_gemm_dims, num_a_blocks, num_b_blocks, is_partial, num_partitions)
        mem_cc, wr_cc, unroll_factor = self.get_memory_cycles(curr_gemm_dims, num_a_blocks, num_b_blocks, is_c_mat_used, num_cu=num_cu_util,
                                               is_partial=is_partial, num_partitions=num_partitions)
        if alu_cc > mem_cc:
            cycles = num_rounds * alu_cc + wr_cc  # Assumes write out cycles are exposed for ALU bound kernel
            alu_cc *= num_rounds
            mem_cc *=num_rounds
        else:
            cycles = num_rounds * mem_cc  # Write out cycles are part of mem_cc
            mem_cc *= num_rounds
            alu_cc *= num_rounds
        cycles += (self.get_tpu_ramp_down_latency(curr_gemm_dims, num_cu_util, num_a_blocks, num_b_blocks, num_rounds) if self.hw_cfg.tpu_en else
                   self.hw_cfg.inst_fetch_bubble)
        #cycles *= 1.125
        alu_util_factor = (alu_cc / cycles) * 100
        chip_util_factor = alu_util_factor * (num_cu_util / avail_cu)
        cycles *= num_act_rounds
        num_cu_util_trail = 0
        num_a_blocks_trail = 0
        num_b_blocks_trail = 0
        num_partitions_trail = 1
        cycles_trail = 0
        unroll_factor_trail = 1
        num_rounds_trail = 1
        mem_cc_trail = 0
        wr_cc_trail = 0
        alu_cc_trail = 0
        main_instr = ("%dx%dx%dx%d" % (
        self.hw_cfg.dl_instr_large_block[INSTR_M_IND], self.hw_cfg.dl_instr_large_block[INSTR_N_IND],
        self.hw_cfg.dl_instr_large_block[INSTR_K_IND], self.hw_cfg.dl_instr_large_block[INSTR_NUM_BLOCKS_IND]))
        #print('{}', format(main_instr))

        # Separate kernel to handle trailing blocks
        if trailing_blocks and not self.sw_opt.disable_trail:
            trail_gemm_dims = []
            if (M - curr_gemm_dims.M) > 0:
                M_trail = M - curr_gemm_dims.M
                N_trail = N
                trail_gemm_dims = GemmDims(M_trail, N_trail, K)
            elif (N - curr_gemm_dims.N) > 0:
                N_trail = N - curr_gemm_dims.N
                M_trail = M
                trail_gemm_dims = GemmDims(M_trail, N_trail, K)
            else:
                assert(0)
            cycles_trail, alu_cc_trail, num_cu_util_trail, num_a_blocks_trail, num_b_blocks_trail, num_partitions_trail, unroll_factor_trail, \
                num_rounds_trail, mem_cc_trail, wr_cc_trail = \
                self.process_trail_blocks(trail_gemm_dims, num_cu, is_c_mat_used, avail_cu, is_partial=is_partial, tpu_inf_flag=tpu_inf_flag)
            total_cycles = (cycles + cycles_trail) #* 1.125
            chip_util_factor = ((alu_cc * (num_cu_util / avail_cu) + alu_cc_trail * (num_cu_util_trail / avail_cu)) / total_cycles) * 100
            cycles = total_cycles

        if trailing_blocks and self.sw_opt.disable_trail and self.sw_opt.TileGranularity:
            Tile0 = math.ceil(self.M / inst_m)
            Tile1 = math.ceil(self.N / inst_n)
            TileGranularity0 = (self.M / inst_m) / Tile0 if M > inst_m else 1
            TileGranularity1 = (self.N / inst_n) / Tile1 if N > inst_n else 1
            Tiles_per_cu = (Tile0 * Tile1) / num_cu
            CUGranularity = Tiles_per_cu / math.ceil(Tiles_per_cu) if Tiles_per_cu > 1 else 1
            cycles = cycles / (TileGranularity0 * TileGranularity1 * CUGranularity)
            chip_util_factor = chip_util_factor * (TileGranularity0 * TileGranularity1 * CUGranularity)

        # Try K-space partitioning
        if chip_util_factor < 95 and (K >= M or K >= N) and not self.batched_gemm and not tpu_inf_flag and not self.sw_opt.disable_kpartitioning:
            num_a_blocks, num_b_blocks, num_partitions, chip_util_factor, alu_util_factor, num_cu_util, cycles, alu_cc, mem_cc, wr_cc, \
            num_a_blocks_trail, num_b_blocks_trail, num_partitions_trail, num_cu_util_trail, unroll_factor, unroll_factor_trail, \
            num_rounds_trail, alu_cc_trail, mem_cc_trail, wr_cc_trail = self.find_opt_partitions(GemmDims(M, N, K), num_cu, is_c_mat_used, avail_cu)


        speedup = self.get_speedup_factor(num_cu_util, chip_util_factor)

        tpu_sub_res = TpuSubResults()

        if self.hw_cfg.tpu_en and not self.sw_opt.training:
            # Resetting here for safety just in case if trail kernel changed the internal GEMM Block dimensions
            _, _, _, _ = self.set_tensor_params(M, N, K)
            vgpr_util_bytes_wt, vgpr_util_bytes_res = self.get_vgpr_utils(curr_gemm_dims, num_a_blocks, num_b_blocks, num_partitions)
            tpu_sub_res = TpuSubResults(vgpr_util_bytes_wt, vgpr_util_bytes_res)

        res = GemmRes(alu_util_factor, chip_util_factor, speedup, cycles, num_cu_util, num_a_blocks, num_b_blocks,
                      num_partitions, tpu_sub_res,
                      alu_cc, mem_cc, num_rounds, total_blocks, num_cu_util_trail, num_a_blocks_trail, num_b_blocks_trail, #Ashish added
                      num_partitions_trail, cycles_trail, wr_cc, main_instr, unroll_factor=unroll_factor, unroll_factor_trail=unroll_factor_trail,\
                      num_rounds_trail=num_rounds_trail, alu_cc_trail=alu_cc_trail, mem_cc_trail=mem_cc_trail, wr_cc_trail=wr_cc_trail)  # Ashish added



        return res
