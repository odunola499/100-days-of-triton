import triton
import triton.language as tl

import torch


def naive_softmax(x:torch.Tensor):
    x_max = x.max(dim = 1)[0]
    safe_x = x-x_max[:,None]
    numerator = torch.exp(safe_x)
    denominator = torch.sum(numerator, dim = 1)
    sm_out = numerator/denominator[:,None]
    return sm_out

@triton.jit
def _softmax_fwd_kernel(
        output_ptr, output_stride,
        input_ptr, input_stride,
        num_cols,
        block_size:tl.constexpr,
):
    row_index = tl.program_id(0)
    row_start_ptr = input_ptr + (row_index * input_stride)
    col_offsets = tl.arange(0, block_size)
    input_pointers = row_start_ptr + col_offsets

    row_mask = col_offsets < num_cols
    row = tl.load(input_pointers, mask = row_mask, other = float('-inf'))
    #start softmax
    row_safe = row - tl.max(row, axis = 0)
    numerator = tl.exp(row_safe)
    denominator = tl.sum(numerator, axis = 0)
    row_sm = numerator / denominator
    #write to hbm
    output_start_ptr = output_ptr + (row_index * output_stride)
    output_pointers = output_start_ptr + col_offsets
    tl.store(output_pointers, row_sm, mask = row_mask)





def softmax(x:torch.Tensor) -> torch.Tensor:
    # 2D tensor
    rows, cols = x.shape
    assert x.dim() == 2, f"Excepted 2D tensor, got {x.dim()}D tensor"
    block_size = triton.next_power_of_2(cols) # used for maxing
    num_warps = 4 #32 threads
    if block_size > 2047:
        num_warps = 8
    if block_size > 4095:
        num_warps = 16

    grid = (rows, ) #paralize through rows
    output = torch.empty_like(x)
    _softmax_fwd_kernel[grid](
        output, output.stride(0), x, x.stride(0), cols, block_size = block_size, num_warps=num_warps
    )
    return output

