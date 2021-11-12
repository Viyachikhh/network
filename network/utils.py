import numpy as np


def get_im2col_indices(x_shape, filter_height, filter_width, padding=0, stride=1):
    # Функция для получения правильных размерностей
    N, C, H, W = x_shape

    out_height = int((H + 2 * padding - filter_height) / stride + 1)
    out_width = int((W + 2 * padding - filter_width) / stride + 1)

    i0 = np.repeat(np.arange(filter_height), filter_width)  # (filter_height * filter_width * C)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)  # (out_height, out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * C)  # аналогично i0

    j1 = stride * np.tile(np.arange(out_width), out_height)  # аналогично i1
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), filter_height * filter_width).reshape(-1, 1)

    return k.astype(int), i.astype(int), j.astype(int)


def im2col_indices(x, filter_height, filter_with, padding=0, stride=1):

    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant', constant_values=0)

    k, i, j = get_im2col_indices(x.shape, filter_height, filter_with, padding, stride)
    #print(k.shape, i.shape, j.shape, x_padded.shape)
    cols = x_padded[:, k, i, j]
    #print(cols.shape)
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_with * C, -1)
    #print(cols.shape)
    return cols


def col2im_indices(cols, x_shape, filter_height, filter_width, padding=0,
                   stride=1):
    """
    Функция для получения из двухмерного четырёхмерное
    """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, filter_height, filter_width, padding, stride)
    #print(k, i, j)
    cols_reshaped = cols.reshape(C * filter_height * filter_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

