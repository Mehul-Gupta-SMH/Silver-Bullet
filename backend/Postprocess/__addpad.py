from typing import List, Union
import torch
import torch.nn.functional as F

MAX_SENTENCES = 64
TARGET_SIZE   = 32  # default spatial size for resized feature maps


def resize_matrix(mat: Union[List[list], torch.Tensor], target_size: int = TARGET_SIZE) -> torch.Tensor:
    """Resize an n×m similarity matrix to target_size×target_size via bilinear interpolation.

    Unlike zero-padding, every cell in the output is derived from actual sentence-pair
    signal — there are no artificial zeros from empty padding regions.  Large matrices
    (many sentences) are averaged down; small matrices (few sentences) are interpolated
    up.  This produces a uniform spatial layout that lets the CNN judge similarity from
    the signal structure rather than from how populated the grid is.

    Args:
        mat:         2D list-of-lists or torch.Tensor of shape [n, m].
        target_size: Output side length (square output). Default 32.

    Returns:
        torch.Tensor: shape [target_size, target_size].

    Raises:
        ValueError: If mat is empty or not a proper 2D structure.
    """
    if isinstance(mat, list):
        if not mat or not mat[0]:
            raise ValueError("resize_matrix received an empty matrix.")
        mat = torch.tensor(mat, dtype=torch.float32)

    mat = mat.float()                    # [n, m]
    mat = mat.unsqueeze(0).unsqueeze(0)  # [1, 1, n, m] — F.interpolate needs BCHW
    resized = F.interpolate(
        mat,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0)  # [target_size, target_size]


def pad_matrix(mat: List[list], target_size: int = MAX_SENTENCES, pad_value: int = 0) -> torch.Tensor:
    """Pad (or truncate) a 2D matrix to a square target_size × target_size tensor.

    .. deprecated::
        Use ``resize_matrix`` instead.  Zero-padding leaves the majority of the spatial
        grid empty for short texts, causing the CNN to deflate similarity scores.
        ``resize_matrix`` uses bilinear interpolation so every output cell carries signal.

    If the input is larger than target_size in either dimension it is silently
    truncated — the first target_size rows/columns are kept.

    Args:
        mat (list of list of float): The input 2D matrix (n rows × m cols).
        target_size (int): Desired output size for both dimensions (default 64).
        pad_value (int | float): Fill value for padding (default 0).
    Returns:
        torch.Tensor: Shape [target_size, target_size].
    Raises:
        ValueError: If mat is empty or not a proper 2D list.
    """
    if not mat or not mat[0]:
        raise ValueError("pad_matrix received an empty matrix.")

    # Truncate rows / cols that exceed target_size
    rows = mat[:target_size]
    rows = [row[:target_size] for row in rows]

    x = torch.tensor(rows, dtype=torch.float32)  # [n, m]  n,m <= target_size

    # Pad: F.pad order is (left, right, top, bottom)
    pad_cols = target_size - x.shape[1]
    pad_rows = target_size - x.shape[0]

    x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, n, m]
    x_padded = F.pad(x, (0, pad_cols, 0, pad_rows), mode="constant", value=pad_value)

    return x_padded.squeeze(0).squeeze(0)  # [target_size, target_size]


if __name__ == "__main__":
    # resize_matrix smoke tests
    small = [[1.0, 0.5], [0.3, 0.9], [0.1, 0.7]]   # 3x2
    out = resize_matrix(small, 32)
    print(out.shape)   # torch.Size([32, 32])

    large = [[float(i * 70 + j) for j in range(70)] for i in range(70)]   # 70x70
    out2 = resize_matrix(large, 32)
    print(out2.shape)  # torch.Size([32, 32])
