from typing import List
import torch
import torch.nn.functional as F

MAX_SENTENCES = 64


def pad_matrix(mat: List[list], target_size: int = MAX_SENTENCES, pad_value: int = 0) -> torch.Tensor:
    """Pad (or truncate) a 2D matrix to a square target_size × target_size tensor.

    If the input is larger than target_size in either dimension it is silently
    truncated — the first target_size rows/columns are kept.  This avoids hard
    crashes when a text has more than MAX_SENTENCES sentences.

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
    matrix = [[1, 2], [3, 4]]
    padded = pad_matrix(matrix, 64, pad_value=0)
    print(padded)
    print(padded.shape)  # torch.Size([64, 64])

    # Truncation smoke test
    big = [[float(i * 70 + j) for j in range(70)] for i in range(70)]
    truncated = pad_matrix(big, 64)
    print(truncated.shape)  # torch.Size([64, 64])
