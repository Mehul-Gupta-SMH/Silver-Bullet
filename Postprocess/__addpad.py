from typing import List
import torch
import torch.nn.functional as F


def pad_matrix(mat: List[list], target_size: int = 64, pad_value: int = 0) -> torch.tensor:
    """
    Pad a 2D matrix to a target size with a specified pad value.
    Args:
        mat (list of list of int/float): The input 2D matrix.
        target_size (int): The desired size for both dimensions after padding.
        pad_value (int/float): The value to use for padding.
    Returns:
        list of list of int/float: The padded 2D matrix.
    Raises:
        ValueError: If the input matrix is larger than the target size in any dimension.
    """
    # Convert to tensor
    x = torch.tensor(mat, dtype=torch.float32)  # shape [2, 2]

    # Conv layers expect 4D: [batch, channel, height, width]
    x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, 2, 2]

    # Pad: (left, right, top, bottom)
    x_padded = F.pad(x,
                     (0, target_size - len(mat[0]), 0, target_size-len(mat)),
                     mode="constant",
                     value=pad_value)

    return x_padded.squeeze(0).squeeze(0)  # [64, 64]


if __name__ == "__main__":
    # Example
    matrix = [
        [1, 2],
        [3, 4]
    ]

    padded_matrix = pad_matrix(matrix, 64, pad_value=0)

    print(padded_matrix)
    print(padded_matrix.shape)  # Should be (64, 64)
