import torch

def transpose_matrix(a) -> torch.Tensor:
    """
    Transpose a 2D matrix `a` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a transposed tensor.
    """
    a_t = torch.as_tensor(a)  # Convert input to a PyTorch tensor
    if a_t.dim() != 2:
        raise ValueError("Input must be a 2D matrix.")
    return a_t.t()  # Transpose the tensor


# Example usage
if __name__ == "__main__":
    # Input as a Python list
    matrix_list = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    print("Transposed (from list):")
    print(transpose_matrix(matrix_list))

    # Input as a NumPy array
    import numpy as np
    matrix_np = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    print("\nTransposed (from NumPy array):")
    print(transpose_matrix(matrix_np))

    # Input as a PyTorch tensor
    matrix_tensor = torch.tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])
    print("\nTransposed (from PyTorch tensor):")
    print(transpose_matrix(matrix_tensor))