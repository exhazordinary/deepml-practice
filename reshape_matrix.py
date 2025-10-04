import numpy as np

def reshape_matrix(a: list[list[int | float]], new_shape: tuple[int, int]) -> list[list[int | float]]:
    np_array = np.array(a)
    if np_array.size != new_shape[0] * new_shape[1]:
        return []

    reshaped_matrix = np_array.reshape(new_shape)
    return reshaped_matrix.tolist()