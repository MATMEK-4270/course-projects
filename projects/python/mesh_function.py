import numpy as np


def mesh_function(f: callable, t: np.ndarray):
    f_array = np.zeros_like(t, dtype="float64")
    for i, val in enumerate(t):
        f_array[i] = f(val)
    return f_array

def func(t):
    t = np.array(t, dtype="float64")
    return np.piecewise(t, [(0 <= t) * (t <= 3), (3 < t) * (t <= 4)], [lambda x: np.exp(-x), lambda x: np.exp(-3 * x)])

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)
