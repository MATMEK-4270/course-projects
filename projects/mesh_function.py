import numpy as np
from collections.abc import Callable


def mesh_function(f: Callable[[float], float], t: float) -> np.ndarray:
    tuple= np.zeros(len(t))
    for i in range(len(t)):
        tuple[i] = f(t[i])
    return tuple
        
    raise NotImplementedError


def func(t: float) -> float:
    if 0<= t <= 3:
        return np.exp(-t)
    elif 3<t<=4:
        return np.exp(-3*t)
    else:
        raise ValueError
    raise NotImplementedError



def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    
    assert np.allclose(fun, f)

if __name__ == "__main__":
    test_mesh_function()
T=np.arange(0,4.1,0.1)    
ph= mesh_function(func,T)
import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))

plt.plot(T,ph)
plt.title("strainer",fontsize=16,fontweight ="black")
plt.legend(["mesh"])
plt.xticks(np.arange(0,4.1,0.1))
plt.xticks(fontsize=10,fontweight="bold")
plt.yticks(fontsize=10,fontweight="bold")
plt.show()
