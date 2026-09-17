import numpy as np
import sympy as sp
from poisson import Poisson
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

x, y = sp.symbols("x,y")

# Below we create a solver that reuses some of the implementation from
# the 1D solver in poisson.py.


class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with Dirichlet boundary conditions.
    """

    def __init__(self, Lx: float, Ly: float):
        self.px = Poisson(Lx)  # we can reuse some of the code from the 1D case
        self.py = Poisson(Ly)

    def create_mesh(self, Nx: int, Ny: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a 2D Cartesian mesh

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in x-direction
        Ny : int
            The number of uniform intervals in y-direction
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh
        """
        xi = self.px.create_mesh(Nx)
        yj = self.py.create_mesh(Ny)
        xij, yij = np.meshgrid(xi, yj, indexing="ij", sparse=True)
        return xij, yij

    def laplace(self, Nx: int, Ny: int) -> sparse.lil_matrix:
        """Return a vectorized Laplace operator

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in x-direction
        Ny : int
            The number of uniform intervals in y-direction

        Returns
        -------
        A : scipy sparse LIL matrix
            The vectorized Laplace operator
        """
        D2x = self.px.D2(Nx, self.px.L / Nx)
        D2y = self.py.D2(Ny, self.py.L / Ny)
        Ix = sparse.eye(Nx + 1)
        Iy = sparse.eye(Ny + 1)
        return (sparse.kron(D2x, Iy) + sparse.kron(Ix, D2y)).tolil()

    def assemble(self, Nx: int, Ny: int, f: sp.Expr, ue: sp.Expr) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return assembled coefficient matrix A and right hand side vector b

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in x-direction
        Ny : int
            The number of uniform intervals in y-direction
        f : Sympy expression
            The right hand side as a Sympy expression in x and y
        ue : Sympy expression
            The exact solution as a Sympy expression in x and y

        Returns
        -------
        A : scipy sparse CSR matrix
            Coefficient matrix
        b : 1D array
            Right hand side vector
        """
        A = self.laplace(Nx, Ny)
        bnds = self.get_boundary_indices(Nx, Ny)
        for i in bnds:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()
        b = np.zeros((Nx + 1, Ny + 1))
        xij, yij = self.create_mesh(Nx, Ny)
        b[:, :] = self.meshfunction(f, xij, yij)
        # Set boundary conditions
        uij = self.meshfunction(ue, xij, yij)
        b.ravel()[bnds] = uij.ravel()[bnds]
        return A, b

    def meshfunction(self, u: sp.Expr, xij: np.ndarray, yij: np.ndarray) -> np.ndarray:
        """Return Sympy function as mesh function

        Parameters
        ----------
        u : Sympy function

        Returns
        -------
        array - The input function as a mesh function
        """
        return sp.lambdify((x, y), u)(xij, yij)

    def get_boundary_indices(self, Nx: int, Ny: int) -> np.ndarray:
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((Nx + 1, Ny + 1), dtype=bool)
        B[1:-1, 1:-1] = 0
        return np.where(B.ravel() == 1)[0]

    def l2_error(self, u: np.ndarray, ue: sp.Expr) -> float:
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy expression
            The exact solution

        Returns
        -------
        float - The l2-error

        """
        Nx, Ny = u.shape[0] - 1, u.shape[1] - 1
        dx = self.px.L / Nx
        dy = self.py.L / Ny
        xij, yij = self.create_mesh(Nx, Ny)
        return np.sqrt(dx * dy * np.sum((u - self.meshfunction(ue, xij, yij)) ** 2))

    def __call__(self, Nx: int, Ny: int, ue: sp.Expr) -> np.ndarray:
        """Solve Poisson's equation with a given manufactured solution

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in x-direction
        Ny : int
            The number of uniform intervals in y-direction
        ue : Sympy expression
            The exact solution

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(Nx, Ny, sp.diff(ue, x, 2) + sp.diff(ue, y, 2), ue)
        return sparse_linalg.spsolve(A, b.ravel()).reshape((Nx + 1, Ny + 1))


def convergence_rates(ue, m=6):
    E = []
    h = []
    N0 = 8
    for _ in range(m):
        sol = Poisson2D(1, 1)
        u = sol(N0, N0, ue)
        E.append(sol.l2_error(u, ue))
        h.append(sol.px.L / N0)
        N0 *= 2
    r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m, 1)]
    return r, np.array(E), np.array(h)


def test_poisson2d():
    r, _, _ = convergence_rates(sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y)))
    assert abs(r[-1] - 2) < 1e-2


if __name__ == "__main__":
    test_poisson2d()
