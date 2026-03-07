# LBM-based absolute permeability computation.
import numpy as np

__all__ = ["permeability_lbm", "PermeabilityResult"]


class PermeabilityResult:
    """Container for LBM permeability simulation results."""

    def __init__(
        self, im, axis, porosity, k_lu, u_darcy, u_pore, velocity, k_m2=None, k_mD=None
    ):
        """
        Parameters
        ----------
        im : ndarray
            The boolean image used in the simulation.
        axis : int
            The axis along which flow was computed.
        porosity : float
            Pore volume fraction.
        k_lu : float
            Permeability in lattice units (voxels^2).
        u_darcy : float
            Darcy (superficial) velocity in lattice units.
        u_pore : float
            Mean pore-space velocity in lattice units.
        velocity : ndarray, shape (nx, ny, nz, 3)
            Steady-state velocity field.
        k_m2 : float or None
            Permeability in m^2 (if dx_m was provided).
        k_mD : float or None
            Permeability in milliDarcy (if dx_m was provided).
        """
        self.im = im
        self.axis = axis
        self.porosity = porosity
        self.k_lu = k_lu
        self.u_darcy = u_darcy
        self.u_pore = u_pore
        self.velocity = velocity
        self.k_m2 = k_m2
        self.k_mD = k_mD

    def __repr__(self):
        parts = f"PermeabilityResult(k_lu={self.k_lu:.6e}, axis={self.axis}"
        if self.k_mD is not None:
            parts += f", k_mD={self.k_mD:.4f}"
        return parts + ")"


# Pressure BCs used by the flow solver
_RHO_IN = 1.00
_RHO_OUT = 0.99
_CS2 = 1.0 / 3.0  # D3Q19 speed of sound squared
_MD_CONVERSION = 9.869233e-16  # 1 milliDarcy in m^2


def permeability_lbm(
    im,
    *,
    axis: int,
    nu: float = 1 / 6,
    tol: float = 1e-3,
    n_steps: int = 100_000,
    sparse: bool = False,
    dx_m: float = None,
) -> PermeabilityResult:
    """Compute absolute permeability using LBM (D3Q19 MRT).

    Solves creeping (Stokes) flow on the pore space of a 3D binary image
    using the Lattice Boltzmann Method with a D3Q19 MRT collision
    operator. Permeability is extracted via Darcy's law.

    Parameters
    ----------
    im : ndarray, shape (nx, ny, nz)
        Binary image. True (or 1) = pore, False (or 0) = solid.
    axis : int
        Axis along which to apply the pressure gradient
        (0=x, 1=y, 2=z).
    nu : float
        Kinematic viscosity in lattice units. Default 1/6.
        Relaxation time tau_f = 3*nu + 0.5.
    tol : float
        Convergence tolerance on relative velocity change.
        Default 1e-3.
    n_steps : int
        Maximum number of LBM iterations. Default 100000.
    sparse : bool
        If True, use Taichi sparse storage. Default False.
    dx_m : float or None
        Physical voxel size in metres. If provided, results include
        permeability in m^2 and milliDarcy.

    Returns
    -------
    result : PermeabilityResult
        Contains k_lu, porosity, u_darcy, u_pore, velocity field,
        and optionally k_m2 and k_mD.

    Raises
    ------
    RuntimeError
        If the image has no pore voxels.
    """
    from ._lbm._flow_solver import solve_flow

    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    solid = (im == 0).astype(np.int8)
    if solid.sum() == solid.size:
        raise RuntimeError("Image has no pore voxels.")

    velocity, density = solve_flow(
        solid,
        axis=axis,
        nu=nu,
        n_steps=n_steps,
        tol=tol,
        sparse=sparse,
    )

    # Compute permeability via Darcy's law
    pore_mask = solid == 0
    porosity = float(pore_mask.sum()) / pore_mask.size
    v_flow = velocity[..., axis]
    L = im.shape[axis]
    u_darcy = float(np.mean(v_flow))
    u_pore = float(np.mean(v_flow[pore_mask]))
    grad_P = (_RHO_IN - _RHO_OUT) * _CS2 / L
    mu = nu  # dynamic viscosity; rho ~ 1 in LBM
    k_lu = u_darcy * mu / grad_P

    # Physical units
    k_m2 = k_mD = None
    if dx_m is not None:
        k_m2 = k_lu * dx_m**2
        k_mD = k_m2 / _MD_CONVERSION

    return PermeabilityResult(
        im=np.asarray(im, dtype=bool),
        axis=axis,
        porosity=porosity,
        k_lu=k_lu,
        u_darcy=u_darcy,
        u_pore=u_pore,
        velocity=velocity,
        k_m2=k_m2,
        k_mD=k_mD,
    )
