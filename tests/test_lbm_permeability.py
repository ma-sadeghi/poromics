# Tests for LBM-based permeability solver.
import numpy as np
import pytest
from poromics.simulation import TransientFlow
from poromics import permeability_lbm


def _make_cylinder(L, N, R):
    """Create a boolean image with a cylindrical pore along axis 0."""
    im = np.zeros((L, N, N), dtype=bool)
    yc, zc = N / 2, N / 2
    for j in range(N):
        for k in range(N):
            if (j - yc + 0.5) ** 2 + (k - zc + 0.5) ** 2 <= R**2:
                im[:, j, k] = True
    return im


def _make_parallel_plates(L, N, h):
    """Create a boolean image with parallel plates (walls in y, periodic in z).

    Flow along axis 0. Solid walls bound the channel in y.
    """
    im = np.zeros((L, N, N), dtype=bool)
    y_start = (N - h) // 2
    im[:, y_start : y_start + h, :] = True
    return im


def _make_rectangle(L, N, h, w):
    """Create a boolean image with a rectangular duct (h in y, w in z).

    Flow along axis 0. Centered in the y-z plane.
    """
    im = np.zeros((L, N, N), dtype=bool)
    y_start = (N - h) // 2
    z_start = (N - w) // 2
    im[:, y_start : y_start + h, z_start : z_start + w] = True
    return im


def _make_equilateral_triangle(L, N, side):
    """Create a boolean image with an equilateral triangle duct.

    Flow along axis 0. Triangle centered in y-z plane, with base
    parallel to z-axis.
    """
    im = np.zeros((L, N, N), dtype=bool)
    h_tri = side * np.sqrt(3) / 2
    yc = N / 2
    zc = N / 2
    # Vertices: top=(yc-h/3*2, zc), bottom-left=(yc+h/3, zc-side/2),
    #           bottom-right=(yc+h/3, zc+side/2)
    y_top = yc - 2 * h_tri / 3
    y_bot = yc + h_tri / 3
    for j in range(N):
        for k in range(N):
            y = j + 0.5
            z = k + 0.5
            # Check if (y,z) is inside the triangle
            # Triangle: base at y=y_bot from z=zc-side/2 to z=zc+side/2
            #           apex at y=y_top, z=zc
            if y < y_top or y > y_bot:
                continue
            # At height y, the width of the triangle
            frac = (y_bot - y) / (y_bot - y_top)
            half_w = (side / 2) * (1 - frac)
            if abs(z - zc) <= half_w:
                im[:, j, k] = True
    return im


def _local_dp_dx(density, solid, x_lo, x_hi):
    """Compute pore-averaged pressure gradient between two x-slices."""
    cs2 = 1.0 / 3.0
    pore_2d = solid[0, :, :] == 0
    rho_lo = density[x_lo][pore_2d].mean()
    rho_hi = density[x_hi][pore_2d].mean()
    return (rho_hi - rho_lo) * cs2 / (x_hi - x_lo)


def _run_and_get_Q(solid, L, N, nu, n_steps=20000, tol=1e-4):
    """Run LBM, return (Q_lbm, dp_dx_local, pore_2d, velocity, density)."""
    im = solid == 0
    solver = TransientFlow(im, axis=0, nu=nu, voxel_size=1.0)
    solver.run(n_steps=n_steps, tol=tol)
    velocity = solver._solver.get_velocity()
    density = solver._solver.get_density()
    # Measure dp/dx in the interior (20%-80%) to avoid inlet/outlet BC artifacts.
    # The equilibrium pressure BC creates a sharp pressure jump at the
    # boundary faces; the fully-developed region has a uniform gradient.
    x_lo = int(0.2 * L)
    x_hi = int(0.8 * L)
    dp_dx = _local_dp_dx(density, solid, x_lo, x_hi)
    mid_x = L // 2
    pore_2d = solid[mid_x, :, :] == 0
    Q_lbm = float(np.sum(velocity[mid_x, :, :, 0][pore_2d]))
    return Q_lbm, dp_dx, pore_2d, velocity, density


class CylinderFlowTest:
    """Compare LBM flow in a cylinder against Hagen-Poiseuille."""

    L, N, R, NU = 100, 40, 10, 1.0 / 6.0

    @pytest.fixture(scope="class")
    def lbm_result(self):
        im = _make_cylinder(self.L, self.N, self.R)
        solid = (~im).astype(np.int8)
        return _run_and_get_Q(solid, self.L, self.N, self.NU)

    def test_parabolic_profile(self, lbm_result):
        """Velocity at midplane should follow a parabolic (1 - r²/R²) profile."""
        _, _, pore_2d, velocity, _ = lbm_result
        mid_x = self.L // 2
        vx = velocity[mid_x, :, :, 0]
        yc, zc = self.N / 2, self.N / 2
        r2_vals, vx_vals = [], []
        for j in range(self.N):
            for k in range(self.N):
                if pore_2d[j, k]:
                    r2_vals.append((j - yc + 0.5) ** 2 + (k - zc + 0.5) ** 2)
                    vx_vals.append(vx[j, k])
        r2_arr = np.array(r2_vals)
        vx_arr = np.array(vx_vals)
        A = np.column_stack([np.ones_like(r2_arr), -r2_arr])
        coeffs = np.linalg.lstsq(A, vx_arr, rcond=None)[0]
        fitted = A @ coeffs
        ss_res = np.sum((vx_arr - fitted) ** 2)
        ss_tot = np.sum((vx_arr - vx_arr.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot
        assert r_squared > 0.99, f"R²={r_squared:.4f}"

    def test_Q_vs_hagen_poiseuille(self, lbm_result):
        """Flow rate at midplane should match HP: Q = |dp/dx| * pi*R^4 / (8*nu)."""
        Q_lbm, dp_dx, pore_2d, _, _ = lbm_result
        R_eff = np.sqrt(pore_2d.sum() / np.pi)
        Q_hp = abs(dp_dx) * np.pi * R_eff**4 / (8 * self.NU)
        error = abs(Q_lbm - Q_hp) / Q_hp
        assert error < 0.10, f"Q error={error:.2%}"


class ParallelPlatesFlowTest:
    """Compare LBM flow between parallel plates against plane Poiseuille."""

    L, N, H, NU = 100, 40, 20, 1.0 / 6.0

    @pytest.fixture(scope="class")
    def lbm_result(self):
        im = _make_parallel_plates(self.L, self.N, self.H)
        solid = (~im).astype(np.int8)
        return _run_and_get_Q(solid, self.L, self.N, self.NU)

    def test_parabolic_profile(self, lbm_result):
        """Velocity should follow plane Poiseuille parabolic u(y) ∝ y*(h-y)."""
        _, _, _, velocity, _ = lbm_result
        mid_x = self.L // 2
        mid_z = self.N // 2
        vx = velocity[mid_x, :, mid_z, 0]
        y_start = (self.N - self.H) // 2
        y_end = y_start + self.H
        y_pore = np.arange(y_start, y_end, dtype=float)
        vx_pore = vx[y_start:y_end]
        y0, y1 = y_start - 0.5, y_end - 0.5
        parab = (y_pore - y0) * (y1 - y_pore)
        c = float(np.sum(vx_pore * parab) / np.sum(parab**2))
        fitted = c * parab
        ss_res = np.sum((vx_pore - fitted) ** 2)
        ss_tot = np.sum((vx_pore - vx_pore.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot
        assert r_squared > 0.99, f"R²={r_squared:.4f}"

    def test_Q_vs_plane_poiseuille(self, lbm_result):
        """Flow rate per unit width should match q = |dp/dx| * h^3 / (12*nu)."""
        Q_lbm, dp_dx, _, velocity, _ = lbm_result
        # Q per unit z-width at midplane
        mid_x = self.L // 2
        mid_z = self.N // 2
        y_start = (self.N - self.H) // 2
        y_end = y_start + self.H
        q_lbm = float(np.sum(velocity[mid_x, y_start:y_end, mid_z, 0]))
        q_hp = abs(dp_dx) * self.H**3 / (12 * self.NU)
        error = abs(q_lbm - q_hp) / q_hp
        assert error < 0.10, f"q error={error:.2%}"


class RectangularDuctFlowTest:
    """Compare LBM flow in a rectangular duct against Boussinesq series."""

    L, N, H, W, NU = 100, 50, 20, 30, 1.0 / 6.0

    @pytest.fixture(scope="class")
    def lbm_result(self):
        im = _make_rectangle(self.L, self.N, self.H, self.W)
        solid = (~im).astype(np.int8)
        return _run_and_get_Q(solid, self.L, self.N, self.NU)

    def test_Q_vs_rectangular_duct(self, lbm_result):
        """Flow rate should match Boussinesq series solution within 10%.

        Q = |dp/dx|/(12*nu) * h^3*l * [1 - 192*h/(pi^5*l) * sum_n tanh(β_n*l/2)/n^5]
        where β_n = n*pi/h, n = 1,3,5,...
        (h = shorter side, l = longer side)
        """
        Q_lbm, dp_dx, _, _, _ = lbm_result
        short = min(self.H, self.W)
        long = max(self.H, self.W)
        correction = 0.0
        for n in range(1, 200, 2):
            beta_n = n * np.pi / short
            correction += np.tanh(beta_n * long / 2) / n**5
        correction *= 192 * short / (np.pi**5 * long)
        q_rect = abs(dp_dx) / (12 * self.NU) * short**3 * long * (1 - correction)
        error = abs(Q_lbm - q_rect) / q_rect
        assert error < 0.10, f"Q error={error:.2%}"


class TriangularDuctFlowTest:
    """Compare LBM flow in an equilateral triangle against Boussinesq."""

    L, N, SIDE, NU = 100, 60, 24, 1.0 / 6.0

    @pytest.fixture(scope="class")
    def lbm_result(self):
        im = _make_equilateral_triangle(self.L, self.N, self.SIDE)
        solid = (~im).astype(np.int8)
        return _run_and_get_Q(solid, self.L, self.N, self.NU)

    def test_Q_vs_equilateral_triangle(self, lbm_result):
        """Flow rate should match Boussinesq's formula within 15%.

        For equilateral triangle with side 2h/sqrt(3):
          Q = G * h^4 / (60*sqrt(3)*mu)
        Here h = side * sqrt(3) / 2 (the triangle height).
        The Boussinesq formula uses the half-height parameter.
        Equivalently: Q = G * sqrt(3)/320 * a^4  where a = side length.

        Wider tolerance (15%) because voxelized triangle has significant
        staircase error at the boundaries.
        """
        Q_lbm, dp_dx, _, _, _ = lbm_result
        a = self.SIDE
        Q_tri = abs(dp_dx) * np.sqrt(3) * a**4 / (320 * self.NU)
        error = abs(Q_lbm - Q_tri) / Q_tri
        assert error < 0.15, f"Q error={error:.2%}"


class PermeabilityAPITest:
    """Test the public permeability_lbm() API."""

    def test_k_positive_finite(self):
        """Permeability of open space should be positive and finite."""
        im = np.ones((20, 20, 20), dtype=bool)
        result = permeability_lbm(im, axis=0, voxel_size=1.0, n_steps=5000, tol=1e-3)
        assert result.k > 0
        assert np.isfinite(result.k)
        assert result.porosity == 1.0

    def test_no_pores(self):
        """Fully solid image should raise RuntimeError."""
        im = np.zeros((10, 10, 10), dtype=bool)
        with pytest.raises(RuntimeError):
            permeability_lbm(im, axis=0, voxel_size=1.0)

    def test_invalid_axis(self):
        """Invalid axis should raise ValueError."""
        im = np.ones((10, 10, 10), dtype=bool)
        with pytest.raises(ValueError):
            permeability_lbm(im, axis=3, voxel_size=1.0)

    def test_k_scales_with_voxel_size(self):
        """Permeability should scale with voxel_size²."""
        im = np.ones((20, 20, 20), dtype=bool)
        dx = 1e-6
        r1 = permeability_lbm(im, axis=0, voxel_size=1.0, n_steps=5000, tol=1e-3)
        r2 = permeability_lbm(im, axis=0, voxel_size=dx, n_steps=5000, tol=1e-3)
        assert r2.k == pytest.approx(r1.k * dx**2, rel=1e-10)


class RescaleTest:
    """Test PermeabilityResult.rescale()."""

    @pytest.fixture()
    def base_result(self):
        im = np.ones((20, 20, 20), dtype=bool)
        return permeability_lbm(im, axis=0, voxel_size=1e-6, n_steps=5000, tol=1e-3)

    def test_rescale_identity(self, base_result):
        """Rescaling with the original parameters reproduces the result."""
        r2 = base_result.rescale(voxel_size=1e-6, nu=1e-6)
        assert r2.k == pytest.approx(base_result.k, rel=1e-10)
        assert r2.u_darcy == pytest.approx(base_result.u_darcy, rel=1e-10)
        assert r2.u_pore == pytest.approx(base_result.u_pore, rel=1e-10)
        np.testing.assert_allclose(r2.velocity, base_result.velocity, rtol=1e-10)

    def test_rescale_k_scales_with_voxel_size(self, base_result):
        """k should scale with voxel_size²."""
        dx2 = 2e-6
        r2 = base_result.rescale(voxel_size=dx2, nu=1e-6)
        assert r2.k == pytest.approx(base_result.k * 4.0, rel=1e-10)

    def test_rescale_velocity_changes_with_nu(self, base_result):
        """Velocity should change when nu changes."""
        r2 = base_result.rescale(voxel_size=1e-6, nu=2e-6)
        assert r2.u_darcy != pytest.approx(base_result.u_darcy, rel=1e-3)
        assert r2.k == pytest.approx(base_result.k, rel=1e-10)

    def test_rescale_pressure_requires_rho(self, base_result):
        """Pressure should be None when rho is not provided."""
        r2 = base_result.rescale(voxel_size=1e-6, nu=1e-6)
        assert r2.pressure is None
        assert r2.kinematic_pressure is not None

    def test_rescale_pressure_with_rho(self, base_result):
        """Pressure should be an array when rho is provided."""
        r2 = base_result.rescale(voxel_size=1e-6, nu=1e-6, rho=1000.0)
        assert r2.pressure is not None
        assert r2.pressure.shape == base_result.im.shape
        np.testing.assert_allclose(
            r2.pressure, 1000.0 * r2.kinematic_pressure, rtol=1e-12
        )

    def test_rescale_is_chainable(self, base_result):
        """Rescaling a rescaled result should work."""
        r2 = base_result.rescale(voxel_size=2e-6, nu=2e-6)
        r3 = r2.rescale(voxel_size=1e-6, nu=1e-6)
        assert r3.k == pytest.approx(base_result.k, rel=1e-10)


class PressureUnitsTest:
    """Regression tests for pressure-field unit handling."""

    def test_transient_flow_pressure_raises_without_rho(self):
        im = np.ones((10, 10, 10), dtype=bool)
        solver = TransientFlow(im, axis=0, nu=1e-6, voxel_size=1e-6)
        solver.run(n_steps=100, tol=None)
        with pytest.raises(RuntimeError, match="rho"):
            _ = solver.pressure

    def test_transient_flow_kinematic_pressure_always_available(self):
        im = np.ones((10, 10, 10), dtype=bool)
        solver = TransientFlow(im, axis=0, nu=1e-6, voxel_size=1e-6)
        solver.run(n_steps=100, tol=None)
        Pk = solver.kinematic_pressure
        assert Pk.shape == im.shape
        assert np.isfinite(Pk).all()

    def test_transient_flow_pressure_scales_with_rho(self):
        im = np.ones((10, 10, 10), dtype=bool)
        s1 = TransientFlow(im, axis=0, nu=1e-6, voxel_size=1e-6, rho=1.0)
        s1.run(n_steps=200, tol=None)
        s2 = TransientFlow(im, axis=0, nu=1e-6, voxel_size=1e-6, rho=1000.0)
        s2.run(n_steps=200, tol=None)
        np.testing.assert_allclose(s2.pressure, 1000.0 * s1.pressure, rtol=1e-10)
        np.testing.assert_allclose(
            s1.pressure, 1.0 * s1.kinematic_pressure, rtol=1e-10
        )

    def test_permeability_lbm_pressure_none_without_rho(self):
        im = np.ones((10, 10, 10), dtype=bool)
        result = permeability_lbm(im, axis=0, voxel_size=1e-6,
                                  n_steps=500, tol=1e-3)  # fmt: skip
        assert result.pressure is None
        assert result.kinematic_pressure is not None
        assert result.kinematic_pressure.shape == im.shape

    def test_permeability_lbm_pressure_pa_with_rho(self):
        im = np.ones((10, 10, 10), dtype=bool)
        result = permeability_lbm(im, axis=0, voxel_size=1e-6, rho=1000.0,
                                  n_steps=500, tol=1e-3)  # fmt: skip
        assert result.pressure is not None
        np.testing.assert_allclose(
            result.pressure, 1000.0 * result.kinematic_pressure, rtol=1e-10
        )


class ConvergenceSignalTest:
    """Regression tests for exposing convergence state on results."""

    def test_converged_true_on_easy_case(self):
        im = np.ones((10, 10, 10), dtype=bool)
        result = permeability_lbm(im, axis=0, voxel_size=1e-6,
                                  n_steps=10000, tol=1e-2)  # fmt: skip
        assert result.converged is True
        assert result.n_iterations is not None
        assert result.n_iterations > 0

    def test_converged_false_when_steps_exhausted(self):
        from loguru import logger as _loguru

        messages = []
        handler_id = _loguru.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            im = np.ones((10, 10, 10), dtype=bool)
            result = permeability_lbm(
                im, axis=0, voxel_size=1e-6, n_steps=1, tol=1e-12,
            )
        finally:
            _loguru.remove(handler_id)
        assert result.converged is False
        assert result.n_iterations > 0
        assert any("did not converge" in m for m in messages)
