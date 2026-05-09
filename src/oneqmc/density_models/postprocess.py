from typing import Sequence

import jax
import numpy as np
import jax.numpy as jnp
from jax import lax

from ..molecule import Molecule
from .analysis import ScoreMatchingDensityModel, get_dft_grid
from .operators import AutoDiffDerivativeOperator, NumericallyStableKSPotentialOperator
from ..thirdparty.cubetools import write_cube

# FIXME
import logging
log = logging.getLogger(__name__)


def create_npz_density_file(
    density_model: ScoreMatchingDensityModel,
    mol: Molecule,
    output_path: str,
    levels: Sequence[int],
):
    data = {}
    for level in levels:
        grid_r, _ = get_dft_grid(mol, level)
        derivatives_up = jax.vmap(
            AutoDiffDerivativeOperator(density_model.unnormalized_log_density_up, ("grad", "lap"))
        )(grid_r)
        derivatives_down = jax.vmap(
            AutoDiffDerivativeOperator(density_model.unnormalized_log_density_down, ("grad", "lap"))
        )(grid_r)
        data[f"{level}_rho_up"] = np.array(
            jax.vmap(density_model.spin_up_density)(grid_r), dtype=np.float64
        )
        data[f"{level}_rho_down"] = np.array(
            jax.vmap(density_model.spin_down_density)(grid_r), dtype=np.float64
        )
        data[f"{level}_grad_log_rho_up"] = np.array(derivatives_up[..., :3], np.float64)
        data[f"{level}_grad_log_rho_down"] = np.array(derivatives_down[..., :3], np.float64)
        data[f"{level}_lap_log_rho_up"] = np.array(derivatives_up[..., 3], np.float64)
        data[f"{level}_lap_log_rho_down"] = np.array(derivatives_down[..., 3], np.float64)
        data[f"{level}_effective_minus_external_potential_up"] = np.array(
            jax.vmap(
                NumericallyStableKSPotentialOperator(
                    mol.to_mol_conf(len(mol.charges)).nuclei,
                    density_model.unnormalized_log_density_up,
                    AutoDiffDerivativeOperator,
                )
            )(grid_r),
            dtype=np.float64,
        )
        data[f"{level}_effective_minus_external_potential_down"] = np.array(
            jax.vmap(
                NumericallyStableKSPotentialOperator(
                    mol.to_mol_conf(len(mol.charges)).nuclei,
                    density_model.unnormalized_log_density_down,
                    AutoDiffDerivativeOperator,
                )
            )(grid_r),
            dtype=np.float64,
        )
    with open(output_path, "wb") as f:
        np.savez(f, **data)


def create_cube_density_file(
    density_model: ScoreMatchingDensityModel,
    mol: Molecule,
    # nx: int,
    # ny: int,
    # nz: int,
    output_path: str,
):
    # FIXME
    nx = ny = nz = 101
    cube_size = np.array([12, 12, 12], dtype=np.float64)

    step = np.empty(3)
    step[0] = cube_size[0] / (nx - 1)
    step[1] = cube_size[1] / (ny - 1)
    step[2] = cube_size[2] / (nz - 1)

    Bohr2Ang = 0.5_291_772_109_03
    mass_center = np.array([0.516931383, -0.055573255, 0.008948753]) / Bohr2Ang
    origin = mass_center - cube_size / 2
    meta = {
        "org": origin,
        "xvec": np.array((step[0], 0, 0)),
        "yvec": np.array((0, step[1], 0)),
        "zvec": np.array((0, 0, step[2])),
        "atoms": tuple(zip(mol.charges, mol.coords)),
    }
    log.info(f"{step = }")
    log.info(f"{meta = }")

    # # TODO: Support xvec, yvec and zvec as 3-component vectors?
    # grid_z = jnp.linspace(origin[2], origin[2] + cube_size[2], nz)
    # log.info(f"{grid_z.shape = }")

    # def rho_xy(x, y):
    #     # TODO: Support xvec, yvec and zvec as 3-component vectors?
    #     grid_x = jnp.ones_like(grid_z) * x[0]
    #     grid_y = jnp.ones_like(grid_z) * y[1]
    #     grid_r = jnp.vstack((grid_x, grid_y, grid_z)).T
    #     # log.info(f"{x = } { y = } {grid_z.shape = } {grid_x.shape = } {grid_y.shape = } {grid_r.shape = }")
    #     rho = jax.vmap(density_model.__call__)(grid_r)
    #     # log.info(f"{rho.shape = }")
    #     return np.array(rho)

    # write_cube((nx, ny, nz), rho_xy, meta, output_path)

    # Create rectangular 3D grid
    # Code adapted from notebooks/05_alkane_scalability
    def get_coords(box, boxorig):
        xs = np.linspace(0, 1, nx)
        ys = np.linspace(0, 1, ny)
        zs = np.linspace(0, 1, nz)
        frac_coords = np.stack(np.meshgrid(xs, ys, zs), axis=-1)
        # permuting x<->y is necessary to match weird ordering of cube format
        return np.einsum("yxzi,ij->xyzj", frac_coords, box) + boxorig

    grid_r = get_coords(np.diag(cube_size), origin).reshape(-1, 3)
    # log.info(f"{orig_shape = } {grid_r.shape = }")
    # log.info(f"{grid_r = }")
    rho = lax.map(density_model.__call__, grid_r, batch_size=10000)
    rho.reshape(nx, ny, nz)
    with open(output_path, "wb") as f:
        np.save(f, np.asarray(rho))

    # # rho = np.array(
    # #     jax.vmap(density_model.__call__)(grid_r), dtype=np.float64
    # # )
    # rho = jax.vmap(density_model.__call__)(grid_r)
    # log.info(f"{rho.shape = }")

    # # derivatives = jax.vmap(
    # #     AutoDiffDerivativeOperator(density_model.unnormalized_log_density, ("grad", "lap"))
    # # )(grid_r)
    # write_cube(rho, meta, output_path)
