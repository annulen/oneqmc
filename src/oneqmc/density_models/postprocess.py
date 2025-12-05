from typing import Sequence

import jax
import numpy as np

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
        "xvec": (step[0], 0, 0),
        "yvec": (0, step[1], 0),
        "zvec": (0, 0, step[2]),
        "atoms": tuple(zip(mol.charges, mol.coords)),
    }
    log.info(f"{step = }")
    log.info(f"{meta = }")

    # Create rectangular 3D grid
    # Use mgrid?? E.g. np.mgrid[-2:2.1, -2:2.1, -2:2.1].reshape(3,-1).T
    # x = np.linspace(, 2, num=nx)
    # y = np.linspace(-2, 2, num=ny)
    # z = np.linspace(-2, 2, num=nz)
    def end_value(i):
        return origin[i] + cube_size[i] + step[i]

    grid_r = np.mgrid[
        origin[0]:end_value(0):step[0],
        origin[1]:end_value(1):step[1],
        origin[2]:end_value(2):step[2],
    ].reshape(3, -1).T
    log.info(f"{grid_r.shape = }")
    log.info(f"{grid_r = }")

    # rho = np.array(
    #     jax.vmap(density_model.__call__)(grid_r), dtype=np.float64
    # )
    rho = jax.vmap(density_model.__call__)(grid_r)
    log.info(f"{rho.shape = }")

    # derivatives = jax.vmap(
    #     AutoDiffDerivativeOperator(density_model.unnormalized_log_density, ("grad", "lap"))
    # )(grid_r)
    write_cube(rho, meta, output_path)
