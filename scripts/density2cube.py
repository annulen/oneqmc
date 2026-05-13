#!/usr/bin/env python

import numpy as np
import sys
from oneqmc.thirdparty.cubetools import write_cube_orig

### FIXME ###

import logging
log = logging.getLogger(__name__)


def make_cube(volume_data, output_path):
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
        "atoms": tuple((2, [0,0,0])) #tuple(zip(mol.charges, mol.coords)),
    }
    log.info(f"{step = }")
    log.info(f"{meta = }")

    volume_data = volume_data.reshape(nx, ny, nz)
    print(volume_data.shape)
    write_cube_orig(volume_data, meta, output_path)

#############


def load_volume_data(f) -> np.ndarray:
    return np.load(f)


def main(args):
    assert len(args) == 2
    # with open(args[0]) as f:
    volume_data = load_volume_data(args[0])
    print(volume_data.shape)
    make_cube(volume_data, args[1])


if __name__ == "__main__":
    main(sys.argv[1:])