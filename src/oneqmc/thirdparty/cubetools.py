#------------------------------------------------------------------------------
# Module: cubetools
#------------------------------------------------------------------------------
#
# Description:
# Module to work with Gaussian cube format files
# (see http://paulbourke.net/dataformats/cube/)
#
#------------------------------------------------------------------------------
#
# What does it do:
# * Read/write cube files to/from numpy arrays (dtype=float*)
# * Read/write pairse of cube files to/from numpy arrays (dtype=complex*)
# * Provides a CubeFile object, to be used when cubefiles with 
#   constant and static data is required. It simulates the readline method
#   of a file object with a cube file opened, without creating a file
#
#------------------------------------------------------------------------------
#
# Dependency: numpy
#
#------------------------------------------------------------------------------
#
# Author: P. R. Vaidyanathan (aditya95sriram <at> gmail <dot> com)
# Date: 25th June 2017
#
#------------------------------------------------------------------------------
#
# MIT License
# 
# Copyright (c) 2019 P. R. Vaidyanathan
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#------------------------------------------------------------------------------

def _putline(*args):
    """
    Generate a line to be written to a cube file where 
    the first field is an int and the remaining fields are floats.
    
    params:
        *args: first arg is formatted as int and remaining as floats
    
    returns: formatted string to be written to file with trailing newline
    """
    s = "{0:^ 8d}".format(args[0])
    s += "".join("{0:< 12.6f}".format(arg) for arg in args[1:])
    return s + "\n"


def write_cube(data_shape, data_xy, meta, fname):
    """
    Write volumetric data to cube file along
    
    params:
        data_shape: nx, ny, nz
        data_xy: callback providing data vector for fixed x and y
        meta: dict containing metadata with following keys
            atoms: list of atoms in the form (mass, [position])
            org: origin
            xvec,yvec,zvec: lattice vector basis
        fname: filename of cubefile (existing files overwritten)
    
    returns: None
    """
    with open(fname, "w") as cube:
        # first two lines are comments
        cube.write(" Cubefile created by cubetools.py\n  source: none\n")
        natm = len(meta['atoms'])
        nx, ny, nz = data_shape
        cube.write(_putline(natm, *meta['org'])) # 3rd line #atoms and origin
        cube.write(_putline(nx, *meta['xvec']))
        cube.write(_putline(ny, *meta['yvec']))
        cube.write(_putline(nz, *meta['zvec']))
        for atom_mass, atom_pos in meta['atoms']:
            cube.write(_putline(atom_mass, *atom_pos)) #skip the newline
        for i in range(nx):
            x = meta['org'] + i * meta['xvec']
            for j in range(ny):
                y = meta['org'] + j * meta['yvec']
                data = data_xy(x, y)
                for k in range(nz):
                    if (i or j or k) and k%6==0:
                        cube.write("\n")
                    cube.write(" {0: .5E}".format(data[k]))
