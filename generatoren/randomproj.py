#!/usr/bin/env python3

# author: Daniel Schmidt (daniel.schmidt@uni-bonn.de), University of Bonn
# date  : july 2021
# copyright: tba

"""
Achlioptas [1] proposes a method to project n points in R^d down to R^k with 
k << d while approximately maintaining pairwise distances. The method uses
a random projection matrix. More precisely, it holds for any two points 
u,v in R^d and their respective projections pu, pv that 

    (1-eps) ||u - v||^2 <= ||pu - pv||^2 <= (1+eps) ||u-v||^2

with probability at least 1-n^{-beta} if 

    k >= (4+2beta) / (eps^2/2 - eps^3/3) ln n     [with 0 < eps <= 1.5]

The guarantee works with either a dense or a sparse random projection matrix.
The dense matrix is computationally cheaper to draw, potentially, but the 
projection itself is more expensive and more prone to numerical errors. 

This script reads points from the standard input, projects them down to
a target dimension using Achlioptas method, and prints the result to the 
standard output. By default, the script uses the sparse projection matrix.

Expected input format is:
<n_points> <dim>
<point1>
<point2>
....

Each <point> is expected with <dim> many coordinates, separated by spaces.

[1] Achlioptas, Dimitris. 
"Database-Friendly Random Projections: Johnson-Lindenstrauss with Binary Coins." 
Journal of Computer and System Sciences, Special Issue on PODS 2001, 66, no. 4,
2003. pp. 671–87. https://doi.org/10.1016/S0022-0000(03)00025-4.
"""

import numpy as np
import sys
import argparse
import math
import enum

def gen_random_dense_proj(n_rows, n_cols, rng):
    """
    Generates a dense random projection matrix.

    This method returns a matrix with n_rows rows and n_cols columns that is 
    filled with random entries. Each entry is set to 1 with probability 1/2, or
    to -1 otherwise.

    Parameters
    ----------
    n_rows : int
        number of rows of the random projection matrix.
    
    n_cols : int
        number of columns of the projection matrix.

    rng : numpy.random.RandomState
        a numpy random generator (used to generate the random entries)
    """
    return rng.choice([-1, 1], size=(n_rows, n_cols))

def gen_random_sparse_proj(n_rows, n_cols, rng):
    """
    Generates a sparse random projection matrix.

    This method returns a matrix with n_rows rows and n_cols columns that is 
    filled with random entries. Each entry is set to 1 with probability 1/6, or
    to 0 with probability 2/3, or to -1 otherwise.

    Parameters
    ----------
    n_rows : int
        number of rows of the random projection matrix.
    
    n_cols : int
        number of columns of the projection matrix.

    rng : numpy.random.RandomState
        a numpy random generator (used to generate the random entries)
    """
    return rng.choice([-1, 0, 0, 0, 0, 1], size=(n_rows, n_cols))

def read_matrix(infile):
    """
    Reads a matrix (points) from a file like object.
    
    Expects one row (point) per line. Coordinates should be separated with 
    spaces.
    """
    return np.loadtxt(infile)

def fit_target_dim(n_points, eps, beta):
    """
    Computes the minimum dimension necessary for Achlioptas guarantee.

    Parameters
    ----------
    n_points : int
        number of projected points

    eps : float
        distances are maintained with a factor of (1+eps)

    beta : float
        desired success probability is 1-n^{-beta}


    Returns
    -------
    int
        The minimum dimension d such that projecting n points to dimension
        d provably maintains pair-wise distances within a factor of (1+eps) with 
        probability of at least 1-n^{-beta}. 
    """
    min_dim = (
        (4.0+2.0*beta) / 
        (
            ((eps**2) / 2.0) -
            ((eps**3) / 3.0)
        )
        * math.log(n_points)
    )
    
    return max(int( math.ceil(min_dim) + 0.5), 1)

class Method(enum.Enum):
    """Possible projection methods for Achlioptas approach"""
    SPARSE=gen_random_sparse_proj
    DENSE=gen_random_dense_proj

def reduce_dimension_all(
        point_matrix, 
        target_dim, 
        seed=None, 
        method=Method.SPARSE
    ):

    """
    Projects the points to target_dimension with Achlioptas method.

    This method expects a matrix containing one point per row. It draws a 
    random projection matrix, scales it with 1/sqrt(target_dim) and applies the
    resulting random projection. The projection skewes pairwise distances
    by at most a factor of (1+eps) if the target dimension is in the order of 
    log n. More precisely, the target dimension needs to be at least
    (4+2beta) / (eps^2/2 - eps^3/3) ln n in order to successfuly project n
    points with a probability of 1-n^{-beta}.

    The method raises a ValueError if the target dimension is greater than the
    dimension of the input points.

    Parameters
    ----------
    point_matrix : 2d numpy.array
        input points, one point per row
    
    target_dim : int
        all points will be projected down to this dimension

    seed : int
        seed for the random generator, or none to use a random seed

    method : Method
        selects the desired projection method (SPARSE or DENSE)

    Returns
    -------
    2d numpy.array
        projected points with target dimension, one point per row
    """
    if target_dim > point_matrix.shape[1]:
        raise ValueError("Target dimension is greater than input dimension")

    proj_matrix = draw_projection_matrix(input_dim, target_dim, seed, method)
    return np.matmul(point_matrix, proj_matrix)

def as_points(infile, separator=" "):
    """Generates a point as numpy.array for each line of a  file-like infile"""
    for line in infile:
        yield np.fromstring(line, sep=separator)

def reduce_dim(
        points, 
        input_dim, 
        target_dim, 
        seed=None, 
        method=Method.SPARSE
    ):
    """
    Reads points and projects them to target_dimension with Achlioptas method.

    This is a generator method that reads points one by one. It draws a 
    random projection matrix, scales it with 1/sqrt(target_dim) and applies the
    resulting random projection to any point. Yields one projected point for
    each input point.
    
    The projection skewes pairwise distances by at most a factor of (1+eps) if 
    the target dimension is in the order of log n. More precisely, the target 
    dimension needs to be at least (4+2beta) / (eps^2/2 - eps^3/3) ln n in order
     to successfuly project n points with a probability of 1-n^{-beta}.

    The method raises a ValueError if the target dimension is greater than the
    dimension of the input points. The method assumes that all points are 
    of the same input dimension. 

    Parameters
    ----------
    points : iterable
        an iterable of the points to be projected

    input_dim: int
        dimension of the input points    

    target_dim : int
        all points will be projected down to this dimension

    seed : int
        seed for the random generator, or none to use a random seed

    method : Method
        selects the desired projection method (SPARSE or DENSE)
    """
    proj_matrix = draw_projection_matrix(input_dim, target_dim, seed, method)

    for p in points:
        yield project_point(proj_matrix, p)

def draw_projection_matrix(
        input_dim, 
        target_dim, 
        seed=None, 
        method=Method.SPARSE
    ):
    """Draws a random projection matrix for Achlioptas method.
    
       The matrix is chosen to project points of the input dimension to 
       the chosen target dimension. If the dense method is chosen, each entry
       of the matrix is 1/sqrt(target_dim) with probability 1/2 or 
       -1/sqrt(target_dim) otherwise. If the sparse method is chosen (default),
       each entry is 1/sqrt(target_dim) with probality 1/6, or 
       -1/sqrt(target_dim) with proability 1/6, or 0 with the remaining 
       probability of 2/3.

       The method assumes that target_dim <= input_dim and otherwise raises a 
       ValueError.

       Parameters
       ----------
       input_dim : int 
            Dimension of the input points, i.e., number of rows of the 
            random matrix.

       target_dim : int
            The matrix will map to a space of this dimension, i.e. number of 
            columns of the random matrix.

        seed : int or None (default: None)
            Seed for the numpy default random generator

        method : Method
            Which of Achlioptas proposed methods to use.

        Returns
        -------
        2d numpy.array
            Random projection matrix with input_dim rows and target_dim columns 
    """

    if target_dim > input_dim:
        raise ValueError("Target dimension is greater than input dimension")

    rng = np.random.default_rng(seed)
    scale_factor = (1.0/math.sqrt(target_dim))
    return scale_factor*method(input_dim, target_dim, rng) 

def project_point(matrix, point):
    """Projects a single point using a given projection matrix"""

    return np.matmul(point, matrix)

def add_projections_args(arg_parser):
    """Adds all arguments to the parser needed to control the projection"""

    arg_parser.add_argument(
        "--seed", "-s",
        type=int,
        help="Seed for the random number generator"
    )
    arg_parser.add_argument(
        "--dense-projection",
        help="uses the dense projection method",
        action='store_const',
        const=Method.DENSE,
        default=Method.SPARSE
    )

    # parser for fixed-dim subcommand:
    sub_parsers = arg_parser.add_subparsers(
        help="Handling of target dimension",
        required=True,
        dest="dimension_mode"
    )
    fixed_dim_parser = sub_parsers.add_parser("fixed-dim", 
        help="project to fixed dimension"
    )
    fixed_dim_parser.add_argument(
         "target_dim", 
         type=int, 
         help="Points will be projected down to have this dimension."
    )
    
    # parser for auto-dim subcommand:
    auto_dim_parser = sub_parsers.add_parser("auto-dim",
        help="""
             Compute suitable target dimension such that projecting n 
             points skews pairwise distances by at most (1+eps) with probability
             n^{-beta}.
             """
    )
    auto_dim_parser.add_argument(
        "--eps",
        type=float,
        help="pairwise distances will be skewed by at most 1+eps w.h.p.",
        default=1.0
    )
    auto_dim_parser.add_argument(
        "--beta",
        type=float,
        help="desired success probability of at least 1-n^{-beta}",
        default=0.5
    )


def add_output_args(arg_parser):
    """Adds all arguments to the parser needed to control the output"""

    arg_parser.add_argument(
            "--precision", 
            type=int, 
            help="precision of numbers in the output", 
            default=3
    )
    arg_parser.add_argument(
            "--delimiter", 
            help="string used as delimiter for the coordinates", 
            default=" "
    )

def _build_arg_parser():
    """Builds the parser for the command line arguments"""
    
    arg_parser = argparse.ArgumentParser(description=
        """
        Reads points from the standard input and projects them down to a given
        target dimension. The projection matrix is filled with -1 and 1
        uniformly at random. Prints the projected points into the standard 
        output.
        """
    )
    add_projections_args(arg_parser)
    add_output_args(arg_parser)
    
    return arg_parser

if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    format_str = "{{:.{:}f}}".format(args.precision)

    try:
        # read number of points and input dimension from first line
        n_points, input_dim = map(int, sys.stdin.readline().split())

        target_dim = 0
        if args.dimension_mode == "fixed-dim":
            target_dim = args.target_dim
        else:
            target_dim = fit_target_dim(n_points, args.eps, args.beta)

        print(n_points, target_dim)

        for proj_point in reduce_dim(
            as_points(sys.stdin), 
            input_dim,
            target_dim,
            args.seed,
            args.dense_projection  
        ):
            print(args.delimiter.join( 
                [format_str.format(x) for x in proj_point] 
            ))

    except ValueError as e:
        print("Error:", e)
