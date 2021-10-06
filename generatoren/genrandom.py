#!/usr/bin/env python3

# author: Daniel Schmidt (daniel.schmidt@uni-bonn.de), University of Bonn
# date  : july 2021
# copyright: tba

"""Generates random points with a given dimension and sparsity"""

import numpy as np
import argparse

def _build_arg_parser():
    """Builds the parser for the command line arguments"""
    
    arg_parser = argparse.ArgumentParser(description=
        """
        Compares the pairwise distances in the input distances with the 
        pairwise distances in a projected instances. Assumes the i-th point
        in the projected instance is the projection of the i-th point of the
        input instance.
        """
    )
    arg_parser.add_argument(
        "n_points", 
        help="number of generated points",
        type=int
    )
    arg_parser.add_argument(
        "dimension", 
        help="dimension of the generated points",
        type=int
    )
    arg_parser.add_argument(
        "--sparsity",
        help="percentage of non-zero coordinates in generated points",
        default=0.5
    )
    arg_parser.add_argument(
        "--seed",
        help="seed for the random number generator",
    )
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


    return arg_parser


def random_points_sparse(n_points, dim, p=0.5, seed=None):
    """
    Return n_points many sparse random points in dimension dim. 

    Each coordinate is non-zero with probability p.
    """
    
    rng = np.random.default_rng(seed)
    random_points = np.empty((n_points, dim))
    for _ in range(n_points):
        random_point = np.empty(dim)
        for x in range(dim):
            if rng.random() > p:
                random_point[x] = 0.0
            else:
                random_point[x] = rng.random()
        
        yield random_point


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    format_str = "{{:.{:}f}}".format(args.precision)

    print(args.n_points, args.dimension)

    for point in random_points_sparse(args.n_points, 
                                      args.dimension, 
                                      args.sparsity
    ):
        print(args.delimiter.join( 
            [format_str.format(x) for x in point] 
        ))