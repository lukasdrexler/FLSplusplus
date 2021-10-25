#!/usr/bin/env python3

# author: Daniel Schmidt (daniel.schmidt@uni-bonn.de), University of Bonn
# date  : july 2021
# copyright: tba

"""A testing script for the dimensionality reduction via random projection"""

import sys
import argparse
import itertools
import math
import numpy as np

import randomproj


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
    randomproj.add_projections_args(arg_parser)
    randomproj.add_output_args(arg_parser)

    return arg_parser


def read_instance(infilename):
    """Reads points from a file"""

    with open(infilename) as infile:
        infile.readline()  # ignore first line
        return np.fromfile(infile)


def sed(p, q):
    """Squared euclidean distance of p and q"""

    return np.linalg.norm(p - q)


def build_error_histogram(true_points, proj_points, precision=0.01):
    """
    For each pair of points, compares true distance to projected distance.

    This method iterates over all pairs of points. For each pair p and q, it 
    computes the true distance ||p-q|| of p and q in the original space; and the 
    distance ||pi(p) - pi(q)|| in the projected space. It then counts how many
    points have a distance error ||pi(p) - pi(q)|| / ||p-q|| of x and builds 
    a histogram of the frequencies. The histogram rounds down to the given
    precision.
    """

    histogram = dict()

    for (true_p, true_q), (proj_p, proj_q) in zip(
            itertools.combinations(true_points, 2),
            itertools.combinations(proj_points, 2)
    ):

        true_dist = sed(true_p, true_q)
        proj_dist = sed(proj_p, proj_q)

        if true_dist == 0:
            histogram[math.inf] += 1
        else:
            dist_error = proj_dist / true_dist

            bucket = precision * math.floor(dist_error / precision)
            if not bucket in histogram:
                histogram[bucket] = 0

            histogram[bucket] += 1

    return histogram


if __name__ == "__main__":
    try:
        args = _build_arg_parser().parse_args()
        n_points, dim = map(int, sys.stdin.readline().split())

        target_dim = 0
        if args.dimension_mode == "fixed-dim":
            target_dim = args.target_dim
        else:
            target_dim = randomproj.fit_target_dim(
                n_points,
                args.eps,
                args.beta
            )

        print("Projecting to dimension", target_dim)

        # duplicate input iterator so that one of them may be projected
        points, points_dup = itertools.tee(randomproj.as_points(sys.stdin))
        proj_points = randomproj.reduce_dim(points_dup,
                                            dim,
                                            target_dim,
                                            args.seed,
                                            args.dense_projection
                                            )

        histogram = build_error_histogram(points, proj_points, precision=0.01)

        print(
            """
        The following histogram shows how many of the {} pairwise distances
        had an error (projected_distance / true_distance) of [x,x+0.01) percent 
        after the projection.

        The points were projected down to {} dimensions.

        n_points: {} | dim: {} | eps: {} | beta: {}

        """.format(
                int(0.5 * n_points * (n_points - 1)),
                target_dim,
                n_points,
                dim,
                args.eps if args.dimension_mode == "auto-dim" else "---",
                args.beta if args.dimension_mode == "auto-dim" else "---",
            )
        )

        print("{:>7}  {:>6}".format("error", "#points"))
        for key in sorted(histogram.keys()):
            print("{:>6.2f}%: {:>6}".format(key, histogram[key]))

        print()

    except ValueError as e:
        print("Error:", e)
