#!/usr/bin/env python3

# author: Daniel Schmidt (daniel.schmidt@uni-bonn.de), University of Bonn
# date  : july 2021
# copyright: tba

"""
A generator for a simplex-based class of worst-case instances for kmeans++.

Arthur and Vassilvitskii [1] prove that the seeding of the kmeans++ algorithm
is at best 2*ln(k) competitive in expectation by describing a worst-case 
example for the algorithm. In his Bachelor's thesis [2], Jan Stallmann constructs
two families of instances from Arthur's and Vassilvitskii's proof. This 
generator builds a point set according to the second, simplex based 
construction and prints it to the standard output.

The construction takes as parameters a number of points n, a number of centers k
and two scaling constants D and d. We assume that n is a multiple of k and 
generate k clusters with n/k points each. In each cluster, the points are 
arranged as the vertices of a (n/k)-simplex (here, for simplicity, a l-simplex  
is the convex hull of the l unit vectors in R^l [*]). We position the clusters 
such that their centroids form the vertices of an appropriately scaled 
k-simplex. More precisely, the outer simplex used to determine the position of 
the clusters is scaled such that all its edges have length D; the simplices that
place the points of the clusters are scaled such that all edges have 
length d [**]. The inner simplices are placed in orthogonal dimensions such that
any two points in the same cluster have distance d while any two points from
different clusters have distance D.

[1] D. Arthur and S. Vassilvitskii. 
    K-means++: the advantages of careful seeding.
    Proceedings of the 18th annual ACM-SIAM symposium on Discrete algorithms,
    January 2007, pp. 1027–1035
    https://dl.acm.org/doi/10.5555/1283383.1283494

[2] Jan Stallmann.
    Benchmarkinstanzen für das k-means Problem.
    Bachelor's Thesis, University of Bonn, 2014.
    Advisor: Melanie Schmidt. In german.

[*] It is more common to define the l-simplex as the convex hull of the l+1 
    unit vectors in R^{l+1}. This is because the convex hull of l unit vectors
    is a l-1 dimensional object. The difference comes from the fact that the
    l-simplex does not contain the origin and is affinely embedded into R^{l}.
    We could also define the l-simplex in R^{l-1}, but the vertices are more
    difficult to describe then.

[**] Contrary to intuition, an l-simplex has edge length sqrt(2) even though
     it is built from unit vectors.
"""
import numpy as np
import itertools
import argparse
import math


def generate_simplex(n_vertices, scaling=1.0):
    """
    Generates a list of the vertices of a regular simplex.

    A standard simplex with l vertices is the convex hull of the l unit vectors 
    in R^l. The output of this method is a list of the vertices of a standard 
    simplex with n_vertices many vertices scaled by the scaling factor.

    Parameters
    ----------
    n_vertices: int 
        number of vertices of the simplex.

    scaling: float
        the simplex is scaled by this factor; i.e., each edge in the simplex has
        length sqrt(2)*scaling.
    
    Returns
    -------
    np.array
        a matrix of the simplex vertices
    """
    return scaling * np.eye(n_vertices)


def generate_simplex_instance(
        outside_distance,
        inside_distance,
        n_clusters,
        n_points):
    """
    Generates a simplex based worst-case example for kmeans++.

    This is a generator method that yields n_points many points, each having 
    (n_clusters+n_points) dimensions. The points are grouped in n_clusters 
    clusters with (n_points/n_clusters) each. The method assumes that n_points 
    is a multiple of n_clusters and raises a ValueError if this is not the case. 

    In order to build cluster i, the method will use the vertex set of an inner
    simplex with (n_points/n_clusters) many vertices and arrange it around
    a larger, outer simplex with n_clusters many vertices. The inner simplex 
    is a standard simplex with an edge length given by inside_distance, the outer 
    simplex is a standard simplex with an edge length given by 
    outside_distance. 

    Hence, each generated point describes one vertex of some of the inner 
    simplices. 

    Consider the j-th vertex v_ij of the inner simplex that makes up the i-th
    cluster. We use the first n_clusters coordinates of v_ij to place the 
    point in the i-th cluster; consequently, v_ij has its first n_clusters 
    many coordinates set to zero except for coordinate i where it has an
    appropriately scaled non-zero value. 
    The subsequent n_points many coordinates of v_ij are grouped into 
    groups of cluster_size = (n_points/n_clusters). For v_ij, all groups but
    group i are set to zero; group i contains the j-th vertex of a simplex 
    with cluster_size vertices (appropriately scaled and translated such that 
    the centroid of the simplex lies in the i-th vertex of the outer 
    simplex).

    Parameters 
    ----------
    outside_distance: float
        Edge length of the outer simplex: The clusters are placed such that 
        any points from different clusters have this distance
    
    inside_distance: float
        Edge length of the inner simplex: The points inside each cluster are 
        placed such that this is their pairwise distance.
    
    n_clusters: int
        Number of clusters in the output instance. This is the number of 
        vertices of the outer simplex.
    
    n_points: int
        Total number of points in the instance. The outer simplex has 
        (n_points / n_clusters) vertices.
    """
    if n_points % n_clusters != 0:
        raise ValueError(
            "n_points={} must be a multiple of n_clusters={}"
                .format(n_points, n_clusters)
        )

    n_points_per_cluster = n_points // n_clusters
    # scale inner simplex such that all points in the 
    # simplex have distance inside_distance
    inner_scale = inside_distance / math.sqrt(2)

    # scale outer simplex such that all points in
    # different cluster have pairwise distance
    # outside_distance
    outer_scale = math.sqrt(
        0.5 * (
                outside_distance * outside_distance
                - (n_points - n_clusters) / float(n_points)
                * inside_distance * inside_distance
        )
    )

    # centroid of the inner simplex
    inner_centroid = np.full(
        n_points_per_cluster,
        (1.0 / n_points_per_cluster) * inner_scale
    )

    # generate vertices of inner simplex, centered around origin
    inner_simplex = generate_simplex(
        n_points_per_cluster,
        inner_scale
    ) - inner_centroid

    # generate vertices of outer simplex
    outer_simplex = generate_simplex(n_clusters, outer_scale)

    # now, generate the actual instance
    for v_idx, outer_vertex in enumerate(outer_simplex):
        for inner_vertex in inner_simplex:
            # each inner simplex occupies n_points_per_cluster many
            # dimensions. Dimensions occupied by other inner 
            # simpleces need to be filled with zeroes

            # number of inner simplices that were generated 
            # before this one is v_idx
            leading_dims = n_points_per_cluster * v_idx

            # there are n_clusters many inner simplices; hence, 
            # the number of simplices that will be generated 
            # after this one is (n_clusters - (v_idx + 1)).
            trailing_dims = n_points_per_cluster * (n_clusters - v_idx - 1)

            # pad with zeroes accordingly
            padded_vertex = np.pad(
                inner_vertex,  # array to pad
                (leading_dims, trailing_dims),  # pad (before, after)
                mode='constant',  # pad with constant
                constant_values=0  # value for padding
            )

            # need to prepend the coordinates of the vertex of the
            # outer simplex
            yield np.concatenate((outer_vertex, padded_vertex))


def _build_arg_parser():
    """Builds the parser for the command line arguments"""

    arg_parser = argparse.ArgumentParser(description=
                                         """
        Prints a simplex based worst-case example for kmeans++. The example 
        consists of an outer simplex that contains an inner simplex in each of 
        its vertices. The edge length of the inner and outer simplex may be 
        controlled, as well as the simplices' respective dimension. The example 
        is printed to the standard output. 
        """
                                         )
    arg_parser.add_argument(
        "n_clusters",
        type=int,
        help="number k of clusters in the instance"
    )
    arg_parser.add_argument(
        "n_points",
        type=int,
        help="number n of points in the instance"
    )
    arg_parser.add_argument(
        "outer_dist",
        type=float,
        help="length D of the edges of the outer simplex"
    )
    arg_parser.add_argument(
        "inner_dist",
        type=float,
        help="length d of the edges of the inner simplex"
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


if __name__ == "__main__":
    """Prints a simplex based worst-case example for kmeans++ to stdout"""

    args = _build_arg_parser().parse_args()

    format_str = "{{:.{:}f}}".format(args.precision)

    try:
        # print number of points and dimension of output points
        print(args.n_points, args.n_clusters + args.n_points)

        for p in generate_simplex_instance(
                args.outer_dist,
                args.inner_dist,
                args.n_clusters,
                args.n_points
        ):
            print(args.delimiter.join([format_str.format(x) for x in p]))

    except ValueError as e:
        print("Error:", e)
