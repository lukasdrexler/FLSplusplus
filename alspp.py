import numpy as np
import argparse
from sklearn import cluster


def _build_arg_parser():
    """Builds the parser for the command line arguments"""

    arg_parser = argparse.ArgumentParser(description=
                                         """
        Run the alspp algorithm using some predefined parameters for depth, normal_iterations 
        and search_iterations. Output the centers and inertia.
        Some possible heuristic parameters can also be applied for decreased running time.
        """
                                         )
    arg_parser.add_argument(
        "-f", "--file",
        type=argparse.FileType('r'),
        required=True,
        help="file which contains all points separated by rows"
    )
    arg_parser.add_argument(
        "-k", "--n_centers",
        type=int,
        required=True,
        help="number of centers"
    )
    arg_parser.add_argument(
        "-d", "--depth",
        type=int,
        default=3,
        help="depth used in alspp for looking into further iterations"
    )
    arg_parser.add_argument(
        "-n", "--normal_iterations",
        type=int,
        default=3,
        help="number of steps where we run loyds algrorithm and then start to sample one or more new candidates"
    )
    arg_parser.add_argument(
        "-s", "--search_steps",
        type=int,
        default=1,
        help="number of iterations we try to exchange some center by sampling a new point as candidate"
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

    # parameter which specifies how much information is given
    group = arg_parser.add_mutually_exclusive_group()
    group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='print quiet'
    )
    group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='print verbose'
    )
    return arg_parser


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()

    try:
        X = np.genfromtxt(args.file)
        als_pp = cluster.KMeans(init='k-means++',
                                n_clusters=args.n_centers,
                                n_init=1,
                                algorithm='als++',
                                depth=args.depth,
                                search_steps=args.search_steps,
                                norm_it=args.normal_iterations,
                                random_state=None)
        als_pp.fit(X)

        print("Dataset: {}, k: {}, depth: {}, norm_it: {}, search_steps: {}".format(args.file.name, args.n_centers, args.depth, args.normal_iterations, args.search_steps))

        if args.quiet:
            print(als_pp.inertia_)
        elif args.verbose:
            centers = als_pp.cluster_centers_
            print("Calculated centers:")
            for i in range(len(centers)):
                print(centers[i])
            print("Inertia: {}".format(als_pp.inertia_))
        else:
            print("Inertia of calculated clustering = {}".format(als_pp.inertia_))
    except ValueError as e:
        print("Error:", e)
