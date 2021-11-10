from sklearn import cluster
import argparse
import numpy as np



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
        normal_kmeans = cluster.KMeans(init='k-means++',
                                n_clusters=args.n_centers,
                                n_init=1,
                                algorithm='elkan',
                                random_state=None)
        normal_kmeans.fit(X)


        print("Inertia of kMeans++: {}".format(normal_kmeans.inertia_))
    except ValueError as e:
        print("Error:", e)
