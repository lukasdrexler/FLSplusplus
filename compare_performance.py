import numpy as np
import argparse
from sklearn import cluster
import time


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

    arg_parser.add_argument(
        "-r", "--random_state",
        type=int,
        help="Random state",
        default=None
    )

    arg_parser.add_argument(
        "-rep", "--repeats",
        type=int,
        help="Number of repeats for alspp",
        default=1
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


def get_random_state():
    return np.random.randint(0, 1000000)


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()

    print("\nDataset: {}, k: {}, depth: {}, norm_it: {}, search_steps: {}".format(args.file.name, args.n_centers, args.depth, args.normal_iterations, args.search_steps))

    try:
        X = np.genfromtxt(args.file)

        if args.random_state is None:
            start_random_state = get_random_state()
        else:
            start_random_state = args.random_state

        als_pp_time = 0
        best_als_pp_inertia = np.infty
        random_state = start_random_state

        for i in range(args.repeats):

            start = time.time()

            als_pp = cluster.KMeans(init='k-means++',
                                    n_clusters=args.n_centers,
                                    n_init=1,
                                    algorithm='als++',
                                    depth=args.depth,
                                    search_steps=args.search_steps,
                                    norm_it=args.normal_iterations,
                                    random_state=random_state,
                                    verbose=args.verbose)
            als_pp.fit_new(X)

            # we define our time limit for the other algorithms and compare their inertia values
            als_pp_time += (time.time() - start)
            if best_als_pp_inertia > als_pp.inertia_:
                if best_als_pp_inertia != np.infty:
                    print("alspp old inertia: {} new inertia: {}".format(best_als_pp_inertia, als_pp.inertia_))
                best_als_pp_inertia = als_pp.inertia_
            random_state = get_random_state()

        print("ALSpp inertia: {}".format(als_pp.inertia_))
        print("ALSpp used time: {}".format(als_pp_time))

        current_time = 0
        best_inertia_normal = np.infty
        repeats = 0
        random_state =start_random_state
        while current_time < als_pp_time:
            start = time.time()
            normal_kmeans = cluster.KMeans(init='k-means++',
                                           n_clusters=args.n_centers,
                                           n_init=1,
                                           algorithm='elkan',
                                           random_state=random_state)
            normal_kmeans.fit(X)
            normal_kmeans_time = time.time() - start
            current_time += normal_kmeans_time
            if best_inertia_normal > normal_kmeans.inertia_:
                if best_inertia_normal != np.infty:
                    print("kmeans old inertia: {} new inertia: {}".format(best_inertia_normal, normal_kmeans.inertia_))
                best_inertia_normal = normal_kmeans.inertia_
            random_state = get_random_state()
            repeats += 1

        print("best normal kmeans inertia in same time: {}".format(best_inertia_normal))
        print("number of repeats: {}".format(repeats))

        # if args.quiet:
        #     print(als_pp.inertia_)
        # elif args.verbose:
        #     centers = als_pp.cluster_centers_
        #     print("Calculated centers:")
        #     for i in range(len(centers)):
        #         print(centers[i])
        #     print("Inertia ALSPP: {}".format(als_pp.inertia_))
        # else:
        #     print("Inertia of ALSPP = {}".format(als_pp.inertia_))

    except Exception as e:
        print("Inertia of ALSPP = -1")
        f = open("errors.txt", "a")
        f.write("Dataset: {}, k: {}, depth: {}, norm_it: {}, search_steps: {}\n".format(args.file.name, args.n_centers, args.depth, args.normal_iterations, args.search_steps))
        f.write(str(e) + "\n\n")
        f.close()
