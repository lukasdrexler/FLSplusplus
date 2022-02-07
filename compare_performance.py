import numpy as np
import argparse
from sklearn import cluster
import time
# from resource import getrusage as resource_usage, RUSAGE_SELF
# from resource import *

import matplotlib.pyplot as plt


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
    return np.random.randint(1, 100000000)


def get_random_states(size=10):
    return np.random.randint(1, 100000000, size=size)


def alspp_compare(verbose=False):
    if args.random_state is None:
        start_random_state = get_random_state()
    else:
        start_random_state = args.random_state
    als_pp_time = 0
    best_inertia_alspp = np.infty
    random_state = start_random_state

    random_states = np.append(random_state, get_random_states(size=100000))
    final_repeats = {"kmeans": 0,
                     "alspp": args.repeats,
                     "lspp": 0}
    history = {"kmeans": {},
               "alspp": {},
               "lspp": {}}

    algorithms = ["alspp", "lspp", "kmeans"]
    for algorithm in algorithms:
        history[algorithm]["times"] = []
        history[algorithm]["cum_times"] = []
        history[algorithm]["best_inertia"] = []

    for run in range(args.repeats):
        current_state = random_states[run]

        start = time.time()

        als_pp = cluster.KMeans(init='k-means++',
                                n_clusters=args.n_centers,
                                n_init=1,
                                algorithm='als++',
                                depth=args.depth,
                                search_steps=args.search_steps,
                                norm_it=args.normal_iterations,
                                random_state=current_state,
                                verbose=False)
        als_pp.fit_new(X)

        # we define our time limit for the other algorithms and compare their inertia values
        current_time = time.time() - start
        history["alspp"]["times"].append(current_time)
        als_pp_time += current_time
        history["alspp"]["cum_times"].append(als_pp_time)
        if best_inertia_alspp > als_pp.inertia_:
            if best_inertia_alspp != np.infty:
                if verbose:
                    print("alspp old inertia: {} new inertia: {}".format(best_inertia_alspp, als_pp.inertia_))
            best_inertia_alspp = als_pp.inertia_
        history["alspp"]["best_inertia"].append(best_inertia_alspp)

    if verbose:
        print("ALSpp inertia: {}".format(als_pp.inertia_))
        print("ALSpp used time: {}".format(als_pp_time))
    current_time = 0
    best_inertia_normal = np.infty
    repeats = 0

    while current_time < als_pp_time:
        current_state = random_states[repeats]
        start = time.time()
        normal_kmeans = cluster.KMeans(init='k-means++',
                                       n_clusters=args.n_centers,
                                       n_init=1,
                                       algorithm='elkan',
                                       random_state=current_state)
        normal_kmeans.fit(X)
        normal_kmeans_time = time.time() - start
        history["kmeans"]["times"].append(normal_kmeans_time)
        current_time += normal_kmeans_time
        history["kmeans"]["cum_times"].append(current_time)
        if best_inertia_normal > normal_kmeans.inertia_:
            if best_inertia_normal != np.infty:
                if verbose:
                    print("kmeans old inertia: {} new inertia: {}".format(best_inertia_normal, normal_kmeans.inertia_))
            best_inertia_normal = normal_kmeans.inertia_
        history["kmeans"]["best_inertia"].append(best_inertia_normal)
        repeats += 1
    if verbose:
        print("best normal kmeans inertia in same time: {}".format(best_inertia_normal))
        print("number of repeats: {}".format(repeats))

    final_repeats["kmeans"] = repeats

    # comparison to lspp

    current_time = 0
    best_inertia_lspp = np.infty
    repeats = 0

    while current_time < als_pp_time:
        current_state = random_states[repeats]
        start = time.time()
        lspp = cluster.KMeans(init='k-means++',
                              n_clusters=args.n_centers,
                              n_init=1,
                              algorithm='lspp',
                              random_state=current_state,
                              z=25)
        lspp.fit_new(X)
        lspp_time = time.time() - start
        history["lspp"]["times"].append(lspp_time)
        current_time += lspp_time
        history["lspp"]["cum_times"].append(current_time)
        if best_inertia_lspp > lspp.inertia_:
            if best_inertia_lspp != np.infty:
                if verbose:
                    print("lspp old inertia: {} new inertia: {}".format(best_inertia_lspp, lspp.inertia_))
            best_inertia_lspp = lspp.inertia_
        history["lspp"]["best_inertia"].append(best_inertia_lspp)
        repeats += 1

    if verbose:
        print("best lspp inertia in same time: {}".format(best_inertia_lspp))
        print("number of repeats: {}".format(repeats))

    final_repeats["lspp"] = repeats

    wins = {"kmeans": 0,
            "alspp": 0,
            "lspp": 0}

    # If some inertia value is identical there is no clear winner and we skip the results
    if best_inertia_normal == best_inertia_alspp or best_inertia_normal == best_inertia_lspp or best_inertia_alspp == best_inertia_lspp:
        return wins

    inertias = [best_inertia_alspp, best_inertia_lspp, best_inertia_normal]
    inertias_winner = (2 * np.ones(3) - np.argsort(inertias)).astype(int)

    wins["alspp"] = inertias_winner[0]
    wins["lspp"] = inertias_winner[1]
    wins["kmeans"] = inertias_winner[2]

    return wins, final_repeats, history

    # if best_als_pp_inertia < best_inertia_normal:
    #     return 1, 0
    # else:
    #     return 0, 1


def plot_results(h, steps=100):
    my_trials = len(h)
    algorithms = ["alspp", "lspp", "kmeans"]

    # we take the largest time as a limit and then cut the plot in x steps
    time_max = 0
    for i in range(my_trials):
        his = h[i]
        for alg in algorithms:
            current_cum_time = his[alg]["cum_times"][-1]
            if current_cum_time > time_max:
                time_max = current_cum_time

    avg_results = {}

    for alg in algorithms:
        avg_results[alg] = {"x": [],
                            "y": []}

    # collect all times where the inertia values did change
    times = []

    for i in range(my_trials):
        for alg in algorithms:
            current_best = np.infty

            his = h[i][alg]
            for j in range(len(his["best_inertia"])):
                if his["best_inertia"][j] < current_best:
                    times.append(his["cum_times"][j])
                    current_best = his["best_inertia"][j]

    # for current_time in np.arange(0, time_max + time_max/steps, time_max/steps):
    times = np.sort(times)
    for current_time in times:
        for alg in algorithms:
            cum_best = 0
            found_values = 0
            for i in range(my_trials):
                # find the best inertia value for this run, which did take at most current_time time
                his = h[i]
                best_value = np.infty
                for j in range(len(his[alg]["cum_times"])):
                    cum_time = his[alg]["cum_times"][j]
                    if cum_time <= current_time and best_value > his[alg]["best_inertia"][j]:
                        best_value = his[alg]["best_inertia"][j]
                    elif cum_time > current_time:
                        break
                if best_value != np.infty:
                    # we add for the current run the best found inertia value and average in the end over the found values
                    cum_best += best_value
                    found_values += 1

            if found_values != 0:
                avg_best = cum_best / found_values
                avg_results[alg]["x"].append(current_time)
                avg_results[alg]["y"].append(avg_best)

    # we plot our results
    for alg in algorithms:
        plt.plot(avg_results[alg]["x"], avg_results[alg]["y"], label=alg)

    plt.legend(loc="upper right")
    plt.ylabel("avg min inertia")
    plt.xlabel("time constraint")

    plt.show()


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()

    print("\nDataset: {}, k: {}, depth: {}, norm_it: {}, search_steps: {}".format(args.file.name, args.n_centers,
                                                                                  args.depth, args.normal_iterations,
                                                                                  args.search_steps))

    trials = 10

    # try:
    X = np.genfromtxt(args.file)

    alspp_win_sum = k_means_win_sum = lspp_win_sum = 0

    histories = []

    for iteration in range(trials):
        print("###### Run {} ######".format(iteration))
        # alspp_win, k_means_win = alspp_compare(verbose=True)
        wins, final_repeats, history = alspp_compare(verbose=True)
        if wins["alspp"] == 2:
            alspp_win_sum += 1
        elif wins["lspp"] == 2:
            lspp_win_sum += 1
        elif wins["kmeans"] == 2:
            k_means_win_sum += 1

        histories.append(history)

    print("#Wins ALSPP: {} , #Wins LSPP: {} , #Wins kMeans: {}".format(alspp_win_sum, lspp_win_sum, k_means_win_sum))

    plot_results(histories)

    # except Exception as e:
    #     print("Inertia of ALSPP = -1")
    #     f = open("errors.txt", "a")
    #     f.write("Dataset: {}, k: {}, depth: {}, norm_it: {}, search_steps: {}\n".format(args.file.name, args.n_centers,
    #                                                                                     args.depth,
    #                                                                                     args.normal_iterations,
    #                                                                                     args.search_steps))
    #     f.write(str(e) + "\n\n")
    #     f.close()
