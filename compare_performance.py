import os
from os import path, listdir
from os.path import isfile, join
import sys

import numpy as np
import argparse
from sklearn import cluster

if "linux" in sys.platform:
    from resource import getrusage as resource_usage, RUSAGE_SELF
    from time import time as timestamp
elif "win" in sys.platform:
    import time

import matplotlib.pyplot as plt

import copy


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

    arg_parser.add_argument(
        "-p", "--plot",
        action='store_true',
        help="Only plot current file"
    )

    arg_parser.add_argument(
        "-t", "--trials",
        type=int,
        help="Number of iterations we average over",
        default=5
    )

    arg_parser.add_argument(
        "-ng", "--nogreedy",
        action='store_true',
        help="No Greedy D2 Sampling"
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


class Timestamp:
    def __init__(self):
        self.user = None
        self.sys = None
        self.real = None
        self.own_time = 0
        if "linux" in sys.platform:
            self.os = "linux"
        elif "win" in sys.platform:
            self.os = "win"

    def timestamp(self):
        if self.os == "linux":
            start_time, start_resources = timestamp(), resource_usage(RUSAGE_SELF)
            return {'real': start_time,
                    'sys': start_resources.ru_stime,
                    'user': start_resources.ru_utime}
        elif self.os == "win":
            return {'real': time.time()}

    def make_timestamp(self):
        times = self.timestamp()
        if self.os == "linux":
            self.real = times['real']
            self.sys = times['sys']
            self.user = times['user']
        elif self.os == "win":
            self.real = times['real']

    def get_elapsed_time(self):
        times = self.timestamp()
        if self.os == "linux":
            return (times['sys'] + times['user']) - (self.sys + self.user)
        elif self.os == "win":
            return times['real'] - self.real

    def add_time(self, elapsed_time):
        self.own_time += elapsed_time

    def get_time(self):
        return self.own_time

    def set_time(self, mytime):
        self.own_time = mytime


class Experiment:
    def __init__(self, dataset, trials, n_centers, depth, norm_it, best_of, greedy):
        self.dataset = dataset
        self.trials = trials
        self.n_centers = n_centers
        self.depth = depth
        self.norm_it = norm_it
        self.best_of = best_of
        self.greedy = greedy

        # greedy_value represents the actual value we need to use to get the desired result
        self.greedy_value = None
        if not greedy:
            self.greedy_value = 1

        self.histories = {}
        for i in range(self.best_of):
            self.histories[i] = []

        # we create our dictionary hierarchy if not existing
        # directory of main in pycharm-environment
        compare_dir = os.path.dirname(__file__)
        # subfolder in which the plots are saved in
        compare_dir = path.join(compare_dir, 'Compare Performance')
        if not path.exists(compare_dir):
            os.makedirs(compare_dir)
        compare_dir = path.join(compare_dir, dataset)
        if not path.exists(compare_dir):
            os.makedirs(compare_dir)
        compare_dir = path.join(compare_dir, "trials_{}_bo_{}".format(self.trials, self.best_of))
        if not path.exists(compare_dir):
            os.makedirs(compare_dir)
        compare_dir = path.join(compare_dir, "k_{}_d_{}_ni_{}".format(self.n_centers, self.depth, self.norm_it))
        if not path.exists(compare_dir):
            os.makedirs(compare_dir)

        # if we run normal D2-Sampling and no greedy D2-Sampling we create a subfolder
        if not greedy:
            compare_dir = path.join(compare_dir, "normal_D2_Sampling")
            if not path.exists(compare_dir):
                os.makedirs(compare_dir)

        self.directory = compare_dir

        # create a Log file
        self.logfile = path.join(self.directory, "log.txt")
        fp = open(self.logfile, "w")
        fp.write("Dataset: {}, k: {}, depth: {}, norm_it: {}, best of {}\n".format(self.dataset, self.n_centers, self.depth, self.norm_it, self.best_of))
        if self.greedy:
            fp.write("Greedy D2-Sampling\n")
        else:
            fp.write("Non-greedy D2-Sampling\n")
        fp.close()

    def write_to_log(self, text):
        fp = open(self.logfile, "a")
        fp.write(text + "\n")
        fp.close()

    def run_experiment(self):
        # collect dataset
        X = np.genfromtxt(self.dataset)

        alspp_win_sum = k_means_win_sum = lspp_win_sum = 0

        histories = []

        for iteration in range(self.trials):
            self.write_to_log("###### Run {} ######".format(iteration + 1))

            wins, final_repeats, history = self.compare_algorithms(X, iteration)
            if wins["alspp"] == 2:
                alspp_win_sum += 1
            elif wins["lspp"] == 2:
                lspp_win_sum += 1
            elif wins["kmeans"] == 2:
                k_means_win_sum += 1

            histories.append(history)

        self.write_to_log("#Wins ALSPP: {} , #Wins LSPP: {} , #Wins kMeans: {}".format(alspp_win_sum, lspp_win_sum, k_means_win_sum))

        all_avg_results = {}

        for i in range(self.best_of):
            avg_results = self.save_plot_results(self.histories[i], i+1)
            all_avg_results[i] = avg_results
            self.plot_results(avg_results)

        np.save(path.join(self.directory, "all_avg_results.npy"), all_avg_results)

    def compare_algorithms(self, X, it):
        # initialize time values of all three algorithms
        als_pp_time = 0
        current_kmeans_time = 0
        current_lspp_time = 0

        # make timestamp object for ALSpp (this does not start the timer)
        Alspp_Time = Timestamp()

        # initialize starting inertias of all three algorithms
        best_inertia_alspp = np.infty
        best_inertia_kmeans = np.infty
        best_inertia_lspp = np.infty

        # save number of repeats of kmeans and lspp
        kmeans_repeats = 0
        lspp_repeats = 0

        # create many fixed random states which each algorithm runs consecutively until time limit is reached
        random_states = get_random_states(size=100000)

        # initialize number of repeats and histories for each algorithm
        final_repeats = {"kmeans": 0,
                         "alspp": self.best_of,
                         "lspp": 0}
        history = {"kmeans": {},
                   "alspp": {},
                   "lspp": {}}

        algorithms = ["alspp", "lspp", "kmeans"]
        for algorithm in algorithms:
            history[algorithm]["times"] = []
            history[algorithm]["cum_times"] = []
            history[algorithm]["best_inertia"] = []

        # after each run of ALSPP we use the elapsed time for normal kmeans and lspp
        for run in range(self.best_of):

            self.write_to_log("Current iteration of ALSPP: {}".format(run))

            current_state = random_states[run]

            # Initializing each algorithm is not counted to the elapsed time
            als_pp = cluster.KMeans(init='k-means++',
                                    n_clusters=self.n_centers,
                                    n_init=1,
                                    algorithm='als++',
                                    depth=self.depth,
                                    search_steps=1,
                                    norm_it=self.norm_it,
                                    random_state=current_state,
                                    verbose=False,
                                    tol=0,
                                    n_local_trials=self.greedy_value)

            # start the clock
            Alspp_Time.make_timestamp()

            # run ALSPP on dataset
            als_pp.fit_new(X)

            # we define our time limit for the other algorithms and compare their inertia values
            current_time = Alspp_Time.get_elapsed_time()
            als_pp_time += current_time

            # save elapsed time and cumulative elapsed time
            history["alspp"]["times"].append(current_time)
            history["alspp"]["cum_times"].append(als_pp_time)

            # compare current best inertia value to new one
            if best_inertia_alspp > als_pp.inertia_:
                if best_inertia_alspp != np.infty:
                    self.write_to_log("alspp old inertia: {} new inertia: {}".format(best_inertia_alspp, als_pp.inertia_))
                else:
                    self.write_to_log("Starting with first inertia solution {}".format(als_pp.inertia_))
                best_inertia_alspp = als_pp.inertia_
            history["alspp"]["best_inertia"].append(best_inertia_alspp)

            ##########################
            # normal Kmeans evaluation
            ##########################

            kmeans_time = Timestamp()


            # As long as we are below the alspp cumulative timelimit repeat kmeans
            while current_kmeans_time < als_pp_time:
                if kmeans_repeats > len(random_states):
                    random_states = np.append(random_states, get_random_states(size=100000))

                current_state = random_states[kmeans_repeats]

                normal_kmeans = cluster.KMeans(init='k-means++',
                                               n_clusters=self.n_centers,
                                               n_init=1,
                                               algorithm='elkan',
                                               random_state=current_state,
                                               tol=0,
                                               n_local_trials=self.greedy_value)

                kmeans_time.make_timestamp()

                normal_kmeans.fit_new(X)

                normal_kmeans_time = kmeans_time.get_elapsed_time()
                history["kmeans"]["times"].append(normal_kmeans_time)
                current_kmeans_time += normal_kmeans_time
                history["kmeans"]["cum_times"].append(current_kmeans_time)

                if best_inertia_kmeans > normal_kmeans.inertia_:
                    if best_inertia_kmeans != np.infty:
                        self.write_to_log("kmeans old inertia: {} new inertia: {}".format(best_inertia_kmeans, normal_kmeans.inertia_))
                    else:
                        self.write_to_log("kmeans starting inertia: {}".format(normal_kmeans.inertia_))
                    best_inertia_kmeans = normal_kmeans.inertia_
                history["kmeans"]["best_inertia"].append(best_inertia_kmeans)
                kmeans_repeats += 1

            self.write_to_log("best normal kmeans inertia in same time: {}".format(best_inertia_kmeans))
            self.write_to_log("number of repeats: {}".format(kmeans_repeats))


            #################
            # lspp evaluation
            #################

            Lspp_Time = Timestamp()

            while current_lspp_time < als_pp_time:
                if lspp_repeats > len(random_states):
                    random_states = np.append(random_states, get_random_states(size=100000))
                current_state = random_states[lspp_repeats]

                lspp = cluster.KMeans(init='k-means++',
                                      n_clusters=self.n_centers,
                                      n_init=1,
                                      algorithm='lspp',
                                      random_state=current_state,
                                      z=25,
                                      tol=0,
                                      n_local_trials=self.greedy_value)

                Lspp_Time.make_timestamp()

                lspp.fit_new(X)

                lspp_time = Lspp_Time.get_elapsed_time()
                history["lspp"]["times"].append(lspp_time)
                current_lspp_time += lspp_time
                history["lspp"]["cum_times"].append(current_lspp_time)
                if best_inertia_lspp > lspp.inertia_:
                    if best_inertia_lspp != np.infty:
                        self.write_to_log("lspp old inertia: {} new inertia: {}".format(best_inertia_lspp, lspp.inertia_))
                    else:
                        self.write_to_log("lspp starting inertia: {}".format(lspp.inertia_))
                    best_inertia_lspp = lspp.inertia_
                history["lspp"]["best_inertia"].append(best_inertia_lspp)
                lspp_repeats += 1


            self.write_to_log("best lspp inertia in same time: {}".format(best_inertia_lspp))
            self.write_to_log("number of repeats: {}".format(lspp_repeats))

            # We save the current dictionaries for later processing/plotting
            self.histories[run].append(copy.deepcopy(history))

        final_repeats["kmeans"] = kmeans_repeats
        final_repeats["lspp"] = lspp_repeats

        self.write_to_log("ALSpp final inertia: {}".format(best_inertia_alspp))
        self.write_to_log("ALSpp final used time: {}".format(als_pp_time))
        self.write_to_log("Normal kmeans final inertia: {}".format(best_inertia_kmeans))
        self.write_to_log("LSPP final inertia: {}".format(best_inertia_lspp))

        wins = {"kmeans": 0,
                "alspp": 0,
                "lspp": 0}

        # If some inertia value is identical there is no clear winner and we skip the results
        if best_inertia_kmeans == best_inertia_alspp or best_inertia_kmeans == best_inertia_lspp or best_inertia_alspp == best_inertia_lspp:
            self.write_to_log("Some algorithms had same result!")
            return wins, final_repeats, history

        inertias = [best_inertia_alspp, best_inertia_lspp, best_inertia_kmeans]
        inertias_winner = (2 * np.ones(3) - np.argsort(inertias)).astype(int)

        wins["alspp"] = inertias_winner[0]
        wins["lspp"] = inertias_winner[1]
        wins["kmeans"] = inertias_winner[2]

        np.save(path.join(self.directory, "raw_dictionary.npy"), history)
        np.save(path.join(self.directory, "random_states.npy"), random_states)

        return wins, final_repeats, history

    def save_plot_results(self, h, trial):
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

        avg_results = {"config": {}}

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

        avg_results["alspp"]["config"] = {"d": self.depth,
                                          "n": self.norm_it,
                                          "ss": 1,
                                          "reps": trial}


        #np.save(path.join(self.directory, "avg_results_{}.npy".format(trial)), avg_results)
        self.avg_results = avg_results

        return avg_results

    def plot_results(self, avg_results):
        algorithms = ["alspp", "lspp", "kmeans"]

        #avg_results = self.avg_results

        fig = plt.figure()

        # we plot our results
        for alg in algorithms:
            plt.plot(avg_results[alg]["x"], avg_results[alg]["y"], label=alg)

        plt.legend(loc="upper right")
        plt.ylabel("avg min inertia [cost]")
        plt.xlabel("time constraint [s]")

        # history["alspp"]["config"] = {"d": args.depth,
        # "n": args.normal_iterations,
        # "ss": args.search_steps,
        # "reps": args.repeats}

        if "config" in avg_results["alspp"]:
            d = avg_results["alspp"]["config"]["d"]
            n = avg_results["alspp"]["config"]["n"]
            ss = avg_results["alspp"]["config"]["ss"]
            reps = avg_results["alspp"]["config"]["reps"]

            plt.title("depth:{} norm_it:{} ss:{} best of {}".format(d, n, ss, reps))

        #plt.show()
        image_name = path.join(self.directory, 'best of {}'.format(reps))
        fig.savefig(image_name)

def plot_results(avg_results, current_directory):
    algorithms = ["alspp", "lspp", "kmeans"]

    #avg_results = self.avg_results

    fig = plt.figure()

    # we plot our results
    for alg in algorithms:
        plt.plot(avg_results[alg]["x"], avg_results[alg]["y"], label=alg)

    plt.legend(loc="upper right")
    plt.ylabel("avg min inertia [cost]")
    plt.xlabel("time constraint [s]")

    # history["alspp"]["config"] = {"d": args.depth,
    # "n": args.normal_iterations,
    # "ss": args.search_steps,
    # "reps": args.repeats}

    if "config" in avg_results["alspp"]:
        d = avg_results["alspp"]["config"]["d"]
        n = avg_results["alspp"]["config"]["n"]
        ss = avg_results["alspp"]["config"]["ss"]
        reps = avg_results["alspp"]["config"]["reps"]

        plt.title("depth:{} norm_it:{} ss:{} best of {}".format(d, n, ss, reps))

    #plt.show()
    image_name = path.join(current_directory, 'best of {}'.format(reps))
    fig.savefig(image_name)

def plot_all_results():
    # we create our dictionary hierarchy if not existing
    # directory of main in pycharm-environment
    compare_dir = os.path.dirname(__file__)
    # subfolder in which the plots are saved in
    compare_dir = path.join(compare_dir, 'Compare Performance')
    if path.exists(compare_dir):
        list_directory = listdir(compare_dir)
        for dir in list_directory:
            current_path = path.join(compare_dir, dir)
            rec_create_plots(current_path)

def rec_create_plots(current_path):
    # check if files exist, otherwise do recursive call
    onlyfiles = [f for f in listdir(current_path) if isfile(join(current_path, f))]
    if len(onlyfiles) == 0:
        list_directory = listdir(current_path)
        for dir in list_directory:
            new_path = path.join(current_path, dir)
            rec_create_plots(new_path)
    else:
        # we are in some folder with files => we search for file all_avg_results and generate all plots
        if "all_avg_results.npy" in onlyfiles:
            file_path = path.join(current_path, "all_avg_results.npy")
            all_avg_results = np.load(file_path, allow_pickle=True).item()
            for i in range(len(all_avg_results)):
                avg_results = all_avg_results[i]
                plot_results(avg_results, current_path)

def get_time_stamp():
    if "linux" in sys.platform:
        start_time, start_resources = timestamp(), resource_usage(RUSAGE_SELF)

        return {'real': start_time,
                'sys': start_resources.ru_stime,
                'user': start_resources.ru_utime}
    elif "win" in sys.platform:
        return {'real': time.time()}


def get_elapsed_time(time_dict):
    if "linux" in sys.platform:
        end_resources, end_time = resource_usage(RUSAGE_SELF), timestamp()
        cum_start = time_dict["sys"] + time_dict["user"]
        cum_end = end_resources.ru_stime + end_resources.ru_utime
        return cum_end - cum_start
    elif "win" in sys.platform:
        end_time = time.time()
        return end_time - time_dict["real"]


def alspp_compare(verbose=False):
    if args.random_state is None:
        start_random_state = get_random_state()
    else:
        start_random_state = args.random_state
    als_pp_time = 0
    Alspp_Time = Timestamp()
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

        # start = time.time()

        als_pp = cluster.KMeans(init='k-means++',
                                n_clusters=args.n_centers,
                                n_init=1,
                                algorithm='als++',
                                depth=args.depth,
                                search_steps=args.search_steps,
                                norm_it=args.normal_iterations,
                                random_state=current_state,
                                verbose=False,
                                tol=0)

        Alspp_Time.make_timestamp()

        als_pp.fit_new(X)

        # we define our time limit for the other algorithms and compare their inertia values
        # current_time = time.time() - start
        current_time = Alspp_Time.get_elapsed_time()
        # als_pp_time.add_time(current_time)
        # als_pp_time.add_time(als_pp_time.get_time())
        als_pp_time += current_time

        history["alspp"]["times"].append(current_time)
        # als_pp_time += current_time
        history["alspp"]["cum_times"].append(als_pp_time)
        if best_inertia_alspp > als_pp.inertia_:
            if best_inertia_alspp != np.infty:
                if verbose:
                    print("alspp old inertia: {} new inertia: {}".format(best_inertia_alspp, als_pp.inertia_))
            best_inertia_alspp = als_pp.inertia_
        history["alspp"]["best_inertia"].append(best_inertia_alspp)

    history["alspp"]["config"] = {"d": args.depth,
                                  "n": args.normal_iterations,
                                  "ss": args.search_steps,
                                  "reps": args.repeats}

    if verbose:
        print("ALSpp inertia: {}".format(best_inertia_alspp))
        print("ALSpp used time: {}".format(als_pp_time))
    # current_time = 0
    kmeans_time = Timestamp()
    current_time = 0
    best_inertia_normal = np.infty
    repeats = 0

    while current_time < als_pp_time:
        current_state = random_states[repeats]
        # start = time.time()

        normal_kmeans = cluster.KMeans(init='k-means++',
                                       n_clusters=args.n_centers,
                                       n_init=1,
                                       algorithm='elkan',
                                       random_state=current_state,
                                       tol=0)

        kmeans_time.make_timestamp()

        normal_kmeans.fit(X)
        # normal_kmeans_time = time.time() - start
        normal_kmeans_time = kmeans_time.get_elapsed_time()
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

    Lspp_Time = Timestamp()
    current_time = 0
    best_inertia_lspp = np.infty
    repeats = 0

    while current_time < als_pp_time:
        current_state = random_states[repeats]
        # start = time.time()

        lspp = cluster.KMeans(init='k-means++',
                              n_clusters=args.n_centers,
                              n_init=1,
                              algorithm='lspp',
                              random_state=current_state,
                              z=25,
                              tol=0)

        Lspp_Time.make_timestamp()

        lspp.fit_new(X)
        # lspp_time = time.time() - start
        lspp_time = Lspp_Time.get_elapsed_time()
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
        return wins, final_repeats, history

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

def alspp_compare(verbose=False):
    if args.random_state is None:
        start_random_state = get_random_state()
    else:
        start_random_state = args.random_state
    als_pp_time = 0
    Alspp_Time = Timestamp()
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

        # start = time.time()

        als_pp = cluster.KMeans(init='k-means++',
                                n_clusters=args.n_centers,
                                n_init=1,
                                algorithm='als++',
                                depth=args.depth,
                                search_steps=args.search_steps,
                                norm_it=args.normal_iterations,
                                random_state=current_state,
                                verbose=False,
                                tol=0)

        Alspp_Time.make_timestamp()

        als_pp.fit_new(X)

        # we define our time limit for the other algorithms and compare their inertia values
        # current_time = time.time() - start
        current_time = Alspp_Time.get_elapsed_time()
        # als_pp_time.add_time(current_time)
        # als_pp_time.add_time(als_pp_time.get_time())
        als_pp_time += current_time

        history["alspp"]["times"].append(current_time)
        # als_pp_time += current_time
        history["alspp"]["cum_times"].append(als_pp_time)
        if best_inertia_alspp > als_pp.inertia_:
            if best_inertia_alspp != np.infty:
                if verbose:
                    print("alspp old inertia: {} new inertia: {}".format(best_inertia_alspp, als_pp.inertia_))
            best_inertia_alspp = als_pp.inertia_
        history["alspp"]["best_inertia"].append(best_inertia_alspp)

    history["alspp"]["config"] = {"d": args.depth,
                                  "n": args.normal_iterations,
                                  "ss": args.search_steps,
                                  "reps": args.repeats}

    if verbose:
        print("ALSpp inertia: {}".format(best_inertia_alspp))
        print("ALSpp used time: {}".format(als_pp_time))
    # current_time = 0
    kmeans_time = Timestamp()
    current_time = 0
    best_inertia_normal = np.infty
    repeats = 0

    while current_time < als_pp_time:
        current_state = random_states[repeats]
        # start = time.time()

        normal_kmeans = cluster.KMeans(init='k-means++',
                                       n_clusters=args.n_centers,
                                       n_init=1,
                                       algorithm='elkan',
                                       random_state=current_state,
                                       tol=0)

        kmeans_time.make_timestamp()

        normal_kmeans.fit(X)
        # normal_kmeans_time = time.time() - start
        normal_kmeans_time = kmeans_time.get_elapsed_time()
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

    Lspp_Time = Timestamp()
    current_time = 0
    best_inertia_lspp = np.infty
    repeats = 0

    while current_time < als_pp_time:
        current_state = random_states[repeats]
        # start = time.time()

        lspp = cluster.KMeans(init='k-means++',
                              n_clusters=args.n_centers,
                              n_init=1,
                              algorithm='lspp',
                              random_state=current_state,
                              z=25,
                              tol=0)

        Lspp_Time.make_timestamp()

        lspp.fit_new(X)
        # lspp_time = time.time() - start
        lspp_time = Lspp_Time.get_elapsed_time()
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
        return wins, final_repeats, history

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

def save_plot_results(h, save=True):
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

    avg_results["alspp"]["config"] = {"d": his["alspp"]["config"]["d"],
                                      "n": his["alspp"]["config"]["n"],
                                      "ss": his["alspp"]["config"]["ss"],
                                      "reps": his["alspp"]["config"]["reps"]}

    if save:
        np.save("avg_results.npy", avg_results)

    return avg_results


def plot_results2(file=True, h=None):
    algorithms = ["alspp", "lspp", "kmeans"]

    # if file==True we choose the dictionary from a savefile, otherwise we compute the result
    if file:
        avg_results = np.load("avg_results.npy", allow_pickle='TRUE').item()
    else:
        if h is None:
            return
        else:
            avg_results = save_plot_results(h, save=False)

    # we plot our results
    for alg in algorithms:
        plt.plot(avg_results[alg]["x"], avg_results[alg]["y"], label=alg)

    plt.legend(loc="upper right")
    plt.ylabel("avg min inertia [cost]")
    plt.xlabel("time constraint [s]")

    # history["alspp"]["config"] = {"d": args.depth,
    # "n": args.normal_iterations,
    # "ss": args.search_steps,
    # "reps": args.repeats}

    if "config" in avg_results["alspp"]:
        d = avg_results["alspp"]["config"]["d"]
        n = avg_results["alspp"]["config"]["n"]
        ss = avg_results["alspp"]["config"]["ss"]
        reps = avg_results["alspp"]["config"]["reps"]

        plt.title("depth:{} normit:{} ss:{} best of {}".format(d, n, ss, reps))

    plt.show()


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()

    if args.plot:
        #plot_results()
        plot_all_results()

    else:
        greedy = not args.nogreedy
        #experiment_data = {"dataset": "datasets/pr91.txt", "trials": 10, "n_centers": 7, "depth": 2, "norm_it": 1, "best_of": 5}
        experiment_data = {"dataset": args.file.name, "trials": args.trials, "n_centers": args.n_centers, "depth": args.depth,
                           "norm_it": args.normal_iterations, "best_of": args.repeats, "greedy": greedy}
        test_experiment = Experiment(**experiment_data)
        test_experiment.run_experiment()
        #test_experiment.save_plot_results()

        # print("\nDataset: {}, k: {}, depth: {}, norm_it: {}, search_steps: {}".format(args.file.name, args.n_centers,
        #                                                                               args.depth, args.normal_iterations,
        #                                                                               args.search_steps))

        # test_experiment.write_to_log("\nDataset: {}, k: {}, depth: {}, norm_it: {}, search_steps: {}".format(args.file.name, args.n_centers,
        #                                                                                args.depth, args.normal_iterations,
        #                                                                                args.search_steps))

        # trials = 1
        #
        # # try:
        # X = np.genfromtxt(args.file)
        #
        # alspp_win_sum = k_means_win_sum = lspp_win_sum = 0
        #
        # histories = []
        #
        # for iteration in range(trials):
        #     print("###### Run {} ######".format(iteration + 1))
        #
        #     wins, final_repeats, history = alspp_compare(verbose=True)
        #     if wins["alspp"] == 2:
        #         alspp_win_sum += 1
        #     elif wins["lspp"] == 2:
        #         lspp_win_sum += 1
        #     elif wins["kmeans"] == 2:
        #         k_means_win_sum += 1
        #
        #     histories.append(history)
        #
        # print("#Wins ALSPP: {} , #Wins LSPP: {} , #Wins kMeans: {}".format(alspp_win_sum, lspp_win_sum, k_means_win_sum))
        #
        # save_plot_results(histories)
        #
        # plot_results(histories)

    # except Exception as e:
    #     print("Inertia of ALSPP = -1")
    #     f = open("errors.txt", "a")
    #     f.write("Dataset: {}, k: {}, depth: {}, norm_it: {}, search_steps: {}\n".format(args.file.name, args.n_centers,
    #                                                                                     args.depth,
    #                                                                                     args.normal_iterations,
    #                                                                                     args.search_steps))
    #     f.write(str(e) + "\n\n")
    #     f.close()
