import os
import numpy as np
import argparse

def _build_arg_parser():
    """Builds the parser for the command line arguments"""

    arg_parser = argparse.ArgumentParser(description=
                                         """
        Generates the calls for our experiments.
        """
                                         )
    arg_parser.add_argument(
        "-r", "--runs",
        type=int,
        default=1,
        help="number of consecutive runs with same parameters"
    )

    return arg_parser




def generate_random_state():
    return np.random.randint(0, 1000000)

def write_combinations_to_file(f, n_centers, depths, norm_its, search_steps, n_runs):
    for n_center in n_centers:
        for depth in depths:
            for norm_it in norm_its:
                for search_step in search_steps:
                    for i in range(n_runs):
                        seed = generate_random_state()
                        f.write("python alspp.py -f {} -k {} -d {} -n {} -s {} -r {}\n".format(datapath, n_center, depth, norm_it, search_step, seed))
                        f.write("python normal_kmeans.py -f {} -k {} -r {}\n".format(datapath, n_center, seed))
                        f.write("python -O lspp.py -f {} -k {} -r {} -z {}\n".format(datapath, n_center, seed, 25))


if __name__ == '__main__':

    args = _build_arg_parser().parse_args()

    n_runs = args.runs

    datasets = open('datasets.txt', 'r')

    lines = datasets.readlines()

    # check if file already exists, otherwise skip
    f = open('calls.txt', 'w')

    # for each possible dataset we write the parameters in a line to calls.txt
    for line in lines:
        datapath = line.strip()
        dataset = datapath.split(sep='/')[-1].split(sep='.')[0]

        if dataset == 'pr91':
            n_centers = np.array([4, 8, 16, 100])
            depths = np.array([1, 3, 5])
            # search_steps = np.arange(1, 4).astype('int64')
            search_steps = np.array([1])
            norm_its = np.array([1, 2, 3, 5])

            write_combinations_to_file(f, n_centers, depths, norm_its, search_steps, n_runs)

        elif dataset == 'rectangles':
            n_centers = np.array([4, 9, 36])
            depths = np.array([1, 3, 5])
            # search_steps = np.arange(1, 2).astype('int64')
            search_steps = np.array([1])
            norm_its = np.array([1, 2, 3, 5])

            write_combinations_to_file(f, n_centers, depths, norm_its, search_steps, n_runs)

        elif dataset == 'D31':
            n_centers = np.array([31])
            depths = np.array([1, 3, 5])
            search_steps = np.array([1])
            norm_its = np.array([1, 2, 3, 5])

            write_combinations_to_file(f, n_centers, depths, norm_its, search_steps, n_runs)

        elif dataset == 's2' or dataset == 's3' or dataset == 's4':
            n_centers = np.array([15])
            depths = np.array([1, 3, 5])
            search_steps = np.array([1])
            norm_its = np.array([1, 2, 3, 5])

            write_combinations_to_file(f, n_centers, depths, norm_its, search_steps, n_runs)

        elif dataset == 'unbalance':
            n_centers = np.array([4, 6, 8, 10])
            depths = np.array([1, 3, 5])
            search_steps = np.array([1])
            norm_its = np.array([1, 2, 3, 5])

            write_combinations_to_file(f, n_centers, depths, norm_its, search_steps, n_runs)

        # elif dataset == 'Tower':
        #   n_centers = np.array([20, 40])
        #  depths = np.array([1, 3])
        # search_steps = np.array([1])
        # norm_its = np.array([1, 2])

        # write_combinations_to_file(f, n_centers, depths, norm_its, search_steps)

        #elif dataset == 'clegg':
        #    n_centers = np.array([20, 40])
        #    depths = np.array([1])
        #    search_steps = np.array([1])
        #    norm_its = np.array([1])

        #    write_combinations_to_file(f, n_centers, depths, norm_its, search_steps, n_runs)

        elif dataset == 'unproj':
            n_centers = np.array([10])
            depths = np.array([1])
            search_steps = np.array([1])
            norm_its = np.array([1])

            write_combinations_to_file(f, n_centers, depths, norm_its, search_steps, n_runs)



    f.close()
