import os
import numpy as np

datasets = open('datasets.txt', 'r')

lines = datasets.readlines()

#check if file already exists, otherwise skip
f = open('calls.txt', 'w')


def write_combinations_to_file(f, n_centers, depths, norm_its, search_steps):
    for n_center in n_centers:
        for depth in depths:
            for norm_it in norm_its:
                for search_step in search_steps:
                    f.write("python alspp.py -f {} -k {} -d {} -n {} -s {}\n".format(datapath, n_center, depth, norm_it, search_step))


# for each possible dataset we write the parameters in a line to calls.txt
for line in lines:
    datapath = line.strip()
    dataset = datapath.split(sep='/')[-1].split(sep='.')[0]
    if dataset == 'pr91':
        n_centers = np.array([8])
        depths = np.array([5, 7, 10])
        search_steps = np.arange(1, 4).astype('int64')
        norm_its = np.array([2, 3, 5, 7, 10])

        write_combinations_to_file(f, n_centers, depths, norm_its, search_steps)

    elif dataset == 'rectangles':
        n_centers = np.array([5, 10, 15])
        depths = np.array([2, 3, 5])
        search_steps = np.arange(1, 2).astype('int64')
        norm_its = np.array([2, 3, 5])

        write_combinations_to_file(f, n_centers, depths, norm_its, search_steps)

f.close()
