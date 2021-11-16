import os
import numpy as np


n_runs = 50

datasets = open('datasets.txt', 'r')

lines = datasets.readlines()

#check if file already exists, otherwise skip
f = open('calls.txt', 'w')


def write_combinations_to_file(f, n_centers, depths, norm_its, search_steps):
    for n_center in n_centers:
        for depth in depths:
            for norm_it in norm_its:
                for search_step in search_steps:
                    for i in range (n_runs):
                        f.write("python alspp.py -f {} -k {} -d {} -n {} -s {}\n".format(datapath, n_center, depth, norm_it, search_step))
                        f.write("python normal_kmeans.py -f {} -k {}\n".format(datapath, n_center))


# for each possible dataset we write the parameters in a line to calls.txt
for line in lines:
    datapath = line.strip()
    dataset = datapath.split(sep='/')[-1].split(sep='.')[0]

    if dataset == 'pr91':
        n_centers = np.array([4, 8, 16, 100])
        depths = np.array([1, 3, 5])
        #search_steps = np.arange(1, 4).astype('int64')
        search_steps = np.array([1])
        norm_its = np.array([1, 2, 3, 5])

        write_combinations_to_file(f, n_centers, depths, norm_its, search_steps)

    elif dataset == 'rectangles':
        n_centers = np.array([4, 9, 36])
        depths = np.array([1, 3, 5])
        #search_steps = np.arange(1, 2).astype('int64')
        search_steps = np.array([1])
        norm_its = np.array([1, 2, 3, 5])

        write_combinations_to_file(f, n_centers, depths, norm_its, search_steps)

    elif dataset == 'D31':
        n_centers = np.array([31])
        depths = np.array([1, 3, 5])
        search_steps = np.array([1])
        norm_its = np.array([1, 2, 3, 5])

        write_combinations_to_file(f, n_centers, depths, norm_its, search_steps)

    elif dataset == 's2' or dataset == 's3' or dataset == 's4':
        n_centers = np.array([15])
        depths = np.array([1, 3, 5])
        search_steps = np.array([1])
        norm_its = np.array([1, 2, 3, 5])

        write_combinations_to_file(f, n_centers, depths, norm_its, search_steps)

    elif dataset == 'unbalance':
        n_centers = np.array([4, 6, 8, 10])
        depths = np.array([1, 3, 5])
        search_steps = np.array([1])
        norm_its = np.array([1, 2, 3, 5])

        write_combinations_to_file(f, n_centers, depths, norm_its, search_steps)

    #elif dataset == 'Tower':
     #   n_centers = np.array([20, 40])
      #  depths = np.array([1, 3])
       # search_steps = np.array([1])
        #norm_its = np.array([1, 2])


        
        #write_combinations_to_file(f, n_centers, depths, norm_its, search_steps)

    elif dataset == 'clegg':
        n_centers = np.array([20, 40])
        depths = np.array([1, 3])
        search_steps = np.array([1])
        norm_its = np.array([1, 3])        

        write_combinations_to_file(f, n_centers, depths, norm_its, search_steps)

f.close()
