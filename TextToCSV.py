import pandas as pd
import argparse



def _build_arg_parser():
    """Builds the parser for the command line arguments"""

    arg_parser = argparse.ArgumentParser(description=
                                         """
        Takes a txt-File (in our format) and converts it to a CSV-File.
        """)

    arg_parser.add_argument(
        "-f", "--file",
        type=str,
        required=True,
        help="Input File")

    arg_parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Name of output file"
        )

    return arg_parser


if __name__ == '__main__':

    args = _build_arg_parser().parse_args()


    column_names = ['Call', 'Inertia ALS++', 'Inertia kM++', 'time ALS++', 'time kM++']
    df = pd.DataFrame(columns=column_names)

    with open(args.file) as f:
        d = {}
        runs_count = 1
        current_call = ""
        for line in f:
            
            split_line = line.split()

            if split_line:
                if split_line[0] == 'Dataset:':
                    
                    if current_call != line:
                        
                        current_call = line
                        runs_count = 1
                        key_name = current_call + " run " + str(runs_count)


                    else:

                        runs_count += 1
                        key_name = line + " run " + str(runs_count)
                        

                    d[key_name] = []

                elif split_line[0] == 'Inertia' and split_line[2] == 'ALSPP':
                    d[key_name].append(split_line[-1])

                elif split_line[0] == 'user':
                    d[key_name].append(float(split_line[-1]))

                elif split_line[0] == 'sys':
                    d[key_name][-1] += float(split_line[-1])

                elif split_line[0] == 'Inertia' and split_line[2] == 'kMeans++:':
                    d[key_name].append(split_line[-1])


    new_dict = {}

    for key in d:
        new_dict[key] = pd.Series(d[key], index=["Inertia ALS++", "Time ALS++", "Inertia kMeans++", "Time kMeans++"])


    df = pd.DataFrame(new_dict).transpose()
    df.to_csv(args.output + ".csv")


