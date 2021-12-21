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

    column_names = ['Inertia ALS++', 'Inertia kM++', 'Inertia LS++', 'time ALS++', 'time kM++', 'time LS++']
    df = pd.DataFrame(columns=column_names)

    with open(args.file, 'r') as f:
        d = {}
        runs_count = 1
        current_call = ""

        loop = True
        f.readline()
        f.readline()

        while loop:
            lines = []


            # Read lines batch-wise, i.e. all lines that correspond to a single run of the algorithms

            for i in range(14):
                lines.append(f.readline().rstrip('\n'))
            if lines[0] == "":
                loop = False
                break

            alspp_check = lines[1].split()

            # We only write if there was no error (i.e. the inertia is not -1)
            if float(alspp_check[-1]) != -1:

                # Check whether we are dealing with a new call/configuration
                if current_call != lines[0]:
                    # If it is a new configuration, we reset the counter to 1 and adjust current_call accordingly
                    runs_count = 1
                    key_name = lines[0] + " run " + str(runs_count)
                    current_call = lines[0]
                else:
                    # If it is a repetition of an already known call, we simply increase the counter
                    runs_count += 1
                    key_name = current_call + " run " + str(runs_count)

                # All information for the current call (inertias and times) is going to be stored in d[key_name]
                d[key_name] = []


                # Save inertia of ALSPP solution from split string
                d[key_name].append(lines[1].split()[-1])

                # add user and of from ALSPP in one key
                d[key_name].append(float(lines[3].split()[-1]))
                d[key_name][-1] += (float)(lines[4].split()[-1])


                # Save inertia of kMeans++ solution from split string
                d[key_name].append(lines[5].split()[-1])

                # add user and sys of kMeans++ in one key
                d[key_name].append(float(lines[7].split()[-1]))
                d[key_name][-1] += float(lines[8].split()[-1])


                # add inertia of LS++
                d[key_name].append(lines[9].split()[-1])

                # add user and sys of LS++
                d[key_name].append(float(lines[11].split()[-1]))
                d[key_name][-1] += float(lines[12].split()[-1])


    new_dict = {}

    for key in d:
        new_dict[key] = pd.Series(d[key], index=["Inertia ALS++", "Time ALS++", "Inertia kMeans++", "Time kMeans++", "Inertia LS++", "Time LS++"])

    df = pd.DataFrame(new_dict).transpose()
    df.to_csv(args.output + ".csv")
