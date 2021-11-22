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

    with open(args.file, 'r') as f:
        d = {}
        runs_count = 1
        current_call = ""

        loop = True
        f.readline()
        f.readline()

        while loop:
            lines = []
            for i in range(10):
                lines.append(f.readline())
            if lines[0] == "":
                loop = False
                break

            alspp_check = lines[1].split()
            if float(alspp_check[-1]) != -1:
                if current_call != lines[1]:
                    runs_count = 1
                    key_name = lines[0] + " run " + str(runs_count)
                    current_call = lines[1]
                else:
                    runs_count += 1
                    key_name = lines[0] + " run " + str(runs_count)

                d[key_name] = []

                # Save inertia of ALSPP solution from splitted string
                d[key_name].append(lines[1].split()[-1])

                # add user and sys from ALSPP in one key
                d[key_name].append(float(lines[3].split()[-1]))
                d[key_name][-1] += (float)(lines[4].split()[-1])

                # Save inertia of ALSPP solution from splitted string
                d[key_name].append(lines[5].split()[-1])

                # add user and sys from kMeans++ in one key
                d[key_name].append(float(lines[7].split()[-1]))
                d[key_name][-1] += float(lines[8].split()[-1])

    new_dict = {}

    for key in d:
        new_dict[key] = pd.Series(d[key], index=["Inertia ALS++", "Time ALS++", "Inertia kMeans++", "Time kMeans++"])

    df = pd.DataFrame(new_dict).transpose()
    df.to_csv(args.output + ".csv")
