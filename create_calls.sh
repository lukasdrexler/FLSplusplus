#!/bin/bash
# collect all possible datasets in folder
find datasets/ -type f | sort -V > datasets.txt

# create calls.txt from python script and write to calls.txt
python calls_datasets_generator.py

today="$(date '+%Y-%m-%d')"
now="$(date '+%H:%M:%S')"
echo "Time: ${today} ${now}" >> run.txt
cat calls.txt | xargs -L1 /usr/bin/time -p >> run.txt 2>&1



# remove unnecessary files
# rm -i sorted_datasets.txt

# examples considering redirecting stderr and stdout:
# stdout -> file and stderr -> file2: 		command > file 2>file2
# stdout -> file and then stderr -> stdout: command >file 2>&1
# stdout -> file and stderr -> file :		command &> file (not supported in every shell, bash supports it)
# source: https://askubuntu.com/questions/625224/how-to-redirect-stderr-to-a-file

#/usr/bin/time -p ... > outputs.txt 2> current_time.txt
#cat current_time.txt ... zusammenrechnen >>   " " + outputs.txt
#/usr/bin/time -p ... >> outputs.txt 2> current_time.txt
#cat current_time.txt ... zusammenrechnen >>   " " + outputs.txt
#/usr/bin/time -p ... >> outputs.txt 2> current_time.txt
#cat current_time.txt ... zusammenrechnen >>   " " + outputs.txt

#cat time.txt | cut -d ' ' -f 2 | awk '(NR>1)' | awk '{ total+=$1} END { print total }' | sed 's/^/ /' >> output.txt

