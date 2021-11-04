#!/bin/bash
# collect all possible datasets in folder
find datasets/ -type f | sort -V > sorted_datasets.txt

# create file consisting of the parameter choices
printf "%s\n" {1..3}" "{1..3}" "{1..3} > parameters.txt

# to concatenate both files we add a 1 to the beginning of each line
cat sorted_datasets.txt | sed 's/^/1 /' > a.txt
cat parameters.txt | sed 's/^/1 /' > b.txt

# now we concatenate both using the 1es and then remove the 1es
join -j 1 a.txt b.txt | cut -d ' ' -f 2- > calls.txt

# remove unnecessary files
rm -i a.txt b.txt parameters.txt sorted_datasets.txt

# examples considering redirecting stderr and stdout:
# stdout -> file and stderr -> file2: 		command > file 2>file2
# stdout -> file and then stderr -> stdout: command >file 2>&1
# stdout -> file and stderr -> file :		command &> file (not supported in every shell, bash supports it)
# source: https://askubuntu.com/questions/625224/how-to-redirect-stderr-to-a-file

/usr/bin/time -p ... > outputs.txt 2> current_time.txt
cat current_time.txt ... zusammenrechnen >>   " " + outputs.txt
/usr/bin/time -p ... >> outputs.txt 2> current_time.txt
cat current_time.txt ... zusammenrechnen >>   " " + outputs.txt
/usr/bin/time -p ... >> outputs.txt 2> current_time.txt
cat current_time.txt ... zusammenrechnen >>   " " + outputs.txt

cat time.txt | cut -d ' ' -f 2 | awk '(NR>1)' | awk '{ total+=$1} END { print total }' | sed 's/^/ /' >> output.txt