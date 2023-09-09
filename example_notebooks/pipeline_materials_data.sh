#!/bin/bash

# Set the base directory where "aaa" folder is located
base_dir="/scratch/yll6162/modnet/materials_data"

skip_list=()
# Find all subfolders under "aaa" and iterate through them
for subfolder in "$base_dir"/*; do
    if [ -d "$subfolder" ]; then
    # Check if there are any "ccc.csv" files within the current subfolder
        if [ -n "$(find "$subfolder" -maxdepth 1 -type f -name "*.csv")" ]; then
            property=$(basename "$subfolder")

            if [[ " ${skip_list[*]} " =~ " ${property} " ]]; then
                echo "Skipping folder:  $subfolder"
            else
                echo "Processing folder: $subfolder"
                python materials_data_model.py $base_dir $property
                if [ $? -ne 0 ]; then
                  echo "Previous command exited with a non-zero status. Aborting."
                  exit 1
                fi
            fi
            # # Run your Python script on each "ccc.csv" file within the current subfolder
            # find "$subfolder" -maxdepth 1 -type f -name "ccc.csv" | while read -r ccc_file; do
            #     python your_script.py "$ccc_file"
            # done
        else
            echo "No '.csv' files found in folder: $subfolder"
        fi
    fi
done


