#!/bin/bash

# Set the base directory where "aaa" folder is located
base_dir="/scratch/yll6162/modnet/ibrnet_data"

want_list=('aflow_density' 'aflow_Egap' 'aflow_enthalpy_formation_atom' 'aflow_volume_atom' 'jarvis_bulk_modulus' 'jarvis_e_form' 'jarvis_gap_opt' 'jarvis_gap_tbmbj' 'jarvis_shear_modulus' 'mp_band_gap' 'mp_density' 'mp_e_above_hull' 'mp_formation_energy_per_atom' 'mp_total_magnetization' 'mp_volume' 'oqmd_band_gap' 'oqmd_e_formation_energy' 'oqmd_stability' 'oqmd_volume')

# want_list=('jarvis_bulk_modulus'
#             'jarvis_e_form'
#             'jarvis_gap_opt'
#             'jarvis_shear_modulus'
#             'mp_band_gap'
#             'mp_density'
#             'mp_e_above_hull'
#             'mp_formation_energy_per_atom'
#             'mp_total_magnetization'
#             'mp_volume')
# Find all subfolders under "aaa" and iterate through them
for subfolder in "$base_dir"/*; do
    if [ -d "$subfolder" ]; then
    # Check if there are any "ccc.csv" files within the current subfolder
        if [ -n "$(find "$subfolder" -maxdepth 1 -type f -name "*.csv")" ]; then
            property=$(basename "$subfolder")

            if [[ " ${want_list[*]} " =~ " ${property} " ]]; then
                echo "Processing folder: $subfolder"
                python modnet_pipeline_ibrnet.py $base_dir $property
                if [ $? -ne 0 ]; then
                  echo "Previous command exited with a non-zero status. Aborting."
                  exit 1
                fi
            else
                echo "Skipping folder:  $subfolder"

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


