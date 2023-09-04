#!/bin/bash

# Set the base directory where "aaa" folder is located
base_dir="/scratch/yll6162/modnet/materials_data"

skip_list=('mepsx' 
 'et_c55' 
 'n-powerfact' 
 'mbj_bandgap' 
 'mepsy' 
 'avg_elec_mass' 
 'avg_hole_mass' 
 'bulk_modulus_kv' 
 'density' 
 'dfpt_piezo_max_dielectric' 
 'dfpt_piezo_max_dielectric_electronic' 
 'dfpt_piezo_max_dielectric_ionic' 
 'dfpt_piezo_max_dij' 
 'dfpt_piezo_max_eij' 
 'ehull' 
 'encut' 
 'epsx' 
 'epsy' 
 'epsz' 
 'et_c11' 
 'et_c12' 
 'et_c13' 
 'et_c22' 
 'et_c33' 
 'et_c44' 
 'et_c66' 
 'exfoliation_energy' 
 'formation_energy_peratom' 
 'kpoint_length_unit' 
 'magmom_oszicar' 
 'magmom_outcar' 
 'max_efg' 
 'max_ir_mode' 
 'max_mode' 
 'mepsz' 
 'min_ir_mode' 
 'min_mode' 
 'n_em300k')
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
                python modnet_pipeline.py $base_dir $property

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


