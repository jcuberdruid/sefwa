#!/bin/bash

DIRECTORY="../../data/JxGxSzB3/"

# Array of all class CSV files
FILES=(
    "MI_FF_T2.csv"
    "MI_RLH_T1.csv"
    "MI_RLH_T2.csv"
)

# Loop over each file
for (( i=0; i<${#FILES[@]}; i++ )); do
    for (( j=i+1; j<${#FILES[@]}; j++ )); do
        for (( k=0; k<12; k++)); do
            # Run the Python program with the pair of files
            echo "Processing ${FILES[$i]} and ${FILES[$j]}"
            python3 main.py "$DIRECTORY/${FILES[$i]}" "$DIRECTORY/${FILES[$j]}" $k
        done
    done
done

