#!/bin/bash


# This is a script that runs a series of commands to train, test, and analyze a FAAE model with different hyperparameters.

# Define a list of values for -A and -G
A_VALUES="0.01 0.1 0.25 0.5 0.75 1"
G_VALUES="1 2 5 7 8 9 10 12 15 20"  # usually gamma should not below 1, but here is for try.

# Loop through each combination of -A and -G values
for a in $A_VALUES; do
  for g in $G_VALUES; do
    echo "Running with -A $a and -G $g" # print the current values of -A and -G to the console
    ./bootstrap_train_cvae_supervised_age_gender.py -A $a -G $g -R 0
    ./bootstrap_test_cvae_supervised_age_gender.py -D "ADNI"
    ./bootstrap_cvae_group_analysis_1x1_age_gender.py -D "ADNI" -L 0
  done
done


# Array to hold all possible values for -R
R_values=(0 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 5 10)

# Combinations of -A and -G values
combinations=(
    "0.1 5"
    "0.5 5"
    "0.01 20"
    "0.5 1"
)

# Loop through each combination of -A and -G values
for combo in "${combinations[@]}"
do
    # Split the combination into separate -A and -G values
    set -- $combo
    A_val=$1
    G_val=$2

    # For each combination of -A and -G, iterate over all -R values
    for R in "${R_values[@]}"
    do
        # Execute the training command
        ./bootstrap_train_cvae_supervised_age_gender.py -A $A_val -G $G_val -R $R
        # Execute the test command
        ./bootstrap_test_cvae_supervised_age_gender.py -D "ADNI"
        # Execute the group analysis command
        ./bootstrap_cvae_group_analysis_1x1_age_gender.py -D "ADNI" -L 0
    done
done