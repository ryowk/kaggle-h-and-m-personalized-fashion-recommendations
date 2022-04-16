#!/bin/bash

set -eu -o pipefail


for dataset in 1 10 100;
do
    for week in 1 2 3 4 5 6;
    do
        for dim in 8 16 32 64;
        do
            python train_lfm.py i_i $dataset $week $dim
            python train_lfm.py if_i $dataset $week $dim
            # python train_lfm.py if_f $dataset $week $dim
            # python train_lfm.py if_if $dataset $week $dim
        done
    done
done
