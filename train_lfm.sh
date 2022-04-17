#!/bin/bash

set -eu -o pipefail


for dataset in 1 10 100; do
    for week in 1 2 3 4 5; do
        for dim in 8 16 32; do
            for model_type in i_i if_i if_f if_if; do
                path=artifacts/lfm/lfm_${model_type}_dataset${dataset}_week${week}_dim${dim}_model.pkl
                if [ -e $path ]; then
                    echo skip $path
                else
                    python train_lfm.py $model_type $dataset $week $dim
                fi
            done
        done
    done
done
