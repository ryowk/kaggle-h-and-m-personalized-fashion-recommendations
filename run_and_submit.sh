#!/bin/bash

set -eu -o pipefail

notebook_path=$1
message=$2
echo $notebook_path

jupyter nbconvert --to python $notebook_path
python_path=${notebook_path%.*}.py

echo start running $python_path
python $python_path

kaggle competitions submit -c h-and-m-personalized-fashion-recommendations -f submission.csv -m "$message"
