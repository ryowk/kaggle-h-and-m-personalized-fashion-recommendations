#!/bin/bash

set -eu -o pipefail

notebook_path=$1
message=$2
echo $notebook_path

output_notebook_path=${notebook_path%.*}.out.ipynb

papermill --log-output $notebook_path $output_notebook_path

kaggle competitions submit -c h-and-m-personalized-fashion-recommendations -f submission.csv -m "$message"
