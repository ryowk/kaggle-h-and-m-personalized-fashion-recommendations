#!/bin/bash

set -eu -o pipefail

export PYTHONPATH=.

for i in `seq 1 32`
do
    python ./experiments/repurchase.py &
done
