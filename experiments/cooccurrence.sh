#!/bin/bash

set -eu -o pipefail

for i in `seq 1 32`
do
    echo submit $i
    python ./experiments/cooccurrence.py &
    sleep 2
done
