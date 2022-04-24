#!/bin/bash

set -eu -o pipefail

python transform_data.py
./train_lfm.sh
python create_user_features.py
