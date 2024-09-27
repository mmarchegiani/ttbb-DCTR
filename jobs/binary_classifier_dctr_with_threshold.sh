#!/bin/bash
TTHBB_SPANET_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttHbb_SPANet
TTBB_DCTR_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttbb-DCTR
NUM_GPU=1

# Create venv in local job dir
python -m venv myenv --system-site-packages
source myenv/bin/activate

# Install ttHbb_SPANet in virtual environment
cd $TTHBB_SPANET_DIR
pip install -e .
cd -

cd $TTBB_DCTR_DIR
pip install -e .
cd -

# Launch training
python $TTBB_DCTR_DIR/scripts/train.py --cfg $1 --log_dir $2 --threshold
