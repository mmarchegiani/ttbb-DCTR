# Script to plot the 1D reweighting map based on the number of jet
# The script takes the reweitghing map file in yaml format and plots the reweighting map for each mask
# on the same plot. Each mask has a different color and is labeled by the corresponding tthbb cut.

import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from ttbb_dctr.lib.plotting import plot_reweighting_map

plt.switch_backend("agg")
hep.style.use("CMS")

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help="Input reweighting map file", required=True)
parser.add_argument('-o', '--output', type=str, help="Output file", default=None, required=False)

args = parser.parse_args()

if not args.input.endswith(".yaml"):
    raise ValueError("Input file should be in yaml format.")

if args.output is None:
    args.output = args.input.replace(".yaml", ".png")

plot_reweighting_map(args.input, args.output)
