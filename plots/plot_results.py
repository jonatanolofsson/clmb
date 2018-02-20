"""Utility to plot output."""
import os
import sys
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CLMB_DIR = os.path.dirname(THIS_DIR)

sys.path.append(CLMB_DIR)

import clmb


def draw(args):
    """Draw."""
    targets = {}

    with open(args.filename, 'r') as file_:
        i = 0
        for line in file_:
            i += 1
            data = json.loads(line)
            for t in data["post"]["targets"]:
                if t["id"] not in targets:
                    targets[t["id"]] = clmb.Target(t)
                targets[t["id"]].add_state(t)
            clmb.plt.subplot(4, 3, i)
            clmb.plt.title(data["t"])
            clmb.plot_traces((targets[t["id"]] for t in data["post"]["targets"]), covellipse=True, r_values=True)
            # clmb.plt.axis([-1, 11, -2, 12])


def parse_args(*argv):
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', default="filter.log")
    parser.add_argument('--output', default="lmb.png")
    parser.add_argument('--show', action="store_true")
    return parser.parse_args(argv)


def main(*argv):
    """Main."""
    args = parse_args(*argv)
    draw(args)
    if args.show:
        plt.show()
    else:
        plt.gcf().savefig(args.output, bbox_inches='tight')


if __name__ == '__main__':
    main(*sys.argv[1:])
