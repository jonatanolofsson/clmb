#!/usr/bin/env python3
"""Test LMB tracker, C++/Python interface."""

import argparse
import asyncio
import concurrent.futures
import logging
import os
import sys
import lmb
import cf
import matplotlib.pyplot as plt
import numpy as np
from lmb import plot
import json

logging.basicConfig(level=logging.DEBUG)
_LOGGER = logging.getLogger(__name__)


class Application:
    """Main application."""

    def __init__(self, loop=None, executor=None):
        """Init."""
        _LOGGER.debug("Application init.")
        self._loop = loop or asyncio.get_event_loop()
        self._executor = executor or \
            concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.time = 0.0
        self.tracker = lmb.Tracker()

    async def _get(self, *args):
        """Get result from async executor."""
        task = self._loop.run_in_executor(
            self._executor, *args)
        completed, _ = await asyncio.wait([task])
        result = [task.result() for task in completed][0]
        return result

    async def predict(self, model, time=None, last_time=None):
        """Step the model forward in time."""
        last_time = last_time or self.time
        self.time = time or self.time + 1.0
        await self._get(self.tracker.predict, model, self.time, last_time)

    async def correct(self, sensor, scan, time):
        """Step the model forward in time."""
        await self._get(self.tracker.correct, sensor, scan, time)

    async def get_targets(self):
        """Step the model forward in time."""
        return await self._get(self.tracker.get_targets)

    async def enof_targets(self):
        """Expected number of targets."""
        return await self._get(self.tracker.enof_targets)

    async def nof_targets(self, rlim):
        """Number of confirmed targets (integer)."""
        return await self._get(self.tracker.nof_targets, rlim)

    async def run(self):
        """Run application."""
        _LOGGER.debug("Starting application.")
        plt.figure(figsize=(30, 30))
        model = lmb.CV(0.9, 1.5)
        sensor = lmb.PositionSensor()
        sensor.lambdaB = 2
        sensor.pD = 0.8
        targets = [
            np.array([0.0, 0.0, 1, 0.5]),
            np.array([0.0, 10.0, 1, -0.5]),
        ]
        ntargets_true = []
        ntargets_verified = []
        ntargets = []
        plt.subplot(3, 1, 1)
        history = []
        origin = np.array([58.3887657, 15.6965082])
        sensor.fov.from_gaussian(origin, 20 * np.eye(2))
        pre_enof_targets = 0
        ospa, gospa = [], []
        last_time = 0
        gridsize = np.array([-5, 5])
        phdpoints = np.empty((2, 40 * 40 / abs(gridsize[0] * gridsize[1])))
        i = 0
        for x in range(20, -20, gridsize[0]):
            for y in range(-20, 20, gridsize[1]):
                phdpoints[:, i] = cf.ne2ll(np.array([[x], [y]]), origin)
                i = i + 1
        for k in range(12):
            print()
            print("k:", k)
            time = k
            if k > 0:
                await self.predict(model, k, last_time)
                tracker_targets = await self.get_targets()
                # print("Predicted: ", tracker_targets)
                for target in targets:
                    target[0:2] += target[2:]

            reports = [lmb.GaussianReport(
                # np.random.multivariate_normal(t[0:2], np.diag([0.01] * 2)),  # noqa
                cf.ne2ll(t[0:2, np.newaxis], origin),
                np.eye(2) * 0.1, 0.01)
                       for i, t in enumerate(targets)]
            print("Reports: ", reports)
            if k > 0:
                sensor.lambdaB = max(0.2 * (len(reports) - pre_enof_targets), 0.01 * len(reports))
            await self.correct(sensor, reports, k)
            tracker_targets = await self.get_targets()
            history.append({"time": time, "targets": tracker_targets})
            pre_enof_targets = await self.enof_targets()
            ntargets.append(pre_enof_targets)
            ntargets_verified.append(await self.nof_targets(0.7))
            ntargets_true.append(len(targets))
            ll_targets = [np.concatenate((cf.ne2ll(t[0:2], origin), t[2:])) for t in targets]
            ospa.append(self.tracker.ospa(ll_targets, 1, 2))
            gospa.append(self.tracker.gospa(ll_targets, 1, 1))

            phd = self.tracker.pos_phd(phdpoints)
            print("points size: ", phdpoints.shape)
            print("phd size: ", phd.shape)
            with open(f"phdfiles/crosstrack_phd_{k:05}.json", 'w') as phdfile:
                json.dump({"points": phdpoints.tolist(), "gridsize": gridsize.tolist(), "phd": phd.tolist()}, phdfile)

            plot.plot_scan(reports, origin)
            plt.plot([t[0] for t in targets],
                     [t[1] for t in targets],
                     marker='D', color='y', alpha=.5, linestyle='None')
            print("Nof clusters: ", self.tracker.nof_clusters)
        plot.plot_history(history, origin, covellipse=False, min_r=0.7)
        plt.xlim([-1, k + 3])
        plt.ylabel('Tracks')
        plt.subplot(3, 1, 2)
        plt.plot(ntargets, label='Estimate')
        plt.plot(ntargets_true, label='True')
        plt.plot(ntargets_verified, label='Verified', marker='*')
        plt.ylabel('# Targets')
        plt.legend(fancybox=True, framealpha=0.5, loc=4, prop={'size': 10})
        plt.xlim([-1, k + 3])
        plt.subplot(3, 1, 3)
        plt.plot(ospa, label="OSPA")
        plt.plot(gospa, label="GOSPA")
        plt.legend()
        plt.xlim([-1, k + 3])


def parse_args(*argv):
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action="store_true")
    return parser.parse_args(argv)


def main(*argv):
    """Main."""
    _LOGGER.debug("Entering main.")
    args = parse_args(*argv)
    loop = asyncio.get_event_loop()
    app = Application(loop=loop)
    loop.run_until_complete(app.run())
    if args.show:
        plt.show()
    else:
        plt.gcf().savefig(os.path.splitext(os.path.basename(__file__))[0],
                          bbox_inches='tight')


if __name__ == '__main__':
    main(*sys.argv[1:])
