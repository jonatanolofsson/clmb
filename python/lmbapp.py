#!/usr/bin/env python3
"""Test LMB tracker, C++/Python interface."""

import argparse
import asyncio
import concurrent.futures
import logging
import os
import sys
import lmb
import matplotlib.pyplot as plt
import numpy as np
import plot

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

    async def predict(self, model, time=None):
        """Step the model forward in time."""
        self.time = time or self.time + 1.0
        await self._get(self.tracker.predict, model, self.time)

    async def correct(self, sensor, scan=None):
        """Step the model forward in time."""
        scan = scan or []
        await self._get(self.tracker.correct, sensor, scan)

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
        plt.figure(figsize=(10, 10))
        model = lmb.CV(0.9, 0.5)
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
        plt.subplot(2, 1, 1)
        history = []
        for k in range(30):
            # print()
            print("k:", k)
            if k > 0:
                sensor.lambdaB = 0.2
                await self.predict(model, k)
                tracker_targets = await self.get_targets()
                # print("Predicted: ", tracker_targets)
                for target in targets:
                    target[0:2] += target[2:]
            # if k == 5:
                # targets.append(np.array([5.0, 5.0, 1.0, 0.0]))
            # if k % 7 == 0:
                # targets.append(np.random.multivariate_normal(
                    # np.array([k, 7.0, 0.0, 0.0]),
                    # np.diag([1, 0.5, 1, 1])))
            # if k % 9 == 1:
                # del targets[-1]
            # if k == 10:
                # targets.append(np.array([10.0, -30.0, 1.0, -0.5]))
            # if k == 20:
                # targets.append(np.array([k, 0.0, 1.0, 4.0]))

            reports = [lmb.GaussianReport(
                # np.random.multivariate_normal(t[0:2], np.diag([0.01] * 2)),  # noqa
                t[0:2, np.newaxis],
                np.eye(2) * 0.5, 0.001)
                       for i, t in enumerate(targets)]
            sensor.lambdaB = 0.1 * len(reports)
            await self.correct(sensor, reports)
            tracker_targets = await self.get_targets()
            # print("Corrected: ", tracker_targets)
            history.append((k, tracker_targets))
            ntargets.append(await self.enof_targets())
            ntargets_verified.append(await self.nof_targets(0.7))
            ntargets_true.append(len(targets))
            plot.plot_scan(reports)
            plt.plot([t[0] for t in targets],
                     [t[1] for t in targets],
                     marker='D', color='y', alpha=.5, linestyle='None')
        plot.plot_history(history, covellipse=True)
        plt.axis([-1, k + 1, -k - 1, k + 1 + 10])
        plt.ylabel('Tracks')
        plt.subplot(2, 1, 2)
        plt.plot(ntargets, label='Estimate')
        plt.plot(ntargets_true, label='True')
        plt.plot(ntargets_verified, label='Verified')
        plt.ylabel('# Targets')
        plt.legend(fancybox=True, framealpha=0.5, loc=4, prop={'size': 10})
        plot.plt.axis([-1, k + 1, min(ntargets + ntargets_true) - 0.1,
                      max(ntargets + ntargets_true) + 0.1])


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
