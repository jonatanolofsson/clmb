"""Application base."""
from copy import deepcopy
from math import floor
import asyncio
import concurrent.futures
import logging
import os
import time as pytime
import matplotlib.pyplot as plt
import numpy as np
import lmb
from lmb import cf
import phdplanner as ppl
LOGGER = logging.getLogger(__name__)


class Application:
    """Application base class."""
    def __init__(self, loop=None, executor=None):
        """Init."""
        self._loop = loop or asyncio.get_event_loop()
        self._executor = executor or \
            concurrent.futures.ThreadPoolExecutor(max_workers=3)

        self.origin = None
        self.model = None
        self.unknown = None
        self.tracker = None
        self.tparams = lmb.Params()
        self.phdsize = np.array([100, 100])
        self.region = None

        # Scan settings
        self.nof_scans = int(1e15)
        self.scan_offset = 0

        self.fileprefix = ""
        self.ppl_agents = []

    async def _get(self, *args):
        """Get result from async executor."""
        task = self._loop.run_in_executor(
            self._executor, *args)
        completed, _ = await asyncio.wait([task])
        result = [task.result() for task in completed][0]
        return result

    async def predict(self, model, time, last_time):
        """Step the model forward in time."""
        return await self._get(self.tracker.predict, model, time, last_time)

    async def correct(self, sensor, scan, time):
        """Step the model forward in time."""
        return await self._get(self.tracker.correct, sensor, scan, time)

    async def get_targets(self):
        """Step the model forward in time."""
        return await self._get(self.tracker.get_targets)

    async def enof_targets(self, *args):
        """Expected number of targets."""
        return await self._get(self.tracker.enof_targets, *args)

    async def nof_targets(self, *args):
        """Number of confirmed targets (integer)."""
        return await self._get(self.tracker.nof_targets, *args)

    async def scans(self):
        """Override."""
        raise NotImplementedError("Override the method plx. Kthxbye")

    def sample_phd(self):
        """Sample phd."""
        phd = np.copy(self.tracker.pos_phd(self.region, self.phdsize))
        if np.isnan(phd).any():
            raise "NaN"
        phd += self.unknown
        return phd

    def plot_tracks(self, phd, reports, history):
        """Prepare area plot"""
        neregion = self.region.neaabbox(self.origin)
        extent = (neregion.min[1], neregion.max[1], neregion.min[0], neregion.max[0])
        lmb.plot.phd(phd, extent=extent)

        lmb.plot.plot_scan(reports, self.origin, bbox=True, show_cid=True, clustercolor=True)
        lmb.plot.plot_tracks(history, self.origin, covellipse=True, bbox=True, min_r=0.3, clustercolor=True)
        plt.gca().set_xlim(extent[0:2])
        plt.gca().set_ylim(extent[2:4])

    def plot_stats(self, k, nreports, ntargets, ntargets_verified):
        xs = np.arange(k + 1)
        plt.plot(xs, ntargets, label='Estimate')
        nreports = np.array(nreports)
        rmask = np.isfinite(nreports)
        plt.plot(xs[rmask], nreports[rmask], label='Reports')
        plt.plot(xs, ntargets_verified, label='Verified', marker='*')
        plt.ylabel('# Targets')
        plt.legend(fancybox=True, framealpha=0.5, loc=4, prop={'size': 10})

    async def run(self):
        """Run application."""
        raise NotImplementedError("Override the method plx. Kthxbye")
