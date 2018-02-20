"""Helper functions for MHT plots."""

"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import matplotlib.colors
from numpy.random import RandomState
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np

class GM:
    """Gaussian Mixture class."""

    def __init__(self, t):
        """Init."""
        self.c = []
        for c in t["c"]:
            self.c.append((
                c["w"],
                np.array(c["x"]),
                np.matrix(c["P"])
            ))

    def mean(self):
        """Mean."""
        return np.sum(c[0] * c[1] for c in self.c)


class PF:
    """Particle filter class."""

    def __init__(self, t):
        """Init."""
        self.w = np.array(t["w"])
        self.x = np.array(t["x"])

    def mean(self):
        """Mean."""
        return (self.w * self.x).sum(axis=1)[:, np.newaxis]

    def cov(self):
        """Covariance."""
        m = self.mean()
        d = self.x - m
        return (self.w * d).dot(d.T) / (1 - (self.w * self.w).sum())


class Target:
    """Target class."""

    def __init__(self, t):
        """Init."""
        self.id = t["id"]
        self.history = []

    def add_state(self, t):
        """Add state."""
        self.t = t["t"]
        self.cid = t["cid"]
        self.r = t["r"]
        types = {"GM": GM, "PF": PF}
        self.pdf = types[t["pdf"]["type"]](t["pdf"])
        self.history.append((t["la"], self.r, self.pdf))

def eigsorted(cov):
    """Return eigenvalues, sorted."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def cov_ellipse(cov, nstd):
    """Get the covariance ellipse."""
    vals, vecs = eigsorted(cov)
    r1, r2 = nstd * np.sqrt(vals)
    theta = np.arctan2(*vecs[:, 0][::-1])

    return r1, r2, theta

CMAP = matplotlib.colors.ListedColormap(RandomState(0).rand(256*256, 3))


def plot_covell(c, w, x, P):
    """Plot cov ellipse with border and aplha color."""
    ca = plot_cov_ellipse(P[0:2, 0:2], x[0:2])
    ce = plot_cov_ellipse(P[0:2, 0:2], x[0:2])
    ca.set_alpha(w * 0.2)
    ce.set_alpha(w * 0.8)
    ca.set_facecolor(CMAP(c))
    ce.set_facecolor('none')
    ce.set_edgecolor(CMAP(c))
    ce.set_linewidth(1)


def plot_trace(t, c=0, covellipse=True, max_back=None, r_values=False, track_id=False, velocity=False, trace=True, **kwargs):
    """Plot single trace."""
    max_back = max_back or 0
    xs, ys, vxs, vys = [], [], [], []
    for ty, r, pdf in t.history[-max_back:]:
        state = np.squeeze(pdf.mean()).tolist()
        xs.append(state[0])
        ys.append(state[1])
        vxs.append(state[2])
        vys.append(state[3])
    if covellipse:
        if isinstance(pdf, PF):
            plot_covell(c, 1, state, pdf.cov())
        elif isinstance(pdf, GM):
            for w, x, P in pdf.c:
                plot_covell(c, w, x, P)
    if trace:
        plt.plot(xs, ys, color=CMAP(c), **kwargs)
    if r_values:
        plt.text(state[0],state[1], '{0:.2f}'.format(t.r), color=CMAP(c), fontsize=16)
    if track_id:
        plt.text(state[0], state[1], str(t.id), color=CMAP(c))
    if velocity:
        v = np.array([vxs[-1], vys[-1]])
        va = np.sqrt(vxs[-1]*vxs[-1] + vys[-1]*vys[-1])
        if va > 15:
            v *= 15 / va
        plt.plot([xs[-1],xs[-1]+v[0]], [ys[-1],ys[-1]+v[1]], color=CMAP(c), linewidth=3)

def plot_traces(targets, cseed=0, covellipse=True, max_back=None, **kwargs):
    """Plot all targets' traces."""
    for t in targets:
        plot_trace(t, t.id + cseed, covellipse, max_back, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=2, **kwargs):
    """Plot confidence ellipse."""
    r1, r2, theta = cov_ellipse(cov, nstd)
    ellip = Ellipse(xy=pos, width=2*r1, height=2*r2, angle=theta, **kwargs)

    plt.gca().add_artist(ellip)
    return ellip


def plot_scan(scan, covellipse=True, **kwargs):
    """Plot reports from scan."""
    options = {
        'marker': '+',
        'color': 'r',
        'linestyle': 'None'
    }
    options.update(kwargs)
    plt.plot([float(r.z[0]) for r in scan.reports],
             [float(r.z[1]) for r in scan.reports], **options)
    if covellipse:
        for r in scan.reports:
            ca = plot_cov_ellipse(r.R[0:2, 0:2], r.z[0:2])
            ca.set_alpha(0.1)
            ca.set_facecolor(options['color'])


def plot_bbox(obj, cseed=0, **kwargs):
    """Plot bounding box."""
    id_ = getattr(obj, 'id', 0)
    options = {
        'alpha': 0.3,
        'color': CMAP(id_ + cseed)
    }
    options.update(kwargs)
    bbox = obj.bbox()
    plt.gca().add_patch(Rectangle(
        (bbox[0], bbox[2]), bbox[1] - bbox[0], bbox[3] - bbox[2],
        **options))
