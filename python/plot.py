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
from matplotlib.patches import Ellipse, Polygon
import numpy as np
import cf


CMAP = matplotlib.colors.ListedColormap(RandomState(0).rand(256*256, 3))


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


def plot_history(history, origin, c=0, covellipse=True, min_r=0, max_back=None, r_values=False, track_id=False, velocity=False, trace=True, **kwargs):
    """Plot single trace."""
    max_back = max_back or 0
    lines = {}
    for recall in history[-max_back:]:
        for t in recall["targets"]:
            if t.id not in lines:
                lines[t.id] = [[], [], [], []]
            lines[t.id][0].append(t.r)
            lines[t.id][1].append(np.concatenate((cf.ll2ne(t.x[0:2], origin), t.x[2:])))
            lines[t.id][2].append(t.P)
            lines[t.id][3].append(t.cid)
        plot_bbox(recall["fov"].nebbox(origin).corners)
    for t in recall["targets"]:
        if t.r < min_r:
            del lines[t.id]
    for id_, (rs, xs, Ps, cids) in lines.items():
        cl = c + id_
        print("Line: ", id_, " :: ", xs)
        if trace:
            plt.plot([x[0] for x in xs], [x[1] for x in xs], color=CMAP(cl), **kwargs)
            for cid, x in zip(cids, xs):
                plt.plot(x[0], x[1], 's', fillstyle='none', color=CMAP(cid), **kwargs)

        if covellipse:
            ca = plot_cov_ellipse(Ps[-1][0:2, 0:2], xs[-1][0:2], 4)
            ce = plot_cov_ellipse(Ps[-1][0:2, 0:2], xs[-1][0:2], 4)
            ca.set_alpha(0.2)
            ca.set_facecolor(CMAP(cl))
            ce.set_facecolor('none')
            ce.set_edgecolor(CMAP(cl))
            ce.set_linewidth(4)

        if r_values:
            plt.text(xs[-1][0]+20, xs[-1][1], '{0:.2f}'.format(rs[-1]), color=CMAP(cl), fontsize=16)

        if track_id:
            plt.text(xs[-1][0]+20, xs[-1][1]+20, str(id_), color=CMAP(cl), )


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


def plot_scan(reports, origin, covellipse=True, **kwargs):
    """Plot reports from scan."""
    options = {
        'marker': '+',
        'color': 'r',
        'linestyle': 'None'
    }
    options.update(kwargs)
    zs = [cf.ll2ne(r.x[0:2], origin) for r in reports]
    plt.plot([float(z[0]) for z in zs],
             [float(z[1]) for z in zs], **options)
    if covellipse:
        for r in reports:
            ca = plot_cov_ellipse(r.R[0:2, 0:2], cf.ll2ne(r.x[0:2], origin))
            ca.set_alpha(0.1)
            ca.set_facecolor(options['color'])


def plot_bbox(corners, id_=0, cseed=0, **kwargs):
    """Plot bounding box."""
    options = {
        'alpha': 0.3,
        'color': CMAP(id_ + cseed)
    }
    options.update(kwargs)
    plt.gca().add_patch(Polygon(corners.T, **options))
