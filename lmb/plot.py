"""Helper functions for LMB plots."""
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
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from . import cf


CMAP = matplotlib.colors.ListedColormap(RandomState(76).rand(256*256, 3))


def eigsorted(cov):
    """Return eigenvalues, sorted."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def calc_cov_ellipse(cov, nstd):
    """Get the covariance ellipse."""
    vals, vecs = eigsorted(cov)
    r1, r2 = nstd * np.sqrt(vals)
    theta = np.arctan2(*vecs[:, 0][::-1])

    return r1, r2, theta


def tracks(history, origin, c=0, covellipse=True, draw_bbox=False, clustercolor=False, min_r=0, r_values=False, show_id=False, show_cid=False, velocity=False, **kwargs):
    """Plot all tracks present at end time."""
    if not history:
        return
    if history[-1].get("fov"):
        bbox(history[-1]["fov"].nebbox(origin))
    lines = {}
    ids = [t.id for t in history[-1]["targets"] if t.r >= min_r]
    if not ids:
        return
    lines = {i: [] for i in ids}
    for recall in history:
        for t in recall["targets"]:
            if t.id not in ids:
                continue
            nex = np.concatenate((cf.ll2ne(t.x[0:2], origin), t.x[2:]))
            lines[t.id].append((t, nex))
    for track in lines.values():
        cl = CMAP(c + getattr(track[-1][0], "cid" if clustercolor else "id"))
        plt.plot([nex[1] for _, nex in track], [nex[0] for _, nex in track], color=cl, **kwargs)

        t, nex = track[-1]
        plt.plot(nex[1], nex[0], 's', fillstyle='none', color=cl, **kwargs)

        if covellipse:
            ca = cov_ellipse(t.P[0:2, 0:2], nex[0:2], 2)
            ce = cov_ellipse(t.P[0:2, 0:2], nex[0:2], 2)
            ca.set_alpha(0.2)
            ca.set_facecolor(cl)
            ce.set_facecolor('none')
            ce.set_edgecolor(cl)
            ce.set_linewidth(1)

        if draw_bbox:
            bbox(t.nebbox(origin), t.cid if clustercolor else t.id, c=c, **kwargs)

        if r_values:
            plt.text(nex[1]+20, nex[0], '{0:.2f}'.format(t.r), color=cl, fontsize=16)

        if show_id:
            plt.text(nex[1]-20, nex[0]+20, str(t.id), color=cl)

        if show_cid:
            plt.text(nex[1]+20, nex[0]+20, str(t.cid), color=cl)


def clusters(history, origin, c=0):
    """Plot all clusters."""
    clusters = {t.cid: {"cid": t.cid, "targets": [], "reports": []} for t in history[-1]["targets"] + history[-1]["reports"]}
    print("clusters: ", clusters)


def cov_ellipse(cov, pos, nstd=2, **kwargs):
    """Plot confidence ellipse."""
    r1, r2, theta = calc_cov_ellipse(cov, nstd)
    ellip = Ellipse(xy=np.flipud(pos), width=2*r2, height=2*r1, angle=theta, **kwargs)

    plt.gca().add_artist(ellip)
    return ellip


def scan(reports, origin, covellipse=True, draw_bbox=False, clustercolor=False, show_id=False, show_cid=False, c=0, **kwargs):
    """Plot reports from scan."""
    zs = [cf.ll2ne(r.x[0:2], origin) for r in reports]
    for rid, r in enumerate(reports):
        nex = cf.ll2ne(r.x[0:2], origin)
        cl = CMAP(c + (r.cid if clustercolor else rid))
        plt.plot([nex[1]], [nex[0]], marker='+', color='r', **kwargs)
        if covellipse:
            ca = cov_ellipse(r.P[0:2, 0:2], nex)
            ca.set_alpha(0.1)
            ca.set_facecolor(cl)

        if draw_bbox:
            # draw_bbox(r.nebbox(origin), rid, c=c, **kwargs)
            bbox(r.nebbox(origin), r.cid, c=c, **kwargs)

        if show_id:
            plt.text(nex[1]-20, nex[0]+20, str(rid), color=cl)

        if show_cid:
            plt.text(nex[1]-40, nex[0]-40, str(r.cid), color=cl)


def bbox(corners, id_=0, c=0, **kwargs):
    """Plot bounding box."""
    corners = corners.corners if hasattr(corners, "corners") else corners
    options = {
        'alpha': 0.3,
        'color': CMAP(id_ + c)
    }
    options.update(kwargs)
    plt.gca().add_patch(Polygon(np.fliplr(corners.T), **options))


def phd(phd_, *args, **kwargs):
    """Plot phd."""
    img = plt.gca().imshow(phd_, *args, origin='lower', vmin=0, vmax=phd_.max(), **kwargs)
    plt.colorbar(img)
