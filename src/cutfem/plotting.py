"""Thesis figure style, sizes, colours and plot helpers.

Rules followed by every figure:

* Created at its printed size (``figure``), included in LaTeX unscaled.
* No titles: the LaTeX caption says what the figure shows. Panels of a
  multi-panel figure are identified with ``panel_label``.
* Categorical colours in a fixed order (``CATEGORICAL``) for unrelated
  configurations; one light-to-dark ramp (``ordered_colors``) for an ordered
  parameter sweep, with the value 0 (no stabilization) always in ``NEUTRAL``.
* Reference orders shown as slope triangles (``slope_triangle``).
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

STYLE = Path(__file__).with_name("thesis.mplstyle")

# Printed widths in inches
TEXTWIDTH = 15.0 / 2.54     # A4 thesis, 11pt
COLUMNWIDTH = 3.5           # two-column paper
WIDTHS = {"full": TEXTWIDTH, "column": COLUMNWIDTH}

# Colours (validated for colour-vision deficiency, adjacent pairs)
CATEGORICAL = ["#6a4bb0", "#0f9a95", "#c97a1a", "#c2456b"]
NEUTRAL = "#8a8a8a"
INK = "#1a1a1a"
MARKERS = ["o", "s", "^", "D", "v", "P"]
_RAMP = LinearSegmentedColormap.from_list("violet", ["#c9bce8", "#6a4bb0", "#2a1a5e"])


def use_style():
    """Activate the thesis style for all subsequent figures."""
    plt.style.use(STYLE)
    # fontTools complains about the old timestamps of the bundled cmr10 font
    logging.getLogger("fontTools").setLevel(logging.ERROR)


def figure(nrows=1, ncols=1, width="full", aspect=0.75, **kwargs):
    """Create a figure at its printed size.

    Parameters
    ----------
    nrows, ncols : int
        Panel grid.
    width : {"full", "column"} or float
        Printed width: a named width, or a fraction of the text width.
    aspect : float
        Height / width of each panel.
    **kwargs
        Passed to ``plt.subplots`` (e.g. ``sharey=True``).
    """
    w = WIDTHS[width] if isinstance(width, str) else width * TEXTWIDTH
    h = w / ncols * aspect * nrows
    return plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)


def ordered_colors(values):
    """Colours for an ordered parameter sweep (light = small, dark = large).

    The value 0 (typically "no stabilization") is always ``NEUTRAL``.
    """
    nonzero = [v for v in values if v != 0]
    ramp = _RAMP(np.linspace(0.0, 1.0, len(nonzero))) if len(nonzero) > 1 \
        else [_RAMP(0.5)] * len(nonzero)
    colors = iter(ramp)
    return [NEUTRAL if v == 0 else next(colors) for v in values]


def slope_triangle(ax, x, y, order, size=0.2, gap=4.0):
    """Draw a slope triangle of the given order on log-log axes, clear of the data.

    Call after every curve of ``ax`` is drawn. The axis limits are frozen,
    then the triangle and its "1" and order labels are placed at the free
    position closest to the reference curve ``(x, y)``: no line, marker or
    scatter point of the axes comes closer than ``gap`` points (plus the
    marker radius). If no position is free, the y-range is widened step by
    step until one is.

    Parameters
    ----------
    ax : matplotlib Axes with log scales on both axes
    x, y : array_like
        Reference curve the triangle refers to.
    order : float
        Slope (negative for decreasing curves, e.g. -2 for kappa ~ h^-2).
    size : float
        Length of the horizontal leg, as a fraction of the axes width.
    gap : float
        Free space around the triangle and its labels, in points.
    """
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    markers = [line.get_markersize() for line in ax.lines if line.get_marker()
               not in (None, "", "None", " ")]
    clearance = gap + max(markers, default=0.0) / 2

    for _ in range(6):
        scale = _axes_size(ax)
        target = _densify(_to_axes(ax, ax.transData, np.column_stack([x, y])) * scale)
        obstacles = _obstacle_points(ax, scale)
        candidates = _triangle_candidates(ax, order, size, clearance, scale)
        if len(target):
            centers = np.array([c["center"] for c in candidates])
            distance = np.min(np.linalg.norm(
                centers[:, None, :] - target[None, :, :], axis=2), axis=1)
            candidates = [candidates[i] for i in np.argsort(distance)]
        for candidate in candidates:
            if _is_free(candidate, obstacles, clearance):
                _draw_triangle(ax, candidate, order, scale)
                return
        bottom, top = ax.get_ylim()      # nothing free: widen the y-range
        ax.set_ylim(bottom * (bottom / top) ** 0.15, top)

    _draw_triangle(ax, candidates[0], order, scale)   # give up: closest position


def _axes_size(ax):
    """Width and height of the axes in points."""
    bbox = ax.get_window_extent()
    return np.array([bbox.width, bbox.height]) * 72.0 / ax.figure.dpi


def _to_axes(ax, transform, points):
    """Convert points in ``transform`` coordinates to finite axes fractions."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    with np.errstate(all="ignore"):
        out = ax.transAxes.inverted().transform(transform.transform(points))
    return out[np.all(np.isfinite(out), axis=1)]


def _densify(points, step=1.0):
    """Points along a polyline, at most ``step`` apart."""
    if len(points) < 2:
        return points
    pieces = [points[:1]]
    for p, q in zip(points[:-1], points[1:], strict=True):
        n = max(int(np.ceil(np.linalg.norm(q - p) / step)), 1)
        pieces.append(p + np.linspace(0.0, 1.0, n + 1)[1:, None] * (q - p))
    return np.vstack(pieces)


def _obstacle_points(ax, scale):
    """Every line (densified), marker and scatter point of ``ax``, in points."""
    points = [np.empty((0, 2))]
    for line in ax.lines:
        if line.get_visible():
            xy = _to_axes(ax, line.get_transform(), line.get_xydata()) * scale
            points.append(_densify(xy))
    for collection in ax.collections:
        offsets = collection.get_offsets()
        if len(offsets):
            xy = _to_axes(ax, collection.get_offset_transform(), offsets) * scale
            points.append(xy)
    return np.vstack(points)


def _triangle_candidates(ax, order, size, clearance, scale, fontsize=8.0, n=40):
    """Possible triangles (vertices, label boxes) on a grid, in points."""
    width, height = scale
    text_h, char_w, pad = 1.3 * fontsize, 0.6 * fontsize, 2.0
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    dx = size * width
    dy = order * dx * (np.log10(x1 / x0) / width) * (height / np.log10(y1 / y0))
    if abs(dy) > 0.6 * height:           # very steep: shorten the triangle
        dx, dy = dx * 0.6 * height / abs(dy), np.sign(dy) * 0.6 * height
    rise = abs(dy)
    order_w = len(f"{order:g}") * char_w + pad

    left = clearance + (order_w if order < 0 else 0.0)
    right = width - clearance - dx - (order_w if order > 0 else 0.0)
    low, high = clearance + text_h + pad, height - clearance - rise
    candidates = []
    for a in np.linspace(left, right, n):
        for b in np.linspace(low, high, n):
            middle = (b + rise / 2 - text_h / 2, b + rise / 2 + text_h / 2)
            if order > 0:     # right angle bottom-right, order label on the right
                vertices = [(a, b), (a + dx, b), (a + dx, b + rise)]
                order_box = (a + dx + pad, middle[0], a + dx + order_w, middle[1])
            else:             # right angle bottom-left, order label on the left
                vertices = [(a, b + rise), (a + dx, b), (a, b)]
                order_box = (a - order_w, middle[0], a - pad, middle[1])
            one_box = (a + dx / 2 - char_w, b - pad - text_h,
                       a + dx / 2 + char_w, b - pad)
            candidates.append({"vertices": np.array(vertices),
                               "boxes": [one_box, order_box],
                               "center": np.mean(vertices, axis=0)})
    return candidates


def _is_free(candidate, points, clearance):
    """Check that no point is inside or within ``clearance`` of the candidate."""
    from matplotlib.path import Path

    if not len(points):
        return True
    v = candidate["vertices"]
    if Path(v).contains_points(points).any():
        return False
    for p, q in ((v[0], v[1]), (v[1], v[2]), (v[2], v[0])):
        t = np.clip(((points - p) @ (q - p)) / ((q - p) @ (q - p)), 0.0, 1.0)
        distance = np.linalg.norm(points - (p + t[:, None] * (q - p)), axis=1)
        if (distance < clearance).any():
            return False
    for x0, y0, x1, y1 in candidate["boxes"]:
        inside = ((points[:, 0] > x0 - clearance) & (points[:, 0] < x1 + clearance)
                  & (points[:, 1] > y0 - clearance) & (points[:, 1] < y1 + clearance))
        if inside.any():
            return False
    return True


def _draw_triangle(ax, candidate, order, scale):
    """Draw a candidate of :func:`_triangle_candidates` with its labels."""
    v = candidate["vertices"] / scale
    one_box, order_box = (np.reshape(box, (2, 2)) / scale for box in candidate["boxes"])
    ax.add_patch(plt.Polygon(v, closed=True, fill=False, edgecolor=INK,
                             linewidth=0.6, transform=ax.transAxes))
    ax.text(one_box[:, 0].mean(), one_box[1, 1], "1", transform=ax.transAxes,
            ha="center", va="top", fontsize=8)
    ax.text(order_box[0, 0] if order > 0 else order_box[1, 0], order_box[:, 1].mean(),
            f"${order:g}$", transform=ax.transAxes,
            ha="left" if order > 0 else "right", va="center", fontsize=8)


def panel_label(ax, text):
    """Identify a panel (e.g. "(a) no CIP") left-aligned above its axes."""
    ax.set_title(text, loc="left", fontsize=plt.rcParams["legend.fontsize"], pad=3)


def mark_nonfinite(ax, x, y, color):
    """Mark values a log axis cannot show: inf at the top, 0 at the bottom.

    Call after all curves are drawn, so the axis limits are final.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    bottom, top = ax.get_ylim()
    for mask, level, marker in ((np.isposinf(y), top, "x"), (y == 0, bottom, "v")):
        if mask.any():
            ax.scatter(x[mask], np.full(mask.sum(), level), color=color,
                       marker=marker, s=25, clip_on=False, zorder=5)
    ax.set_ylim(bottom, top)


def cut_mesh(ax, geometry, center, radius, margin=None):
    """Draw full cells, cut cells, ghost facets and the immersed circle.

    ``geometry`` comes from :func:`cutfem.core.mesh_geometry`. With
    ``margin``, the view is zoomed on the disk plus that margin.
    """
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    styles = {"full": ("#dcefee", "#9cd0cd", "Full cell"),
              "cut": ("#e6e0f3", "#b3a5dc", "Cut cell")}
    handles = []
    for key, (face, edge, label) in styles.items():
        boxes = np.reshape(geometry[key], (-1, 4))
        ax.add_collection(PatchCollection(
            [Rectangle((b[0], b[1]), b[2], b[3]) for b in boxes],
            facecolor=face, edgecolor=edge, linewidth=0.4))
        handles.append(Patch(facecolor=face, edgecolor=edge, label=label))

    ax.add_collection(LineCollection(np.reshape(geometry["ghost"], (-1, 2, 2)),
                                     colors=CATEGORICAL[2], linewidths=0.9))
    handles.append(Line2D([], [], color=CATEGORICAL[2], label="Ghost facet"))

    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta),
            color=CATEGORICAL[0], linewidth=1.0, linestyle="--")
    handles.append(Line2D([], [], color=CATEGORICAL[0], linestyle="--",
                          label=r"$\Gamma$"))

    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    if margin is not None:
        half = radius + margin
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
    else:
        ax.autoscale_view()
    return handles


def legend_above(fig, ax=None, ncols=None, handles=None):
    """One frameless legend above all panels.

    Built from the curves of ``ax`` (default: first axes), or from explicit
    ``handles``.
    """
    if handles is None:
        handles, _ = (ax or fig.axes[0]).get_legend_handles_labels()
    fig.legend(handles=handles, loc="outside upper center",
               ncols=ncols or len(handles))


def finalize(fig):
    """Uniform finishing touches applied to every figure before saving.

    Matplotlib labels minor ticks of short log axes, and the labels overlap.
    Instead, minor ticks are never labelled, and log axes spanning less than
    1.5 decades get labelled major ticks at 1, 2 and 5 times powers of ten
    (more for very short axes).
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    def sci(value, _):
        exponent = int(np.floor(np.log10(value) + 1e-9))
        mantissa = round(value / 10.0**exponent, 3)
        if mantissa == 1:
            return rf"$10^{{{exponent}}}$"
        return rf"${mantissa:g} \times 10^{{{exponent}}}$"

    for ax in fig.axes:
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale != "log" or not isinstance(axis.get_major_locator(), LogLocator):
                continue  # linear axis, or ticks set explicitly by the script
            axis.set_minor_formatter(NullFormatter())
            lo, hi = sorted(axis.get_view_interval())
            decades = np.log10(hi / lo) if lo > 0 else np.inf
            if decades < 1.5:
                subs = (1.0, 2.0, 5.0) if decades > 0.7 else (1.0, 2.0, 3.0, 5.0, 7.0)
                axis.set_major_locator(LogLocator(subs=subs, numticks=50))
                axis.set_major_formatter(FuncFormatter(sci))


def save(fig, stem, formats=("pdf",)):
    """Save ``fig`` as ``stem.<fmt>`` for every format; returns the paths."""
    finalize(fig)
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(f".{fmt}") for fmt in formats]
    for path in paths:
        fig.savefig(path)
    return paths
