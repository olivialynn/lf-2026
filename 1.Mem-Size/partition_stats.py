"""Per-partition size statistics and HEALPix merge analysis for HATS catalogs.

Three notions of a partition's "size", from cheapest to most expensive to measure:

* on-disk compressed bytes      -> partition_sizes()    (one os.stat per file)
* parquet-uncompressed bytes
  and row counts                -> partition_frame()    (one `_metadata` read)
* exact in-memory arrow size    -> partition_arrow_mb() (full read of every file)

A 1 GB *in-memory* partition budget must be checked against partition_arrow_mb():
on the mem_size_dp2 catalogs, disk bytes understate RAM by ~3x and
parquet-uncompressed bytes by ~1.5-2.3x (decompression, then decoding of
dictionary/RLE pages into plain arrow buffers).

Merge analysis: in HEALPix NESTED ordering a pixel's parent one order coarser is
`npix // 4`. rollup() accumulates every partition into all of its ancestors;
maximal_merge_groups() then finds the coarsest ancestors whose whole subtree still
fits the budget - i.e. genuinely over-split regions that could collapse into a
single coarser partition.

All scans cache to `<cache_dir>/*.npz` (default: `.partition_cache/` next to this
file), so only the first pass over a catalog is slow.
"""

import logging
import os
import re
import warnings
from collections import Counter, defaultdict

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
from cdshealpix import healpix_to_lonlat  # NESTED; same lib hats uses internally
from cdshealpix.nested import vertices
from hats import read_hats
from hats.inspection.visualize_catalog import plot_healpix_map
from human_readable import file_size
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".partition_cache")

_PIX = re.compile(r"Norder=(\d+)/Dir=\d+/Npix=(\d+)")


def quiet():
    """Silence third-party DEBUG/INFO logs and the benign hats 'HEALPix pixels
    smaller than a plot pixel' warning, so notebook output is just the figures."""
    for noisy in ("matplotlib", "numexpr", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message="This plot contains HEALPix pixels smaller")


def _cache_path(cat_dir, suffix, cache_dir):
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    return os.path.join(cache_dir, cat_dir.strip("/").replace("/", "__") + suffix)


def pixel_path(cat_dir, order, npix):
    """On-disk parquet path for a HATS partition (Dir = floor(Npix/10000)*10000)."""
    return f"{cat_dir}/dataset/Norder={order}/Dir={(npix // 10000) * 10000}/Npix={npix}.parquet"


# --------------------------------------------------------------------------- #
# Scanning: three ways to measure partition size                               #
# --------------------------------------------------------------------------- #

def partition_sizes(cat_dir, cache=True, cache_dir=None):
    """(order, npix, size_bytes) per partition: ON-DISK COMPRESSED bytes.

    Pixel list comes from read_hats (reads partition_info.csv only); each
    partition file is then os.stat'ed by its deterministic path. Stat-ing by
    pixel path (rather than os.walk) keeps `_metadata`, the skymaps and
    `data_thumbnail.parquet` out of the numbers. Cached to `*.sizes.npz`.
    """
    ckey = _cache_path(cat_dir, ".sizes.npz", cache_dir)
    if cache and os.path.exists(ckey):
        d = np.load(ckey)
        return d["order"], d["npix"], d["size"]
    pix = read_hats(cat_dir).get_healpix_pixels()
    order = np.array([p.order for p in pix], dtype=np.uint8)
    npix = np.array([p.pixel for p in pix], dtype=np.uint64)
    size = np.array(
        [os.path.getsize(pixel_path(cat_dir, int(o), int(p))) for o, p in zip(order, npix)],
        dtype=np.int64,
    )
    if cache:
        os.makedirs(os.path.dirname(ckey), exist_ok=True)
        np.savez(ckey, order=order, npix=npix, size=size)
    return order, npix, size


def partition_frame(cat_dir, cache=True, cache_dir=None):
    """(order, npix, rows, mb) per partition: ROW COUNTS + PARQUET-UNCOMPRESSED MB.

    Fast path: read the parquet ``_metadata`` index once and aggregate row-group
    ``num_rows`` + ``total_byte_size`` (uncompressed page bytes) per file.
    Fallback for catalogs without ``_metadata``: os.stat each file for on-disk
    compressed bytes; row count is left NaN. Cached to ``*.npz``.

    NOTE: ``total_byte_size`` is *decompressed but still encoded* - it understates
    the decoded arrow in-memory size (see partition_arrow_mb).
    """
    ckey = _cache_path(cat_dir, ".npz", cache_dir)
    if cache and os.path.exists(ckey):
        d = np.load(ckey)
        return d["order"], d["npix"], d["rows"], d["mb"]

    meta = f"{cat_dir}/dataset/_metadata"
    if os.path.exists(meta):
        md = pq.read_metadata(meta)
        rows, byt = defaultdict(float), defaultdict(float)
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            f = rg.column(0).file_path
            rows[f] += rg.num_rows
            byt[f] += rg.total_byte_size
        order, npix, rr, mb = [], [], [], []
        for f in rows:
            o, n = _PIX.search(f).groups()
            order.append(int(o)); npix.append(int(n))
            rr.append(rows[f]);   mb.append(byt[f] / 1e6)
    else:
        pix = read_hats(cat_dir).get_healpix_pixels()
        order = [p.order for p in pix]
        npix = [p.pixel for p in pix]
        rr = [np.nan] * len(pix)
        mb = [os.path.getsize(pixel_path(cat_dir, o, n)) / 1e6 for o, n in zip(order, npix)]

    order = np.asarray(order, np.int64)
    npix = np.asarray(npix, np.int64)
    rr = np.asarray(rr, float)
    mb = np.asarray(mb, float)
    if cache:
        os.makedirs(os.path.dirname(ckey), exist_ok=True)
        np.savez(ckey, order=order, npix=npix, rows=rr, mb=mb)
    return order, npix, rr, mb


def partition_arrow_mb(cat_dir, cache=True, cache_dir=None):
    """(order, npix, mb) per partition: EXACT IN-MEMORY MB as a decoded pyarrow
    Table (``Table.nbytes``) - the ground truth for an in-memory partition budget.

    First run reads every parquet file in full (~93 GB across the three
    mem_size_dp2 builds, minutes on NFS) - cached to ``*.arrow.npz``, so it is
    slow exactly once per catalog. Don't call this on Gaia casually (765 GB).
    """
    ckey = _cache_path(cat_dir, ".arrow.npz", cache_dir)
    if cache and os.path.exists(ckey):
        d = np.load(ckey)
        return d["order"], d["npix"], d["mb"]
    pix = read_hats(cat_dir).get_healpix_pixels()
    order = np.array([p.order for p in pix], np.int64)
    npix = np.array([p.pixel for p in pix], np.int64)
    mb = np.array(
        [pq.read_table(pixel_path(cat_dir, int(o), int(p))).nbytes / 1e6
         for o, p in zip(order, npix)],
        float,
    )
    if cache:
        os.makedirs(os.path.dirname(ckey), exist_ok=True)
        np.savez(ckey, order=order, npix=npix, mb=mb)
    return order, npix, mb


def ondisk_mb(cat_dir, **kw):
    """On-disk parquet MB per partition (convenience wrapper over partition_sizes)."""
    _, _, size = partition_sizes(cat_dir, **kw)
    return size / 1e6


# --------------------------------------------------------------------------- #
# Sky geometry                                                                 #
# --------------------------------------------------------------------------- #

def centers_deg(order, npix):
    """HEALPix pixel centers (NESTED) -> (ra, dec) in degrees."""
    lon, lat = healpix_to_lonlat(np.asarray(npix, np.uint64), np.asarray(order, np.uint8))
    return lon.deg % 360.0, lat.deg


def footprint_zoom(order, npix, pad=1.4):
    """(center SkyCoord, fov Quantity) enclosing the pixel centers, or (None, None)
    for near-full-sky footprints. The unit-vector mean centre handles footprints
    that straddle RA = 0."""
    ra, dec = centers_deg(order, npix)
    ra_r, dec_r = np.radians(ra), np.radians(dec)
    v = np.column_stack([np.cos(dec_r) * np.cos(ra_r),
                         np.cos(dec_r) * np.sin(ra_r),
                         np.sin(dec_r)]).mean(axis=0)
    v /= np.linalg.norm(v)
    cen = SkyCoord(ra=np.degrees(np.arctan2(v[1], v[0])) % 360 * u.deg,
                   dec=np.degrees(np.arcsin(v[2])) * u.deg)
    sep = cen.separation(SkyCoord(ra=ra * u.deg, dec=dec * u.deg)).deg.max()
    if sep > 45:
        return None, None  # too wide to zoom sensibly - keep the Mollweide view
    return cen, 2 * pad * sep * u.deg


# --------------------------------------------------------------------------- #
# Merge analysis (HEALPix subtree roll-up)                                     #
# --------------------------------------------------------------------------- #

def rollup(order, npix, value):
    """Accumulate every partition into all of its coarser HEALPix ancestors
    (NESTED parent = npix // 4 per order step).

    Returns four dicts keyed by (order, npix), covering leaves AND ancestors:
    total value under the key, leaf-partition count, and the deepest (hi) /
    shallowest (lo) leaf order contributing to it.
    """
    total, leaves = defaultdict(float), defaultdict(int)
    hi, lo = defaultdict(int), defaultdict(lambda: 99)
    for o, p, v in zip(order.astype(int), npix.astype(int), value):
        k = (int(o), int(p))
        total[k] += v; leaves[k] += 1
        hi[k] = max(hi[k], k[0]); lo[k] = min(lo[k], k[0])
    for o in range(int(order.max()), 0, -1):
        for (oo, p) in [k for k in total if k[0] == o]:
            par = (o - 1, p >> 2)
            total[par] += total[(oo, p)]
            leaves[par] += leaves[(oo, p)]
            hi[par] = max(hi[par], hi[(oo, p)])
            lo[par] = min(lo[par], lo[(oo, p)])
    return total, leaves, hi, lo


def maximal_merge_groups(total, leaves, thresh):
    """Coarsest ancestors whose whole subtree (>1 partition) stays under `thresh`.

    'Maximal' = the subtree's own parent is NOT mergeable, so groups don't nest
    and each over-split region is counted exactly once.
    """
    mergeable = {k for k, t in total.items() if leaves[k] > 1 and t < thresh}
    return [k for k in mergeable if (k[0] - 1, k[1] >> 2) not in mergeable]


def children_under(order, npix, mb, po, pp):
    """All leaf partitions (order >= po) whose HEALPix ancestor at order po is pp."""
    sel = (order >= po) & ((npix >> (2 * (order - po))) == pp)
    return order[sel], npix[sel], mb[sel]


def merge_candidates(name, cat_dir, thresh_mb, show=15, plot=True, cache_dir=None):
    """Find HEALPix subtrees that could collapse to a single coarser partition
    while staying < thresh_mb IN MEMORY (exact arrow sizes). Prints a summary +
    the biggest candidates and (optionally) plots. Returns (maximal, total, leaves).
    """
    order, npix, mb = partition_arrow_mb(cat_dir, cache_dir=cache_dir)
    total, leaves, _, _ = rollup(order, npix, mb)
    maximal = maximal_merge_groups(total, leaves, thresh_mb)
    saved = sum(leaves[k] - 1 for k in maximal)
    n = len(order)

    print(f"{name}: {n:,} partitions, threshold {thresh_mb:.0f} MB (arrow in-memory)")
    print(f"  {len(maximal):,} maximal merge groups -> eliminate {saved:,} partitions "
          f"({100 * saved / n:.1f}%) -> new count {n - saved:,}")
    maximal.sort(key=lambda k: leaves[k], reverse=True)
    print(f"\n  top {show} candidates (parent order/npix : #partitions -> 1, merged size):")
    for o, p in maximal[:show]:
        print(f"    order {o:>2d}  npix {p:>10d} : {leaves[(o, p)]:>4d} -> 1   ({total[(o, p)]:6.0f} MB)")

    if plot:
        merged = np.array([total[k] for k in maximal])
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
        a1.hist(merged, bins=40, color="#4C72B0", edgecolor="white")
        a1.axvline(thresh_mb, color="#C44E52", ls="--", lw=2, label=f"{thresh_mb:.0f} MB target")
        a1.set_xlabel("Merged partition size (MB, in-memory)")
        a1.set_ylabel("# merge groups")
        a1.set_title("Aggregated partition sizes (all < target)")
        a1.legend(); a1.grid(alpha=0.3)
        a2.bar(["current", "after merge"], [n, n - saved], color=["#C44E52", "#55A868"])
        for i, v in enumerate([n, n - saved]):
            a2.text(i, v, f"{v:,}", ha="center", va="bottom")
        a2.set_ylabel("# partitions")
        a2.set_title(f"{name}: partition count, {100 * saved / n:.0f}% reducible")
        a2.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        plt.show()

    return maximal, total, leaves


def aggregation_report(name, cat_dir, thresh_mb, show=10, cache_dir=None):
    """Aggregation depth (HEALPix levels each merge collapses) + a before/after
    partition-size figure, in exact arrow in-memory MB. Equivalent to iterating
    single-level merges to convergence; depth >= 2 are the 'aggregate twice or
    more' cases. Returns (maximal, depth).
    """
    order, npix, mb = partition_arrow_mb(cat_dir, cache_dir=cache_dir)
    total, leaves, hi, _ = rollup(order, npix, mb)
    maximal = maximal_merge_groups(total, leaves, thresh_mb)
    depth = {k: hi[k] - k[0] for k in maximal}  # HEALPix levels collapsed

    # Before/after partition-size arrays: merged groups + untouched leaves.
    maxset = set(maximal)
    after = [total[k] for k in maximal]
    for o, p, m in zip(order.astype(int), npix.astype(int), mb):
        oo, pp, anc = o, p, None
        while oo > 0:
            oo, pp = oo - 1, pp >> 2
            if (oo, pp) in maxset:
                anc = True
                break
        if anc is None:
            after.append(m)
    after = np.array(after)

    dc = Counter(depth.values())
    twice = sum(v for d, v in dc.items() if d >= 2)
    print(f"{name}: aggregation depth (HEALPix levels collapsed):")
    for d in sorted(dc):
        print(f"   {d} level(s): {dc[d]:,} groups")
    print(f"   -> {twice:,} groups aggregate >=2 levels ('twice or more')")
    print("\n   deepest examples (parent order/npix : depth, #parts -> 1, size):")
    for o, p in sorted(maximal, key=lambda k: (depth[k], leaves[k]), reverse=True)[:show]:
        print(f"     order {o:>2d} npix {p:>9d} : depth {depth[(o, p)]}  {leaves[(o, p)]:>4d} -> 1  "
              f"({total[(o, p)]:5.0f} MB, child orders up to {hi[(o, p)]})")

    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.logspace(np.log10(min(mb.min(), after.min())),
                       np.log10(max(mb.max(), after.max())), 45)
    ax.hist(mb, bins=bins, histtype="step", lw=2.4, color="#C44E52",
            label=f"before  ({len(mb):,} parts, median {np.median(mb):.0f} MB)")
    ax.hist(after, bins=bins, histtype="step", lw=2.4, color="#55A868",
            label=f"after   ({len(after):,} parts, median {np.median(after):.0f} MB)")
    ax.axvline(thresh_mb, color="k", ls="--", lw=1.6, alpha=0.7, label=f"{thresh_mb:.0f} MB target")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Partition size (MB, arrow in-memory)")
    ax.set_ylabel("# partitions")
    ax.set_title(f"{name}: partition sizes before vs after HEALPix aggregation (< {thresh_mb:.0f} MB)",
                 fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return maximal, depth


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #

def plot_partition_hist(name, cat_dir, bins=30, color="#4C72B0", cache_dir=None):
    """Histogram of ON-DISK parquet bytes per partition."""
    _, _, size = partition_sizes(cat_dir, cache_dir=cache_dir)
    mb = size / 1e6

    fig, ax = plt.subplots(figsize=(10, 6))
    counts, edges, _ = ax.hist(mb, bins=bins, color=color, edgecolor="white")
    ax.set_xlabel("Partition file size (MB)")
    ax.set_ylabel("Number of partitions")
    ax.set_title(
        f"{name}: partition size distribution\n"
        f"{len(mb):,} partitions, {file_size(int(size.sum()))} total, "
        f"median {np.median(mb):.0f} MB",
        fontsize=13,
    )
    ax.grid(axis="y", alpha=0.3)
    if len(counts):
        peak = counts.argmax()
        ax.annotate(
            f"{int(counts[peak]):,} partitions",
            xy=((edges[peak] + edges[peak + 1]) / 2, counts[peak]),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=10, color="#333333",
        )
    fig.tight_layout()
    plt.show()


def plot_extremes_on_sky(name, cat_dir, n=20, cache_dir=None):
    """Scatter the n smallest / n largest partitions (on-disk bytes) over the
    catalog footprint, on RA/Dec axes centred on the footprint (RA re-centred via
    the circular mean, so regions straddling RA = 0 stay contiguous)."""
    order, npix, size = partition_sizes(cat_dir, cache_dir=cache_dir)
    idx = np.argsort(size)
    small, large = idx[:n], idx[-n:]

    ra, dec = centers_deg(order, npix)
    ra0 = np.degrees(np.arctan2(np.sin(np.radians(ra)).mean(),
                                np.cos(np.radians(ra)).mean())) % 360.0
    dra = ((ra - ra0 + 180) % 360) - 180

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(dra, dec, s=14, c="#BBBBBB", zorder=2,
               label=f"other partitions ({len(size):,} total)")
    ax.scatter(dra[small], dec[small], s=70, c="#4C72B0", edgecolor="white", lw=0.4,
               zorder=3, label=f"{n} smallest  (≤ {file_size(int(size[small].max()))})")
    ax.scatter(dra[large], dec[large], s=80, c="#C44E52", marker="D", edgecolor="white",
               lw=0.4, zorder=4, label=f"{n} largest  (≥ {file_size(int(size[large].min()))})")
    ax.set_xlabel(f"RA − {ra0:.0f}°  [deg]")
    ax.set_ylabel("Dec  [deg]")
    ax.set_title(f"{name}: partition-size extremes on sky  ({len(size):,} partitions)")
    ax.invert_xaxis()  # RA increases to the left, as on sky
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    plt.show()


def plot_partition_map(name, cat_dir, by="rows", cmap="viridis", mark=None, cache_dir=None):
    """Tile map of a HATS catalog coloured by per-partition row count or size (MB,
    parquet-uncompressed). Small footprints are auto-zoomed (TAN projection);
    full-sky catalogs use Mollweide. `mark=(order, npix)` overlays that HEALPix
    pixel as a red tile to locate a region of interest.
    """
    order, npix, rows, mb = partition_frame(cat_dir, cache_dir=cache_dir)
    if by == "rows" and np.isfinite(rows).any():
        values, unit = rows, "rows / partition"
    else:
        values, unit = mb, "MB / partition"

    cen, fov = footprint_zoom(order, npix)
    if cen is None:
        fig, ax = plot_healpix_map(values, ipix=npix, depth=order, projection="MOL", cmap=cmap)
    else:
        fig, ax = plot_healpix_map(values, ipix=npix, depth=order, projection="TAN",
                                   center=cen, fov=fov, cmap=cmap)
    ax.set_title(f"{name}: {unit}   ({len(values):,} partitions, orders "
                 f"{order.min()}-{order.max()})", pad=18)

    if mark is not None:
        mo, mp = mark
        tr = ax.get_transform("world")
        vlon, vlat = vertices(np.array([mp], np.uint64), np.uint8(mo))
        vx = np.append(np.array(vlon.deg)[0], np.array(vlon.deg)[0][0])
        vy = np.append(np.array(vlat.deg)[0], np.array(vlat.deg)[0][0])
        ax.fill(vx, vy, transform=tr, facecolor="red", edgecolor="red", lw=2.0,
                alpha=0.6, zorder=10, label=f"merge example (order {mo}, npix {mp})")
        ax.legend(loc="lower right")

    plt.show()
    return order, npix, rows, mb


def plot_size_pdf_overlay(overlay, colors, xscale, bins=40):
    """Overlay per-catalog partition-size PDFs (on-disk MB). Per catalog:
    dotted line = median, dashed line = max/4 - the size a partition would have
    after one more HEALPix split (4 children), where an upper-edge dropoff is
    expected. `overlay` maps name -> array of MB."""
    allmb = np.concatenate(list(overlay.values()))
    if xscale == "log":
        edges = np.logspace(np.log10(allmb.min()), np.log10(allmb.max()), bins)
    else:
        edges = np.linspace(0, allmb.max(), bins)

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, s in overlay.items():
        c = colors[name]
        ax.hist(s, bins=edges, density=True, histtype="step", lw=2.4, color=c,
                label=f"{name}  (n={len(s):,}, median {np.median(s):.0f} MB, max {s.max():.0f} MB)")
        ax.axvline(np.median(s), color=c, ls=":", lw=1.6, alpha=0.8)
        ax.axvline(s.max() / 4, color=c, ls="--", lw=1.8, alpha=0.9)
    ax.set_xscale(xscale)
    ax.set_xlabel("Partition file size (MB, on disk)")
    ax.set_ylabel("Probability density  (per MB)")
    ax.set_title(f"Partition size PDF ({xscale}-x)\n"
                 "dotted = median,   dashed = max/4 (expected HEALPix-split dropoff)",
                 fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    plt.show()


def print_size_table(overlay):
    """Comparison table of on-disk partition sizes. `overlay` maps name -> MB array."""
    print(f"{'catalog':16s} {'n':>9s} {'total':>10s} {'median':>10s} {'mean':>10s} {'max':>10s} {'max/4':>10s}")
    for name, s in overlay.items():
        print(f"{name:16s} {len(s):>9,d} {file_size(int(s.sum() * 1e6)):>10s} "
              f"{np.median(s):>7.0f} MB {s.mean():>7.0f} MB {s.max():>7.0f} MB {s.max() / 4:>7.0f} MB")


def plot_size_by_order(name, cat_dir, cmap="viridis", min_n=5, drop_orders=(), cache_dir=None):
    """Overlay per-HEALPix-order partition-size PDFs (on-disk MB): one figure per
    catalog, log-x and linear-x panels side by side. Dotted lines = per-order
    medians; a fixed-order grid would put them ~4x apart, a size-targeted build
    keeps them flat. Orders with fewer than `min_n` partitions are skipped as too
    noisy."""
    order, _, size = partition_sizes(cat_dir, cache_dir=cache_dir)
    mb = size / 1e6
    keep = ~np.isin(order, list(drop_orders))
    order, mb = order[keep], mb[keep]
    orders = [int(o) for o in np.unique(order) if (order == o).sum() >= min_n]
    cmo = plt.get_cmap(cmap)
    lo, hi = min(orders), max(orders)
    meds = {o: float(np.median(mb[order == o])) for o in orders}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, xscale in zip(axes, ("log", "linear")):
        if xscale == "log":
            bins = np.logspace(np.log10(mb.min()), np.log10(mb.max()), 45)
        else:
            bins = np.linspace(0, mb.max(), 45)

        for o in orders:
            s = mb[order == o]
            c = cmo((o - lo) / max(1, hi - lo))
            ax.hist(s, bins=bins, density=True, histtype="step", lw=2.2, color=c,
                    label=f"order {o}  (n={len(s):,}, median {meds[o]:.0f} MB)")
            ax.axvline(meds[o], color=c, ls=":", lw=1.4, alpha=0.9)
        ax.set_xscale(xscale)
        ax.set_xlabel("Partition file size (MB, on disk)")
        ax.set_ylabel("Probability density (per order)")
        ax.set_title(f"{xscale}-x", fontsize=13)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=11)  # same lines in both panels; label once

    dropped = f"  [orders {sorted(drop_orders)} dropped]" if drop_orders else ""
    fig.suptitle(f"{name}: partition size by HEALPix order{dropped} — "
                 "flat medians ⇒ size-targeted build (÷4 area cancelled by ×4 density)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()

    ratios = ", ".join(f"{meds[orders[i]] / meds[orders[i + 1]]:.1f}x"
                       for i in range(len(orders) - 1))
    print(f"{name}: consecutive median ratios (order o / o+1) = {ratios}   (4.0x = fixed-grid expectation)")


def plot_merge_example(name, cat_dir, thresh_mb, parent=None,
                       want_children=(6, 18), label_below=9, cache_dir=None):
    """Draw all partitions under one mergeable parent as HEALPix tiles, coloured
    by exact in-memory size and labelled with order/npix + size (MB), inside the
    parent-pixel outline. Tiles at order >= `label_below` are left unlabelled
    (too small/crowded). If `parent` (order, npix) is None, auto-pick a
    mixed-order group with `want_children[0..1]` partitions, relaxing gracefully
    if none exists; returns None (with a message) if the catalog has no merge
    groups at all. Returns the chosen `(order, npix)`.
    """
    order, npix, mb = partition_arrow_mb(cat_dir, cache_dir=cache_dir)
    order = order.astype(int); npix = npix.astype(int)

    if parent is None:
        total, leaves, hi, lo = rollup(order, npix, mb)
        maximal = maximal_merge_groups(total, leaves, thresh_mb)
        if not maximal:
            print(f"{name}: no merge groups under {thresh_mb:.0f} MB - nothing to draw.")
            return None
        cand = [k for k in maximal
                if want_children[0] <= leaves[k] <= want_children[1] and hi[k] > lo[k]]
        if not cand:  # small catalogs may lack big mixed-order groups - relax
            cand = [k for k in maximal if hi[k] > lo[k]] or list(maximal)
        # most distinct child orders, then fullest
        cand.sort(key=lambda k: (len(np.unique(children_under(order, npix, mb, *k)[0])),
                                 total[k]), reverse=True)
        parent = cand[0]

    po, pp = parent
    co, cp, cm = children_under(order, npix, mb, po, pp)
    merged = cm.sum()

    lon0, lat0 = healpix_to_lonlat(np.array([pp], np.uint64), np.uint8(po))
    lon0 = float(lon0.deg[0]); lat0 = float(lat0.deg[0]); cosd = np.cos(np.radians(lat0))

    def poly_xy(o, p):
        lon, lat = vertices(np.array([p], np.uint64), np.uint8(o))
        lon = np.array(lon.deg)[0]; lat = np.array(lat.deg)[0]
        lon = ((lon - lon0 + 180) % 360) - 180
        return np.column_stack([lon * cosd, lat])

    norm = mpl.colors.Normalize(cm.min(), cm.max())
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(12, 11))
    pats = []
    for o, p, m in zip(co, cp, cm):
        xy = poly_xy(o, p); pats.append(Polygon(xy, closed=True))
        if o < label_below:  # deepest tiles are too small/crowded to label
            ax.text(xy[:, 0].mean(), xy[:, 1].mean(), f"o{o}/{p}\n{m:.0f} MB",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if norm(m) < 0.6 else "black")
    pc = PatchCollection(pats, cmap=cmap, norm=norm, edgecolor="white", lw=1.2, alpha=0.95)
    pc.set_array(cm); ax.add_collection(pc)
    ax.add_patch(Polygon(poly_xy(po, pp), closed=True, fill=False, edgecolor="red", lw=3.5))
    fig.colorbar(pc, ax=ax, shrink=0.8).set_label("partition size (MB, arrow in-memory)")
    ax.set_xlabel("Δ RA · cos(Dec)  [deg]")
    ax.set_ylabel("Dec  [deg]")
    ax.set_title(f"{name}: {len(cm)} partitions (orders {co.min()}–{co.max()})\n"
                 f"combinable into one order-{po} tile (npix {pp}) = {merged:.0f} MB "
                 f"in memory (< {thresh_mb:.0f} MB)")
    ax.autoscale_view(); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.tight_layout(); plt.show()
    return parent
