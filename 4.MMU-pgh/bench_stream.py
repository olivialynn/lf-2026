#!/usr/bin/env python3
"""Worker-scaling benchmark for lsdb's InfiniteStream over hf://.

Runs one sweep point per worker count, holding one partition per worker so the
measurement isolates per-worker throughput. Each point gets a fresh LocalCluster,
since n_workers cannot be changed on a live cluster.

A raw single-connection byte read is taken before and after the sweep. That is the
control: Hugging Face throughput varies between sessions (we have measured a 1.6x
spread), so sweep points are only comparable against a rate measured the same day.

Outputs, written incrementally so a killed job still leaves usable data:
    <out>/chunks_w<N>.csv   per-chunk timings for each point
    <out>/summary.csv       one row per point
    <out>/meta.json         host, versions, both control readings
"""

import argparse
import json
import os
import threading
import urllib.request
import platform
import socket
import sys
from datetime import datetime, timezone
from time import perf_counter

import fsspec
import pandas as pd
from dask.distributed import Client, LocalCluster, wait

import lsdb
from lsdb.streams import InfiniteStream

CATALOG_URL = "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north/"

# A known partition file, used only for the raw-read control.
PARTITION_URL = (
    "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north/"
    "mmu_ssl_legacysurvey_north/dataset/Norder=4/Dir=0/Npix=1005.parquet"
)

# Measured from this catalog's parquet footers: 50.74 MB compressed against
# 56.62 MB uncompressed for one row group. Image floats barely compress, so
# decoded bytes proxy wire bytes to about 11%.
WIRE_RATIO = 0.896

# HF's own speed-test object, the same one `hf speedtest` pulls. It is a synthetic,
# edge-cached blob, so it measures the health of the network path -- NOT what a real
# repo file will give you. Measured side by side on one node: 137 MB/s single stream
# against 24 MB/s for an actual dataset parquet. Record both; read the gap as "how
# much of the path is available" versus "what this repo's objects actually serve".
CDN_PROBE_URL = "https://aws.cdn.hf.co/fast/5gb"


def log(msg):
    """Print with a timestamp and flush, so the slurm .out follows the job live."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _stream_bytes(url, nbytes, warm_bytes, byte_range=None):
    """Continuous GET; discard `warm_bytes` of TCP ramp, then time `nbytes`."""
    headers = {"User-Agent": "curl/8"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    r = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)
    try:
        got = 0
        while got < warm_bytes:
            b = r.read(1 << 20)
            if not b:
                break
            got += len(b)
        t0 = perf_counter()
        got = 0
        while got < nbytes:
            b = r.read(1 << 20)
            if not b:
                break
            got += len(b)
        return got, perf_counter() - t0
    finally:
        r.close()


def cdn_control(streams, per_stream_bytes, warm_bytes):
    """Aggregate MB/s pulling HF's speed-test object over `streams` connections.

    This is the path-health control. A low number here means the network or the CDN
    is the problem; a healthy number with slow sweep points means the limit is in how
    this repo's objects are served, which no amount of client tuning will fix.
    """
    out = [None] * streams

    def work(i):
        out[i] = _stream_bytes(CDN_PROBE_URL, per_stream_bytes, warm_bytes)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(streams)]
    t0 = perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = perf_counter() - t0
    return sum(n for n, _ in out) / 1e6 / wall


def raw_read_control(warmup_bytes, read_bytes):
    """Single-connection raw byte read, no parquet decode.

    The first read pays for redirect resolution to the CDN, the TLS handshake and
    the initial block fetch -- around 10x slower than steady state -- so it is
    burned before the clock starts.
    """
    with fsspec.open(PARTITION_URL).open() as f:
        t0 = perf_counter()
        f.read(warmup_bytes)
        warmup_s = perf_counter() - t0

        t0 = perf_counter()
        buf = f.read(read_bytes)
        raw_s = perf_counter() - t0

    mb = len(buf) / 1e6
    return {
        "warmup_mbs": (warmup_bytes / 1e6) / warmup_s,
        "raw_mbs": mb / raw_s,
        "raw_mb": mb,
        "raw_s": raw_s,
    }


def run_stream_benchmark(catalog, client, n_iters, partitions_per_chunk, seed):
    """Pull n_iters chunks, timing each iteration's phases separately.

    Returns one row per chunk:
      wait_s  -- blocked on the future (phase A); unhidden worker-side latency
      post_s  -- shuffle + submit (phases B and C); main-thread dead time
      nbytes  -- decoded in-memory size of the chunk
    """
    stream = InfiniteStream(
        catalog=catalog,
        client=client,
        partitions_per_chunk=partitions_per_chunk,
        seed=seed,
    )
    # iter() submits chunk 0, so chunk 0's clock effectively starts on this line.
    stream_iter = iter(stream)

    records = []
    for k in range(n_iters):
        t0 = perf_counter()
        wait(stream_iter.future)  # blocks without consuming the future
        t1 = perf_counter()
        chunk = next(stream_iter)  # result() already satisfied; shuffle, then submit
        t2 = perf_counter()

        records.append(
            {
                "chunk": k,
                "wait_s": t1 - t0,
                "post_s": t2 - t1,
                "rows": len(chunk),
                # deep=True accounts the Arrow buffers behind the nested `image`
                # column, so this is real bytes rather than a pointer count.
                "nbytes": int(chunk.memory_usage(deep=True).sum()),
            }
        )
        log(
            f"    chunk {k}: wait {t1 - t0:6.2f}s  post {t2 - t1:5.2f}s  "
            f"{records[-1]['nbytes'] / 1e6:7.1f} MB  {records[-1]['rows']:6d} rows"
        )
        # Drop the chunk before the next one lands. Without this, two large frames
        # are briefly alive at once at the moment next() rebinds the name.
        del chunk

    # InfiniteStream always leaves one chunk in flight. Cancel it, or cluster
    # shutdown has to kill a live task and stalls ~4 s at every sweep point.
    if stream_iter.future is not None:
        stream_iter.future.cancel()

    return pd.DataFrame(records)


def summarize(df):
    """Steady-state metrics for one sweep point, excluding the cold chunk 0."""
    steady = df[df["chunk"] > 0]
    if steady.empty:
        raise ValueError("need at least 2 chunks: chunk 0 is excluded as cold")

    mb = steady["nbytes"] / 1e6
    wall_s = (steady["wait_s"] + steady["post_s"]).sum()

    return {
        "chunks": len(steady),
        "decoded_mb": mb.sum(),
        "mb_per_chunk": mb.mean(),
        "wall_s": wall_s,
        "s_per_chunk": wall_s / len(steady),
        "wait_s_mean": steady["wait_s"].mean(),
        "post_s_mean": steady["post_s"].mean(),
        "post_pct": 100 * steady["post_s"].sum() / wall_s,
        # end to end, as the stream actually delivers
        "wire_mbs": WIRE_RATIO * mb.sum() / wall_s,
        # during the fetch only -- the ceiling if post_s were overlapped
        "fetch_wire_mbs": WIRE_RATIO * mb.sum() / steady["wait_s"].sum(),
        "rows_s": steady["rows"].sum() / wall_s,
    }


def mem_per_worker(total_mb, n_workers):
    """Split the allocation across workers, leaving headroom for the client."""
    # The client process holds a chunk plus the shuffled copy, so it needs a real
    # share -- reserve 20% rather than dividing the whole allocation among workers.
    return f"{int(total_mb * 0.80 / n_workers)}MB"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="worker counts to sweep (default: 1 2 4 8)")
    ap.add_argument("--iters", type=int, default=5,
                    help="chunks per point, including the discarded cold chunk (default: 5)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="reports/sweep",
                    help="output directory (default: reports/sweep)")
    ap.add_argument("--local-dir", default=None,
                    help="dask scratch dir (default: $TMPDIR, else /tmp)")
    ap.add_argument("--mem-total-mb", type=int, default=None,
                    help="total memory to divide among workers "
                         "(default: $SLURM_MEM_PER_NODE, else 128 GB)")
    ap.add_argument("--warmup-mb", type=int, default=8)
    ap.add_argument("--raw-mb", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    local_dir = args.local_dir or os.environ.get("TMPDIR", "/tmp")
    total_mb = args.mem_total_mb or int(os.environ.get("SLURM_MEM_PER_NODE", 128 * 1024))

    meta = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "executable": sys.executable,
        # ~/lsdb is a source checkout that can shadow the installed package -- record
        # which one actually got imported so a surprising result can be traced.
        "lsdb_file": lsdb.__file__,
        "lsdb_version": getattr(lsdb, "__version__", "unknown"),
        "catalog": CATALOG_URL,
        "workers": args.workers,
        "iters": args.iters,
        "mem_total_mb": total_mb,
        "local_dir": local_dir,
    }
    log(f"host {meta['host']}  job {meta['slurm_job']}  lsdb {meta['lsdb_version']}")
    log(f"lsdb from {meta['lsdb_file']}")
    log(f"{total_mb} MB total, dask scratch in {local_dir}")

    log("opening catalog ...")
    cat = lsdb.open_catalog(CATALOG_URL)
    meta["npartitions"] = int(cat.npartitions)
    log(f"catalog has {cat.npartitions} partitions")

    log("path-health control: HF speed-test object ...")
    meta["cdn_control"] = {
        "url": CDN_PROBE_URL,
        "mbs_1_stream": cdn_control(1, 150 * 1024**2, 16 * 1024**2),
        "mbs_8_streams": cdn_control(8, 60 * 1024**2, 16 * 1024**2),
    }
    log(f"  {meta['cdn_control']['mbs_1_stream']:.1f} MB/s 1 stream, "
        f"{meta['cdn_control']['mbs_8_streams']:.1f} MB/s 8 streams "
        "-- upper bound for the path, not a target for real repo files")

    log("raw-read control (before): real repo file ...")
    meta["control_before"] = raw_read_control(args.warmup_mb * 1024**2, args.raw_mb * 1024**2)
    log(f"  {meta['control_before']['raw_mbs']:.2f} MB/s single connection, no decode")

    rows = []
    for n in args.workers:
        mem = mem_per_worker(total_mb, n)
        log(f"=== {n} worker(s), {n} partition(s) per chunk, {mem} each ===")

        cluster = LocalCluster(
            n_workers=n,
            threads_per_worker=1,
            memory_limit=mem,
            processes=True,
            local_directory=local_dir,
            dashboard_address=None,  # no browser here; avoids port clashes
        )
        client = Client(cluster)
        try:
            df = run_stream_benchmark(cat, client, args.iters, partitions_per_chunk=n, seed=args.seed)
            df.to_csv(f"{args.out}/chunks_w{n}.csv", index=False)

            s = summarize(df)
            s["workers"] = n
            s["partitions_per_chunk"] = n
            rows.append(s)

            log(f"  -> {s['wire_mbs']:6.2f} MB/s end to end,"
                f" {s['fetch_wire_mbs']:6.2f} MB/s during fetch,"
                f" post_s {s['post_pct']:.1f}% of wall")

            # rewrite the summary after every point, so a killed job keeps its results
            pd.DataFrame(rows).to_csv(f"{args.out}/summary.csv", index=False)
        except Exception as exc:  # a failed point should not kill the sweep
            log(f"  !! {n} workers FAILED: {type(exc).__name__}: {exc}")
        finally:
            client.close()
            cluster.close()

    log("raw-read control (after) ...")
    meta["control_after"] = raw_read_control(args.warmup_mb * 1024**2, args.raw_mb * 1024**2)
    log(f"  {meta['control_after']['raw_mbs']:.2f} MB/s single connection, no decode")

    meta["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(f"{args.out}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if rows:
        summary = pd.DataFrame(rows)
        base = summary.loc[summary["workers"] == min(summary["workers"]), "wire_mbs"].iloc[0]
        summary["scaling_vs_min"] = summary["wire_mbs"] / base

        log("")
        log("=== sweep summary ===")
        cols = ["workers", "wire_mbs", "fetch_wire_mbs", "post_pct", "s_per_chunk", "scaling_vs_min"]
        print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"), flush=True)
        log("")
        log(f"repo-file control: before {meta['control_before']['raw_mbs']:.2f} MB/s, "
            f"after {meta['control_after']['raw_mbs']:.2f} MB/s "
            "-- a wide gap means HF drifted mid-sweep and the points are not comparable")
        log(f"path health: {meta['cdn_control']['mbs_8_streams']:.1f} MB/s at 8 streams "
            "on HF's cached probe object -- the network is not the limit below this")
        summary.to_csv(f"{args.out}/summary.csv", index=False)
    else:
        log("no points completed")

    log(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
