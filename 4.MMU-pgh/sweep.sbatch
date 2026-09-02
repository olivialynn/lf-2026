#!/bin/bash
#SBATCH --account=rubin:default
#SBATCH --partition=roma
#SBATCH --qos=preemptable
# roma mixes EL8 and EL9 nodes; the lsst-scipipe conda env is built for EL9.
#SBATCH --constraint=OS_VER:9.6
#SBATCH --job-name=mmu-stream-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=reports/sweep-%j.out
#SBATCH --error=reports/sweep-%j.err

# Worker-scaling sweep for InfiniteStream over hf://.
# Submit from the project directory:   sbatch sweep.sbatch

set -euo pipefail

PROJECT=/sdf/home/o/olynn/lf-2026/4.MMU-pgh

# The LSST conda python, called by absolute path. No conda activation needed, and
# it still picks up the pip --user installs in ~/.local/lib/python3.13.
PYTHON=/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python3

# cd into the project so sys.path[0] is this directory. Running from $HOME would
# let the ~/lsdb source checkout shadow the installed package.
cd "$PROJECT"
mkdir -p reports

# Keep dask's spill files on node-local scratch, not on the shared home filesystem.
export TMPDIR="${SLURM_TMPDIR:-/lscratch/$USER/slurm-${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"

# Threaded BLAS inside dask workers oversubscribes the allocation; one thread each.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "node    : $(hostname)"
echo "job     : ${SLURM_JOB_ID}"
echo "cpus    : ${SLURM_CPUS_PER_TASK}"
echo "mem     : ${SLURM_MEM_PER_NODE} MB"
echo "tmpdir  : ${TMPDIR}"
echo

"$PYTHON" bench_stream.py \
    --workers 1 2 4 8 16 \
    --iters 5 \
    --out "reports/sweep-${SLURM_JOB_ID}"

echo
echo "results in reports/sweep-${SLURM_JOB_ID}/"

# Clean up node-local scratch; dask spill files can be large.
rm -rf "$TMPDIR"
