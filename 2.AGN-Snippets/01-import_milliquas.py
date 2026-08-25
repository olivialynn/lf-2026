"""Import Milliquas v8 into a HATS collection (catalog + margin + index)."""

import tempfile
from pathlib import Path

from dask.distributed import Client
from hats_import import CollectionArguments
from hats_import.pipeline import pipeline_with_client
 
# Raw catalog downloaded from wherever:
DATA_IN = Path("/Users/orl/code/lsdb-plus/liv-lf/10 - EzTaoX/milliquas_raw")
 
# Where the collection should live. The collection dir itself is created
# under here, named after output_artifact_name.
DATA_OUT = Path("/Users/orl/code/lsdb-plus/liv-lf/10 - EzTaoX/hats")
 
# Any space where Dask can spill:
TMP_PATH = Path("/Users/orl/code/lsdb-plus/tmp")


def build_collection(data_in, data_out, tmp_path, resume=True):
    args = (
        CollectionArguments(
            output_artifact_name="milliquas_v8",
            output_path=data_out,
            tmp_dir=tmp_path,
            resume=resume,
        )
        .catalog(
            input_path=data_in,
            file_reader="fits",
            ra_column="RA",
            dec_column="DEC",
            sort_columns="NAME",
            pixel_threshold=100_000,
            highest_healpix_order=7,
        )
        .add_margin(margin_threshold=10.0, is_default=True)
        .add_index(indexing_column="NAME")
    )
 
    with tempfile.TemporaryDirectory(dir=tmp_path) as dask_tmp:
        with Client(
            n_workers=8, threads_per_worker=1, local_directory=dask_tmp
        ) as client:
            pipeline_with_client(args, client)


if __name__ == "__main__":
    # resume=False wipes intermediates and output for a clean re-run,
    # which is what the commented-out shutil.rmtree was doing by hand.
    build_collection(DATA_IN, DATA_OUT, TMP_PATH, resume=True)
 
