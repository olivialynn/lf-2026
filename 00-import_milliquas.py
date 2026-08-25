import tempfile

from dask.distributed import Client
from hats_import.catalog.arguments import ImportArguments
from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments
import hats_import.pipeline as runner


def main_pipeline(data_in, data_out, tmp_path):
    tmp_dir = tempfile.TemporaryDirectory(dir=tmp_path).name
    with Client(n_workers=8, threads_per_worker=1, local_directory=tmp_dir) as client:
        args = ImportArguments(
            output_artifact_name="Milliquas_v8",
            input_path=data_in,
            file_reader="fits",
            ra_column="RA",
            dec_column="DEC",
            sort_columns="NAME",
            pixel_threshold=1_000_000,
            highest_healpix_order=7,
            output_path=data_out,
        )
        runner(args)


def margin_pipeline(cat_in, margin_out, tmp_path):
    tmp_dir = tempfile.TemporaryDirectory(dir=tmp_path).name
    with Client(n_workers=8, threads_per_worker=1, local_directory=tmp_dir) as client:
        args = MarginCacheArguments(
            input_catalog_path=cat_in,
            output_path=margin_out,
            margin_threshold=10.0,
            output_artifact_name="Milliquas_v8_10arcs",
        )
        runner(args)


if __name__ == "__main__":
    data_in = "/Users/orl/code/lsdb-plus/liv-lf/10 - EzTaoX/milliquas_raw"
    data_out = "/Users/orl/code/lsdb-plus/liv-lf/10 - EzTaoX/milliquas_hats"
    tmp_path = "/Users/orl/code/lsdb-plus/tmp"

    # Remove data_out if it exists
    import shutil
    import os

    # if os.path.exists(data_out):
    #     shutil.rmtree(data_out)

    # main_pipeline(data_in, data_out, tmp_path)

    # cat_in = "/sdf/data/rubin/u/olynn/AGNs/hats/Milliquas_v8"
    # margin_out = "/sdf/data/rubin/u/olynn/AGNs/hats/Milliquas_v8_margin"
    cat_in = data_out + "/Milliquas_v8"
    margin_out = data_out + "_margin"

    margin_pipeline(cat_in, margin_out, tmp_path)
