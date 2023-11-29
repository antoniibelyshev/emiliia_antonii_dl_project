from .utils import AbstractDataset

import zipfile
import pyarrow.parquet as pq


class DefaultDataset(AbstractDataset):
    def __init__(self, workdir, filename, zip_filename=None, trunc_dim=35, valid_size=0.2, batch_size=20):
        super().__init__(workdir, filename, zip_filename=zip_filename, trunc_dim=trunc_dim, valid_size=valid_size, batch_size=batch_size)

    def read(self, workdir, filename, zip_filename=None):
        if zip_filename is None:
            return pq.ParquetFile(workdir + filename).read().to_pandas()
        else:
            with zipfile.ZipFile(workdir + zip_filename, "r") as zipf:
                with zipf.open(filename) as train:
                    return pq.ParquetFile(train).read().to_pandas()

    def transform_to_canonical(self, dataset):
        return dataset