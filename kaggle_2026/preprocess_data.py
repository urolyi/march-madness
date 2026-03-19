import logging
import pathlib

from uro_cbb.preprocess import preprocess_data as _preprocess_data
from uro_cbb.preprocess import preprocess_womens_data as _preprocess_womens_data

MODULE_DIR = pathlib.Path(__file__).parent.absolute()
DATA_DIR = MODULE_DIR.parent / "data"
KAGGLE_DATA_DIR = MODULE_DIR.parent / "data/kaggle_2026/raw"


def preprocess_data(year: int):
    return _preprocess_data(year, data_dir=DATA_DIR, kaggle_data_dir=KAGGLE_DATA_DIR)


def preprocess_womens_data(year: int):
    return _preprocess_womens_data(
        year, data_dir=DATA_DIR, kaggle_data_dir=KAGGLE_DATA_DIR
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = preprocess_womens_data(2026)
    print(df)
