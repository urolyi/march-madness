import logging
import pathlib
import time

import uro_cbb.barttorvik as barttorvik


def download_barttorvik_data(year: int):
    logging.info(f"Downloading Mens Barttorvik Data for {year}")
    data = barttorvik.download_barttorvik_data(year)
    out_dir = pathlib.Path("./data/barttorvik/raw/mens/")
    out_dir.mkdir(parents=True, exist_ok=True)
    data.write_parquet(out_dir / f"barttorvik_{year}.parquet")
    data.write_csv(out_dir / f"barttorvik_{year}.csv")


def download_womens_barttorvik_data(year: int):
    logging.info(f"Downloading Womens Barttorvik Data for {year}")
    data = barttorvik.download_womens_barttorvik_data(year)
    out_dir = pathlib.Path("./data/barttorvik/raw/womens/")
    out_dir.mkdir(parents=True, exist_ok=True)
    data.write_parquet(out_dir / f"barttorvik_{year}.parquet")
    data.write_csv(out_dir / f"barttorvik_{year}.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for year in range(2017, 2014, -1):
        if year == 2020:
            continue
        download_barttorvik_data(year)
        time.sleep(2)
        download_womens_barttorvik_data(year)
        time.sleep(2)
