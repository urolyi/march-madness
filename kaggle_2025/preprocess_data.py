import logging
import pathlib

import polars as pl

import uro_cbb.bball_ref as bball_ref


def preprocess_data(year: int):
    logging.info(f"Preprocessing data for {year}")

    games_df = pl.read_parquet(
        pathlib.Path(
            f"./data/bball_ref/raw/tournament_games/tournament_games_{year}.parquet"
        )
    )
    games_df = games_df.with_columns(
        pl.col("Team1").map_elements(
            lambda team_name: bball_ref.name_to_kaggle(team_name), return_dtype=str
        ),
        pl.col("Team2").map_elements(
            lambda team_name: bball_ref.name_to_kaggle(team_name), return_dtype=str
        ),
    )

    barttorvik_df = pl.read_parquet(
        pathlib.Path(f"./data/barttorvik/raw/barttorvik_data_{year}.parquet")
    )

    advanced_stats_df = pl.read_parquet(
        pathlib.Path(
            f"./data/bball_ref/raw/advanced_stats/advanced_stats_{year}.parquet"
        )
    )

    basic_stats_df = pl.read_parquet(
        pathlib.Path(f"./data/bball_ref/raw/basic_stats/basic_stats_{year}.parquet")
    )

    if year != 2017:
        kenpom_df = pl.read_parquet(
            pathlib.Path(f"./data/kenpom/raw/kenpom_data_{year}.parquet")
        )
        # merge kenpom_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Load tournament games
