import logging
import pathlib
import random
import time

import polars as pl

import uro_cbb.bball_ref as bball_ref


def download_box_score_and_totals(row: pl.DataFrame):
    link = row.select(pl.col("Box Score Link")).item()
    box_score = bball_ref.download_box_score(link)
    team_total = (
        row.select(
            pl.col("Team1").alias("Team"),
            pl.col("Score1").alias("PTS"),
            pl.col("Score2").alias("OPP_PTS"),
            pl.col("Box Score Link").alias("link"),
        )
        .vstack(
            row.select(
                pl.col("Team2").alias("Team"),
                pl.col("Score2").alias("PTS"),
                pl.col("Score1").alias("OPP_PTS"),
                pl.col("Box Score Link").alias("link"),
            )
        )
        .join(
            bball_ref.extract_totals_from_box_score(box_score).with_columns(
                pl.lit(link).alias("link"),
            ),
            on=["link", "PTS"],
            how="inner",
        )
    )
    return box_score, team_total


def download_tournament_game_and_totals(year: int, is_womens: bool = False):
    logging.info(f"Downloading tournament games for {year}")
    if is_womens:
        tournament_games_df = bball_ref.download_womens_basic_tournament_games(year)
        out_dir = pathlib.Path("./data/bball_ref/raw/womens/tournament_games/")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        tournament_games_df = bball_ref.download_basic_tournament_games(year)
        out_dir = pathlib.Path("./data/bball_ref/raw/tournament_games/")
        out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing tournament games for {year}")
    tournament_games_df.write_parquet(
        out_dir / f"tournament_games_{year}.parquet",
    )
    tournament_games_df.write_csv(
        out_dir / f"tournament_games_{year}.csv",
    )

    box_scores = []
    team_totals = []
    # For each game download the box score and extract the totals
    logging.info(f"Downloading box scores for {year} tournament games")
    for i in range(len(tournament_games_df)):
        row = tournament_games_df[i]
        box_score, team_total = download_box_score_and_totals(row)
        box_scores.append(box_score)
        team_totals.append(team_total)
        time.sleep(random.randint(2, 5))

    team_totals_df = (
        pl.concat(team_totals)
        .select(
            pl.exclude(["link", "Player", "GmSc", "FG%", "2P%", "3P%", "FT%"]),
        )
        .with_columns(pl.lit(1).alias("G"))
        .group_by(["Team"])
        .agg(pl.all().sum())
        .sort("G")
    )
    if is_womens:
        out_dir = pathlib.Path("./data/bball_ref/raw/womens/tournament_totals/")
    else:
        out_dir = pathlib.Path("./data/bball_ref/raw/tournament_totals/")
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing team totals for {year}")
    team_totals_df.write_parquet(
        out_dir / f"tournament_team_total_stats_{year}.parquet",
    )
    team_totals_df.write_csv(
        out_dir / f"tournament_team_total_stats_{year}.csv",
    )
    return team_totals_df


def download_advanced_stats(year: int):
    assert year != 2020, "COVID cancelled the 2020 tournament"
    assert year != 2017, "WIP: No archived advanced stats for 2017"
    logging.info(f"Downloading advanced stats for {year}")

    advanced_stats_df = (
        bball_ref.download_archive_basketball_reference_advanced_stats_data(year)
    )
    out_dir = pathlib.Path("./data/bball_ref/raw/advanced_stats/")
    out_dir.mkdir(parents=True, exist_ok=True)
    advanced_stats_df.write_parquet(
        out_dir / f"advanced_stats_{year}.parquet",
    )
    advanced_stats_df.write_csv(
        out_dir / f"advanced_stats_{year}.csv",
    )
    return advanced_stats_df


def download_basic_stats(year: int):
    logging.info(f"Downloading basic stats for {year}")
    basic_stats_df = bball_ref.download_basketball_reference_stats_data(year)

    out_dir = pathlib.Path("./data/bball_ref/raw/basic_stats/")
    out_dir.mkdir(parents=True, exist_ok=True)
    basic_stats_df.write_parquet(
        out_dir / f"basic_stats_{year}.parquet",
    )
    basic_stats_df.write_csv(
        out_dir / f"basic_stats_{year}.csv",
    )
    return basic_stats_df


def download_basic_opponent_stats(year: int):
    logging.info(f"Downloading basic opponent stats for {year}")
    basic_stats_df = bball_ref.download_basketball_reference_stats_data(year)
    basic_stats_df = bball_ref.remove_tournament_games(
        basic_stats_df,
        pl.read_parquet(
            pathlib.Path(
                f"./data/bball_ref/raw/tournament_games/tournament_games_{year}.parquet"
            )
        ),
    )
    out_dir = pathlib.Path("./data/bball_ref/raw/basic_opponent_stats/")
    out_dir.mkdir(parents=True, exist_ok=True)
    basic_stats_df.write_parquet(
        out_dir / f"basic_opponent_stats_{year}.parquet",
    )
    basic_stats_df.write_csv(
        out_dir / f"basic_opponent_stats_{year}.csv",
    )
    return basic_stats_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logging.info("Downloading Basketball Reference Data")
    # Download tournament games
    for year in range(2017, 2014, -1):
        if year == 2020:
            continue
        download_tournament_game_and_totals(year, is_womens=True)
        # Rudimentary rate limiting
        time.sleep(random.randint(5, 10))

    # Download advanced stats
    # for year in range(2025, 2014, -1):
    #     # Skip 2020 because it was cancelled due to COVID
    #     if year == 2020 or year == 2017:
    #         continue
    #     download_advanced_stats(year)
    #     # Rudimentary rate limiting
    #     time.sleep(random.randint(10, 20))

    # Download basic stats
    # module_dir = pathlib.Path(__file__).parent
    # for year in range(2015, 2025):
    #     if year == 2020:
    #         continue
    #     future_stats_df = download_basic_stats(year)
    #     tournament_totals_df = pl.read_parquet(
    #         pathlib.Path(
    #             module_dir.parent
    #             / f"data/bball_ref/raw/tournament_totals/tournament_team_total_stats_{year}.parquet"
    #         )
    #     )
    #     future_info_removed_df = bball_ref.remove_post_season_games(
    #         tournament_totals_df, future_stats_df
    #     )
    #     out_dir = module_dir.parent / "data/bball_ref/raw/basic_stats/"
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     future_info_removed_df.write_parquet(
    #         out_dir / f"basic_stats_{year}.parquet",
    #     )
    #     future_info_removed_df.write_csv(
    #         out_dir / f"basic_stats_{year}.csv",
    #     )
    #     # Rudimentary rate limiting
    #     time.sleep(random.randint(5, 10))
