import logging
import pathlib
import re

import polars as pl


def _merge_with_kaggle_names(
    df: pl.DataFrame, kaggle_names_df: pl.DataFrame, team_col: str
) -> pl.DataFrame:
    kaggle_names_df = kaggle_names_df.with_columns(
        pl.col("Team").alias("__KAGGLE_TEAM_SPELLING_VARIATION"),
        pl.col("TeamID").alias("__KAGGLE_TEAM_ID"),
        pl.col("TeamName").alias("__KAGGLE_TEAM_NAME"),
    ).select(pl.exclude("Team", "TeamID", "TeamName"))
    processed_df = (
        df.with_columns(
            pl.col(team_col).str.to_lowercase().alias(f"{team_col}_lower"),
        )
        .join(
            kaggle_names_df,
            left_on=f"{team_col}_lower",
            right_on="__KAGGLE_TEAM_SPELLING_VARIATION",
            how="inner",
        )
        .with_columns(
            pl.col("__KAGGLE_TEAM_NAME").alias(team_col),
            pl.col("__KAGGLE_TEAM_ID").alias(f"{team_col}_id"),
        )
        .select(
            pl.exclude(
                [
                    f"{team_col}_lower",
                    "__KAGGLE_TEAM_SPELLING_VARIATION",
                    "__KAGGLE_TEAM_ID",
                    "__KAGGLE_TEAM_NAME",
                ]
            )
        )
    )
    if processed_df.shape[0] != df.shape[0]:
        df_names = set(df.select(pl.col(team_col)).to_series())
        kaggle_names = set(
            kaggle_names_df.select(
                pl.col("__KAGGLE_TEAM_SPELLING_VARIATION")
            ).to_series()
        )
        missing_names = df_names - kaggle_names
        logging.error(f"Missing names: {missing_names}")
    return processed_df


def preprocess_data(year: int):
    logging.info(f"Preprocessing data for {year}")
    module_dir = pathlib.Path(__file__).parent.absolute()
    kaggle_names_df = pl.read_csv(
        module_dir.parent / "data/kaggle_2025/raw/MTeamSpellings.csv",
        schema={"Team": pl.Utf8, "TeamID": pl.Int16},
    ).join(
        pl.read_csv(module_dir.parent / "data/kaggle_2025/raw/MTeams.csv").select(
            pl.col("TeamID"), pl.col("TeamName")
        ),
        on="TeamID",
    )

    games_df = (
        pl.read_parquet(
            module_dir.parent
            / f"data/bball_ref/raw/tournament_games/tournament_games_{year}.parquet"
        )
        .with_columns(
            (pl.col("Score1") > pl.col("Score2")).cast(pl.Int8).alias("Result"),
        )
        .pipe(_merge_with_kaggle_names, kaggle_names_df, "Team1")
        .pipe(_merge_with_kaggle_names, kaggle_names_df, "Team2")
    )

    barttorvik_df = (
        pl.read_parquet(
            module_dir.parent / f"data/barttorvik/raw/mens/barttorvik_{year}.parquet"
        )
        .with_columns(pl.col("team").str.to_lowercase().alias("team"))
        .pipe(_merge_with_kaggle_names, kaggle_names_df, "team")
        # .select(pl.exclude("team"))
    )

    # advanced_stats_df = pl.read_parquet(
    #     pathlib.Path(
    #         f"./data/bball_ref/raw/advanced_stats/advanced_stats_{year}.parquet"
    #     )
    # )

    basic_stats_df = (
        pl.read_parquet(
            module_dir.parent
            / f"data/bball_ref/raw/basic_stats/basic_stats_{year}.parquet"
        )
        .with_columns(pl.col("School").str.to_lowercase().alias("School"))
        .pipe(_merge_with_kaggle_names, kaggle_names_df, "School")
    )

    processed_df = (
        games_df.join(
            barttorvik_df,
            left_on=["Team1_id"],
            right_on=["team_id"],
            suffix="_1",
            how="left",
        )
        .join(
            barttorvik_df,
            left_on=["Team2_id"],
            right_on=["team_id"],
            suffix="_2",
            how="left",
        )
        .join(
            basic_stats_df,
            left_on=["Team1_id"],
            right_on=["School_id"],
            suffix="_1",
            how="left",
        )
        .join(
            basic_stats_df,
            left_on=["Team2_id"],
            right_on=["School_id"],
            suffix="_2",
            how="left",
        )
    )
    assert len(processed_df) == len(games_df), "Merge dropped rows"
    if year != 2017:
        kenpom_df = (
            pl.read_parquet(
                module_dir.parent / f"data/kenpom/raw/kenpom_{year}.parquet"
            )
            .with_columns(pl.col("Team").str.to_lowercase().alias("Team"))
            .pipe(_merge_with_kaggle_names, kaggle_names_df, "Team")
        )
        processed_df = processed_df.join(
            kenpom_df,
            left_on=["Team1_id"],
            right_on=["Team_id"],
            suffix="_1",
            how="left",
        ).join(
            kenpom_df,
            left_on=["Team2_id"],
            right_on=["Team_id"],
            suffix="_2",
            how="left",
        )
    return processed_df


def _clean_womens_team_name(team_name: str) -> str:
    # Remove seed information, checkmarks, and other non-team name content
    # Pattern looks for things like "10 seed, ✅" or "(H) 115 Northern Iowa"
    cleaned = re.sub(
        r"\s+\d+\s+seed,\s+[✅❌]|\s+\(.\)\s+\d+.*|\s+\(won\)", "", team_name
    )
    return cleaned.strip()


def preprocess_womens_data(year: int):
    logging.info(f"Preprocessing data for {year}")
    module_dir = pathlib.Path(__file__).parent.absolute()
    kaggle_names_df = pl.read_csv(
        module_dir.parent / "data/kaggle_2025/raw/WTeamSpellings.csv",
        schema={"Team": pl.Utf8, "TeamID": pl.Int16},
    ).join(
        pl.read_csv(module_dir.parent / "data/kaggle_2025/raw/WTeams.csv").select(
            pl.col("TeamID"), pl.col("TeamName")
        ),
        on="TeamID",
    )
    barttorvik_df = (
        pl.read_parquet(
            module_dir.parent / f"data/barttorvik/raw/womens/barttorvik_{year}.parquet"
        )
        .with_columns(
            pl.col("team")
            .map_elements(_clean_womens_team_name, return_dtype=str)
            .str.to_lowercase()
            .alias("team"),
        )
        .pipe(_merge_with_kaggle_names, kaggle_names_df, "team")
        # .select(pl.exclude("team"))
    )
    games_df = (
        pl.read_parquet(
            module_dir.parent
            / f"data/bball_ref/raw/womens/tournament_games/tournament_games_{year}.parquet"
        )
        .with_columns(
            (pl.col("Score1") > pl.col("Score2")).cast(pl.Int8).alias("Result"),
        )
        .pipe(_merge_with_kaggle_names, kaggle_names_df, "Team1")
        .pipe(_merge_with_kaggle_names, kaggle_names_df, "Team2")
    )
    processed_df = games_df.join(
        barttorvik_df,
        left_on=["Team1_id"],
        right_on=["team_id"],
        suffix="_1",
        how="left",
    ).join(
        barttorvik_df,
        left_on=["Team2_id"],
        right_on=["team_id"],
        suffix="_2",
        how="left",
    )
    return processed_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Load tournament games
    # df = preprocess_data(2024)
    df = preprocess_womens_data(2024)
    print(df)
