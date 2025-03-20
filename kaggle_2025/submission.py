import pathlib
import pickle

import polars as pl

module_dir = pathlib.Path(__file__).parent.absolute()


def _create_matchup_preds_df(
    model_results,
    team_df,
    kaggle_teams,
) -> pl.DataFrame:
    matchups_df = (
        kaggle_teams.join(
            kaggle_teams,
            how="cross",
            suffix="_2",
        )
        .filter(
            pl.col("TeamID") < pl.col("TeamID_2"),
        )
        .join(
            team_df,
            on=["TeamID"],
            how="left",
        )
        .join(
            team_df.select(pl.exclude("Team")),
            left_on=["TeamID_2"],
            right_on=["TeamID"],
            suffix="_2",
            how="left",
        )
        .fill_null(pl.lit(0))
    )
    return matchups_df.with_columns(
        pred=model_results.predict(matchups_df.to_pandas()).values
    ).select(
        (
            "2025_"
            + pl.col("TeamID").cast(pl.Utf8)
            + "_"
            + pl.col("TeamID_2").cast(pl.Utf8)
        ).alias("ID"),
        pl.col("pred").alias("Pred"),
    )


def main(
    mens_model_path: pathlib.Path = module_dir.parent
    / "kaggle_2025/models/mens_barttorvik_kenpom.pkl",
    womens_model_path: pathlib.Path = module_dir.parent
    / "kaggle_2025/models/womens_barttorvik.pkl",
    mens_data_path: pathlib.Path = module_dir.parent
    / "data/kaggle_2025/cleaned/mens/barttorvik_kenpom.csv",
    womens_data_path: pathlib.Path = module_dir.parent
    / "data/kaggle_2025/cleaned/womens/barttorvik.csv",
    out_path: pathlib.Path = module_dir.parent
    / "data/kaggle_2025/cleaned/submissions/submission_base_mens_with_kenpom.csv",
    # overrides: list[tuple[str, float]] = [],
):
    mens_model_results = pickle.load(open(mens_model_path, "rb"))
    womens_model_results = pickle.load(open(womens_model_path, "rb"))
    mens_team_df = pl.read_csv(mens_data_path)
    womens_team_df = pl.read_csv(womens_data_path)

    mens_kaggle_teams = pl.read_csv(
        module_dir.parent / "data/kaggle_2025/raw/MTeams.csv"
    ).select(pl.col("TeamID"))
    womens_kaggle_teams = pl.read_csv(
        module_dir.parent / "data/kaggle_2025/raw/WTeams.csv"
    ).select(pl.col("TeamID"))

    mens_matchups_df = _create_matchup_preds_df(
        mens_model_results,
        mens_team_df,
        mens_kaggle_teams,
    )
    womens_matchups_df = _create_matchup_preds_df(
        womens_model_results,
        womens_team_df,
        womens_kaggle_teams,
    )
    submission_df = (
        pl.concat([mens_matchups_df, womens_matchups_df])
        .join(
            pl.read_csv(
                module_dir.parent / "data/kaggle_2025/raw/SampleSubmissionStage2.csv"
            ),
            on="ID",
            how="inner",
            suffix="_2",
        )
        .select(
            pl.col("ID"),
            pl.col("Pred"),
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.write_csv(out_path)


if __name__ == "__main__":
    # typer.run(main)
    main()
