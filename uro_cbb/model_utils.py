import numpy as np
import polars as pl
import pydantic


class ModelResults(pydantic.BaseModel):
    brier_score_mean: float
    brier_score_mean_with_perfect: float
    log_loss_mean: float
    log_loss_mean_with_perfect: float


def compute_log_loss(
    data_df: pl.DataFrame,
    actual_col: str = "Result",
    pred_col: str = "pred",
    cap: float = 0.999,
):
    return (
        data_df.with_columns(
            min_prob=pl.lit(1 - cap),
            max_prob=pl.lit(cap),
        )
        .with_columns(
            pred_capped=pl.min_horizontal(
                pl.max_horizontal(pred_col, "min_prob"), "max_prob"
            )
        )
        .with_columns(
            log_loss=(
                -pl.col(actual_col) * (pl.col("pred_capped").log())
                - (1 - pl.col(actual_col)) * (1 - pl.col("pred_capped")).log()
            )
        )
        .select(
            pl.exclude(
                [
                    "min_prob",
                    "max_prob",
                    "pred_capped",
                ]
            )
        )
    )


def compute_brier_score(
    data_df: pl.DataFrame,
    actual_col: str = "Result",
    pred_col: str = "pred",
):
    return data_df.with_columns(
        brier_score=(pl.col(actual_col) - pl.col(pred_col)) ** 2
    )


def evaluate_model(
    test_df: pl.DataFrame,
    actual_col: str = "Result",
    pred_col: str = "pred",
    cap: float = 0.999,
):
    brier_score = compute_brier_score(test_df, actual_col, pred_col)
    log_loss = compute_log_loss(test_df, actual_col, pred_col, cap)
    brier_score_mean = brier_score.select(pl.col("brier_score").mean()).item()
    brier_score_mean_with_perfect = brier_score_mean - (1 - 0.5) ** 2 / len(test_df)
    log_loss_mean = log_loss.select(pl.col("log_loss").mean()).item()
    log_loss_mean_with_perfect = log_loss_mean + (np.log(0.5) / len(test_df))
    return ModelResults(
        brier_score_mean=brier_score_mean,
        brier_score_mean_with_perfect=brier_score_mean_with_perfect,
        log_loss_mean=log_loss_mean,
        log_loss_mean_with_perfect=log_loss_mean_with_perfect,
    )


if __name__ == "__main__":
    pass
