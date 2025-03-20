import dataclasses
import itertools
import logging
import traceback

import bs4
import pandas as pd
import polars as pl
import pydantic
import requests

import uro_cbb.constants as constants

ADVANCED_STATS_ARCHIVE_URLS = {
    2025: "https://www.sports-reference.com/cbb/seasons/men/2025-advanced-school-stats.html",
    2024: "https://web.archive.org/web/20240320074354/https://www.sports-reference.com/cbb/seasons/men/2024-advanced-school-stats.html",
    2023: "https://web.archive.org/web/20230315164208/https://www.sports-reference.com/cbb/seasons/men/2023-advanced-school-stats.html",
    2022: "https://web.archive.org/web/20220316172443/https://www.sports-reference.com/cbb/seasons/men/2022-advanced-school-stats.html",
    2021: "https://web.archive.org/web/20210315003857/https://www.sports-reference.com/cbb/seasons/2021-advanced-school-stats.html",
    2019: "https://web.archive.org/web/20190314232750/https://www.sports-reference.com/cbb/seasons/2019-advanced-school-stats.html",
    2018: "https://web.archive.org/web/20180314201431/https://www.sports-reference.com/cbb/seasons/2018-advanced-school-stats.html",
    2017: "",  # missing
    2016: "https://web.archive.org/web/20160316031341/http://www.sports-reference.com/cbb/seasons/2016-advanced-school-stats.html",
    2015: "https://web.archive.org/web/20150318205808/http://www.sports-reference.com/cbb/seasons/2015-advanced-school-stats.html",
}

GAME_TEAM_NAME_MAPPING = {
    "UNC": "North Carolina",
    "UMBC": "Maryland-Baltimore County",
    "VCU": "Virginia Commonwealth",
    "Saint Mary's": "Saint Mary's (CA)",
    "ETSU": "East Tennessee State",
    "USC": "Southern California",
    "SMU": "Southern Methodist",
    "St. Joseph's": "Saint Joseph's",
    "St. Peter's": "Saint Peter's",
    "UConn": "Connecticut",
    "Pitt": "Pittsburgh",
    "Ole Miss": "Mississippi",
    "LSU": "Louisiana State",
    "BYU": "Brigham Young",
    "UMass": "Massachusetts",
    "UNLV": "Nevada-Las Vegas",
    "LIU-Brooklyn": "Long Island University",
    "Detroit": "Detroit Mercy",
    "Southern Miss": "Southern Mississippi",
    "Texas A&M;": "Texas A&M",
    "North Carolina A&T;": "North Carolina A&T",
    "UCSB": "UC Santa Barbara",
    "UC-Irvine": "UC Irvine",
    "UC-Davis": "UC Davis",
    "Penn": "Pennsylvania",
}

## SCHEMAS ##
ADVANCED_STATS_SCHEMA = {
    "School": pl.Utf8,
    "G": pl.Int16,
    "W": pl.Int16,
    "L": pl.Int16,
    "W-L%": pl.Float32,
    "SRS": pl.Float32,
    "SOS": pl.Float32,
    "_BLANK1": pl.Utf8,
    "ConfW": pl.Int16,
    "ConfL": pl.Int16,
    "_BLANK2": pl.Utf8,
    "HomeW": pl.Int16,
    "HomeL": pl.Int16,
    "_BLANK3": pl.Utf8,
    "AwayW": pl.Int16,
    "AwayL": pl.Int16,
    "_BLANK4": pl.Utf8,
    "Tm.": pl.Int16,
    "Opp.": pl.Int16,
    "_BLANK5": pl.Utf8,
    "Pace": pl.Float32,
    "ORtg": pl.Float32,
    "FTr": pl.Float32,
    "3PAr": pl.Float32,
    "TS%": pl.Float32,
    "TRB%": pl.Float32,
    "AST%": pl.Float32,
    "STL%": pl.Float32,
    "BLK%": pl.Float32,
    "eFG%": pl.Float32,
    "TOV%": pl.Float32,
    "ORB%": pl.Float32,
    "FT/FGA": pl.Float32,
}

STATS_SCHEMA = {
    "School": pl.Utf8,
    "G": pl.Int16,
    "W": pl.Int16,
    "L": pl.Int16,
    "W-L%": pl.Float32,
    "SRS": pl.Float32,
    "SOS": pl.Float32,
    "_BLANK1": pl.Utf8,
    "ConfW": pl.Int16,
    "ConfL": pl.Int16,
    "_BLANK2": pl.Utf8,
    "HomeW": pl.Int16,
    "HomeL": pl.Int16,
    "_BLANK3": pl.Utf8,
    "AwayW": pl.Int16,
    "AwayL": pl.Int16,
    "_BLANK4": pl.Utf8,
    "Tm.": pl.Int16,
    "Opp.": pl.Int16,
    "_BLANK5": pl.Utf8,
    "MP": pl.Int16,
    "FG": pl.Int16,
    "FGA": pl.Int16,
    "FG%": pl.Float32,
    "3P": pl.Int16,
    "3PA": pl.Int16,
    "3P%": pl.Float32,
    "FT": pl.Int16,
    "FTA": pl.Int16,
    "FT%": pl.Float32,
    "ORB": pl.Int16,
    "TRB": pl.Int16,
    "AST": pl.Int16,
    "STL": pl.Int16,
    "BLK": pl.Int16,
    "TOV": pl.Int16,
    "PF": pl.Int16,
}

BOX_SCORE_SCHEMA = {
    "Player": pl.Utf8,
    "MP": pl.Int16,
    "FG": pl.Int16,
    "FGA": pl.Int16,
    "FG%": pl.Float32,
    "2P": pl.Int16,
    "2PA": pl.Int16,
    "2P%": pl.Float32,
    "3P": pl.Int16,
    "3PA": pl.Int16,
    "3P%": pl.Float32,
    "FT": pl.Int16,
    "FTA": pl.Int16,
    "FT%": pl.Float32,
    "ORB": pl.Int16,
    "DRB": pl.Int16,
    "TRB": pl.Int16,
    "AST": pl.Int16,
    "STL": pl.Int16,
    "BLK": pl.Int16,
    "TOV": pl.Int16,
    "PF": pl.Int16,
    "PTS": pl.Int16,
    "GmSc": pl.Float32,
}

ADVANCED_BOX_SCORE_SCHEMA = {
    "Player": pl.Utf8,
    "MP": pl.Int16,
    "TS%": pl.Float32,
    "eFG%": pl.Float32,
    "3PAr": pl.Float32,
    "FTr": pl.Float32,
    "ORB%": pl.Float32,
    "DRB%": pl.Float32,
    "TRB%": pl.Float32,
    "AST%": pl.Float32,
    "STL%": pl.Float32,
    "BLK%": pl.Float32,
    "TOV%": pl.Float32,
    "USG%": pl.Float32,
    "ORtg": pl.Float32,
    "DRtg": pl.Float32,
    "BPM": pl.Float32,
}


## Dataclasses ##
@pydantic.dataclasses.dataclass(slots=True)
class PostseasonGame:
    team1_name: str
    team2_name: str
    score1: int
    score2: int
    box_score_link: str | None = None


@dataclasses.dataclass(slots=True)
class PostSeasonBoxScore:
    basic_box_score1: pd.DataFrame
    basic_box_score2: pd.DataFrame
    advanced_box_score1: pd.DataFrame
    advanced_box_score2: pd.DataFrame


## Library Functions ##
def create_session_with_retries(
    retries=0,
    backoff_factor=0.1,
    status_forcelist=(500, 502, 503, 504),
    allowed_methods=None,
):
    """
    Creates a requests session with retry capability.

    Args:
        retries: Maximum number of retries
        backoff_factor: Backoff factor for exponential backoff
        status_forcelist: HTTP status codes that should trigger a retry
        allowed_methods: HTTP methods that should be retried

    Returns:
        requests.Session with retry configuration
    """
    if allowed_methods is None:
        allowed_methods = ["HEAD", "GET", "OPTIONS"]

    retry = requests.adapters.Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
    )

    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def try_to_get_contents(data: bs4.element.Tag) -> str | None:
    if hasattr(data, "contents"):
        if data.contents:
            return data.contents[0]
    return None


def _parse_basketball_reference_stats_row(
    potential_row: bs4.element.Tag,
) -> list[str]:
    table_data_elements = potential_row.find_all("td", attrs={"class": "right"})
    # For year <= 2016 table attributes changed a bit
    if not table_data_elements:
        # skip first because rank is now a td element not th
        table_data_elements = potential_row.find_all("td", attrs={"align": "right"})[1:]
    if not table_data_elements:
        return []
    return [try_to_get_contents(table_data) for table_data in table_data_elements]


def _parse_basketball_reference_school(potential_row: bs4.element.Tag) -> str | None:
    return try_to_get_contents(potential_row.find("a"))


def download_archive_basketball_reference_advanced_stats_data(
    year: int,
) -> pd.DataFrame:
    with create_session_with_retries() as session:
        response = session.get(ADVANCED_STATS_ARCHIVE_URLS[year])
        response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser")

    # Goes to the <tbody> section of the code
    table = soup.find("tbody")
    # List of all the <tr> within the <tbody>
    table_rows = table.find_all("tr")

    data_rows = []
    for potential_row in table_rows:
        if parsed_ref_school := _parse_basketball_reference_school(potential_row):
            data_rows.append(
                [parsed_ref_school]
                + _parse_basketball_reference_stats_row(potential_row)
            )
    columns = list(ADVANCED_STATS_SCHEMA.keys())
    if year <= 2019:
        BLANK_COLUMNS = ("_BLANK1", "_BLANK2", "_BLANK3", "_BLANK4")
        if year <= 2016:
            BLANK_COLUMNS += ("_BLANK5",)

        columns = [column for column in columns if column not in BLANK_COLUMNS]
    untyped_df = pd.DataFrame(data_rows, columns=columns)
    return pl.from_pandas(
        untyped_df[[column for column in untyped_df.columns if "_BLANK" not in column]],
        schema_overrides=ADVANCED_STATS_SCHEMA,
    )


def _download_basketball_reference_stats_data(
    stats_url: str,
    year: int,
) -> pd.DataFrame:
    with create_session_with_retries() as session:
        response = session.get(stats_url, headers=constants.HEADERS)
        response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser")

    # Goes to the <tbody> section of the code
    table = soup.find("tbody")
    # List of all the <tr> within the <tbody>
    table_rows = table.find_all("tr")

    data_rows = []
    for potential_row in table_rows:
        if parsed_ref_school := _parse_basketball_reference_school(potential_row):
            data_rows.append(
                [parsed_ref_school]
                + _parse_basketball_reference_stats_row(potential_row)
            )
    columns = list(STATS_SCHEMA.keys())
    untyped_df = pd.DataFrame(data_rows, columns=columns)
    return pl.DataFrame(
        untyped_df[[column for column in untyped_df.columns if "_BLANK" not in column]],
        schema_overrides=STATS_SCHEMA,
    )


def download_basketball_reference_stats_data(year: int) -> pd.DataFrame:
    stats_url = (
        f"https://www.sports-reference.com/cbb/seasons/men/{year}-school-stats.html"
    )
    return _download_basketball_reference_stats_data(stats_url, year)


def download_womens_basketball_reference_stats_data(year: int) -> pd.DataFrame:
    stats_url = (
        f"https://www.sports-reference.com/cbb/seasons/women/{year}-school-stats.html"
    )
    return _download_basketball_reference_stats_data(stats_url, year)


def download_basketball_reference_opponent_stats_data(year: int) -> pd.DataFrame:
    stats_url = (
        f"https://www.sports-reference.com/cbb/seasons/men/{year}-opponent-stats.html"
    )
    return _download_basketball_reference_stats_data(stats_url, year)


def download_womens_basketball_reference_opponent_stats_data(year: int) -> pd.DataFrame:
    stats_url = (
        f"https://www.sports-reference.com/cbb/seasons/women/{year}-opponent-stats.html"
    )
    return _download_basketball_reference_stats_data(stats_url, year)


def _try_to_parse_round(round: bs4.element.Tag) -> list[PostseasonGame]:
    games = []
    for child in round.children:
        if not isinstance(child, bs4.element.Tag):
            continue
        link_elements = child.find_all("a")
        if len(link_elements) < 4:
            continue
        try:
            game = PostseasonGame(
                link_elements[0].contents[0],
                link_elements[2].contents[0],
                int(link_elements[1].contents[0]),
                int(link_elements[3].contents[0]),
                "https://www.sports-reference.com" + link_elements[1].attrs["href"],
            )
            games.append(game)
        except Exception:
            logging.warning(f"Failed to parse postseasongame {child}")
            logging.warning(traceback.format_exc())
            continue
    return games


def download_basic_tournament_games(year: int) -> pd.DataFrame:
    return _download_basic_tournament_games(
        year, f"https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html"
    )


def download_womens_basic_tournament_games(year: int) -> pd.DataFrame:
    return _download_basic_tournament_games(
        year, f"https://www.sports-reference.com/cbb/postseason/women/{year}-ncaa.html"
    )


def _download_basic_tournament_games(year: int, url: str) -> pd.DataFrame:
    """Downloads the postseason bracket results for a given year from postseason/{year}"""
    with create_session_with_retries() as session:
        response = session.get(url)
        response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    rounds = soup.find_all("div", attrs={"class": "round"})

    rounds_parsed = [
        _try_to_parse_round(round) for round in rounds
    ]  # Gives us a list of data [[team1, score1], [team2, score2]]
    games_data = [
        (
            game.team1_name,
            game.team2_name,
            game.score1,
            game.score2,
            game.box_score_link,
        )
        for game in itertools.chain.from_iterable(rounds_parsed)
    ]  # combines all matchups
    # 2021 had one less game because of covid
    # Oregon vs. VCU was declared a no contest
    # if year == 2021 or year == 2023:
    #     assert len(games_data) >= 62, (
    #         f"Expected at least 62 games, got {len(games_data)} games"
    #     )
    # else:
    #     assert len(games_data) == 63, f"Expected 63 games, got {len(games_data)} games"
    if len(games_data) < 63:
        logging.warning(f"Expected 63 games, got {len(games_data)} games")
    untyped_df = pd.DataFrame(
        games_data, columns=["Team1", "Team2", "Score1", "Score2", "Box Score Link"]
    )
    return pl.DataFrame(
        untyped_df,
        schema={
            "Team1": pl.Utf8,
            "Team2": pl.Utf8,
            "Score1": pl.Int16,
            "Score2": pl.Int16,
            "Box Score Link": pl.Utf8,
        },
    )


def _try_to_parse_box_score(
    box_score_element: bs4.element.Tag,
) -> pd.DataFrame | None:
    try:
        table = box_score_element.find("tbody")
        table_header_element = box_score_element.find("thead").find_all("tr")[1]
        table_header = [
            col_header.contents[0]
            for col_header in table_header_element
            if isinstance(col_header, bs4.element.Tag)
        ]
        rows = table.find_all("tr")
        data = []
        for row in rows:
            row_data = [try_to_get_contents(data) for data in row.find_all("td")]
            if row_data:
                player_name = row.find("a").contents[0]
                data.append([player_name] + row_data)
        data.append(
            [
                try_to_get_contents(data)
                for data in box_score_element.find("tfoot").find("tr")
            ]
        )
        return pd.DataFrame(data, columns=table_header)
    except Exception:
        logging.warning(f"Failed to parse basic box score {box_score_element}")
        logging.warning(traceback.format_exc())
        return None


def download_box_score(box_score_link: str) -> PostSeasonBoxScore:
    with create_session_with_retries() as session:
        response = session.get(box_score_link, headers=constants.HEADERS)
        response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    boxscore_elements = soup.select('[id*="all_box"]')
    assert len(boxscore_elements) == 4, (
        f"Expected 4 box scores (basic and advanced for each team), got {len(boxscore_elements)}"
    )
    basic_df1, basic_df2 = (
        _try_to_parse_box_score(boxscore_elements[0]).rename(
            {"Starters": "Player"}, axis=1
        ),
        _try_to_parse_box_score(boxscore_elements[2]).rename(
            {"Starters": "Player"}, axis=1
        ),
    )
    advanced_df1, advanced_df2 = (
        _try_to_parse_box_score(boxscore_elements[1]).rename(
            {"Starters": "Player"}, axis=1
        ),
        _try_to_parse_box_score(boxscore_elements[3]).rename(
            {"Starters": "Player"}, axis=1
        ),
    )
    basic_schema = {column: BOX_SCORE_SCHEMA[column] for column in basic_df1.columns}
    advanced_schema = {
        column: ADVANCED_BOX_SCORE_SCHEMA[column] for column in advanced_df1.columns
    }
    basic_box_score1 = pl.DataFrame(basic_df1, schema=basic_schema)
    basic_box_score2 = pl.DataFrame(basic_df2, schema=basic_schema)
    advanced_box_score1 = pl.DataFrame(advanced_df1, schema=advanced_schema)
    advanced_box_score2 = pl.DataFrame(advanced_df2, schema=advanced_schema)
    return PostSeasonBoxScore(
        basic_box_score1,
        basic_box_score2,
        advanced_box_score1,
        advanced_box_score2,
    )


def extract_totals_from_box_score(box_score_df: pl.DataFrame) -> pl.DataFrame:
    basic1_total = box_score_df.basic_box_score1[-1]
    basic2_total = box_score_df.basic_box_score2[-1]
    return basic1_total.vstack(basic2_total)


def remove_post_season_games(
    tournament_df: pl.DataFrame, stats_df: pl.DataFrame
) -> pl.DataFrame:
    # stacking on any converted names and since we inner join later it will still only include up to one matching row
    tournament_df = tournament_df.with_columns(
        pl.col("Team")
        .map_elements(
            lambda team: GAME_TEAM_NAME_MAPPING.get(team, team), return_dtype=str
        )
        .alias("School")
    ).filter(~pl.col("School").is_null())
    return (
        stats_df.join(tournament_df, on="School", how="inner")
        .with_columns(
            (pl.col("G") - pl.col("G_right")).alias("G"),
            (pl.col("W") - pl.col("G_right") + 1).alias("W"),
            (pl.col("L") - 1).alias("L"),
            (pl.col("Tm.") - pl.col("PTS")).alias("PTS"),
            (pl.col("Opp.") - pl.col("OPP_PTS")).alias("OPP_PTS"),
            (pl.col("MP") - pl.col("MP_right") / 5).alias("MP"),
            (pl.col("FG") - pl.col("FG_right")).alias("FG"),
            (pl.col("FGA") - pl.col("FGA_right")).alias("FGA"),
            (pl.col("3P") - pl.col("3P_right")).alias("3P"),
            (pl.col("3PA") - pl.col("3PA_right")).alias("3PA"),
            (pl.col("FT") - pl.col("FT_right")).alias("FT"),
            (pl.col("FTA") - pl.col("FTA_right")).alias("FTA"),
            (pl.col("ORB") - pl.col("ORB_right")).alias("ORB"),
            (pl.col("TRB") - pl.col("TRB_right")).alias("TRB"),
            (pl.col("AST") - pl.col("AST_right")).alias("AST"),
            (pl.col("STL") - pl.col("STL_right")).alias("STL"),
            (pl.col("BLK") - pl.col("BLK_right")).alias("BLK"),
            (pl.col("TOV") - pl.col("TOV_right")).alias("TOV"),
            (pl.col("PF") - pl.col("PF_right")).alias("PF"),
        )
        .with_columns(
            (pl.col("W") / (pl.col("W") + pl.col("L"))).alias("W-L%"),
            (pl.col("FG") / pl.col("FGA")).alias("FG%"),
            (pl.col("3P") / pl.col("3PA")).alias("3P%"),
            (pl.col("FT") / pl.col("FTA")).alias("FT%"),
        )
        .select(pl.exclude("^.*_right$"))
        .select(
            pl.col("School"),
            pl.col("G"),
            pl.col("W"),
            pl.col("L"),
            pl.col("W-L%"),
            pl.col("PTS"),
            pl.col("OPP_PTS"),
            pl.col("ConfW"),
            pl.col("ConfL"),
            pl.col("HomeW"),
            pl.col("HomeL"),
            pl.col("AwayW"),
            pl.col("AwayL"),
            pl.col("MP"),
            pl.col("FG"),
            pl.col("FGA"),
            pl.col("FG%"),
            pl.col("3P"),
            pl.col("3PA"),
            pl.col("3P%"),
            pl.col("FT"),
            pl.col("FTA"),
            pl.col("FT%"),
            pl.col("ORB"),
            pl.col("TRB"),
            pl.col("AST"),
            pl.col("STL"),
            pl.col("BLK"),
            pl.col("TOV"),
            pl.col("PF"),
        )
    )


if __name__ == "__main__":
    # advanced_stats_df = download_archive_basketball_reference_advanced_stats_data(2015)
    # # print(advanced_stats_df)
    # advanced_stats_df = download_archive_basketball_reference_advanced_stats_data(2024)
    # print(advanced_stats_df)
    # advanced_stats_df = download_archive_basketball_reference_advanced_stats_data(2024)
    # print(advanced_stats_df)
    basic_stats_df = download_basketball_reference_stats_data(2019)
    print(basic_stats_df)
    # postseason_games_df = download_basic_tournament_games(2021)
    # postseason_box_score = download_box_score(
    #     "https://www.sports-reference.com/cbb/boxscores/2022-03-17-14-baylor.html"
    # )
    # print(postseason_box_score)
