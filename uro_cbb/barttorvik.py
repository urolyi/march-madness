import datetime

import bs4
import polars as pl
import pydantic
import requests

import uro_cbb.constants as constants

TABLE_SCHEMA = {
    "rk": pl.Int16,
    "team": pl.Utf8,
    "conf": pl.Utf8,
    "g": pl.Int16,
    "rec": pl.Utf8,
    "adjoe": pl.Float32,
    "adjde": pl.Float32,
    "barthag": pl.Float32,
    "efg%": pl.Float32,
    "efgd%": pl.Float32,
    "tor": pl.Float32,
    "tord": pl.Float32,
    "orb": pl.Float32,
    "drb": pl.Float32,
    "ftr": pl.Float32,
    "ftrd": pl.Float32,
    "2p%": pl.Float32,
    "2p%d": pl.Float32,
    "3p%": pl.Float32,
    "3p%d": pl.Float32,
    "3pr": pl.Float32,
    "3prd": pl.Float32,
    "2pr": pl.Float32,
    "adj t.": pl.Float32,
    "wab": pl.Float32,
}


class GetRequest(pydantic.BaseModel):
    year: int
    sort: str | None = None
    hteam: str | None = None
    t2value: str | None = None
    conlimit: str = "All"
    state: str = "All"
    begin: datetime.date = pydantic.Field(
        default_factory=lambda data: datetime.date(data["year"] - 1, 11, 1)
    )
    end: datetime.date = pydantic.Field(
        default_factory=lambda data: datetime.date(data["year"], 5, 1)
    )
    top: int = 0
    revquad: int = 0
    quad: int = 5
    venue: str = "All"
    type: str = "All"
    mingames: int = 0


def get_content(tag: bs4.Tag) -> str:
    if hasattr(tag, "contents"):
        if len(tag.contents) == 0:
            return None
        return tag.contents[0]
    return None


def _parse_barttorvik_table_cell(cell: bs4.Tag) -> str:
    if cell.find("a"):
        return cell.find("a").text
    return get_content(cell)


def _parse_barttorvik_table_row(row: bs4.Tag) -> dict:
    return [_parse_barttorvik_table_cell(td_tag) for td_tag in row.find_all("td")]


def _download_barttorvik_data(
    year: int,
    barttorvik_url: str = "https://barttorvik.com/trank.php",
) -> pl.DataFrame:
    request = GetRequest(
        year=year,
        end=constants.TOURNAMENT_START_DATE_MAP[year] - datetime.timedelta(days=1),
    )
    response = requests.get(
        barttorvik_url,
        params=request.model_dump(),
        headers=constants.HEADERS,
    )
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser")

    # Get table header for columns and create schema dict
    table_header = soup.find("thead").find_all("tr")[-1]
    col_names = [th_tag.text for th_tag in table_header.find_all("th")]
    schema = {col.lower(): TABLE_SCHEMA[col.lower()] for col in col_names}

    # Get table body and rows
    table_body = soup.find("tbody")
    rows = table_body.select("tr:not(.extraheader)")

    # Parse rows and create dataframe
    data = []
    for row in rows:
        data.append(_parse_barttorvik_table_row(row))
    return pl.DataFrame(data, schema=schema)


def download_barttorvik_data(year: int) -> pl.DataFrame:
    return _download_barttorvik_data(year)


def download_womens_barttorvik_data(year: int) -> pl.DataFrame:
    return _download_barttorvik_data(
        year, barttorvik_url="https://barttorvik.com/womens/trank.php"
    )


if __name__ == "__main__":
    data = download_barttorvik_data(2024)
    print(data)
