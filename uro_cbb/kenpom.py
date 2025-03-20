import bs4
import polars as pl
import requests

KENPOM_ARCHIVE_URLS = {
    2025: "https://web.archive.org/web/20250318045354/kenpom.com",
    2024: "https://web.archive.org/web/20240319172443/https://kenpom.com",
    2023: "https://web.archive.org/web/20230315235513/https://kenpom.com",
    2022: "https://web.archive.org/web/20220316172443/https://kenpom.com",
    2021: "https://web.archive.org/web/20210317122320/https://kenpom.com/",
    2019: "https://web.archive.org/web/20190320235656/https://kenpom.com",
    2018: "https://web.archive.org/web/20180314201532/https://kenpom.com",
    2017: "https://web.archive.org/web/20170315194730/https://kenpom.com",
    2016: "https://web.archive.org/web/20160315203312/https://kenpom.com",
    2015: "https://web.archive.org/web/20150318080652/https://kenpom.com",
}


def _parse_kenpom_school(potential_row: bs4.element.Tag) -> str | None:
    table_data = potential_row.find_all("td")
    if len(table_data) == 0:
        return None
    if school_table_data := table_data[1].find("a"):
        return school_table_data.contents[0]
    return None


def _parse_kenpom_row(potential_row: bs4.element.Tag) -> list[str]:
    return [
        table_data.contents[0]
        for table_data in potential_row.find_all("td")[4:]
        if table_data.find("span") is None
    ]


kenpom_row_schema = {
    "Team": pl.Utf8,
    "AdjEM": pl.Float64,
    "AdjO": pl.Float64,
    "AdjD": pl.Float64,
    "AdjT": pl.Float64,
    "Luck": pl.Float64,
    "SOS_AdjEM": pl.Float64,
    "SOS_AdjO": pl.Float64,
    "SOS_AdjD": pl.Float64,
    "NCSOS_Net": pl.Float64,
}


def download_kenpom_data(year: int) -> pl.DataFrame:
    ken_pom_url = KENPOM_ARCHIVE_URLS[year]
    response = requests.get(ken_pom_url)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, features="html.parser")
    kenpom_table_row_elements = (
        soup.find("table", attrs={"id": "ratings-table"}).find("tbody").find_all("tr")
    )

    data_rows = []
    for potential_row in kenpom_table_row_elements:
        if kenpom_school := _parse_kenpom_school(potential_row):
            kenpom_row = [kenpom_school] + _parse_kenpom_row(potential_row)
            assert len(kenpom_row) == len(kenpom_row_schema)
            data_rows.append(kenpom_row)

    return pl.DataFrame(data_rows, schema=kenpom_row_schema)
