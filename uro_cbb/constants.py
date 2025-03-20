import datetime

TOURNAMENT_START_DATE_MAP = {
    2025: datetime.date(2025, 3, 20),
    2024: datetime.date(2024, 3, 21),
    2023: datetime.date(2023, 3, 16),
    2022: datetime.date(2022, 3, 17),
    2021: datetime.date(2021, 3, 19),
    # 2020: datetime.date(2020, 3, 19), COVID YEAR
    2019: datetime.date(2019, 3, 21),
    2018: datetime.date(2018, 3, 15),
    2017: datetime.date(2017, 3, 16),
    2016: datetime.date(2016, 3, 17),
    2015: datetime.date(2015, 3, 19),
}

# o3-mini-high suggested headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    # "Referer": "https://www.google.com/",
}
