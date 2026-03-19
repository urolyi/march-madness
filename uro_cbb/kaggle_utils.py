import json
import pathlib
from dataclasses import dataclass, field

import polars as pl


@dataclass
class KaggleNameMapper:
    """Loads and caches Kaggle team name/ID mappings from a given data directory."""

    kaggle_data_dir: pathlib.Path
    _spellings_to_id: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _id_to_name: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _womens_spellings_to_id: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _womens_id_to_name: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def _load_mens(self):
        if self._spellings_to_id:
            return
        spellings_path = self.kaggle_data_dir / "spellings_to_id.json"
        id_map_path = self.kaggle_data_dir / "id_kaggle_name_map.json"
        with open(spellings_path) as f:
            self._spellings_to_id = json.load(f)
        with open(id_map_path) as f:
            self._id_to_name = json.load(f)

    def _load_womens(self):
        if self._womens_spellings_to_id:
            return
        spellings_path = self.kaggle_data_dir / "womens_spellings_to_id.json"
        id_map_path = self.kaggle_data_dir / "womens_id_kaggle_name_map.json"
        with open(spellings_path) as f:
            self._womens_spellings_to_id = json.load(f)
        with open(id_map_path) as f:
            self._womens_id_to_name = json.load(f)

    def name_to_kaggle_name(self, team_name: str) -> str:
        self._load_mens()
        kaggle_id = self._spellings_to_id.get(team_name.lower())
        if kaggle_id is None:
            raise ValueError(f"Team name {team_name} not found in spellings map")
        return self._id_to_name[str(kaggle_id)]

    def name_to_kaggle_id(self, team_name: str) -> int:
        self._load_mens()
        kaggle_id = self._spellings_to_id.get(team_name.lower())
        if kaggle_id is None:
            raise ValueError(f"Team name {team_name} not found in spellings map")
        return kaggle_id

    def womens_name_to_kaggle_name(self, team_name: str) -> str:
        self._load_womens()
        kaggle_id = self._womens_spellings_to_id.get(team_name.lower())
        if kaggle_id is None:
            raise ValueError(f"Team name {team_name} not found in womens spellings map")
        return self._womens_id_to_name[str(kaggle_id)]

    def womens_name_to_kaggle_id(self, team_name: str) -> int:
        self._load_womens()
        kaggle_id = self._womens_spellings_to_id.get(team_name.lower())
        if kaggle_id is None:
            raise ValueError(f"Team name {team_name} not found in womens spellings map")
        return kaggle_id

    def load_mens_kaggle_names_df(self) -> pl.DataFrame:
        return pl.read_csv(
            self.kaggle_data_dir / "MTeamSpellings.csv",
            schema={"Team": pl.Utf8, "TeamID": pl.Int16},
        ).join(
            pl.read_csv(self.kaggle_data_dir / "MTeams.csv").select(
                pl.col("TeamID"), pl.col("TeamName")
            ),
            on="TeamID",
        )

    def load_womens_kaggle_names_df(self) -> pl.DataFrame:
        return pl.read_csv(
            self.kaggle_data_dir / "WTeamSpellings.csv",
            schema={"Team": pl.Utf8, "TeamID": pl.Int16},
        ).join(
            pl.read_csv(self.kaggle_data_dir / "WTeams.csv").select(
                pl.col("TeamID"), pl.col("TeamName")
            ),
            on="TeamID",
        )


def generate_json_mappings(kaggle_data_dir: pathlib.Path, output_dir: pathlib.Path):
    """Generate the JSON mapping files from raw Kaggle CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Men's mappings
    spellings_df = pl.read_csv(
        kaggle_data_dir / "MTeamSpellings.csv",
        schema={"Team": pl.Utf8, "TeamID": pl.Int16},
    )
    teams_df = pl.read_csv(kaggle_data_dir / "MTeams.csv").select(
        pl.col("TeamID"), pl.col("TeamName")
    )

    spellings_to_id = {
        row["Team"]: row["TeamID"]
        for row in spellings_df.iter_rows(named=True)
    }
    id_to_name = {
        str(row["TeamID"]): row["TeamName"]
        for row in teams_df.iter_rows(named=True)
    }

    with open(output_dir / "spellings_to_id.json", "w") as f:
        json.dump(spellings_to_id, f, indent=2)
    with open(output_dir / "id_kaggle_name_map.json", "w") as f:
        json.dump(id_to_name, f, indent=2)

    # Women's mappings
    w_spellings_df = pl.read_csv(
        kaggle_data_dir / "WTeamSpellings.csv",
        schema={"Team": pl.Utf8, "TeamID": pl.Int16},
    )
    w_teams_df = pl.read_csv(kaggle_data_dir / "WTeams.csv").select(
        pl.col("TeamID"), pl.col("TeamName")
    )

    w_spellings_to_id = {
        row["Team"]: row["TeamID"]
        for row in w_spellings_df.iter_rows(named=True)
    }
    w_id_to_name = {
        str(row["TeamID"]): row["TeamName"]
        for row in w_teams_df.iter_rows(named=True)
    }

    with open(output_dir / "womens_spellings_to_id.json", "w") as f:
        json.dump(w_spellings_to_id, f, indent=2)
    with open(output_dir / "womens_id_kaggle_name_map.json", "w") as f:
        json.dump(w_id_to_name, f, indent=2)
