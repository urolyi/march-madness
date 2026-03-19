import pathlib

from uro_cbb.kaggle_utils import KaggleNameMapper, generate_json_mappings

MODULE_DIR = pathlib.Path(__file__).parent.absolute()
KAGGLE_DATA_DIR = MODULE_DIR.parent / "data/kaggle_2026/raw"

_mapper = KaggleNameMapper(kaggle_data_dir=MODULE_DIR)


def name_to_kaggle_name(team_name: str) -> str:
    return _mapper.name_to_kaggle_name(team_name)


def name_to_kaggle_id(team_name: str) -> int:
    return _mapper.name_to_kaggle_id(team_name)


def womens_name_to_kaggle_name(team_name: str) -> str:
    return _mapper.womens_name_to_kaggle_name(team_name)


def womens_name_to_kaggle_id(team_name: str) -> int:
    return _mapper.womens_name_to_kaggle_id(team_name)


if __name__ == "__main__":
    generate_json_mappings(
        kaggle_data_dir=KAGGLE_DATA_DIR,
        output_dir=MODULE_DIR,
    )
    print("Generated JSON mapping files in", MODULE_DIR)
