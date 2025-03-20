import json
import pathlib

# Module-level variable to store the mapping data
SPELLINGS_KAGGLE_ID_MAP = None
ID_KAGGLE_NAME_MAP = None
WOMENS_SPELLINGS_KAGGLE_ID_MAP = None
WOMENS_ID_KAGGLE_NAME_MAP = None


def _load_spellings_map():
    """Helper function to load the spellings map once and cache it"""
    global SPELLINGS_KAGGLE_ID_MAP
    if SPELLINGS_KAGGLE_ID_MAP is None:
        module_dir = pathlib.Path(__file__).parent.absolute()
        with open(module_dir / "spellings_kaggle_id_map.json", "r") as f:
            SPELLINGS_KAGGLE_ID_MAP = json.load(f)


def _load_id_map():
    """Helper function to load the id map once and cache it"""
    global ID_KAGGLE_NAME_MAP
    if ID_KAGGLE_NAME_MAP is None:
        module_dir = pathlib.Path(__file__).parent.absolute()
        with open(module_dir / "id_kaggle_name_map.json", "r") as f:
            ID_KAGGLE_NAME_MAP = json.load(f)


def _load_womens_spellings_map():
    """Helper function to load the spellings map once and cache it"""
    global WOMENS_SPELLINGS_KAGGLE_ID_MAP
    if WOMENS_SPELLINGS_KAGGLE_ID_MAP is None:
        module_dir = pathlib.Path(__file__).parent.absolute()
        with open(module_dir / "womens_spellings_kaggle_id_map.json", "r") as f:
            WOMENS_SPELLINGS_KAGGLE_ID_MAP = json.load(f)


def _load_womens_id_map():
    """Helper function to load the id map once and cache it"""
    global WOMENS_ID_KAGGLE_NAME_MAP
    if WOMENS_ID_KAGGLE_NAME_MAP is None:
        module_dir = pathlib.Path(__file__).parent.absolute()
        with open(module_dir / "womens_id_kaggle_name_map.json", "r") as f:
            WOMENS_ID_KAGGLE_NAME_MAP = json.load(f)


def name_to_kaggle_name(team_name: str) -> str:
    _load_spellings_map()
    _load_id_map()
    if team_name.lower() not in SPELLINGS_KAGGLE_ID_MAP:
        raise ValueError(f"Team name {team_name} not found in spellings map")
    kaggle_id = SPELLINGS_KAGGLE_ID_MAP[team_name.lower()]
    return ID_KAGGLE_NAME_MAP[kaggle_id]


def name_to_kaggle_id(team_name: str) -> str:
    _load_spellings_map()
    if team_name.lower() not in SPELLINGS_KAGGLE_ID_MAP:
        raise ValueError(f"Team name {team_name} not found in spellings map")
    return SPELLINGS_KAGGLE_ID_MAP[team_name.lower()]


def womens_name_to_kaggle_name(team_name: str) -> str:
    _load_womens_spellings_map()
    _load_womens_id_map()
    if team_name.lower() not in WOMENS_SPELLINGS_KAGGLE_ID_MAP:
        raise ValueError(f"Team name {team_name} not found in spellings map")
    kaggle_id = WOMENS_SPELLINGS_KAGGLE_ID_MAP[team_name.lower()]
    return WOMENS_ID_KAGGLE_NAME_MAP[kaggle_id]


def womens_name_to_kaggle_id(team_name: str) -> str:
    _load_womens_spellings_map()
    if team_name.lower() not in WOMENS_SPELLINGS_KAGGLE_ID_MAP:
        raise ValueError(f"Team name {team_name} not found in spellings map")
    return WOMENS_SPELLINGS_KAGGLE_ID_MAP[team_name.lower()]
