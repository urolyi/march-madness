import pathlib
import random
import time

import uro_cbb.kenpom as kenpom

if __name__ == "__main__":
    module_dir = pathlib.Path(__file__).parent.absolute()
    for year in range(2025, 2024, -1):
        if year == 2020:
            continue
        kp_df = kenpom.download_kenpom_data(year)
        out_dir = module_dir.parent / "data/kenpom/raw"
        out_dir.mkdir(parents=True, exist_ok=True)
        kp_df.write_csv(out_dir / f"kenpom_{year}.csv")
        kp_df.write_parquet(out_dir / f"kenpom_{year}.parquet")
        time.sleep(random.randint(2, 5))
