# Kaggle March Madness 2026
Files used for Kaggle's 2026 March Madness competition.

## Setup

1. Download the Kaggle 2026 competition dataset and place it in `data/kaggle_2026/raw/`
2. Run download scripts to fetch external data:
   ```bash
   python kaggle_2026/download_barttorvik.py
   python kaggle_2026/download_kenpom.py
   python kaggle_2026/download_bball_ref.py
   ```
3. Generate JSON mapping files:
   ```bash
   python kaggle_2026/kaggle_utils.py
   ```
4. Run `notebooks/kaggle_2026_save_cleaned_team_data.ipynb` to preprocess, train models, and save cleaned data
5. Generate submission:
   ```bash
   python kaggle_2026/submission.py
   ```
