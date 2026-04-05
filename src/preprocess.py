import pandas as pd
import numpy as np
import re
import os

# ── Column name mapping ──────────────────────────────────────────────────────
# The CSV from Kaggle uses these exact column names (lowercase with spaces)
# We rename them to snake_case for easier coding
COLUMN_MAP = {
    'Id':                       'id',
    'ProductId':                'product_id',
    'UserId':                   'user_id',
    'ProfileName':              'profile_name',
    'HelpfulnessNumerator':     'helpful_votes',
    'HelpfulnessDenominator':   'total_votes',
    'Score':                    'score',
    'Time':                     'time',
    'Summary':                  'summary',
    'Text':                     'text',
}

def load_data(path: str) -> pd.DataFrame:
    print("Loading dataset...")
    df = pd.read_csv(path)
    df.rename(columns=COLUMN_MAP, inplace=True)
    print(f"  Raw shape : {df.shape}")
    print(f"  Columns   : {list(df.columns)}")
    return df


# ── Filter: only reviews with enough votes ───────────────────────────────────
# Minimum 5 votes so the helpfulness ratio is statistically meaningful.
# e.g. 1/1 = 100% but that's just one person — not reliable.
def filter_valid_helpfulness(df: pd.DataFrame, min_votes: int = 5) -> pd.DataFrame:
    print(f"\nFiltering: keeping rows with total_votes >= {min_votes} ...")
    df = df[df['total_votes'] >= min_votes].copy()
    # Sanity check — numerator must not exceed denominator
    df = df[df['helpful_votes'] <= df['total_votes']]
    print(f"  After filter : {df.shape}")
    return df


# ── Create target label ──────────────────────────────────────────────────────
# helpfulness_score = helpful_votes / total_votes  →  range [0.0, 1.0]
# This is the value our ML model will learn to predict.
def create_label(df: pd.DataFrame) -> pd.DataFrame:
    print("\nCreating helpfulness_score label...")
    df['helpfulness_score'] = (
        df['helpful_votes'] / df['total_votes']
    ).round(4)
    return df


# ── Clean text ───────────────────────────────────────────────────────────────
# We intentionally KEEP: ! ? . , ' " — these carry emotional signals
# We REMOVE: HTML tags, URLs, special symbols, extra whitespace
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>',        ' ', text)   # HTML tags
    text = re.sub(r'http\S+|www\S+', ' ', text)   # URLs
    text = re.sub(r'[^\w\s!?.,\'\"-]', ' ', text) # special chars
    text = re.sub(r'\s+',             ' ', text).strip()
    return text

def apply_text_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    print("\nCleaning text columns...")
    df['text']    = df['text'].apply(clean_text)
    df['summary'] = df['summary'].apply(clean_text)
    return df


# ── Filter: remove very short reviews ────────────────────────────────────────
# Reviews with fewer than 20 words have almost no NLP signal to extract.
def filter_short_reviews(df: pd.DataFrame, min_words: int = 20) -> pd.DataFrame:
    print(f"\nRemoving reviews shorter than {min_words} words...")
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df = df[df['word_count'] >= min_words].copy()
    print(f"  After filter : {df.shape}")
    return df


# ── Remove duplicate reviews ─────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("\nRemoving duplicate review texts...")
    before = len(df)
    df = df.drop_duplicates(subset=['text']).copy()
    print(f"  Removed {before - len(df)} duplicates")
    return df


# ── Stratified sampling ───────────────────────────────────────────────────────
# We split reviews into 3 score buckets and sample equally from each.
# This prevents the model from being biased toward one type of review.
#
#   low  : score 0.00 – 0.39  (not helpful reviews)
#   mid  : score 0.40 – 0.74  (moderately helpful)
#   high : score 0.75 – 1.00  (very helpful reviews)
def stratified_sample(df: pd.DataFrame, n: int = 10000) -> pd.DataFrame:
    print(f"\nStratified sampling → target {n} rows...")
    df['stratum'] = pd.cut(
        df['helpfulness_score'],
        bins=[0.0, 0.40, 0.75, 1.01],
        labels=['low', 'mid', 'high'],
        include_lowest=True
    )
    print("  Distribution before sampling:")
    print(df['stratum'].value_counts().to_string())

    per_stratum = n // 3
    sampled = (
        df.groupby('stratum', observed=True)
          .apply(lambda g: g.sample(min(len(g), per_stratum), random_state=42), include_groups=False)
          .reset_index(drop=True)
    )
    print(f"  Final sample size : {len(sampled)}")
    return sampled


# ── Save final dataset ────────────────────────────────────────────────────────
KEEP_COLS = [
    'id', 'product_id', 'user_id', 'score', 'time',
    'summary', 'text', 'word_count',
    'helpful_votes', 'total_votes', 'helpfulness_score'
]

def save_sample(df: pd.DataFrame, out_path: str) -> None:
    dir_path = os.path.dirname(out_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    df[KEEP_COLS].to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}  |  shape: {df[KEEP_COLS].shape}")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_preprocessing(
    raw_path : str = "Reviews.csv",
    out_path : str = "reviews_sample.csv",
    n_sample : int = 10000,
    min_votes: int = 5,
    min_words: int = 20,
) -> pd.DataFrame:

    df = load_data(raw_path)
    df = filter_valid_helpfulness(df, min_votes)
    df = create_label(df)
    df = apply_text_cleaning(df)
    df = filter_short_reviews(df, min_words)
    df = remove_duplicates(df)
    df = stratified_sample(df, n_sample)
    save_sample(df, out_path)

    print("\n── Score distribution in final sample ──")
    print(df['helpfulness_score'].describe().round(3))
    return df


if __name__ == "__main__":
    run_preprocessing()