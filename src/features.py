"""
Feature extraction — 3 components:

  Component 1 → Text Structure
      Sentiment, readability, review length,
      product feature mentions, pros/cons mentions

  Component 2 → Grammar Structure
      Emotion intensity, POS tag diversity,
      grammatical consistency, opinion diversity

  Component 3 → User Activity Signals
      Reviewer history, review frequency,
      rating deviation (spam detection)
"""

import re
import math
import pandas as pd
import numpy as np
import spacy
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Load models once (expensive to reload every row) ─────────────────────────
print("Loading NLP models...")
nlp      = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.add_pipe("sentencizer")   # lightweight sentence splitting
vader    = SentimentIntensityAnalyzer()
print("Models loaded.\n")


# ════════════════════════════════════════════════════════════════════════════
# COMPONENT 1 — TEXT STRUCTURE FEATURES
# Focuses on WHAT the reviewer said about the product
# ════════════════════════════════════════════════════════════════════════════

# Keywords that indicate product feature mentions
FEATURE_KEYWORDS = [
    'quality', 'price', 'size', 'weight', 'color', 'colour', 'material',
    'design', 'flavor', 'flavour', 'taste', 'smell', 'texture', 'packaging',
    'battery', 'screen', 'sound', 'speed', 'performance', 'durability',
    'value', 'shipping', 'delivery', 'customer service', 'warranty',
    'ingredients', 'fresh', 'organic', 'natural',
]

# Keywords for pros mentions (positive aspect signals)
PROS_KEYWORDS = [
    'love', 'great', 'excellent', 'amazing', 'best', 'perfect', 'fantastic',
    'wonderful', 'awesome', 'outstanding', 'superb', 'highly recommend',
    'worth', 'impressed', 'delicious', 'fresh',
]

# Keywords for cons mentions (negative aspect signals)
CONS_KEYWORDS = [
    'bad', 'terrible', 'awful', 'worst', 'horrible', 'disappoint', 'waste',
    'poor', 'broken', 'defective', 'damaged', 'cheap', 'overpriced', 'bland',
    'stale', 'expired', 'not worth', 'do not recommend', 'return',
]

def extract_text_structure(text: str, summary: str = "") -> dict:
    text_lower = text.lower()

    # 1. Sentiment scores using VADER
    #    compound : overall sentiment  (-1 = very negative, +1 = very positive)
    #    pos/neg/neu : proportion of each type
    vs = vader.polarity_scores(text)

    # 2. Readability — Flesch Reading Ease (0–100, higher = easier to read)
    #    Reviews that are easy to read tend to be more helpful
    flesch     = textstat.flesch_reading_ease(text)
    flesch_kincaid = textstat.flesch_kincaid_grade(text)

    # 3. Length features
    words      = text.split()
    word_count = len(words)
    char_count = len(text)
    sentences  = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sent_count = max(len(sentences), 1)
    avg_word_len   = sum(len(w) for w in words) / max(word_count, 1)
    avg_sent_len   = word_count / sent_count

    # 4. Product feature mentions count
    feature_count = sum(1 for kw in FEATURE_KEYWORDS if kw in text_lower)

    # 5. Pros / cons mention balance
    pros_count = sum(1 for kw in PROS_KEYWORDS if kw in text_lower)
    cons_count = sum(1 for kw in CONS_KEYWORDS if kw in text_lower)

    # Both pros AND cons → more balanced → tends to be more helpful
    has_both_sides = int(pros_count > 0 and cons_count > 0)

    # 6. Summary vs body sentiment agreement
    #    If summary and body have similar sentiment, review is more coherent
    if summary and isinstance(summary, str):
        summary_vs = vader.polarity_scores(summary)
        sentiment_agreement = 1 - abs(vs['compound'] - summary_vs['compound']) / 2
    else:
        sentiment_agreement = 0.5

    return {
        # Sentiment
        'vader_compound'       : vs['compound'],
        'vader_positive'       : vs['pos'],
        'vader_negative'       : vs['neg'],
        'vader_neutral'        : vs['neu'],
        'sentiment_agreement'  : round(sentiment_agreement, 4),
        # Readability
        'flesch_ease'          : flesch,
        'flesch_grade'         : flesch_kincaid,
        # Length
        'word_count'           : word_count,
        'char_count'           : char_count,
        'sentence_count'       : sent_count,
        'avg_word_length'      : round(avg_word_len, 3),
        'avg_sentence_length'  : round(avg_sent_len, 3),
        # Content quality
        'feature_mentions'     : feature_count,
        'pros_count'           : pros_count,
        'cons_count'           : cons_count,
        'has_both_sides'       : has_both_sides,
    }


# ════════════════════════════════════════════════════════════════════════════
# COMPONENT 2 — GRAMMAR STRUCTURE FEATURES
# Focuses on HOW the reviewer wrote (linguistic quality)
# ════════════════════════════════════════════════════════════════════════════

# Strong emotion / intensity words
EMOTION_WORDS = [
    'absolutely', 'completely', 'totally', 'extremely', 'incredibly',
    'really', 'very', 'definitely', 'certainly', 'strongly',
    'highly', 'deeply', 'truly', 'genuinely', 'honestly',
    'love', 'hate', 'adore', 'despise', 'obsessed', 'disgusted',
]

def extract_grammar_structure(text: str) -> dict:
    text_lower = text.lower()
    words      = text.split()
    word_count = max(len(words), 1)

    # 1. POS (Part-of-Speech) tag diversity using spaCy
    #    A review with nouns, verbs, adjectives, adverbs is richer
    #    than one made up of mostly adjectives ("Great! Amazing! Loved it!")
    doc      = nlp(text)
    pos_tags = [token.pos_ for token in doc if not token.is_space]
    pos_set  = set(pos_tags)
    total_pos = max(len(pos_tags), 1)

    noun_ratio  = pos_tags.count('NOUN')  / total_pos
    verb_ratio  = pos_tags.count('VERB')  / total_pos
    adj_ratio   = pos_tags.count('ADJ')   / total_pos
    adv_ratio   = pos_tags.count('ADV')   / total_pos
    pos_diversity = len(pos_set) / 17   # spaCy has 17 POS tags → normalize to 0–1

    # 2. Emotion intensity
    emotion_count = sum(1 for w in EMOTION_WORDS if w in text_lower)
    emotion_ratio = emotion_count / word_count

    # 3. Grammatical consistency signals
    #    ALL CAPS words → shouting → lower quality signal
    all_caps_words  = sum(1 for w in words if w.isupper() and len(w) > 2)
    all_caps_ratio  = all_caps_words / word_count

    #    Excessive punctuation e.g. "!!!!" or "????" → less formal
    exclaim_count   = text.count('!')
    question_count  = text.count('?')
    exclaim_ratio   = exclaim_count / word_count
    question_ratio  = question_count / word_count

    # 4. Lexical diversity (Type-Token Ratio)
    #    Unique words / total words → higher = richer vocabulary
    unique_words    = len(set(w.lower() for w in words))
    lexical_diversity = unique_words / word_count

    # 5. Paragraph structure — does the reviewer use line breaks to organize?
    paragraph_count = max(len([p for p in text.split('\n') if p.strip()]), 1)

    # 6. Presence of numbers (specific data = more credible)
    #    e.g. "used this for 3 months", "paid $25", "rated 4/5"
    number_count    = len(re.findall(r'\b\d+\.?\d*\b', text))
    has_numbers     = int(number_count > 0)

    return {
        # POS diversity
        'noun_ratio'        : round(noun_ratio, 4),
        'verb_ratio'        : round(verb_ratio, 4),
        'adj_ratio'         : round(adj_ratio, 4),
        'adv_ratio'         : round(adv_ratio, 4),
        'pos_diversity'     : round(pos_diversity, 4),
        # Emotion intensity
        'emotion_count'     : emotion_count,
        'emotion_ratio'     : round(emotion_ratio, 4),
        # Writing quality signals
        'all_caps_ratio'    : round(all_caps_ratio, 4),
        'exclaim_count'     : exclaim_count,
        'exclaim_ratio'     : round(exclaim_ratio, 4),
        'question_count'    : question_count,
        'question_ratio'    : round(question_ratio, 4),
        # Vocabulary richness
        'lexical_diversity' : round(lexical_diversity, 4),
        # Structure
        'paragraph_count'   : paragraph_count,
        'number_count'      : number_count,
        'has_numbers'       : has_numbers,
    }


# ════════════════════════════════════════════════════════════════════════════
# COMPONENT 3 — USER ACTIVITY SIGNALS
# Focuses on WHO wrote the review (spam & credibility detection)
# ════════════════════════════════════════════════════════════════════════════

def build_user_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-user statistics from the full sample.
    These stats capture reviewer behaviour patterns.

    Columns added:
      user_review_count   → total reviews by this user
      user_avg_score      → user's average star rating given
      user_score_std      → std dev of user's ratings (low std = suspicious)
      product_avg_score   → average star rating of the product
      score_deviation     → how far this review's score is from product avg
      reviews_per_day     → posting frequency (high = possible spammer)
      is_suspicious       → flag: very high frequency + always same score
    """
    print("  Building user-level stats...")

    # ── User-level aggregation ────────────────────────────────────────────
    user_stats = df.groupby('user_id').agg(
        user_review_count = ('score', 'count'),
        user_avg_score    = ('score', 'mean'),
        user_score_std    = ('score', 'std'),
        user_first_time   = ('time',  'min'),
        user_last_time    = ('time',  'max'),
    ).reset_index()

    # Fill NaN std (users with only 1 review have undefined std)
    user_stats['user_score_std'] = user_stats['user_score_std'].fillna(0)

    # Reviews per day — high rate may indicate spam bot
    user_stats['active_days'] = (
        (user_stats['user_last_time'] - user_stats['user_first_time'])
        / 86400   # seconds → days
    ).clip(lower=1)   # minimum 1 day to avoid division by zero

    user_stats['reviews_per_day'] = (
        user_stats['user_review_count'] / user_stats['active_days']
    ).round(4)

    # Suspicious flag:
    #   posts > 3 reviews/day AND always gives same rating (std == 0)
    user_stats['is_suspicious'] = (
        (user_stats['reviews_per_day'] > 3) &
        (user_stats['user_score_std']  == 0)
    ).astype(int)

    # ── Product-level aggregation ─────────────────────────────────────────
    product_stats = df.groupby('product_id').agg(
        product_avg_score = ('score', 'mean'),
        product_review_count = ('score', 'count'),
    ).reset_index()

    # ── Merge back into main df ───────────────────────────────────────────
    df = df.merge(user_stats[[
        'user_id', 'user_review_count', 'user_avg_score',
        'user_score_std', 'reviews_per_day', 'is_suspicious'
    ]], on='user_id', how='left')

    df = df.merge(product_stats[[
        'product_id', 'product_avg_score', 'product_review_count'
    ]], on='product_id', how='left')

    # Score deviation — how far this review's rating is from the product mean
    # A review that dramatically differs from others may be an outlier/spam
    df['score_deviation'] = (
        abs(df['score'] - df['product_avg_score'])
    ).round(4)

    return df


# ════════════════════════════════════════════════════════════════════════════
# MAIN — Run all 3 components on the full dataset
# ════════════════════════════════════════════════════════════════════════════

def extract_all_features(
    in_path  : str = "reviews_sample.csv",
    out_path : str = "reviews_features.csv",
) -> pd.DataFrame:

    df = pd.read_csv(in_path)
    print(f"Loaded {len(df)} reviews for feature extraction.\n")

    # ── Component 3: user stats (needs full df context) ───────────────────
    print("── Component 3: User Activity Signals ──")
    df = build_user_stats(df)

    # ── Components 1 & 2: row-level NLP features ─────────────────────────
    print("\n── Components 1 & 2: Text + Grammar features ──")
    print("  (This may take 2–5 minutes for 10K rows...)\n")

    text_features    = []
    grammar_features = []

    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"  Processing row {i} / {len(df)} ...")

        text_features.append(
            extract_text_structure(row['text'], row.get('summary', ''))
        )
        grammar_features.append(
            extract_grammar_structure(row['text'])
        )

    df_text    = pd.DataFrame(text_features)
    df_grammar = pd.DataFrame(grammar_features)

    # Combine everything
    df_final = pd.concat(
        [df.reset_index(drop=True), df_text, df_grammar],
        axis=1
    )

    # Drop helper columns not needed for training
    df_final.drop(columns=['stratum'], errors='ignore', inplace=True)

    import os
    dir_path = os.path.dirname(out_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    df_final.to_csv(out_path, index=False)
    print(f"\n✅ Features saved → {out_path}")
    print(f"   Total features : {df_final.shape[1]} columns")
    print(f"   Total rows     : {df_final.shape[0]}")
    return df_final


if __name__ == "__main__":
    extract_all_features()