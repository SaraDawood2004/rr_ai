import joblib
import pandas as pd
import numpy as np
from features import extract_text_structure, extract_grammar_structure
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "helpfulness_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "models", "feature_cols.pkl")

MODEL = joblib.load(MODEL_PATH)
FEATURE_COLS = joblib.load(FEATURE_PATH)

DEFAULT_USER_CONTEXT = {
    'score'               : 3,
    'user_review_count'   : 1,
    'user_avg_score'      : 3.0,
    'user_score_std'      : 0.0,
    'reviews_per_day'     : 0.1,
    'is_suspicious'       : 0,
    'product_avg_score'   : 3.0,
    'product_review_count': 10,
    'score_deviation'     : 0.0,
}


def assign_tag(score: float) -> dict:
    """
    score >= 0.75  → Useful          (green)
    score 0.40–0.74 → Moderately Useful (yellow)
    score <  0.40  → Not Useful      (red)
    """
    if score >= 0.75:
        return {
            "tag"        : "Useful",
            "color"      : "green",
            "description": "This review is highly informative and credible."
        }
    elif score >= 0.40:
        return {
            "tag"        : "Moderately Useful",
            "color"      : "yellow",
            "description": "This review has some useful information."
        }
    else:
        return {
            "tag"        : "Not Useful",
            "color"      : "red",
            "description": "This review lacks sufficient detail or credibility."
        }


def predict_review(
    text         : str,
    summary      : str = "",
    user_context : dict = None,
) -> dict:
    if not text or len(text.strip()) < 10:
        return {"error": "Review text is too short to analyze."}

    ctx = {**DEFAULT_USER_CONTEXT, **(user_context or {})}

    comp1 = extract_text_structure(text, summary)
    comp2 = extract_grammar_structure(text)

    all_feats = {**comp1, **comp2, **ctx}
    row = pd.DataFrame(
        [[all_feats.get(c, 0) for c in FEATURE_COLS]],
        columns=FEATURE_COLS
    )

    raw_score = float(MODEL.predict(row)[0])
    score     = round(max(0.0, min(1.0, raw_score)), 4)
    tag_info  = assign_tag(score)

    return {
        "score"      : score,
        "percent"    : round(score * 100, 1),
        **tag_info,
        "breakdown"  : {
            "text_structure"   : comp1,
            "grammar_structure": comp2,
            "user_signals"     : {k: ctx[k] for k in DEFAULT_USER_CONTEXT},
        }
    }


if __name__ == "__main__":
    samples = [
        {
            "label"  : "Detailed helpful review",
            "text"   : """I have been using this product for 3 months now and I can
            confidently say it is one of the best purchases I have made this year.
            The quality is excellent and the flavor is natural. It was delivered in
            good packaging and arrived fresh. My only complaint is the price has
            gone up slightly, but overall it is absolutely worth it.""",
            "summary": "Excellent product, great quality and value"
        },
        {
            "label"  : "Unhelpful vague review",
            "text"   : "This is great!! Love it love it love it!!! Amazing!!!! Will buy again.",
            "summary": "Great!!!"
        },
        {
            "label"  : "Balanced review",
            "text"   : """The taste is good and ingredients are clean. However I was
            disappointed by the packaging — mine arrived damaged. Customer service
            sent a replacement quickly. Worth buying from a local store instead.""",
            "summary": "Good product, poor delivery"
        },
    ]

    for rev in samples:
        result = predict_review(rev['text'], rev['summary'])
        print(f"\n{'='*50}")
        print(f"Type   : {rev['label']}")
        print(f"Score  : {result['score']} ({result['percent']}%)")
        print(f"Tag    : {result['tag']}  [{result['color']}]")
        print(f"Desc   : {result['description']}")