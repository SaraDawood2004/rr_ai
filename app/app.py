"""
app/app.py — Flask REST API
Endpoints:
  POST /api/predict        → analyze a single review
  POST /api/predict-bulk   → analyze a list of reviews and rank them
  GET  /                   → serve the frontend
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from predict import predict_review

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app) 


# ── Serve frontend ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ── Single review prediction ──────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()

    text    = data.get('text', '').strip()
    summary = data.get('summary', '').strip()

    if not text or len(text.split()) < 5:
        return jsonify({"error": "Review text is too short."}), 400

    user_context = data.get('user_context', None)
    result = predict_review(text, summary, user_context)

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


# ── Bulk review ranking ───────────────────────────────────────────────────────
@app.route('/api/predict-bulk', methods=['POST'])
def predict_bulk():
    """
    Accepts a list of reviews, scores each one,
    and returns them sorted by helpfulness score (descending).

    Request body:
    {
      "reviews": [
        { "id": 1, "text": "...", "summary": "..." },
        ...
      ]
    }
    """
    data    = request.get_json()
    reviews = data.get('reviews', [])

    if not reviews:
        return jsonify({"error": "No reviews provided."}), 400

    results = []
    for rev in reviews:
        text    = rev.get('text', '').strip()
        summary = rev.get('summary', '').strip()
        rev_id  = rev.get('id', None)

        if not text or len(text.split()) < 5:
            results.append({
                "id"     : rev_id,
                "error"  : "Too short to analyze",
                "score"  : 0,
                "percent": 0,
                "tag"    : "Not Useful",
                "emoji"  : "❌",
                "color"  : "red",
            })
            continue

        res = predict_review(text, summary)
        res['id'] = rev_id
        results.append(res)

    # Sort by score descending
    results.sort(key=lambda r: r.get('score', 0), reverse=True)
    return jsonify({"ranked_reviews": results, "total": len(results)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    print(f"🚀 ReviewRanker API running → http://localhost:{port}")
    app.run(debug=debug, host='0.0.0.0', port=port)