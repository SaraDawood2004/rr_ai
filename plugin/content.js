/**
 * content.js — ReviewRanker v2
 * Supports: Amazon, Flipkart, Walmart, Best Buy, eBay, Myntra
 * Fixes: all reviews loaded, multi-site, color dots, re-analyze trigger
 */

const API_URL = 'http://localhost:5000/api/predict-bulk'

// ─────────────────────────────────────────────────────────────────────────────
// SITE ADAPTERS — selectors for each supported website
// Each adapter defines how to find reviews on that specific site
// ─────────────────────────────────────────────────────────────────────────────
const SITE_ADAPTERS = {
  amazon: {
    match       : () => location.hostname.includes('amazon'),
    reviewList  : '[data-hook="review"]',
    bodyEl      : '[data-hook="review-body"] span',
    titleEl     : '[data-hook="review-title"] span:last-child',
    starEl      : '[data-hook="review-star-rating"] span.a-icon-alt, .review-rating span.a-icon-alt',
    voteEl      : '[data-hook="helpful-vote-statement"]',
    starPattern : /(\d+(\.\d+)?)\s+out\s+of/i,
  },
  flipkart: {
    match       : () => location.hostname.includes('flipkart'),
    reviewList  : '._27M-vq, .t-ZTKy, ._1AtVbE .col',
    bodyEl      : '._6K-7Co, .qwjRop',
    titleEl     : '._2-N8zT, .RcXBOT',
    starEl      : '._3LWZlK, .XQDdHH',
    voteEl      : '._3c3Px5 span',
    starPattern : /^(\d+(\.\d+)?)/,
  },
  walmart: {
    match       : () => location.hostname.includes('walmart'),
    reviewList  : '[itemprop="review"], .Grid-col--fixed-true',
    bodyEl      : '[itemprop="reviewBody"], .review-text',
    titleEl     : '.review-title, [itemprop="name"]',
    starEl      : '.seo-avg-rating, [itemprop="ratingValue"]',
    voteEl      : '.review-footer-helpful-count',
    starPattern : /(\d+(\.\d+)?)/,
  },
  bestbuy: {
    match       : () => location.hostname.includes('bestbuy'),
    reviewList  : '.review-item, .ugc-review',
    bodyEl      : '.ugc-review-body p, .review-item-body',
    titleEl     : '.review-title, .ugc-review-title',
    starEl      : '.ugc-review-rating, .c-ratings-reviews-v4',
    voteEl      : '.review-helpful-count',
    starPattern : /(\d+(\.\d+)?)/,
  },
  ebay: {
    match       : () => location.hostname.includes('ebay'),
    reviewList  : '.ebay-review-section, .review-item',
    bodyEl      : '.review-item-content p, .ebay-review-item-desc',
    titleEl     : '.review-item-title',
    starEl      : '.ebay-star-rating, .reviews-stars',
    voteEl      : '.review-helpful',
    starPattern : /(\d+(\.\d+)?)/,
  },
  myntra: {
    match       : () => location.hostname.includes('myntra'),
    reviewList  : '.user-review-main, .detailed-reviews-userReview',
    bodyEl      : '.user-review-reviewTextWrapper, .detailed-reviews-userText',
    titleEl     : '.user-review-title',
    starEl      : '.user-review-starRating span, .detailed-reviews-rating',
    voteEl      : null,
    starPattern : /(\d+(\.\d+)?)/,
  },
}

function getAdapter() {
  return Object.values(SITE_ADAPTERS).find(a => a.match()) || null
}


// ─────────────────────────────────────────────────────────────────────────────
// SCRAPER — works with whichever adapter is active
// ─────────────────────────────────────────────────────────────────────────────
function scrapeReviews(adapter) {
  const nodes = document.querySelectorAll(adapter.reviewList)
  if (!nodes.length) return []

  const reviews = []

  nodes.forEach((node, idx) => {
    const bodyEl  = node.querySelector(adapter.bodyEl)
    const titleEl = node.querySelector(adapter.titleEl)
    if (!bodyEl) return

    const text    = bodyEl.innerText.trim()
    const summary = titleEl ? titleEl.innerText.trim() : ''
    if (text.split(' ').length < 3) return   // skip too-short entries

    // Star rating
    let score = 3
    const starEl = node.querySelector(adapter.starEl)
    if (starEl) {
      const raw   = starEl.getAttribute('aria-label') || starEl.innerText || ''
      const match = raw.match(adapter.starPattern)
      if (match) score = Math.min(5, Math.max(1, parseFloat(match[1])))
    }

    // Helpful votes
    let helpful_votes = 0
    if (adapter.voteEl) {
      const voteEl = node.querySelector(adapter.voteEl)
      if (voteEl) {
        const m = voteEl.textContent.match(/(\d+)/)
        if (m) helpful_votes = parseInt(m[1])
      }
    }

    const is_suspicious = (score === 1 || score === 5) && helpful_votes === 0 ? 1 : 0

    reviews.push({
      id      : idx,
      element : node,
      text,
      summary,
      user_context: {
        score,
        user_review_count    : 1,
        user_avg_score       : score,
        user_score_std       : 0,
        reviews_per_day      : 0.1,
        is_suspicious,
        product_avg_score    : 0,   // filled below
        product_review_count : nodes.length,
        score_deviation      : 0,   // filled below
      }
    })
  })

  // Compute product avg + deviation for all scraped reviews
  if (reviews.length) {
    const avg = reviews.reduce((s, r) => s + r.user_context.score, 0) / reviews.length
    reviews.forEach(r => {
      r.user_context.product_avg_score = parseFloat(avg.toFixed(2))
      r.user_context.score_deviation   = parseFloat(
        Math.abs(r.user_context.score - avg).toFixed(2)
      )
    })
  }

  return reviews
}


// ─────────────────────────────────────────────────────────────────────────────
// BADGE STYLES — color dot system (no text tags, just colored indicator)
// Green = Useful, Yellow = Moderately useful, Red = Not useful
// ─────────────────────────────────────────────────────────────────────────────
const BADGE_STYLES = {
  green : {
    dot     : '#22c55e',
    bg      : 'rgba(34,197,94,0.10)',
    border  : 'rgba(34,197,94,0.30)',
    label   : 'Useful',
    emoji   : '🟢',
  },
  orange: {
    dot     : '#eab308',
    bg      : 'rgba(234,179,8,0.10)',
    border  : 'rgba(234,179,8,0.30)',
    label   : 'Moderately Useful',
    emoji   : '🟡',
  },
  red   : {
    dot     : '#ef4444',
    bg      : 'rgba(239,68,68,0.10)',
    border  : 'rgba(239,68,68,0.30)',
    label   : 'Not Useful',
    emoji   : '🔴',
  },
}

// Map old color names coming from Flask API → badge style key
function resolveColor(apiColor, score) {
  if (score >= 0.75) return 'green'
  if (score >= 0.40) return 'orange'
  return 'red'
}


// ─────────────────────────────────────────────────────────────────────────────
// BADGE INJECTION
// ─────────────────────────────────────────────────────────────────────────────
function injectStyles() {
  if (document.getElementById('rr-styles')) return
  const s = document.createElement('style')
  s.id = 'rr-styles'
  s.textContent = `
    @keyframes rr-spin { to { transform: rotate(360deg); } }
    @keyframes rr-pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
    .rr-badge {
      display: inline-flex !important;
      align-items: center !important;
      gap: 7px !important;
      margin: 7px 0 5px !important;
      padding: 4px 11px !important;
      border-radius: 99px !important;
      font-size: 11.5px !important;
      font-weight: 600 !important;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
      letter-spacing: .25px !important;
      cursor: default !important;
      user-select: none !important;
      border: 1px solid !important;
      vertical-align: middle !important;
    }
    .rr-dot {
      width: 9px !important;
      height: 9px !important;
      border-radius: 50% !important;
      flex-shrink: 0 !important;
      display: inline-block !important;
    }
    .rr-pct {
      opacity: .65 !important;
      font-weight: 500 !important;
      font-size: 10.5px !important;
    }
    .rr-spinner {
      width: 9px !important; height: 9px !important;
      border: 1.5px solid rgba(150,150,150,.3) !important;
      border-top-color: #888 !important;
      border-radius: 50% !important;
      animation: rr-spin .65s linear infinite !important;
      display: inline-block !important;
      flex-shrink: 0 !important;
    }
  `
  document.head.appendChild(s)
}

function insertBadge(node, badge) {
  // Try to insert right after the star rating row
  const anchors = [
    '[data-hook="review-star-rating"]',
    '.review-rating', '._3LWZlK', '.XQDdHH',
    '.ugc-review-rating', '.ebay-star-rating',
    '.user-review-starRating',
  ]
  for (const sel of anchors) {
    const el = node.querySelector(sel)
    if (el) { el.insertAdjacentElement('afterend', badge); return }
  }
  node.prepend(badge)  // fallback
}

function makeLoadingBadge() {
  const b = document.createElement('span')
  b.className = 'rr-badge rr-loading'
  b.style.cssText = `background:rgba(80,80,120,0.08);border-color:rgba(108,99,255,0.2);color:#888`
  b.innerHTML = `<span class="rr-spinner"></span> ReviewRanker...`
  return b
}

function makeResultBadge(result) {
  const colorKey = resolveColor(result.color, result.score)
  const style    = BADGE_STYLES[colorKey]

  const b = document.createElement('span')
  b.className = 'rr-badge rr-result'
  b.style.cssText = `background:${style.bg};border-color:${style.border};color:${style.dot}`
  b.title = `ReviewRanker: ${result.description}`
  // Badge format: colored dot + "Helpfulness Score: XX%"
  b.innerHTML = `
    <span class="rr-dot" style="background:${style.dot}"></span>
    Helpfulness Score: <strong style="margin-left:3px">${result.percent}%</strong>
  `
  return b
}


// ─────────────────────────────────────────────────────────────────────────────
// RE-ORDER — move DOM nodes so best reviews appear first
// ─────────────────────────────────────────────────────────────────────────────
function reorderReviews(reviews, resultMap) {
  const parent = reviews[0]?.element?.parentNode
  if (!parent) return
  const sorted = [...reviews].sort(
    (a, b) => (resultMap[b.id]?.score || 0) - (resultMap[a.id]?.score || 0)
  )
  sorted.forEach(r => parent.appendChild(r.element))
}


// ─────────────────────────────────────────────────────────────────────────────
// SCROLL WATCHER — detects when new reviews load (infinite scroll / pagination)
// Marks already-processed reviews so they are not re-sent to the API
// ─────────────────────────────────────────────────────────────────────────────
let observer = null

function watchForNewReviews(adapter) {
  if (observer) observer.disconnect()

  observer = new MutationObserver(() => {
    const unprocessed = [...document.querySelectorAll(adapter.reviewList)]
      .filter(el => !el.dataset.rrDone)

    if (unprocessed.length >= 1) {
      // Debounce — wait 800ms after DOM settles before re-running
      clearTimeout(window.__rrTimer)
      window.__rrTimer = setTimeout(() => analyzeAll(adapter), 800)
    }
  })

  observer.observe(document.body, { childList: true, subtree: true })
}


// ─────────────────────────────────────────────────────────────────────────────
// MAIN ANALYZE FUNCTION
// Called on load AND whenever new reviews appear (scroll/pagination)
// Exposed as window.__rrRun so the popup button can trigger it too
// ─────────────────────────────────────────────────────────────────────────────
async function analyzeAll(adapter) {
  if (!adapter) return

  // Only process reviews that haven't been analyzed yet
  const allNodes  = [...document.querySelectorAll(adapter.reviewList)]
  const freshNodes = allNodes.filter(el => !el.dataset.rrDone)
  if (!freshNodes.length) return

  // Build review objects only from fresh nodes
  const reviews = scrapeReviews(adapter).filter(r => !r.element.dataset.rrDone)
  if (!reviews.length) return

  console.log(`[ReviewRanker] Analyzing ${reviews.length} new reviews on ${location.hostname}`)

  // Show loading badges + mark as in-progress
  reviews.forEach(r => {
    r.element.dataset.rrDone = 'pending'
    const old = r.element.querySelector('.rr-badge')
    if (old) old.remove()
    insertBadge(r.element, makeLoadingBadge())
  })

  try {
    const res  = await fetch(API_URL, {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({
        reviews: reviews.map(r => ({
          id          : r.id,
          text        : r.text,
          summary     : r.summary,
          user_context: r.user_context,
        }))
      }),
      signal: AbortSignal.timeout(30000),  // 30s timeout
    })

    const data = await res.json()
    if (!data.ranked_reviews) throw new Error('Bad response')

    const resultMap = {}
    data.ranked_reviews.forEach(r => resultMap[r.id] = r)

    reviews.forEach(r => {
      const loading = r.element.querySelector('.rr-loading')
      if (loading) loading.remove()

      const result = resultMap[r.id]
      if (result) {
        insertBadge(r.element, makeResultBadge(result))
        r.element.dataset.rrDone = 'done'
      } else {
        delete r.element.dataset.rrDone
      }
    })

    reorderReviews(reviews, resultMap)

  } catch (err) {
    reviews.forEach(r => {
      const loading = r.element.querySelector('.rr-loading')
      if (loading) loading.remove()
      delete r.element.dataset.rrDone
    })
    console.warn('[ReviewRanker] API error:', err.message)
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// RE-ANALYZE TRIGGER (called by popup "Analyze reviews on this page" button)
// Clears all previous results and re-runs everything fresh
// ─────────────────────────────────────────────────────────────────────────────
window.__rrReanalyze = function() {
  const adapter = getAdapter()
  if (!adapter) return
  // Reset all processed flags and remove old badges
  document.querySelectorAll(adapter.reviewList).forEach(el => {
    delete el.dataset.rrDone
    el.querySelectorAll('.rr-badge').forEach(b => b.remove())
  })
  analyzeAll(adapter)
}

// Simple trigger for popup (calls reanalyze)
window.__rrRun = window.__rrReanalyze


// ─────────────────────────────────────────────────────────────────────────────
// BOOT
// ─────────────────────────────────────────────────────────────────────────────
;(function boot() {
  const adapter = getAdapter()
  if (!adapter) {
    console.log('[ReviewRanker] Site not supported.')
    return
  }
  console.log(`[ReviewRanker] Active on ${location.hostname}`)
  injectStyles()

  // Initial run after page settles
  setTimeout(() => analyzeAll(adapter), 2000)

  // Watch for dynamically loaded reviews (pagination, infinite scroll)
  watchForNewReviews(adapter)
})()