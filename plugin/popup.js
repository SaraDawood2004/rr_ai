// popup.js — ReviewRanker
// All JS is in this external file to satisfy Chrome Extension CSP rules.

const API = 'https://reviewranker-u5vf.onrender.com/'

const STYLES = {
  green : { dot:'#22c55e', bg:'rgba(34,197,94,.10)',  border:'rgba(34,197,94,.30)',  label:'Useful' },
  yellow: { dot:'#eab308', bg:'rgba(234,179,8,.10)',  border:'rgba(234,179,8,.30)',  label:'Moderately Useful' },
  red   : { dot:'#ef4444', bg:'rgba(239,68,68,.10)',  border:'rgba(239,68,68,.30)',  label:'Not Useful' },
}

function resolveStyle(score) {
  if (score >= 0.75) return STYLES.green
  if (score >= 0.40) return STYLES.yellow
  return STYLES.red
}

// ── Grab DOM refs ─────────────────────────────────────────────
const inp        = document.getElementById('inp')
const btnAnalyze = document.getElementById('btn-analyze')
const spinAnalyze= document.getElementById('spin-analyze')
const btnPage    = document.getElementById('btn-page')
const spinPage   = document.getElementById('spin-page')
const resultEl   = document.getElementById('result')
const toastEl    = document.getElementById('toast')

// ── Server health check ───────────────────────────────────────
async function checkServer() {
  const dot  = document.getElementById('status-dot')
  const text = document.getElementById('status-text')
  try {
    await fetch(`${API}/`, { method: 'HEAD', signal: AbortSignal.timeout(2000) })
    dot.className    = 'dot on'
    text.textContent = 'AI server active'
  } catch {
    dot.className    = 'dot off'
    text.textContent = 'Server offline — run: python app/app.py'
    btnAnalyze.disabled = true
    btnPage.disabled    = true
  }
}

// ── Single review analysis ────────────────────────────────────
async function analyzeText() {
  const text = inp.value.trim()
  if (!text) { showToast('Please paste a review text first.'); return }
  if (text.split(' ').length < 5) { showToast('Review too short — add more detail.'); return }

  setLoading(btnAnalyze, spinAnalyze, true)
  hideResult()

  try {
    const res = await fetch(`${API}/api/predict`, {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ text }),
      signal : AbortSignal.timeout(15000),
    })
    if (!res.ok) throw new Error('Server error ' + res.status)
    const d = await res.json()
    if (d.error) { showToast(d.error); return }
    renderResult(d)
  } catch (e) {
    showToast('Could not reach server. Is Flask running?')
  } finally {
    setLoading(btnAnalyze, spinAnalyze, false)
  }
}

// ── Re-analyze page reviews ───────────────────────────────────
function analyzePageReviews() {
  setLoading(btnPage, spinPage, true)

  chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
    const tab = tabs[0]
    if (!tab || !tab.id) {
      showToast('Could not access the current tab.')
      setLoading(btnPage, spinPage, false)
      return
    }

    const supported = ['amazon', 'flipkart', 'walmart', 'bestbuy', 'ebay', 'myntra']
    const url       = tab.url || ''
    if (!supported.some(s => url.includes(s))) {
      showToast('Open a supported product review page first.')
      setLoading(btnPage, spinPage, false)
      return
    }

    chrome.scripting.executeScript(
      {
        target: { tabId: tab.id },
        func  : () => {
          if (typeof window.__rrReanalyze === 'function') {
            window.__rrReanalyze()
            return 'ok'
          }
          return 'not_ready'
        },
      },
      results => {
        setLoading(btnPage, spinPage, false)
        const val = results?.[0]?.result
        if (val === 'ok') {
          window.close()   // close popup so user sees the live updates
        } else {
          showToast('ReviewRanker not ready — reload the product page first.')
        }
      }
    )
  })
}

// ── Render result card ────────────────────────────────────────
function renderResult(d) {
  const style = resolveStyle(d.score)

  resultEl.style.background  = style.bg
  resultEl.style.borderColor = style.border
  resultEl.classList.remove('result-hidden')

  document.getElementById('r-dot').style.background = style.dot
  document.getElementById('r-label').textContent    = style.label
  document.getElementById('r-label').style.color    = style.dot
  document.getElementById('r-pct').textContent      = 'Score: ' + d.percent + '%'
  document.getElementById('r-pct').style.color      = style.dot
  document.getElementById('r-desc').textContent     = d.description

  const bar = document.getElementById('r-bar')
  bar.style.background = style.dot
  bar.style.width      = '0%'
  setTimeout(() => { bar.style.width = d.percent + '%' }, 60)

  if (d.breakdown) {
    const ts = d.breakdown.text_structure   || {}
    const gs = d.breakdown.grammar_structure|| {}
    const us = d.breakdown.user_signals     || {}
    document.getElementById('r-breakdown').innerHTML =
      bk('Sentiment',  fmtSentiment(ts.vader_compound)) +
      bk('Readability',fmtRead(ts.flesch_ease)) +
      bk('Words',      ts.word_count ?? '—') +
      bk('Features',   ts.feature_mentions ?? '—') +
      bk('Balanced',   ts.has_both_sides ? '✓ Yes' : '✗ No') +
      bk('Suspicious', us.is_suspicious   ? '⚠ Yes' : '✓ No')
  }
}

// ── Helpers ───────────────────────────────────────────────────
function bk(k, v) {
  return `<div class="bk"><span class="bk-k">${k}</span><span class="bk-v">${v}</span></div>`
}
function setLoading(btn, spinner, on) {
  btn.disabled = on
  if (on) {
    spinner.style.display = 'inline-block'
    btn.querySelector('.btn-label').style.display = 'none'
  } else {
    spinner.style.display = 'none'
    btn.querySelector('.btn-label').style.display = ''
  }
}
function hideResult() { resultEl.classList.add('result-hidden') }
function showToast(msg) {
  toastEl.textContent = msg
  toastEl.classList.remove('result-hidden')
  clearTimeout(window.__toastTimer)
  window.__toastTimer = setTimeout(() => toastEl.classList.add('result-hidden'), 3500)
}
function fmtSentiment(v) { return v > 0.4 ? 'Positive' : v < -0.4 ? 'Negative' : 'Neutral' }
function fmtRead(v)       { return v >= 70 ? 'Easy'     : v >= 50  ? 'Moderate' : 'Complex'  }

// ── Wire up buttons ───────────────────────────────────────────
btnAnalyze.addEventListener('click', analyzeText)
btnPage.addEventListener('click', analyzePageReviews)
inp.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); analyzeText() }
})

// ── Boot ──────────────────────────────────────────────────────
checkServer()