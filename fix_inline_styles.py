"""Remove all inline styles from index.html and replace with CSS classes."""
from pathlib import Path

BASE = Path('d:/Option-Pricing-Using-Monte-Carlo-Simulation-Deep-Learning/frontend')
html_path = BASE / 'index.html'
content = html_path.read_text(encoding='utf-8')
original_len = len(content)

exact = [
    # Dashboard card version info
    ('<div class="card" style="margin-top:0.5rem">', '<div class="card mt-05">'),
    # Result badge
    ('<span class="result-badge" id="resultBadge" style="display:none">', '<span class="result-badge hidden" id="resultBadge">'),
    # Pricing results
    ('<div class="metrics-row" id="pricingResults" style="display:none">', '<div class="metrics-row hidden" id="pricingResults">'),
    # Greeks quick
    ('<div class="greeks-grid" id="greeksQuick" style="display:none">', '<div class="greeks-grid hidden" id="greeksQuick">'),
    # Greek chart wrap
    ('<div class="chart-container" style="display:none" id="greekChartWrap">', '<div class="chart-container hidden" id="greekChartWrap">'),
    # MC charts wrap
    ('<div class="charts-grid" id="mcChartsWrap" style="display:none">', '<div class="charts-grid hidden" id="mcChartsWrap">'),
    # DL news field + vol models field (both have margin-top:0.6rem)
    ('<div class="field" style="margin-top:0.6rem">', '<div class="field mt-06">'),
    # Optional span (DL label)
    ('<span style="color:var(--text-tertiary);font-weight:400">', '<span class="text-tertiary-lt">'),
    # DL textarea (padding 0.65rem)
    ('style="width:100%;resize:vertical;font-family:inherit;font-size:0.9rem;padding:0.65rem;border-radius:8px;border:1px solid var(--border);background:var(--bg-card);color:var(--text-primary)"',
     'class="form-textarea"'),
    # Sentiment textarea (padding 0.75rem)
    ('style="width:100%;resize:vertical;font-family:inherit;font-size:0.9rem;padding:0.75rem;border-radius:8px;border:1px solid var(--border);background:var(--bg-card);color:var(--text-primary)"',
     'class="form-textarea"'),
    # DL results hidden
    ('<div class="dl-results" id="dlResults" style="display:none">', '<div class="dl-results hidden" id="dlResults">'),
    # DL train status
    ('<div class="card" id="dlTrainStatus" style="display:none;margin-top:1rem">', '<div class="card hidden mt-1" id="dlTrainStatus">'),
    # DL comp chart wrap
    ('<div class="chart-container" id="compChartWrap" style="display:none">', '<div class="chart-container hidden" id="compChartWrap">'),
    # Vol engine status card
    ('<div class="card" id="volEngineStatus" style="margin-bottom:1.2rem">', '<div class="card mb-12" id="volEngineStatus">'),
    # Metrics row margin 0
    ('<div class="metrics-row" style="margin:0">', '<div class="metrics-row m-0">'),
    # Vol model checks
    ('<div style="display:flex;gap:0.7rem;flex-wrap:wrap;margin-top:0.3rem" id="volModelChecks">', '<div class="vol-model-checks" id="volModelChecks">'),
    # Vol btn group margin-top:1rem
    ('<div class="btn-group" style="margin-top:1rem">', '<div class="btn-group mt-1">'),
    # Vol train progress
    ('<div id="volTrainProgress" style="display:none;margin-top:0.8rem;">', '<div id="volTrainProgress" class="hidden mt-08">'),
    # Spinner row
    ('<div style="display:flex;align-items:center;gap:0.6rem">', '<div class="spinner-row">'),
    # Small spinner
    ('<div class="spinner" style="width:18px;height:18px;border:2px solid rgba(99,102,241,.3);border-top-color:#6366f1;border-radius:50%;animation:spin .8s linear infinite">',
     '<div class="spinner-sm">'),
    # Vol train message
    ('<span style="color:var(--text-secondary);font-size:0.85rem" id="volTrainMsg">', '<span class="text-secondary-sm" id="volTrainMsg">'),
    # Vol comparison card
    ('<div class="card" id="volComparisonCard" style="display:none;margin-top:1.2rem">', '<div class="card hidden mt-12" id="volComparisonCard">'),
    # Overflow-x:auto plain div (multiple occurrences)
    ('<div style="overflow-x:auto">', '<div class="overflow-x-auto">'),
    # Vol baseline row
    ('<div class="metrics-row" style="margin-top:0.8rem" id="volBaselineRow">', '<div class="metrics-row mt-08" id="volBaselineRow">'),
    # Vol feature card
    ('<div class="card" id="volFeatureCard" style="display:none;margin-top:1.2rem">', '<div class="card hidden mt-12" id="volFeatureCard">'),
    # IV Prediction card (only non-hidden mt-12 card)
    ('<div class="card" style="margin-top:1.2rem">', '<div class="card mt-12">'),
    # ML results
    ('<div class="metrics-row" id="mlResults" style="display:none;margin-top:1rem">', '<div class="metrics-row hidden mt-1" id="mlResults">'),
    # Sentiment field margin-bottom
    ('<div class="field" style="margin-bottom:1rem">', '<div class="field mb-1">'),
    # Sentiment results
    ('<div id="sentimentResults" style="display:none;margin-top:1rem">', '<div id="sentimentResults" class="hidden mt-1">'),
    # Sent bullish
    ('<div class="metric-value" id="sentBullish" style="color:#00e5a0">', '<div class="metric-value text-bullish" id="sentBullish">'),
    # Sent bearish
    ('<div class="metric-value" id="sentBearish" style="color:#ff5c7c">', '<div class="metric-value text-bearish" id="sentBearish">'),
    # Gauge labels
    ('<span style="color:#ff5c7c">Very Bearish</span>', '<span class="text-bearish">Very Bearish</span>'),
    ('<span style="color:var(--text-secondary)">Neutral</span>', '<span class="text-secondary">Neutral</span>'),
    ('<span style="color:#00e5a0">Very Bullish</span>', '<span class="text-bullish">Very Bullish</span>'),
    # varResults
    ('<div id="varResults" style="display:none;margin-top:1rem">', '<div id="varResults" class="hidden mt-1">'),
    # varPctLoss color
    ('<div class="metric-value" id="varPctLoss" style="color:#ff5c7c">', '<div class="metric-value text-bearish" id="varPctLoss">'),
    # Risk bars width:0%
    (' style="width:0%"', ''),
    # RAG follow-ups
    ('<div class="follow-ups" id="followUps" style="display:none">', '<div class="follow-ups hidden" id="followUps">'),
    # RAG meta
    ('<div class="rag-meta" id="ragMeta" style="display:none">', '<div class="rag-meta hidden" id="ragMeta">'),
    # Cache badge
    ('<span class="rag-badge rag-badge-cache" id="cacheBadge" style="display:none">\u26a1 cached</span>',
     '<span class="rag-badge rag-badge-cache hidden" id="cacheBadge">\u26a1 cached</span>'),
    # Market stream log (complex)
    ('<div class="result-block" id="mktStreamLog" style="margin-top:0.75rem;max-height:300px;overflow-y:auto;font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;display:none"></div>',
     '<div class="result-block mkt-stream-log hidden" id="mktStreamLog"></div>'),
    # Market chain container
    ('<div id="mktChainContainer" style="overflow-x:auto">', '<div id="mktChainContainer" class="overflow-x-auto">'),
    # Market chain table hidden
    ('<table class="data-table" id="mktChainTable" style="display:none">', '<table class="data-table hidden" id="mktChainTable">'),
    # Mispricing results
    ('<div id="mispResults" style="display:none;margin-top:1rem">', '<div id="mispResults" class="hidden mt-1">'),
    # Mispricing scan results
    ('<div id="mispScanResults" style="display:none;margin-top:1rem">', '<div id="mispScanResults" class="hidden mt-1">'),
    # Regime field span 2
    ('<div class="field" style="grid-column: span 2">', '<div class="field col-span-2">'),
    # Regime results
    ('<div id="regimeResults" style="display:none;margin-top:1rem">', '<div id="regimeResults" class="hidden mt-1">'),
    # SHAP results
    ('<div id="shapResults" style="display:none;margin-top:1rem">', '<div id="shapResults" class="hidden mt-1">'),
    # shapNarrative pre-wrap
    ('<div id="shapNarrative" class="result-block" style="white-space:pre-wrap">', '<div id="shapNarrative" class="result-block pre-wrap">'),
    # btn-note
    ('<div class="btn-note" style="font-size:0.7rem;color:var(--text-muted);margin-top:0.25rem">', '<div class="btn-note">'),
    # bench results
    ('<div id="benchResults" style="display:none;margin-top:1rem">', '<div id="benchResults" class="hidden mt-1">'),
    # overflow-x:auto with margin-top:0.5rem (arb + pf table wrappers)
    ('<div style="overflow-x:auto;margin-top:0.5rem">', '<div class="overflow-x-auto mt-05">'),
    # Quant result cards with display:none + margin-top:0.75rem (specific first)
    ('<div class="card" style="margin-top:0.75rem;display:none" id="pinnsResults">', '<div class="card hidden mt-075" id="pinnsResults">'),
    ('<div id="pinnsNarrative" class="result-block" style="margin-top:0.5rem">', '<div id="pinnsNarrative" class="result-block mt-05">'),
    ('<div class="card" style="margin-top:0.75rem;display:none" id="hedgeResults">', '<div class="card hidden mt-075" id="hedgeResults">'),
    ('<div id="hedgeNarrative" class="result-block" style="margin-top:0.5rem">', '<div id="hedgeNarrative" class="result-block mt-05">'),
    ('<div class="card" style="margin-top:0.75rem;display:none" id="vsResults">', '<div class="card hidden mt-075" id="vsResults">'),
    ('<div class="chart-container" style="margin-top:0.75rem"><canvas id="vsChart"></canvas></div>',
     '<div class="chart-container mt-075"><canvas id="vsChart"></canvas></div>'),
    ('<div id="vsNarrative" class="result-block" style="margin-top:0.5rem">', '<div id="vsNarrative" class="result-block mt-05">'),
    ('<div class="card" style="margin-top:0.75rem;display:none" id="jdResults">', '<div class="card hidden mt-075" id="jdResults">'),
    ('<div id="jdNarrative" class="result-block" style="margin-top:0.5rem">', '<div id="jdNarrative" class="result-block mt-05">'),
    ('<div class="card" style="margin-top:0.75rem;display:none" id="arbResults">', '<div class="card hidden mt-075" id="arbResults">'),
    ('<div class="card" style="margin-top:0.75rem;display:none" id="uqResults">', '<div class="card hidden mt-075" id="uqResults">'),
    ('<div id="uqNarrative" class="result-block" style="margin-top:0.5rem">', '<div id="uqNarrative" class="result-block mt-05">'),
    # Portfolio risk btn-group margin
    ('<div class="btn-group" style="margin-top:0.75rem">', '<div class="btn-group mt-075">'),
    ('<div class="card" style="margin-top:0.75rem;display:none" id="gmcResults">', '<div class="card hidden mt-075" id="gmcResults">'),
    ('<div id="gmcNarrative" class="result-block" style="margin-top:0.5rem">', '<div id="gmcNarrative" class="result-block mt-05">'),
    ('<div class="card" style="margin-top:0.75rem;display:none" id="pfResults">', '<div class="card hidden mt-075" id="pfResults">'),
    ('<div id="pfNarrative" class="result-block" style="margin-top:0.5rem">', '<div id="pfNarrative" class="result-block mt-05">'),
    # Generic visible cards (run AFTER specific display:none variants above)
    ('<div class="card" style="margin-top:0.75rem">', '<div class="card mt-075">'),
    ('<div class="card" style="margin-top:1rem">', '<div class="card mt-1">'),
]

warnings = []
for old, new in exact:
    count = content.count(old)
    if count == 0:
        warnings.append(f'NOT FOUND: {repr(old[:80])}')
    else:
        print(f'  [{count}x] {repr(old[:70])}')
    content = content.replace(old, new)

# Check remaining style= attributes
remaining = content.count('style=')
print(f'\nRemaining style= attributes: {remaining}')
if remaining > 0:
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if 'style=' in line:
            print(f'  L{i}: {line.strip()[:140]}')

if warnings:
    print('\nWARNINGS (not found — check manually):')
    for w in warnings:
        print(f'  {w}')

html_path.write_text(content, encoding='utf-8')
print(f'\nWrote {len(content)} chars (was {original_len}). Delta: {len(content)-original_len}')
