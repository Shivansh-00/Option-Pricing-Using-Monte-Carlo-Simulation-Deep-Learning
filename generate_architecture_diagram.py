"""
OptionQuant — IEEE Publication-Quality Architecture Diagram  (v3 — Enhanced)
=============================================================================
- Generous whitespace between every layer & box
- Larger, crisper fonts; no text crowding
- Shadow-style depth on every box
- Colour-coded data-flow arrows with clear labels
- Gridded sub-panels with contrasting headers
- 300 DPI, 12 × 18 in  →  3600 × 5400 px
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ═══════════════════════════════════════════════════════════════
#  GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif", "Georgia", "serif"],
    "font.size":   9,
    "text.usetex": False,
})

W, H   = 120, 190       # coordinate space
FW, FH = 12, 18         # figure inches
DPI    = 300

fig, ax = plt.subplots(figsize=(FW, FH), dpi=DPI)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("auto")
ax.axis("off")
fig.patch.set_facecolor("#FDFDFE")

# ═══════════════════════════════════════════════════════════════
#  PALETTE
# ═══════════════════════════════════════════════════════════════
P = dict(
    user     = "#C8E6C9",
    fe       = "#BBDEFB",
    nginx    = "#FFE0B2",
    api      = "#D1C4E9",
    price    = "#B2DFDB",
    ml       = "#FFF9C4",
    dl       = "#F8BBD0",
    rag      = "#DCEDC8",
    data     = "#D7CCC8",
    ext      = "#CFD8DC",
    bdr      = "#263238",
    shadow   = "#B0BEC5",
    arrow    = "#37474F",
    title    = "#0D47A1",
    sub      = "#1565C0",
    txt      = "#212121",
    ltxt     = "#546E7A",
    white    = "#FFFFFF",
    hdr_price= "#00796B",
    hdr_ml   = "#F9A825",
    hdr_dl   = "#C62828",
    hdr_rag  = "#33691E",
)

ML = 5          # margin left
CW = W - 2*ML  # content width = 110

# ═══════════════════════════════════════════════════════════════
#  PRIMITIVES
# ═══════════════════════════════════════════════════════════════

def box(x, y, w, h, title, bg, sub=None, *,
        fs=8.5, fs2=6, bold=True, lw=0.8, rad=0.3,
        shadow=True, border=None):
    """Rounded box with drop-shadow and optional subtitle."""
    if shadow:
        s = FancyBboxPatch((x+0.25, y-0.25), w, h,
            boxstyle=f"round,pad=0.1,rounding_size={rad}",
            fc=P["shadow"], ec="none", lw=0, zorder=2, alpha=0.25)
        ax.add_patch(s)
    p = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0.1,rounding_size={rad}",
        fc=bg, ec=border or P["bdr"], lw=lw, zorder=3)
    ax.add_patch(p)
    cx, cy = x + w/2, y + h/2
    fw = "bold" if bold else "normal"
    if sub:
        ax.text(cx, cy + 0.9, title, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=P["txt"], zorder=4)
        ax.text(cx, cy - 1.1, sub, ha="center", va="center",
                fontsize=fs2, color="#555", fontstyle="italic",
                zorder=4, linespacing=1.35)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=P["txt"], zorder=4)


def layer(x, y, w, h, label, bg="#F5F6FA", lc=None):
    """Dashed layer background."""
    p = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.25,rounding_size=0.6",
        fc=bg, ec="#90A4AE", lw=0.6,
        linestyle=(0, (6, 3)), zorder=1, alpha=0.5)
    ax.add_patch(p)
    ax.text(x + 1.2, y + h - 0.9, label, ha="left", va="top",
            fontsize=7.5, fontweight="bold", fontstyle="italic",
            color=lc or P["ltxt"], zorder=2)


def arrow(x1, y1, x2, y2, label=None, *, c=None, lw=0.9,
          sty="-|>", rad=0, fs=6, ms=9):
    """Arrow with optional label."""
    c = c or P["arrow"]
    cs = f"arc3,rad={rad}"
    a = FancyArrowPatch((x1,y1),(x2,y2),
        arrowstyle=sty, color=c, lw=lw,
        mutation_scale=ms, zorder=5, connectionstyle=cs)
    ax.add_patch(a)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.7, label, ha="center", va="bottom",
                fontsize=fs, color=c, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.92))


def row_xs(n, bw, gap=2):
    """Centre n boxes of width bw with gap inside CW."""
    total = n*bw + (n-1)*gap
    s = ML + (CW - total)/2
    return [s + i*(bw+gap) for i in range(n)]

def hline(y):
    ax.plot([ML, W-ML], [y, y], color="#CBD5E1", lw=0.4, zorder=0)


# ═══════════════════════════════════════════════════════════════
#  TITLE
# ═══════════════════════════════════════════════════════════════
ax.text(W/2, 188, "OptionQuant — System Architecture",
        ha="center", va="center", fontsize=18, fontweight="bold", color=P["title"])
ax.text(W/2, 185.5,
        "Intelligent Option Pricing Platform  ·  Monte Carlo Simulation  ·  "
        "Deep Learning  ·  RAG-based Explainability",
        ha="center", va="center", fontsize=9, color=P["sub"])
hline(184)

# ═══════════════════════════════════════════════════════════════
#  LAYER 1 — CLIENT   (y = 177 – 182)
# ═══════════════════════════════════════════════════════════════
L1y = 177
layer(ML, L1y, CW, 6, "Layer 1 — Client Tier", "#E8F5E9")
bw1 = 28
xs1 = row_xs(2, bw1, gap=26)
box(xs1[0], L1y+1, bw1, 4, "Web Browser", P["user"],
    "Interactive SPA Dashboard  ·  9 Analytical Sections", fs=9.5)
box(xs1[1], L1y+1, bw1, 4, "REST / API Client", P["user"],
    "Programmatic JSON Access  ·  External Integration", fs=9.5)

# ═══════════════════════════════════════════════════════════════
#  LAYER 2 — PRESENTATION   (y = 161 – 175)
# ═══════════════════════════════════════════════════════════════
L2y = 161
layer(ML, L2y, CW, 14.5, "Layer 2 — Presentation & API Gateway", "#E3F2FD")

# Nginx
box(ML+15, L2y+9.5, 80, 4,
    "Nginx Reverse Proxy   (Port 3000 → 80)", P["nginx"],
    "Gzip Level 6  ·  Security Headers (XSS, Frame, CSP)  ·  7-day Static Cache  ·  "
    "Keepalive 32  ·  300 s Read Timeout",
    fs=9.5, fs2=6.5)

# Frontend — 5 boxes
fbw = 20
fxs = row_xs(5, fbw, gap=2.5)
fy = L2y + 1.2
fh = 6
box(fxs[0], fy, fbw, fh, "SPA Shell", P["fe"],
    "index.html  ·  login.html\n50x.html  ·  Section Router", fs=8)
box(fxs[1], fy, fbw, fh, "Application Logic", P["fe"],
    "app.js  (1 359 LOC)\nAuth Guard  ·  API Wrapper\nRetry + Timeout", fs=8)
box(fxs[2], fy, fbw, fh, "Chart Visualization", P["fe"],
    "Canvas Rendering\nComparison Bar Chart\nGreeks Radar  ·  MC Paths", fs=8)
box(fxs[3], fy, fbw, fh, "Styling Engine", P["fe"],
    "CSS3 Dark Theme\nGlassmorphism Effects\nParticle Ambient Orbs", fs=8)
box(fxs[4], fy, fbw, fh, "Auth Manager", P["fe"],
    "JWT Token Storage\nSilent Refresh  (60 s)\nAuto-Redirect on 401", fs=8)

# ═══════════════════════════════════════════════════════════════
#  LAYER 3 — FASTAPI BACKEND   (y = 143 – 159)
# ═══════════════════════════════════════════════════════════════
L3y = 143
layer(ML, L3y, CW, 16.5, "Layer 3 — FastAPI Backend   (Uvicorn · 2 Workers · Python 3.11)", "#EDE7F6")

# Middleware
box(ML+3, L3y+12, CW-6, 3.2,
    "Middleware:    CORS  │  Request-ID (UUID) Tracking  │  Latency Monitoring  │"
    "  JWT Authentication  │  Exception Handler  (422 / 400 / 500)",
    "#D1C4E9", bold=False, fs=7.8, lw=0.5, shadow=False)

# API routes — 5 boxes
abw = 20
axs = row_xs(5, abw, gap=2.5)
ay = L3y + 1.5
ah = 8.5
box(axs[0], ay, abw, ah, "/api/v1/auth", P["api"],
    "POST  /signup\nPOST  /login\nPOST  /refresh\nPOST  /logout\nGET   /me", fs=8.5, fs2=5.8)
box(axs[1], ay, abw, ah, "/api/v1/pricing", P["api"],
    "POST  /bs\nPOST  /mc\nPOST  /mc/detailed\nPOST  /mc/compare\nPOST  /greeks", fs=8.5, fs2=5.8)
box(axs[2], ay, abw, ah, "/api/v1/ml", P["api"],
    "POST  /iv-predict\nPOST  /vol/train\nGET   /vol/status\nGET   /vol/models", fs=8.5, fs2=5.8)
box(axs[3], ay, abw, ah, "/api/v1/dl", P["api"],
    "POST  /forecast\nPOST  /train\nPOST  /predict-volatility\nPOST  /market-sentiment\nGET   /status", fs=8.5, fs2=5.8)
box(axs[4], ay, abw, ah, "/api/v1/ai", P["api"],
    "POST  /explain\nGET   /rag/health\nGET   /rag/metrics\nGET   /rag/stats", fs=8.5, fs2=5.8)

# ═══════════════════════════════════════════════════════════════
#  LAYER 4 — CORE ENGINES   (y = 62 – 140)
# ═══════════════════════════════════════════════════════════════
L4y = 62
L4h = 79
layer(ML, L4y, CW, L4h, "Layer 4 — Core Computational Engines", "#E0F2F1")

# half-panel dimensions
HW = 53          # half width
HG = 4           # gap between halves
SL = ML + 1      # sub left
SR = ML + HW + HG  # sub right
SW = HW          # sub width

# ─────────────────────────────────────────────────────────
#  4A — PRICING & RISK   (top-left)
# ─────────────────────────────────────────────────────────
PAy = L4y + 54
PAh = 22
layer(SL, PAy, SW, PAh, "4A — Pricing & Risk Engine", "#E0F2F1", P["hdr_price"])

# Header bar
box(SL+1, PAy+PAh-4.5, SW-2, 2.5,
    "Black-Scholes · Monte Carlo · Greeks · Heston Stochastic Vol · Variance Reduction",
    "#B2DFDB", bold=True, fs=7.5, shadow=False, lw=0.4)

pbw = 15.5
pgap = 2
pleft = SL + (SW - 3*pbw - 2*pgap)/2

# Row 1 — primary engines
pr1 = PAy + 11.5
prh = 5
box(pleft, pr1, pbw, prh, "Black-Scholes\nAnalytical", P["price"],
    "N(d₁), N(d₂)\nCalls & Puts\nClosed-Form Solution", fs=8, fs2=5.5)
box(pleft+pbw+pgap, pr1, pbw, prh, "Monte Carlo\nGBM Engine", P["price"],
    "Batched Vectorized Paths\nConvergence Tracking\nUp to 500 K Paths", fs=8, fs2=5.5)
box(pleft+2*(pbw+pgap), pr1, pbw, prh, "Greeks\nEngine", P["price"],
    "Δ   Γ   V   Θ   ρ\nAnalytical Formulas\nFinite-Difference Bumps", fs=8, fs2=5.5)

# Row 2 — supporting modules
pr2 = PAy + 2.5
pr2h = 6.5
box(pleft, pr2, pbw, pr2h, "Variance\nReduction", P["price"],
    "Antithetic Variates\nControl Variates (β-opt)\nStratified Sampling\n~40-60% Var Reduction", fs=8, fs2=5.2)
box(pleft+pbw+pgap, pr2, pbw, pr2h, "Stochastic\nVolatility", P["price"],
    "Heston Model\nEuler Discretization\nCorrelated Brownian W₁,W₂\nFull Truncation Scheme", fs=8, fs2=5.2)
box(pleft+2*(pbw+pgap), pr2, pbw, pr2h, "Multi-Method\nComparison", P["price"],
    "BS vs Standard MC\nvs Antithetic vs Control\nError Analysis + CI\n4 Estimator Benchmark", fs=8, fs2=5.2)

# Internal arrows
arrow(pleft+pbw, pr1+prh/2, pleft+pbw+pgap, pr1+prh/2, lw=0.5)
arrow(pleft+2*pbw+pgap, pr1+prh/2, pleft+2*(pbw+pgap), pr1+prh/2, lw=0.5)
arrow(pleft+pbw+pgap+pbw/2, pr1, pleft+pbw/2, pr2+pr2h, lw=0.5)
arrow(pleft+pbw+pgap+pbw/2, pr1, pleft+pbw+pgap+pbw/2, pr2+pr2h, lw=0.5)

# ─────────────────────────────────────────────────────────
#  4B — ML VOLATILITY   (top-right)
# ─────────────────────────────────────────────────────────
PBy = PAy
PBh = PAh
layer(SR, PBy, SW, PBh, "4B — ML Volatility Forecasting Pipeline", "#FFF8E1", P["hdr_ml"])

box(SR+1, PBy+PBh-4.5, SW-2, 2.5,
    "7 ML Models · 30+ Features · Walk-Forward CV · GARCH/EWMA Baselines · Regime Detection",
    "#FFF9C4", bold=True, fs=7.5, shadow=False, lw=0.4)

mleft = SR + (SW - 3*pbw - 2*pgap)/2

# Row 1
box(mleft, pr1, pbw, prh, "Synthetic Data\nGenerator", P["ml"],
    "Regime-Switching GBM\n4 Market Regimes\n10-Year Simulation", fs=8, fs2=5.5)
box(mleft+pbw+pgap, pr1, pbw, prh, "Feature\nEngineering", P["ml"],
    "30+ Features · 6 Groups\nParkinson · Yang-Zhang\nRSI · Bollinger · VIX", fs=8, fs2=5.5)
box(mleft+2*(pbw+pgap), pr1, pbw, prh, "Target\nBuilder", P["ml"],
    "Realized Volatility\nParkinson Vol\nGarman-Klass Vol", fs=8, fs2=5.5)

# Row 2
mbw2 = 24
box(mleft, pr2, mbw2, pr2h, "Model Zoo   (7 Models)", P["ml"],
    "Ridge Regression  ·  Lasso\nRandom Forest (200 trees)\nGradient Boosting (300 est)\n"
    "Ensemble Stacking (3-fold CV)\nLSTM (NumPy)  ·  Temporal CNN", fs=8, fs2=5.2)
box(mleft+mbw2+pgap, pr2, SW-2*(SW-3*pbw-2*pgap)/2 - mbw2-pgap, pr2h,
    "Volatility\nEngine", P["ml"],
    "Walk-Forward CV\nBaseline Comparison\n(Historical / GARCH / EWMA)\n"
    "Drift Monitor  ·  Inference", fs=8, fs2=5.2)

# ML arrows
arrow(mleft+pbw, pr1+prh/2, mleft+pbw+pgap, pr1+prh/2, lw=0.5)
arrow(mleft+2*pbw+pgap, pr1+prh/2, mleft+2*(pbw+pgap), pr1+prh/2, lw=0.5)
arrow(mleft+pbw/2, pr1, mleft+12, pr2+pr2h, lw=0.5)
arrow(mleft+2*(pbw+pgap)+pbw/2, pr1, mleft+mbw2+pgap+10, pr2+pr2h, lw=0.5)

# ─────────────────────────────────────────────────────────
#  4C — DEEP LEARNING   (bottom-left)
# ─────────────────────────────────────────────────────────
PCy = L4y + 27
PCh = 24.5
layer(SL, PCy, SW, PCh, "4C — Deep Learning Pipeline", "#FFEBEE", P["hdr_dl"])

box(SL+1, PCy+PCh-4.5, SW-2, 2.5,
    "LSTM · Transformer · Temporal CNN · Hybrid Predictor · Sentiment Analyzer · SPSA Training",
    "#F8BBD0", bold=True, fs=7.5, shadow=False, lw=0.4)

dleft = SL + (SW - 3*pbw - 2*pgap)/2

# Row 1 — DL models
dr1 = PCy + 14
drh = 5.5
box(dleft, dr1, pbw, drh, "LSTM\nNetwork", P["dl"],
    "Multi-Layer · 64 Hidden\nSPSA Gradient-Free Opt\nPure NumPy (No PyTorch)\nEarly Stopping", fs=8, fs2=5.2)
box(dleft+pbw+pgap, dr1, pbw, drh, "Transformer\nEncoder", P["dl"],
    "Multi-Head Self-Attention\n4 Heads × 64 Dim\nGELU + LayerNorm\nSinusoidal Pos. Enc.", fs=8, fs2=5.2)
box(dleft+2*(pbw+pgap), dr1, pbw, drh, "Temporal\nCNN", P["dl"],
    "1D Conv · 16 Filters\nKernel Size = 5 · ReLU\nGlobal Average Pool\nLinear Projection", fs=8, fs2=5.2)

# Row 2 — Hybrid + Sentiment
dr2 = PCy + 6
dr2h = 5.5
dbw2_val = 24
box(dleft, dr2, dbw2_val, dr2h, "Hybrid Predictor", P["dl"],
    "Ensemble Weighting:\n45% Black-Scholes + 25% Monte Carlo\n+ 20% LSTM + 10% Residual\n"
    "Sentiment Nudge · Confidence Score", fs=8, fs2=5.2)
box(dleft+dbw2_val+pgap, dr2, SW - 2*(SW-3*pbw-2*pgap)/2 - dbw2_val - pgap, dr2h,
    "Sentiment\nAnalyzer", P["dl"],
    "Financial Lexicon (80 Terms)\n40% Transformer + 60% Lexicon\nNegation Handling\n"
    "Bullish / Neutral / Bearish", fs=8, fs2=5.2)

# Row 3 — Training
dr3 = PCy + 1.5
dr3h = 3
box(dleft, dr3, dbw2_val, dr3h, "Training Pipeline", P["dl"],
    "Synthetic GBM · SPSA Optimization · MinMax Scaling · Early Stopping", fs=7, fs2=5.5)
box(dleft+dbw2_val+pgap, dr3, SW - 2*(SW-3*pbw-2*pgap)/2 - dbw2_val - pgap, dr3h,
    "Preprocessing", P["dl"],
    "Sequence Windowing (T=30) · Train/Val Split · Normalization", fs=7, fs2=5.5)

# DL arrows
arrow(dleft+pbw/2, dr1, dleft+12, dr2+dr2h, lw=0.5)
arrow(dleft+pbw+pgap+pbw/2, dr1, dleft+dbw2_val+pgap+10, dr2+dr2h, lw=0.5)
arrow(dleft+12, dr2, dleft+12, dr3+dr3h, lw=0.5)

# ─────────────────────────────────────────────────────────
#  4D — RAG EXPLAINABILITY   (bottom-right)
# ─────────────────────────────────────────────────────────
PDy = PCy
PDh = PCh
layer(SR, PDy, SW, PDh, "4D — RAG AI Explainability Pipeline", "#F1F8E9", P["hdr_rag"])

box(SR+1, PDy+PDh-4.5, SW-2, 2.5,
    "13-Stage Pipeline · BM25+ · TF-IDF/Dense Hybrid · RRF Fusion · CoT Prompting · Gemini LLM",
    "#DCEDC8", bold=True, fs=7.5, shadow=False, lw=0.4)

rleft = SR + (SW - 3*pbw - 2*pgap)/2

# Row 1 — Input
rr1 = PDy + 14
rrh = 5.5
box(rleft, rr1, pbw, rrh, "Guard\nRails", P["rag"],
    "Input Validation\nPrompt Injection Detect\nDomain Scope Check\n120 Domain Keywords", fs=8, fs2=5.2)
box(rleft+pbw+pgap, rr1, pbw, rrh, "Query\nEngine", P["rag"],
    "50+ Synonym Expansion\nMulti-Part Decompose\nClassification (5 Types)\nContext Enrichment", fs=8, fs2=5.2)
box(rleft+2*(pbw+pgap), rr1, pbw, rrh, "Document\nChunking", P["rag"],
    "Recursive Splitting\nSemantic Boundaries\nSliding Window\nMarkdown-Aware", fs=8, fs2=5.2)

# Row 2 — Retrieval
rr2 = PDy + 6
rr2h = 5.5
box(rleft, rr2, pbw, rr2h, "Embedding\nEngine", P["rag"],
    "TF-IDF (10 K features)\nSentence-Transformers\nSHA-256 Disk Cache\nAuto-Backend Select", fs=8, fs2=5.2)
box(rleft+pbw+pgap, rr2, pbw, rr2h, "Hybrid\nVector Store", P["rag"],
    "BM25+ Sparse (k₁=1.5)\nDense Cosine Similarity\nRRF Fusion (k=60)\nMetadata Filtering", fs=8, fs2=5.2)
box(rleft+2*(pbw+pgap), rr2, pbw, rr2h, "Advanced\nRetriever", P["rag"],
    "Multi-Hop Iterative\nMulti-Signal Reranking\nNear-Duplicate Dedup\nAdaptive top-k", fs=8, fs2=5.2)

# Row 3 — Generation
rr3 = PDy + 1.5
rr3h = 3
box(rleft, rr3, pbw, rr3h, "Prompt Engine", P["rag"],
    "Chain-of-Thought · Citation Forcing · Conflict Detection", fs=7, fs2=5.5)
box(rleft+pbw+pgap, rr3, pbw, rr3h, "LLM Client", P["rag"],
    "Gemini 2.0 Flash API · Circuit Breaker · Exp. Backoff", fs=7, fs2=5.5)
box(rleft+2*(pbw+pgap), rr3, pbw, rr3h, "Evaluation", P["rag"],
    "Groundedness · Faithfulness · Hallucination Risk Score", fs=7, fs2=5.5)

# RAG internal arrows
for i in range(2):
    arrow(rleft+(i+1)*pbw+i*pgap, rr1+rrh/2, rleft+(i+1)*(pbw+pgap), rr1+rrh/2, lw=0.5)
    arrow(rleft+(i+1)*pbw+i*pgap, rr2+rr2h/2, rleft+(i+1)*(pbw+pgap), rr2+rr2h/2, lw=0.5)
for i in range(3):
    cx = rleft + i*(pbw+pgap) + pbw/2
    arrow(cx, rr1, cx, rr2+rr2h, lw=0.5)
for i in range(3):
    cx = rleft + i*(pbw+pgap) + pbw/2
    arrow(cx, rr2, cx, rr3+rr3h, lw=0.5)
for i in range(2):
    arrow(rleft+(i+1)*pbw+i*pgap, rr3+rr3h/2, rleft+(i+1)*(pbw+pgap), rr3+rr3h/2, lw=0.5)

# ─────────────────────────────────────────────────────────
#  BOTTOM OF L4 — Knowledge Base + Supporting Modules
# ─────────────────────────────────────────────────────────
# KB + Orchestrator (right)
box(SR+1, L4y+14, SW-2, 4,
    "Knowledge Base   (10 Expert Markdown Documents)", P["rag"],
    "Black-Scholes Model · Monte Carlo Simulation · Greeks & Sensitivity · Volatility Modeling\n"
    "Stochastic Vol Models · Variance Reduction · Deep Learning Pricing · American Options\n"
    "Portfolio Hedging · Risk Management",
    fs=8, fs2=5.5)

box(SR+1, L4y+6, SW-2, 5.5,
    "RAG Orchestrator   (explain.py — 13-Stage Pipeline)", P["rag"],
    "(1) Input Validation  >  (2) Cache Check  >  (3) Domain Scope  >  (4) Query Classification\n"
    ">  (5) Hybrid Retrieval (Dense + BM25+)  >  (6) Quality Assessment  >  (7) Evidence Extraction\n"
    ">  (8) Context Budget Enforcement  >  (9) Prompt Assembly (CoT)  >  (10) LLM Generation\n"
    ">  (11) Response Validation  >  (12) Evaluation Metrics  >  (13) Cache Update",
    fs=8.5, fs2=5.5)

arrow(SR+1+(SW-2)/2, L4y+11.5, SR+1+(SW-2)/2, L4y+14, lw=0.6)

# Supporting modules (left)
box(SL+1, L4y+14, SW-2, 4,
    "Feature Engineering & Data Pipeline", P["price"],
    "Log Returns · Realized Volatility · VIX Integration · Volume Analysis\n"
    "Rate Extraction · OHLCV Processing · MarketRow Data Loader · CSV Streaming",
    fs=8, fs2=5.5)

box(SL+1, L4y+6, 25, 5.5,
    "Model Monitor &\nExplainability", P["price"],
    "Prediction Drift Detection\nStable / Warning Status\nFeature Importance (SHAP)\nModel-Agnostic Explain",
    fs=8, fs2=5.2)
box(SL+27, L4y+6, 26, 5.5,
    "Metrics Engine &\nEvent Logger", P["price"],
    "RMSE · MAE · MAPE · R²\nQLIKE Loss · Directional Acc.\nJSON-Line Audit Trail\nUTC Timestamp Logging",
    fs=8, fs2=5.2)

# ═══════════════════════════════════════════════════════════════
#  LAYER 5 — DATA & PERSISTENCE   (y = 40 – 58)
# ═══════════════════════════════════════════════════════════════
L5y = 40
L5h = 18
layer(ML, L5y, CW, L5h, "Layer 5 — Data & Persistence", "#EFEBE9")

dbw3 = 17
dxs2 = row_xs(6, dbw3, gap=1.5)
dy2 = L5y + 7
dh2 = 6.5
box(dxs2[0], dy2, dbw3, dh2, "SQLite\nDatabase", P["data"],
    "Users Table\nToken Blacklist\nRate Limits\nWAL Mode", fs=8, fs2=5.5)
box(dxs2[1], dy2, dbw3, dh2, "Model\nStore", P["data"],
    "lstm_model.pt\ntransformer_model.pt\nDocker Volume\nPersistent", fs=8, fs2=5.5)
box(dxs2[2], dy2, dbw3, dh2, "Event\nLog", P["data"],
    "JSON-Line Format\nUTC Timestamps\nAudit Trail\ndata/processed/", fs=8, fs2=5.5)
box(dxs2[3], dy2, dbw3, dh2, "Embedding\nCache", P["data"],
    "Disk-Backed .npy\nSHA-256 Content Hash\nMemory + File\nAuto-Invalidation", fs=8, fs2=5.5)
box(dxs2[4], dy2, dbw3, dh2, "Response\nCache", P["data"],
    "LRU (128 Entries)\nMD5 Query Keys\nTTL: 600 s\nHit/Miss Stats", fs=8, fs2=5.5)
box(dxs2[5], dy2, dbw3, dh2, "Config\nStore", P["data"],
    ".env Files (2)\nSettings Dataclass\n60+ Parameters\nHot Reload", fs=8, fs2=5.5)

# Volume bar
box(ML+5, L5y+1.5, 50, 3,
    "Docker Volumes:   model_data (persistent)  │  knowledge_base (read-only bind mount)",
    P["data"], bold=False, fs=7.5, lw=0.4, shadow=False)
box(ML+57, L5y+1.5, 50, 3,
    "Local Directories:   data/raw/  ·  data/processed/  │  Logs:  /tmp/optiquant_logs",
    P["data"], bold=False, fs=7.5, lw=0.4, shadow=False)

# ═══════════════════════════════════════════════════════════════
#  LAYER 6 — EXTERNAL & INFRASTRUCTURE   (y = 18 – 38)
# ═══════════════════════════════════════════════════════════════
L6y = 18
L6h = 20
layer(ML, L6y, CW, L6h, "Layer 6 — External Services & Deployment Infrastructure", "#ECEFF1")

ebw2 = 20
exs2 = row_xs(5, ebw2, gap=2.5)
ey2 = L6y + 9
eh2 = 6
box(exs2[0], ey2, ebw2, eh2, "Google Gemini\nAPI", P["ext"],
    "gemini-2.0-flash\nRate Limit: 30 RPM\n100 K TPM Budget\nCircuit Breaker (5/60s)", fs=8.5, fs2=5.5)
box(exs2[1], ey2, ebw2, eh2, "Docker\nCompose", P["ext"],
    "Multi-Container Orch.\nBridge Network\nVolume Management\nService Dependencies", fs=8.5, fs2=5.5)
box(exs2[2], ey2, ebw2, eh2, "Backend\nContainer", P["ext"],
    "Python 3.11-slim\n2 CPU · 2 GB RAM\nNon-Root User\nMulti-Stage Build", fs=8.5, fs2=5.5)
box(exs2[3], ey2, ebw2, eh2, "Frontend\nContainer", P["ext"],
    "Nginx 1.27-alpine\n0.5 CPU · 128 MB\nStatic Serving\nReverse Proxy", fs=8.5, fs2=5.5)
box(exs2[4], ey2, ebw2, eh2, "Health &\nSecurity", P["ext"],
    "/health · /ready\nAuto-Restart: unless-stopped\nSecurity Headers\nCompression", fs=8.5, fs2=5.5)

# Security bars
box(ML+3, L6y+3.5, 52, 3.5,
    "Authentication & Authorization", P["ext"],
    "PBKDF2-HMAC-SHA256 (100 K iterations, 16-byte salt)\n"
    "JWT HS256: Access Token 30 min · Refresh Token 7 days · Token Rotation\n"
    "Password: ≥8 chars, uppercase + lowercase + digit + special character",
    fs=8, fs2=5.5, shadow=False)
box(ML+57, L6y+3.5, 52, 3.5,
    "Rate Limiting & Protection", P["ext"],
    "Login: 20 attempts / 5 min / IP\nLLM Token Bucket: 30 RPM, 100 K TPM\n"
    "Circuit Breaker: CLOSED → OPEN (5 fail) → HALF_OPEN (60s) → CLOSED (2 ok)",
    fs=8, fs2=5.5, shadow=False)

# ═══════════════════════════════════════════════════════════════
#  INTER-LAYER ARROWS   (colour-coded, labelled)
# ═══════════════════════════════════════════════════════════════

# L1 → L2  (Client → Nginx)
arrow(xs1[0]+bw1/2, L1y+1, ML+15+40, L2y+9.5+4,
      "HTTPS Request", lw=1.2, c="#1565C0", fs=7)
arrow(xs1[1]+bw1/2, L1y+1, ML+15+40, L2y+9.5+4,
      "REST API Call", lw=1.2, c="#1565C0", fs=7)

# L2 → Frontend
arrow(ML+15+20, L2y+9.5, fxs[1]+fbw/2, fy+fh,
      "Static Files", lw=0.8, c="#0277BD", fs=6)
arrow(ML+15+60, L2y+9.5, fxs[3]+fbw/2, fy+fh, lw=0.7, c="#0277BD")

# L2 → L3  (Nginx → Backend)
arrow(W/2, L2y+9.5, W/2, L3y+L3y-L3y+16.5,
      "/api/*  Reverse Proxy  (HTTP/1.1)", lw=1.5, c="#1A237E", fs=7.5)

# L3 → L4  (API routes → Core engines)
rc = [axs[i]+abw/2 for i in range(5)]
arrow(rc[0], ay, dxs2[0]+dbw3/2, dy2+dh2,
      "", lw=0.7, c="#7E57C2", rad=0.2)   # auth → SQLite
arrow(rc[1], ay, pleft+pbw+pgap/2, PAy+PAh,
      "", lw=0.8, c=P["hdr_price"], rad=0.0)
arrow(rc[2], ay, mleft+pbw+pgap/2, PBy+PBh,
      "", lw=0.8, c="#E65100", rad=0.0)
arrow(rc[3], ay, dleft+24, PCy+PCh,
      "", lw=0.8, c=P["hdr_dl"], rad=0.1)
arrow(rc[4], ay, rleft+pbw+pgap/2, PDy+PDh,
      "", lw=0.8, c=P["hdr_rag"], rad=-0.08)

# L4 → L5  (Core → Data)
arrow(SL+13, L4y+6, dxs2[1]+dbw3/2, dy2+dh2,
      "", lw=0.6, c="#6D4C41", rad=0.05)
arrow(SR+SW/2-5, L4y+6, dxs2[3]+dbw3/2, dy2+dh2,
      "", lw=0.6, c="#6D4C41", rad=-0.05)
arrow(SR+SW/2+5, L4y+6, dxs2[4]+dbw3/2, dy2+dh2,
      "", lw=0.6, c="#6D4C41", rad=-0.05)

# RAG → Gemini  (cross-layer highlight)
arrow(rleft+pbw+pgap+pbw/2, rr3,
      exs2[0]+ebw2/2, ey2+eh2,
      "Gemini API Call  (HTTPS)", lw=1.1, c="#D32F2F", rad=0.18, fs=6.5)

# ═══════════════════════════════════════════════════════════════
#  LEGEND
# ═══════════════════════════════════════════════════════════════
leg_y2 = 13.5
legs = [
    ("Frontend / SPA",  P["fe"]),
    ("API Gateway",     P["nginx"]),
    ("API Routes",      P["api"]),
    ("Pricing & Risk",  P["price"]),
    ("ML Pipeline",     P["ml"]),
    ("DL Pipeline",     P["dl"]),
    ("RAG Pipeline",    P["rag"]),
    ("Data / Storage",  P["data"]),
    ("External / Infra",P["ext"]),
]
lbw, lbh = 3, 1.5
lgap = 12.2
lstart = ML + (CW - len(legs)*lgap)/2

ax.text(W/2, leg_y2+3.5, "Legend", ha="center", fontsize=8.5,
        fontweight="bold", color=P["txt"])

for i, (lbl, clr) in enumerate(legs):
    lx = lstart + i*lgap
    p = FancyBboxPatch((lx, leg_y2), lbw, lbh,
        boxstyle="round,pad=0.08,rounding_size=0.2",
        fc=clr, ec=P["bdr"], lw=0.5, zorder=3)
    ax.add_patch(p)
    ax.text(lx+lbw+0.8, leg_y2+lbh/2, lbl, ha="left", va="center",
            fontsize=6, color=P["txt"])

# ═══════════════════════════════════════════════════════════════
#  FIGURE CAPTION
# ═══════════════════════════════════════════════════════════════
ax.text(W/2, 10,
    "Fig. 1.  System architecture of OptionQuant — an intelligent option pricing "
    "platform integrating Monte Carlo simulation,\n"
    "deep learning (LSTM, Transformer, Temporal CNN), ML-based volatility "
    "forecasting (7 models, 30+ features), and Retrieval-Augmented\n"
    "Generation (RAG) with a 13-stage pipeline for AI-powered financial "
    "explainability via Google Gemini LLM.",
    ha="center", va="center", fontsize=7, fontstyle="italic", color="#555")

# ═══════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════
out = (r"c:\Users\shiva\Downloads\Option-Pricing-Using-Monte-Carlo-Simulation"
       r"-Deep-Learning\architecture_diagram.png")
fig.savefig(out, bbox_inches="tight", dpi=DPI, facecolor="#FDFDFE",
            edgecolor="none", pad_inches=0.3)
plt.close(fig)
print(f"Saved: {out}")
print(f"Resolution: {DPI} DPI  |  Size: {FW} x {FH} in  |  "
      f"~{FW*DPI} x {FH*DPI} px")
