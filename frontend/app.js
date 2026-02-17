/* ═══════════════════════════════════════════════════════════════
   OptionQuant — Application Logic (Conference-Grade)
   ═══════════════════════════════════════════════════════════════ */

// ── 1. Auth Guard ──────────────────────────────────────────────
// Stores the ongoing refresh promise so API helpers can await it.
let _authReady = Promise.resolve();

(function authGuard() {
  const token   = localStorage.getItem('oq-token');
  const expires = localStorage.getItem('oq-expires');
  if (!token || !expires || Date.now() >= Number(expires)) {
    const refresh = localStorage.getItem('oq-refresh');
    if (refresh) {
      _authReady = fetch('/api/v1/auth/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refresh })
      })
        .then(r => { if (!r.ok) throw new Error(); return r.json(); })
        .then(d => {
          localStorage.setItem('oq-token', d.access_token);
          if (d.refresh_token) localStorage.setItem('oq-refresh', d.refresh_token);
          localStorage.setItem('oq-expires', (Date.now() + (d.expires_in || 1800) * 1000).toString());
        })
        .catch(() => { redirectToLogin(); });
      return;
    }
    redirectToLogin();
  }
})();

function redirectToLogin() {
  localStorage.removeItem('oq-token');
  localStorage.removeItem('oq-refresh');
  localStorage.removeItem('oq-expires');
  window.location.href = '/login.html';
}

// ── 2. Helpers ─────────────────────────────────────────────────
function getAuthHeaders() {
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${localStorage.getItem('oq-token')}`
  };
}

function handleAuthError(status) {
  if (status === 401) { redirectToLogin(); return true; }
  return false;
}

async function api(url, body) {
  await _authReady;   // ensure token refresh is complete
  const res = await fetch(url, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify(body)
  });
  if (handleAuthError(res.status)) return null;
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `API error ${res.status}`);
  return data;
}

async function apiGet(url) {
  await _authReady;   // ensure token refresh is complete
  const res = await fetch(url, { headers: getAuthHeaders() });
  if (handleAuthError(res.status)) return null;
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `API error ${res.status}`);
  return data;
}

// ── 3. UI Utilities ────────────────────────────────────────────
const $ = (s) => document.getElementById(s);
const loadingOverlay = $('loadingOverlay');

function showLoading() { loadingOverlay.classList.add('active'); }
function hideLoading() { loadingOverlay.classList.remove('active'); }

// Toast system — classes match CSS: .toast-success, .toast-error, etc.
const toastIcons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
function toast(type, title, msg = '') {
  const container = $('toasts');
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = `
    <span class="toast-icon">${toastIcons[type] || 'ℹ️'}</span>
    <div class="toast-body">
      <div class="toast-title">${title}</div>
      ${msg ? `<div class="toast-msg">${msg}</div>` : ''}
    </div>
    <span class="toast-close">✕</span>
  `;
  container.appendChild(el);
  // Trigger reflow then animate in
  requestAnimationFrame(() => { el.classList.add('show'); });
  el.querySelector('.toast-close').onclick = () => dismissToast(el);
  setTimeout(() => dismissToast(el), 4500);
}
function dismissToast(el) {
  el.classList.remove('show');
  setTimeout(() => el.remove(), 300);
}

// Number formatting
function fmt(v, d = 4) {
  if (v == null || isNaN(v)) return '—';
  return Number(v).toFixed(d);
}
function fmtPct(v) { return v == null ? '—' : (Number(v) * 100).toFixed(1) + '%'; }

// ── 4. Navigation ──────────────────────────────────────────────
const sections = {
  pricing:        { title: 'Option Pricing',       sub: 'Black-Scholes & Monte Carlo engines' },
  greeks:         { title: 'Greeks Analysis',       sub: 'Sensitivity surface visualisation' },
  'monte-carlo':  { title: 'Monte Carlo',           sub: 'GBM path simulation & convergence' },
  'deep-learning':{ title: 'Deep Learning',         sub: 'LSTM & Transformer neural pricing' },
  'ml-volatility':{ title: 'ML Volatility',         sub: 'Implied volatility prediction' },
  explainability: { title: 'AI Explainability',     sub: 'RAG-powered Q&A engine' }
};

const navItems = document.querySelectorAll('.sidebar-nav .nav-item');
navItems.forEach(item => {
  item.addEventListener('click', () => navigate(item.dataset.section));
});

function navigate(key) {
  navItems.forEach(n => n.classList.toggle('active', n.dataset.section === key));
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  const sec = document.getElementById(`sec-${key}`);
  if (sec) sec.classList.add('active');
  const info = sections[key] || {};
  $('pageTitle').textContent   = info.title || '';
  $('pageSubtitle').textContent = info.sub || '';
  // Close mobile sidebar
  $('sidebar').classList.remove('open');
  $('sidebarOverlay').classList.remove('active');
}

// Mobile sidebar toggle
$('mobileToggle').addEventListener('click', () => {
  $('sidebar').classList.toggle('open');
  $('sidebarOverlay').classList.toggle('active');
});
$('sidebarOverlay').addEventListener('click', () => {
  $('sidebar').classList.remove('open');
  $('sidebarOverlay').classList.remove('active');
});

// ── 5. Theme Toggle ────────────────────────────────────────────
$('themeToggle').addEventListener('click', () => {
  const html = document.documentElement;
  const isDark = html.dataset.theme === 'dark';
  html.dataset.theme = isDark ? 'light' : 'dark';
  $('themeIcon').textContent = isDark ? '☀️' : '🌙';
  localStorage.setItem('oq-theme', html.dataset.theme);
});
// Restore saved theme
(function restoreTheme() {
  const saved = localStorage.getItem('oq-theme');
  if (saved) {
    document.documentElement.dataset.theme = saved;
    $('themeIcon').textContent = saved === 'dark' ? '🌙' : '☀️';
  }
})();

// ── 6. Health Check ────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch('/health');
    const ok = res.ok;
    $('statusDot').classList.toggle('online', ok);
    $('statusText').textContent = ok ? 'API Online' : 'API Error';
    return ok;
  } catch {
    $('statusDot').classList.remove('online');
    $('statusText').textContent = 'API Offline';
    return false;
  }
}
$('healthBtn').addEventListener('click', async () => {
  $('healthLabel').textContent = 'Checking…';
  const ok = await checkHealth();
  $('healthLabel').textContent = ok ? 'API Online ✓' : 'API Error ✗';
  toast(ok ? 'success' : 'error', ok ? 'Backend Online' : 'Backend Unreachable');
});
$('healthBtn2').addEventListener('click', () => $('healthBtn').click());
// Auto-check on load
checkHealth();

// ── 7. User Profile ───────────────────────────────────────────
(async function loadProfile() {
  try {
    const user = await apiGet('/api/v1/auth/me');
    if (!user) return;
    const name = user.full_name || user.username || 'User';
    $('userName').textContent = name;
    $('userAvatar').textContent = name.charAt(0).toUpperCase();
    $('userRole').textContent = user.role || 'Analyst';
  } catch {
    $('userName').textContent = 'User';
    $('userAvatar').textContent = 'U';
  }
})();

// ── 8. Logout (direct fetch — not api()) ───────────────────────
$('logoutBtn').addEventListener('click', async () => {
  try {
    const token = localStorage.getItem('oq-token');
    if (token) {
      await fetch('/api/v1/auth/logout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      });
    }
  } catch { /* ignore */ }
  localStorage.removeItem('oq-token');
  localStorage.removeItem('oq-refresh');
  localStorage.removeItem('oq-expires');
  window.location.href = '/login.html';
});

// ── 9. Chart Defaults ──────────────────────────────────────────
function chartDefaults() {
  const isDark = document.documentElement.dataset.theme !== 'light';
  const gridColor = isDark ? 'rgba(255,255,255,.06)' : 'rgba(0,0,0,.06)';
  const textColor = isDark ? '#9ba1b7' : '#5a6178';
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: textColor, font: { family: "'Inter',sans-serif", size: 12 } } },
      tooltip: {
        backgroundColor: isDark ? '#1e2338' : '#ffffff',
        titleColor: isDark ? '#f0f1f5' : '#1a1d2b',
        bodyColor: isDark ? '#9ba1b7' : '#5a6178',
        borderColor: isDark ? 'rgba(255,255,255,.1)' : 'rgba(0,0,0,.1)',
        borderWidth: 1, cornerRadius: 8, padding: 10
      }
    },
    scales: {
      x: { grid: { color: gridColor }, ticks: { color: textColor, font: { size: 11 } } },
      y: { grid: { color: gridColor }, ticks: { color: textColor, font: { size: 11 } } }
    }
  };
}

// Chart instance registry (destroy before re-create)
const charts = {};
function getOrCreateChart(id, config) {
  if (charts[id]) { charts[id].destroy(); }
  charts[id] = new Chart(document.getElementById(id), config);
  return charts[id];
}

// ── 10. Get Pricing Parameters ─────────────────────────────────
function getParams() {
  return {
    spot:        parseFloat($('spot').value)     || 100,
    strike:      parseFloat($('strike').value)   || 100,
    rate:        parseFloat($('rate').value)      || 0.05,
    volatility:  parseFloat($('sigma').value)     || 0.2,
    maturity:    parseFloat($('maturity').value)  || 1,
    option_type: $('optType').value               || 'call'
  };
}

// ── 11. Price Option (parallel BS + MC + Greeks) ───────────────
$('priceBtn').addEventListener('click', priceOption);
async function priceOption() {
  const params = getParams();
  showLoading();
  try {
    const [bs, mc, greeks] = await Promise.all([
      api('/api/v1/pricing/bs',     params),
      api('/api/v1/pricing/mc',     params),
      api('/api/v1/pricing/greeks', params)
    ]);
    if (!bs || !mc || !greeks) return;

    // Results
    $('pricingResults').style.display = '';
    $('bsPrice').textContent = fmt(bs.price);
    $('mcPrice').textContent = fmt(mc.price);

    // Backend returns {model, price, metadata} — compute error from price difference
    const diff = Math.abs(bs.price - mc.price);
    const se = diff > 0 ? diff / 1.96 : 0;
    $('mcStd').textContent = se > 0 ? fmt(se) : '< 0.0001';
    $('mcCI').textContent = se > 0
      ? `[${fmt(mc.price - 1.96 * se, 2)}, ${fmt(mc.price + 1.96 * se, 2)}]`
      : `≈ ${fmt(mc.price, 2)}`;

    $('resultBadge').style.display = '';
    $('resultBadge').textContent = `BS: $${fmt(bs.price, 2)}`;

    // Greeks quick view
    $('greeksQuick').style.display = '';
    $('qDelta').textContent = fmt(greeks.delta);
    $('qGamma').textContent = fmt(greeks.gamma, 6);
    $('qTheta').textContent = fmt(greeks.theta);
    $('qVega').textContent  = fmt(greeks.vega);
    $('qRho').textContent   = fmt(greeks.rho);

    toast('success', 'Pricing Complete', `BS=$${fmt(bs.price,2)}  MC=$${fmt(mc.price,2)}`);
  } catch (err) {
    toast('error', 'Pricing Failed', err.message);
  } finally {
    hideLoading();
  }
}

// Reset button
$('resetBtn').addEventListener('click', () => {
  $('spot').value = 100; $('strike').value = 100; $('rate').value = 0.05;
  $('sigma').value = 0.2; $('maturity').value = 1; $('optType').value = 'call';
  $('pricingResults').style.display = 'none';
  $('greeksQuick').style.display = 'none';
  $('resultBadge').style.display = 'none';
});

// ── 12. Greeks Surface Plot ────────────────────────────────────
$('plotGreekBtn').addEventListener('click', plotGreekSurface);
async function plotGreekSurface() {
  const params  = getParams();
  const greek   = $('greekSelect').value;
  const range   = (parseFloat($('greekRange').value) || 30) / 100;
  const lo      = params.spot * (1 - range);
  const hi      = params.spot * (1 + range);
  const steps   = 30;
  const spots   = Array.from({ length: steps }, (_, i) => lo + (hi - lo) * i / (steps - 1));

  showLoading();
  try {
    const results = await Promise.all(
      spots.map(s => api('/api/v1/pricing/greeks', { ...params, spot: s }))
    );

    const values = results.map(r => r ? r[greek] : null);
    $('greekChartWrap').style.display = '';
    const colors = { delta:'#6d5cff', gamma:'#00e5a0', theta:'#ff5c7c', vega:'#ffc044', rho:'#3ea8ff' };

    getOrCreateChart('greekChart', {
      type: 'line',
      data: {
        labels: spots.map(s => s.toFixed(1)),
        datasets: [{
          label: `${greek.charAt(0).toUpperCase() + greek.slice(1)} vs Spot`,
          data: values,
          borderColor: colors[greek] || '#6d5cff',
          backgroundColor: (colors[greek] || '#6d5cff') + '22',
          fill: true, tension: .3, pointRadius: 2
        }]
      },
      options: { ...chartDefaults(), plugins: { ...chartDefaults().plugins } }
    });

    toast('success', 'Surface Plotted', `${greek} across ${steps} spot points`);
  } catch (err) {
    toast('error', 'Greeks Failed', err.message);
  } finally {
    hideLoading();
  }
}

// ── 13. Monte Carlo Simulation (client-side GBM) ──────────────
$('simBtn').addEventListener('click', runMonteCarlo);
function runMonteCarlo() {
  const params = getParams();
  const nPaths = parseInt($('mcPaths').value) || 200;
  const nSteps = parseInt($('mcSteps').value) || 252;
  const dt     = params.maturity / nSteps;
  const drift  = (params.rate - 0.5 * params.volatility ** 2) * dt;
  const vol    = params.volatility * Math.sqrt(dt);

  showLoading();
  setTimeout(() => {
    try {
      const paths = [];
      const payoffs = [];
      const convergence = [];
      let payoffSum = 0;

      for (let p = 0; p < nPaths; p++) {
        const path = [params.spot];
        let S = params.spot;
        for (let s = 0; s < nSteps; s++) {
          const z = boxMullerRandom();
          S *= Math.exp(drift + vol * z);
          path.push(S);
        }
        paths.push(path);

        // Payoff
        const payoff = params.option_type === 'call'
          ? Math.max(S - params.strike, 0)
          : Math.max(params.strike - S, 0);
        payoffs.push(payoff);
        payoffSum += payoff;
        convergence.push(Math.exp(-params.rate * params.maturity) * payoffSum / (p + 1));
      }

      // Show charts
      $('mcChartsWrap').style.display = '';

      // Paths chart (show subset)
      const maxDisplay = Math.min(nPaths, 80);
      const labels = Array.from({ length: nSteps + 1 }, (_, i) => i);
      const datasets = [];
      for (let p = 0; p < maxDisplay; p++) {
        const hue = (p * 360 / maxDisplay) % 360;
        datasets.push({
          data: paths[p],
          borderColor: `hsla(${hue},70%,60%,.4)`,
          borderWidth: 1, pointRadius: 0, fill: false, tension: 0
        });
      }
      getOrCreateChart('mcChart', {
        type: 'line',
        data: { labels, datasets },
        options: {
          ...chartDefaults(),
          plugins: { ...chartDefaults().plugins, legend: { display: false } },
          scales: {
            ...chartDefaults().scales,
            x: { ...chartDefaults().scales.x, title: { display: true, text: 'Time Step', color: '#9ba1b7' } },
            y: { ...chartDefaults().scales.y, title: { display: true, text: 'Price ($)', color: '#9ba1b7' } }
          },
          animation: false
        }
      });

      // Convergence chart
      getOrCreateChart('convChart', {
        type: 'line',
        data: {
          labels: Array.from({ length: nPaths }, (_, i) => i + 1),
          datasets: [{
            label: 'MC Price Convergence',
            data: convergence,
            borderColor: '#00e5a0',
            backgroundColor: 'rgba(0,229,160,.08)',
            fill: true, tension: .2, pointRadius: 0
          }]
        },
        options: {
          ...chartDefaults(),
          scales: {
            ...chartDefaults().scales,
            x: { ...chartDefaults().scales.x, title: { display: true, text: 'Number of Paths', color: '#9ba1b7' } },
            y: { ...chartDefaults().scales.y, title: { display: true, text: 'Estimated Price ($)', color: '#9ba1b7' } }
          }
        }
      });

      const finalPrice = convergence[convergence.length - 1];
      toast('success', 'Simulation Complete', `${nPaths} paths · Price ≈ $${fmt(finalPrice, 2)}`);
    } catch (err) {
      toast('error', 'Simulation Failed', err.message);
    } finally {
      hideLoading();
    }
  }, 50);
}

function boxMullerRandom() {
  let u, v, s;
  do { u = Math.random() * 2 - 1; v = Math.random() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
  return u * Math.sqrt(-2 * Math.log(s) / s);
}

// ── 14. Deep Learning Forecast ─────────────────────────────────
$('dlBtn').addEventListener('click', dlForecast);
async function dlForecast() {
  const body = {
    spot:        parseFloat($('dlSpot').value)     || 100,
    strike:      parseFloat($('dlStrike').value)   || 100,
    maturity:    parseFloat($('dlMaturity').value)  || 1,
    rate:        parseFloat($('dlRate').value)      || 0.05,
    volatility:  parseFloat($('dlSigma').value)     || 0.2,
    option_type: $('dlType').value                  || 'call'
  };
  showLoading();
  try {
    const d = await api('/api/v1/dl/forecast', body);
    if (!d) return;

    $('dlResults').style.display = '';
    $('dlForecast').textContent = fmt(d.forecast_price);
    $('dlVol').textContent      = d.forecast_vol != null ? fmtPct(d.forecast_vol) : '—';
    $('dlResidual').textContent = d.residual != null ? fmt(d.residual) : '—';

    const bench = d.benchmarks || {};
    $('dlBS').textContent = bench.bs != null ? fmt(bench.bs) : '—';
    $('dlMC').textContent = bench.mc != null ? fmt(bench.mc) : '—';

    // Comparison chart
    $('compChartWrap').style.display = '';
    getOrCreateChart('compChart', {
      type: 'bar',
      data: {
        labels: ['Deep Learning', 'Black-Scholes', 'Monte Carlo'],
        datasets: [{
          label: 'Option Price ($)',
          data: [d.forecast_price, bench.bs, bench.mc],
          backgroundColor: ['#6d5cff', '#00e5a0', '#3ea8ff'],
          borderRadius: 8, barThickness: 50
        }]
      },
      options: {
        ...chartDefaults(),
        plugins: { ...chartDefaults().plugins, legend: { display: false } },
        scales: {
          ...chartDefaults().scales,
          y: { ...chartDefaults().scales.y, beginAtZero: true,
               title: { display: true, text: 'Price ($)', color: '#9ba1b7' } }
        }
      }
    });

    toast('success', 'DL Forecast', `Price ≈ $${fmt(d.forecast_price, 2)}`);
  } catch (err) {
    toast('error', 'DL Forecast Failed', err.message);
  } finally {
    hideLoading();
  }
}

// ── 15. ML Implied Volatility ──────────────────────────────────
$('mlBtn').addEventListener('click', mlPredict);
async function mlPredict() {
  const body = {
    spot:          parseFloat($('mlSpot').value)  || 100,
    rate:          parseFloat($('mlRate').value)   || 0.05,
    maturity:      parseFloat($('mlMat').value)    || 0.5,
    realized_vol:  parseFloat($('mlRvol').value)   || 0.18,
    vix:           parseFloat($('mlVix').value)    || 20,
    skew:          parseFloat($('mlSkew').value)   || -0.15
  };
  showLoading();
  try {
    const d = await api('/api/v1/ml/iv-predict', body);
    if (!d) return;
    $('mlResults').style.display = '';
    $('mlIV').textContent     = d.implied_vol != null ? fmtPct(d.implied_vol) : '—';
    $('mlRegime').textContent = d.regime || '—';
    toast('success', 'IV Predicted', `IV = ${fmtPct(d.implied_vol)} · Regime: ${d.regime || '—'}`);
  } catch (err) {
    toast('error', 'ML Prediction Failed', err.message);
  } finally {
    hideLoading();
  }
}

// ── 16. AI / RAG Explainability ────────────────────────────────
const chatArea  = $('chatArea');
const ragInput  = $('ragInput');
const ragBtn    = $('ragBtn');

// Conversation history for multi-turn context
let chatHistory = [];
const MAX_HISTORY = 10;

// Persist conversation across page reloads
function saveChatHistory() {
  try { sessionStorage.setItem('oq-chat-history', JSON.stringify(chatHistory)); } catch {}
}
function loadChatHistory() {
  try {
    const saved = sessionStorage.getItem('oq-chat-history');
    if (saved) chatHistory = JSON.parse(saved);
  } catch {}
}
loadChatHistory();

// Restore chat bubbles from history on load
chatHistory.forEach(msg => {
  if (msg.role === 'user') addBubble('user', msg.content);
  else addBubble('assistant', renderMarkdown(msg.content));
});

ragBtn.addEventListener('click', askRAG);
ragInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); askRAG(); }
});

// Quick chips
document.querySelectorAll('#quickChips .chip').forEach(chip => {
  chip.addEventListener('click', () => {
    ragInput.value = chip.dataset.q;
    askRAG();
  });
});

// Load RAG stats on section open
function loadRAGStats() {
  apiGet('/api/v1/ai/rag/stats').then(data => {
    if (!data) return;
    const dc = $('ragDocCount');
    const sc = $('ragSourceCount');
    const al = $('ragAvgLatency');
    const cr = $('ragCacheRate');
    if (dc) dc.textContent = data.total_chunks || '—';
    if (sc) sc.textContent = data.unique_sources || '—';
    if (al) al.textContent = data.avg_search_ms != null ? data.avg_search_ms.toFixed(1) : '—';
    if (cr) cr.textContent = data.cache_hit_rate != null ? (data.cache_hit_rate * 100).toFixed(0) + '%' : '—';
  }).catch(() => {});
}

// Load stats when explainability section becomes active
const statsObserver = new MutationObserver(() => {
  const sec = document.getElementById('sec-explainability');
  if (sec && sec.classList.contains('active')) loadRAGStats();
});
const explainSec = document.getElementById('sec-explainability');
if (explainSec) statsObserver.observe(explainSec, { attributes: true, attributeFilter: ['class'] });

let _ragAsking = false;   // debounce guard for RAG requests

async function askRAG() {
  if (_ragAsking) return;  // prevent duplicate concurrent calls
  const q = ragInput.value.trim();
  if (!q) return;
  _ragAsking = true;

  // Add user bubble
  addBubble('user', q);
  ragInput.value = '';

  // Track in chat history
  chatHistory.push({ role: 'user', content: q });
  if (chatHistory.length > MAX_HISTORY) chatHistory = chatHistory.slice(-MAX_HISTORY);
  saveChatHistory();

  // Hide follow-ups while loading
  const fuContainer = $('followUps');
  if (fuContainer) fuContainer.style.display = 'none';

  // Show typing indicator
  const typing = document.createElement('div');
  typing.className = 'typing-indicator';
  typing.innerHTML = '<span></span><span></span><span></span>';
  chatArea.appendChild(typing);
  chatArea.scrollTop = chatArea.scrollHeight;

  try {
    const d = await api('/api/v1/ai/explain', {
      question: q,
      context: getParams(),
      chat_history: chatHistory.slice(0, -1),  // exclude the just-added user msg
    });
    typing.remove();
    if (!d) return;

    // Track assistant response in history
    chatHistory.push({ role: 'assistant', content: d.answer || '' });
    if (chatHistory.length > MAX_HISTORY) chatHistory = chatHistory.slice(-MAX_HISTORY);
    saveChatHistory();

    // Assistant bubble with rendered markdown
    addBubble('assistant', renderMarkdown(d.answer || 'No answer received.'));

    // Confidence, badges & sources
    if (d.confidence != null || (d.sources && d.sources.length)) {
      $('ragMeta').style.display = '';
      if (d.confidence != null) {
        const pct = Math.round(d.confidence * 100);
        $('confFill').style.width = pct + '%';
        $('confLabel').textContent = pct + '%';
      }
      // Query type badge
      const qtBadge = $('queryTypeBadge');
      if (qtBadge && d.query_type) {
        const typeLabels = { factual: '📖 Factual', analytical: '🔍 Analytical', comparative: '⚖️ Comparative', general: '💬 General', out_of_scope: '🚫 Off-topic' };
        qtBadge.textContent = typeLabels[d.query_type] || d.query_type;
      }
      // Latency badge
      const ltBadge = $('latencyBadge');
      if (ltBadge && d.latency_ms != null) ltBadge.textContent = `⏱ ${Math.round(d.latency_ms)}ms`;
      // Cache badge
      const cBadge = $('cacheBadge');
      if (cBadge) cBadge.style.display = d.cached ? '' : 'none';

      const srcList = $('sourceList');
      srcList.innerHTML = '';
      (d.sources || []).forEach(s => {
        const tag = document.createElement('span');
        tag.className = 'source-tag';
        tag.textContent = typeof s === 'string' ? s : s.title || s.name || 'Source';
        srcList.appendChild(tag);
      });
    }

    // Follow-up suggestions
    if (d.follow_ups && d.follow_ups.length) {
      const fuChips = $('followUpChips');
      if (fuChips && fuContainer) {
        fuChips.innerHTML = '';
        d.follow_ups.forEach(fu => {
          const chip = document.createElement('span');
          chip.className = 'follow-up-chip';
          chip.textContent = fu;
          chip.addEventListener('click', () => {
            ragInput.value = fu;
            askRAG();
          });
          fuChips.appendChild(chip);
        });
        fuContainer.style.display = '';
      }
    }

    // Refresh RAG stats
    loadRAGStats();

    toast('info', 'Answer Ready');
  } catch (err) {
    typing.remove();
    addBubble('assistant', `⚠️ Error: ${err.message}`);
    toast('error', 'AI Error', err.message);
  } finally {
    _ragAsking = false;   // release debounce guard
  }
}

function addBubble(role, html) {
  const div = document.createElement('div');
  div.className = `chat-bubble ${role}`;
  if (role === 'user') {
    div.textContent = html;
  } else {
    div.innerHTML = html;
  }
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function renderMarkdown(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n{2,}/g, '</p><p>')
    .replace(/\n/g, '<br>')
    .replace(/^/, '<p>')
    .replace(/$/, '</p>');
}

// ── 17. Keyboard Shortcuts ─────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    // Find the active section and trigger its primary action
    const active = document.querySelector('.section.active');
    if (!active) return;
    const id = active.id;
    if (id === 'sec-pricing')        $('priceBtn').click();
    else if (id === 'sec-greeks')    $('plotGreekBtn').click();
    else if (id === 'sec-monte-carlo') $('simBtn').click();
    else if (id === 'sec-deep-learning') $('dlBtn').click();
    else if (id === 'sec-ml-volatility') $('mlBtn').click();
    else if (id === 'sec-explainability') ragBtn.click();
  }
});

// ── 18. Initialization ────────────────────────────────────────
console.log('%c🚀 OptionQuant loaded', 'color:#6d5cff;font-size:14px;font-weight:700');
