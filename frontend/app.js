/* ═══════════════════════════════════════════════════════════════
   OptionQuant — Application Logic (Conference-Grade)
   ═══════════════════════════════════════════════════════════════ */

// ── 1. Auth Guard ──────────────────────────────────────────────
// Stores the ongoing refresh promise so API helpers can await it.
let _authReady = Promise.resolve();
let _tokenRefreshTimer = null;

const API_BASE = window.location.protocol === 'file:' ? 'http://localhost:8001' : '';
function apiUrl(path) {
  return `${API_BASE}${path}`;
}

(function authGuard() {
  const token   = localStorage.getItem('oq-token');
  const expires = localStorage.getItem('oq-expires');
  if (!token || !expires || Date.now() >= Number(expires)) {
    const refresh = localStorage.getItem('oq-refresh');
    if (refresh) {
      _authReady = fetch(apiUrl('/api/v1/auth/refresh'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refresh })
      })
        .then(r => { if (!r.ok) throw new Error(); return r.json(); })
        .then(d => {
          localStorage.setItem('oq-token', d.access_token);
          if (d.refresh_token) localStorage.setItem('oq-refresh', d.refresh_token);
          localStorage.setItem('oq-expires', (Date.now() + (d.expires_in || 1800) * 1000).toString());
          scheduleTokenRefresh(d.expires_in || 1800);
        })
        .catch(() => { redirectToLogin(); });
      return;
    }
    redirectToLogin();
  } else {
    // Schedule proactive refresh before expiry
    const remaining = Math.max(0, (Number(expires) - Date.now()) / 1000);
    scheduleTokenRefresh(remaining);
  }
})();

function scheduleTokenRefresh(expiresInSec) {
  if (_tokenRefreshTimer) clearTimeout(_tokenRefreshTimer);
  // Refresh 60s before expiry (or at 75% of lifetime, whichever is sooner)
  const refreshIn = Math.max(10, Math.min(expiresInSec - 60, expiresInSec * 0.75)) * 1000;
  _tokenRefreshTimer = setTimeout(async () => {
    const refresh = localStorage.getItem('oq-refresh');
    if (!refresh) return;
    try {
      const res = await fetch(apiUrl('/api/v1/auth/refresh'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refresh })
      });
      if (!res.ok) throw new Error();
      const d = await res.json();
      localStorage.setItem('oq-token', d.access_token);
      if (d.refresh_token) localStorage.setItem('oq-refresh', d.refresh_token);
      localStorage.setItem('oq-expires', (Date.now() + (d.expires_in || 1800) * 1000).toString());
      scheduleTokenRefresh(d.expires_in || 1800);
    } catch {
      // Silent fail — next API call will trigger redirect if needed
    }
  }, refreshIn);
}

function redirectToLogin() {
  localStorage.removeItem('oq-token');
  localStorage.removeItem('oq-refresh');
  localStorage.removeItem('oq-expires');
  window.location.href = apiUrl('/login.html');
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

async function api(url, body, { retries = 1, timeout = 30000 } = {}) {
  await _authReady;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const res = await fetch(apiUrl(url), {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      clearTimeout(timer);
      if (handleAuthError(res.status)) return null;
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `API error ${res.status}`);
      return data;
    } catch (err) {
      clearTimeout(timer);
      if (err.name === 'AbortError') throw new Error('Request timed out');
      if (attempt < retries) {
        await new Promise(r => setTimeout(r, 500 * (attempt + 1)));
        continue;
      }
      throw err;
    }
  }
}

async function apiGet(url, { retries = 1, timeout = 15000 } = {}) {
  await _authReady;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const res = await fetch(apiUrl(url), { headers: getAuthHeaders(), signal: controller.signal });
      clearTimeout(timer);
      if (handleAuthError(res.status)) return null;
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `API error ${res.status}`);
      return data;
    } catch (err) {
      clearTimeout(timer);
      if (err.name === 'AbortError') throw new Error('Request timed out');
      if (attempt < retries) {
        await new Promise(r => setTimeout(r, 500 * (attempt + 1)));
        continue;
      }
      throw err;
    }
  }
}

// ── 3. UI Utilities ────────────────────────────────────────────
const $ = (s) => document.getElementById(s);
const loadingOverlay = $('loadingOverlay');

function showLoading() { loadingOverlay.classList.add('active'); }
function hideLoading() { loadingOverlay.classList.remove('active'); }

// Toast system — classes match CSS: .toast-success, .toast-error, etc.
const toastIcons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
/** Safe parseFloat — only uses fallback for NaN (empty/invalid), not for 0 */
function pf(id, fallback) {
  const v = parseFloat($(id).value);
  return Number.isNaN(v) ? fallback : v;
}
function toast(type, title, msg = '') {
  const container = $('toasts');
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = `
    <span class="toast-icon">${toastIcons[type] || 'ℹ️'}</span>
    <div class="toast-body">
      <div class="toast-title">${escapeHtml(title)}</div>
      ${msg ? `<div class="toast-msg">${escapeHtml(msg)}</div>` : ''}
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
  dashboard:      { title: 'Dashboard',           sub: 'System overview & quick actions' },
  pricing:        { title: 'Option Pricing',       sub: 'Black-Scholes & Monte Carlo engines' },
  greeks:         { title: 'Greeks Analysis',       sub: 'Sensitivity surface visualisation' },
  'monte-carlo':  { title: 'Monte Carlo',           sub: 'GBM path simulation & convergence' },
  'deep-learning':{ title: 'Deep Learning',         sub: 'LSTM & Transformer neural pricing' },
  'ml-volatility':{ title: 'ML Volatility',         sub: 'Implied volatility prediction' },
  sentiment:      { title: 'Market Sentiment',      sub: 'Financial news NLP analysis' },
  'risk-analytics':{ title: 'Risk Analytics',       sub: 'VaR & risk decomposition' },
  explainability: { title: 'AI Explainability',     sub: 'RAG-powered Q&A engine' },
  'quant-dashboard': { title: 'Quant Intelligence', sub: 'Unified quant ecosystem status' },
  pinns:          { title: 'PINNs Pricing',         sub: 'Physics-informed neural network pricing' },
  'rl-hedging':   { title: 'RL Hedging',            sub: 'Reinforcement learning dynamic hedging' },
  'vol-surface':  { title: 'Vol Surface',            sub: 'Transformer implied vol surface' },
  'jump-diffusion': { title: 'Jump Diffusion',      sub: 'Merton model & regime switching' },
  arbitrage:      { title: 'Arbitrage Scanner',      sub: 'Multi-dimensional arbitrage detection' },
  uncertainty:    { title: 'Uncertainty',             sub: 'Bayesian uncertainty quantification' },
  'gpu-mc':       { title: 'GPU Monte Carlo',        sub: 'CUDA-accelerated MC pricing' },
  'portfolio-risk': { title: 'Portfolio Risk',       sub: 'Portfolio VaR & stress testing' },
  'market-intel': { title: 'Market Intelligence',    sub: 'Real-time market data & option chains' },
  mispricing:     { title: 'Mispricing Scanner',     sub: 'Detect mispriced options vs model fair value' },
  regime:         { title: 'Regime Detection',       sub: 'HMM-based market regime classification' },
  'shap-explain': { title: 'SHAP Explain',           sub: 'Shapley value feature attribution' },
  benchmark:      { title: 'Benchmark',              sub: 'Multi-engine performance profiling' },
};

const navItems = document.querySelectorAll('.sidebar-nav .nav-item');
navItems.forEach(item => {
  item.addEventListener('click', () => navigate(item.dataset.section));
  // Keyboard: Enter or Space triggers navigation
  item.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      navigate(item.dataset.section);
    }
  });
});

function navigate(key) {
  navItems.forEach(n => n.classList.toggle('active', n.dataset.section === key));
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  const sec = document.getElementById(`sec-${key}`);
  if (sec) sec.classList.add('active');
  const info = sections[key] || {};
  $('pageTitle').textContent   = info.title || '';
  $('pageSubtitle').textContent = info.sub || '';
  closeSidebar();
}

// ── 4a. Sidebar Open/Close ─────────────────────────────────────
const sidebarEl  = $('sidebar');
const overlayEl  = $('sidebarOverlay');
const toggleBtn  = $('mobileToggle');
let _sidebarOpen = false;

function openSidebar() {
  if (_sidebarOpen) return;
  _sidebarOpen = true;
  sidebarEl.classList.add('open');
  overlayEl.classList.add('active');
  toggleBtn.setAttribute('aria-expanded', 'true');
  document.body.classList.add('sidebar-open');
  // Focus first nav item for accessibility
  const firstItem = sidebarEl.querySelector('.nav-item');
  if (firstItem) firstItem.focus();
}

function closeSidebar() {
  if (!_sidebarOpen) return;
  _sidebarOpen = false;
  sidebarEl.classList.remove('open');
  overlayEl.classList.remove('active');
  toggleBtn.setAttribute('aria-expanded', 'false');
  document.body.classList.remove('sidebar-open');
}

// Toggle button
toggleBtn.addEventListener('click', () => {
  _sidebarOpen ? closeSidebar() : openSidebar();
});

// Overlay click closes
overlayEl.addEventListener('click', closeSidebar);

// Escape key closes sidebar
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && _sidebarOpen) {
    closeSidebar();
    toggleBtn.focus();
  }
});

// ── 4b. Swipe-to-close gesture (mobile) ────────────────────────
(function initSwipeToClose() {
  let touchStartX = 0;
  let touchStartY = 0;
  let isSwiping = false;

  sidebarEl.addEventListener('touchstart', (e) => {
    const touch = e.touches[0];
    touchStartX = touch.clientX;
    touchStartY = touch.clientY;
    isSwiping = true;
  }, { passive: true });

  sidebarEl.addEventListener('touchmove', (e) => {
    if (!isSwiping || !_sidebarOpen) return;
    const touch = e.touches[0];
    const dx = touch.clientX - touchStartX;
    const dy = Math.abs(touch.clientY - touchStartY);
    // Only horizontal swipe (left), ignore vertical scrolling
    if (dy > Math.abs(dx)) { isSwiping = false; return; }
    if (dx < -40) {
      closeSidebar();
      isSwiping = false;
    }
  }, { passive: true });

  sidebarEl.addEventListener('touchend', () => { isSwiping = false; }, { passive: true });

  // Also allow swipe from left edge to open (on main content)
  let edgeTouchX = 0;
  document.addEventListener('touchstart', (e) => {
    const touch = e.touches[0];
    if (touch.clientX < 24 && !_sidebarOpen) {
      edgeTouchX = touch.clientX;
    } else {
      edgeTouchX = -1;
    }
  }, { passive: true });

  document.addEventListener('touchmove', (e) => {
    if (edgeTouchX < 0) return;
    const touch = e.touches[0];
    if (touch.clientX - edgeTouchX > 60) {
      openSidebar();
      edgeTouchX = -1;
    }
  }, { passive: true });
})();

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
    const ctrl = new AbortController();
    setTimeout(() => ctrl.abort(), 5000);
    const res = await fetch(apiUrl('/health'), { signal: ctrl.signal });
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
      await fetch(apiUrl('/api/v1/auth/logout'), {
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
  sessionStorage.removeItem('oq-chat-history');
  window.location.href = apiUrl('/login.html');
});

// ── 9. Chart Defaults ──────────────────────────────────────────
function chartDefaults(overrides) {
  const isDark = document.documentElement.dataset.theme !== 'light';
  const gridColor = isDark ? 'rgba(255,255,255,.05)' : 'rgba(0,0,0,.05)';
  const textColor = isDark ? '#a1a7c0' : '#4a5068';
  const tooltipBg = isDark ? '#161b2e' : '#ffffff';
  const base = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 700, easing: 'easeOutQuart' },
    interaction: { mode: 'nearest', intersect: false, axis: 'x' },
    plugins: {
      legend: {
        labels: {
          color: textColor,
          font: { family: "'Inter',sans-serif", size: 12, weight: '500' },
          usePointStyle: true,
          pointStyle: 'circle',
          padding: 16,
        },
      },
      tooltip: {
        backgroundColor: tooltipBg,
        titleColor: isDark ? '#f0f1f5' : '#1a1d2b',
        bodyColor: isDark ? '#a1a7c0' : '#4a5068',
        borderColor: isDark ? 'rgba(109,92,255,.3)' : 'rgba(0,0,0,.08)',
        borderWidth: 1,
        cornerRadius: 10,
        padding: { top: 10, bottom: 10, left: 14, right: 14 },
        titleFont: { family: "'Inter',sans-serif", size: 13, weight: '600' },
        bodyFont: { family: "'Inter',sans-serif", size: 12 },
        boxPadding: 6,
        usePointStyle: true,
        displayColors: true,
        caretSize: 6,
      },
    },
    scales: {
      x: {
        grid: { color: gridColor, drawBorder: false },
        ticks: { color: textColor, font: { family: "'Inter',sans-serif", size: 11 }, padding: 6 },
        border: { display: false },
      },
      y: {
        grid: { color: gridColor, drawBorder: false },
        ticks: { color: textColor, font: { family: "'Inter',sans-serif", size: 11 }, padding: 6 },
        border: { display: false },
      },
    },
  };
  return overrides ? deepMergeChart(base, overrides) : base;
}

function deepMergeChart(target, source) {
  const out = { ...target };
  for (const key of Object.keys(source)) {
    if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
      out[key] = deepMergeChart(target[key] || {}, source[key]);
    } else {
      out[key] = source[key];
    }
  }
  return out;
}

function gradientFill(ctx, chartArea, r, g, b) {
  if (!chartArea) return `rgba(${r},${g},${b},0.15)`;
  const grad = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
  grad.addColorStop(0, `rgba(${r},${g},${b},0.28)`);
  grad.addColorStop(0.6, `rgba(${r},${g},${b},0.06)`);
  grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
  return grad;
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
    spot:        pf('spot', 100),
    strike:      pf('strike', 100),
    rate:        pf('rate', 0.05),
    volatility:  pf('sigma', 0.2),
    maturity:    pf('maturity', 1),
    option_type: $('optType').value || 'call'
  };
}

// ── 11. Price Option (parallel BS + MC + Greeks) ───────────────
$('priceBtn').addEventListener('click', priceOption);
async function priceOption() {
  const params = getParams();
  showLoading();
  try {
    const [bsResult, mcResult, greeksResult] = await Promise.allSettled([
      api('/api/v1/pricing/bs',     params),
      api('/api/v1/pricing/mc',     params),
      api('/api/v1/pricing/greeks', params)
    ]);
    const bs = bsResult.status === 'fulfilled' ? bsResult.value : null;
    const mc = mcResult.status === 'fulfilled' ? mcResult.value : null;
    const greeks = greeksResult.status === 'fulfilled' ? greeksResult.value : null;
    if (!bs && !mc && !greeks) { toast('error', 'Pricing Failed', 'All pricing engines failed'); return; }

    // Results
    $('pricingResults').style.display = '';
    $('bsPrice').textContent = bs ? fmt(bs.price) : '—';
    $('mcPrice').textContent = mc ? fmt(mc.price) : '—';

    // Backend returns {model, price, metadata} — use actual std_error & CI from metadata
    if (mc) {
      const meta = mc.metadata || {};
      const se = meta.std_error != null ? meta.std_error : (bs ? Math.abs(bs.price - mc.price) / 1.96 : 0);
      $('mcStd').textContent = se > 0.00005 ? fmt(se) : '< 0.0001';
      const ciLo = meta.ci_lower != null ? meta.ci_lower : mc.price - 1.96 * se;
      const ciHi = meta.ci_upper != null ? meta.ci_upper : mc.price + 1.96 * se;
      $('mcCI').textContent = se > 0.00005
        ? `[${fmt(ciLo, 2)}, ${fmt(ciHi, 2)}]`
        : `≈ ${fmt(mc.price, 2)}`;
    } else {
      $('mcStd').textContent = '—';
      $('mcCI').textContent = '—';
    }

    if (bs) {
      $('resultBadge').style.display = '';
      $('resultBadge').textContent = `BS: $${fmt(bs.price, 2)}`;
    }

    // Greeks quick view
    if (greeks) {
      $('greeksQuick').style.display = '';
      $('qDelta').textContent = fmt(greeks.delta);
      $('qGamma').textContent = fmt(greeks.gamma, 6);
      $('qTheta').textContent = fmt(greeks.theta);
      $('qVega').textContent  = fmt(greeks.vega);
      $('qRho').textContent   = fmt(greeks.rho);
    }

    toast('success', 'Pricing Complete', `BS=$${bs ? fmt(bs.price,2) : '—'}  MC=$${mc ? fmt(mc.price,2) : '—'}`);
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
  const range   = (pf('greekRange', 30)) / 100;
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
    const rgb = { delta:[109,92,255], gamma:[0,229,160], theta:[255,92,124], vega:[255,192,68], rho:[62,168,255] };
    const c = colors[greek] || '#6d5cff';
    const cRGB = rgb[greek] || [109,92,255];

    getOrCreateChart('greekChart', {
      type: 'line',
      data: {
        labels: spots.map(s => s.toFixed(1)),
        datasets: [{
          label: `${greek.charAt(0).toUpperCase() + greek.slice(1)} vs Spot`,
          data: values,
          borderColor: c,
          backgroundColor(ctx) {
            const chart = ctx.chart;
            const { ctx: cx, chartArea } = chart;
            return gradientFill(cx, chartArea, ...cRGB);
          },
          fill: true,
          tension: .35,
          pointRadius: 1.5,
          pointHoverRadius: 6,
          pointBackgroundColor: c,
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: c,
          pointHoverBorderWidth: 2.5,
          borderWidth: 2.5,
        }]
      },
      options: chartDefaults()
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
async function runMonteCarlo() {
  const params = getParams();
  const requestedPaths = parseInt($('mcPaths').value, 10) || 200;
  const requestedSteps = parseInt($('mcSteps').value, 10) || 252;

  if (!Number.isFinite(params.spot) || params.spot <= 0 ||
      !Number.isFinite(params.strike) || params.strike <= 0 ||
      !Number.isFinite(params.volatility) || params.volatility <= 0 ||
      !Number.isFinite(params.maturity) || params.maturity <= 0) {
    toast('warning', 'Invalid Inputs', 'Spot/strike/volatility/maturity must be positive numbers');
    return;
  }

  const nPaths = Math.max(10, Math.min(requestedPaths, 500000));
  const nSteps = Math.max(10, Math.min(requestedSteps, 2000));

  showLoading();
  try {
    // Primary path: backend detailed MC avoids browser UI-thread stalls.
    const d = await api('/api/v1/pricing/mc/detailed', {
      ...params,
      paths: nPaths,
      steps: nSteps,
      method: 'antithetic',
      return_paths: true,
      seed: 42,
    }, { timeout: 180000, retries: 1 });

    if (!d) return;

    const paths = Array.isArray(d.sample_paths) ? d.sample_paths : [];
    const convergence = Array.isArray(d.convergence) ? d.convergence : [];
    const meanPath = Array.isArray(d.mean_path) ? d.mean_path : [];
    const ciLowerPath = Array.isArray(d.ci_lower_path) ? d.ci_lower_path : [];
    const ciUpperPath = Array.isArray(d.ci_upper_path) ? d.ci_upper_path : [];
    const stabilityWarnings = Array.isArray(d.warnings) ? d.warnings : [];

    if (!paths.length || !convergence.length) {
      throw new Error('Simulation returned incomplete chart data');
    }

    renderMonteCarloCharts(paths, convergence, params, {
      meanPath,
      ciLowerPath,
      ciUpperPath,
    });

    if (stabilityWarnings.length) {
      toast('warning', 'Monte Carlo Warning', stabilityWarnings[0]);
    }

    toast('success', 'Simulation Complete', `${d.paths_used || nPaths} paths · Price ≈ $${fmt(d.price, 2)} · ${fmt(d.elapsed_ms, 0)}ms`);
  } catch (apiErr) {
    console.warn('Monte Carlo API fallback to local simulation:', apiErr);

    // Fallback path: local simulation with bounded memory usage.
    const dt = params.maturity / nSteps;
    const drift = (params.rate - 0.5 * params.volatility ** 2) * dt;
    const vol = params.volatility * Math.sqrt(dt);
    const maxDisplay = Math.min(nPaths, 80);
    const paths = [];
    const convergence = [];
    let payoffSum = 0;

    for (let p = 0; p < nPaths; p++) {
      let S = params.spot;
      const keepPath = p < maxDisplay;
      const path = keepPath ? [params.spot] : null;

      for (let s = 0; s < nSteps; s++) {
        const z = boxMullerRandom();
        S *= Math.exp(drift + vol * z);
        if (keepPath) path.push(S);
      }

      if (keepPath) paths.push(path);

      const payoff = params.option_type === 'call'
        ? Math.max(S - params.strike, 0)
        : Math.max(params.strike - S, 0);
      payoffSum += payoff;
      convergence.push(Math.exp(-params.rate * params.maturity) * payoffSum / (p + 1));
    }

    if (!paths.length || !convergence.length) {
      throw new Error('Local simulation produced no data');
    }

    renderMonteCarloCharts(paths, convergence, params, computePathStatistics(paths));
    const finalPrice = convergence[convergence.length - 1];
    toast('info', 'Simulation Complete (Local Mode)', `${nPaths} paths · Price ≈ $${fmt(finalPrice, 2)}`);
  } finally {
    hideLoading();
  }
}

function computePathStatistics(paths) {
  if (!Array.isArray(paths) || !paths.length) {
    return { meanPath: [], ciLowerPath: [], ciUpperPath: [] };
  }

  const valid = paths.filter(p => Array.isArray(p) && p.length > 0);
  if (!valid.length) {
    return { meanPath: [], ciLowerPath: [], ciUpperPath: [] };
  }

  const nSteps = Math.max(...valid.map(p => p.length));
  const nPaths = valid.length;
  const meanPath = new Array(nSteps).fill(0);
  const m2Path = new Array(nSteps).fill(0);

  for (let i = 0; i < nSteps; i++) {
    let count = 0;
    for (let p = 0; p < nPaths; p++) {
      const v = valid[p][i];
      if (!Number.isFinite(v)) continue;
      count += 1;
      const delta = v - meanPath[i];
      meanPath[i] += delta / count;
      const delta2 = v - meanPath[i];
      m2Path[i] += delta * delta2;
    }
  }

  const ciLowerPath = new Array(nSteps).fill(0);
  const ciUpperPath = new Array(nSteps).fill(0);
  for (let i = 0; i < nSteps; i++) {
    const variance = nPaths > 1 ? (m2Path[i] / (nPaths - 1)) : 0;
    const stderr = Math.sqrt(Math.max(variance, 0)) / Math.sqrt(Math.max(nPaths, 1));
    const delta = 1.96 * stderr;
    ciLowerPath[i] = meanPath[i] - delta;
    ciUpperPath[i] = meanPath[i] + delta;
  }

  return { meanPath, ciLowerPath, ciUpperPath };
}

function renderMonteCarloCharts(paths, convergence, params, pathStats = {}) {
  $('mcChartsWrap').style.display = '';

  const maxPathLen = paths.reduce((m, p) => Math.max(m, Array.isArray(p) ? p.length : 0), 0);
  const labels = Array.from({ length: maxPathLen || 1 }, (_, i) => i);
  const datasets = [];
  const palette = ['#6d5cff','#00e5a0','#ff5c7c','#3ea8ff','#ffc044','#22d3ee','#a78bfa','#f87171','#34d399','#fbbf24'];
  const meanPath = Array.isArray(pathStats.meanPath) ? pathStats.meanPath : [];
  const ciLowerPath = Array.isArray(pathStats.ciLowerPath) ? pathStats.ciLowerPath : [];
  const ciUpperPath = Array.isArray(pathStats.ciUpperPath) ? pathStats.ciUpperPath : [];

  for (let p = 0; p < paths.length; p++) {
    const color = palette[p % palette.length];
    datasets.push({
      data: paths[p],
      borderColor: color + '55',
      borderWidth: 1.2,
      pointRadius: 0,
      fill: false,
      tension: 0,
    });
  }

  datasets.push({
    label: `Strike ($${params.strike})`,
    data: Array(labels.length).fill(params.strike),
    borderColor: '#ff5c7c',
    borderWidth: 1.5,
    borderDash: [6, 4],
    pointRadius: 0,
    fill: false,
  });

  if (ciLowerPath.length === labels.length && ciUpperPath.length === labels.length) {
    datasets.push({
      label: '95% CI Lower',
      data: ciLowerPath,
      borderColor: 'rgba(62, 168, 255, 0.1)',
      backgroundColor: 'rgba(62, 168, 255, 0)',
      borderWidth: 0,
      pointRadius: 0,
      fill: false,
    });
    datasets.push({
      label: '95% CI Band',
      data: ciUpperPath,
      borderColor: 'rgba(62, 168, 255, 0.18)',
      backgroundColor: 'rgba(62, 168, 255, 0.16)',
      borderWidth: 0,
      pointRadius: 0,
      fill: '-1',
    });
  }

  if (meanPath.length === labels.length) {
    datasets.push({
      label: 'Mean Path',
      data: meanPath,
      borderColor: '#3ea8ff',
      borderWidth: 2.2,
      pointRadius: 0,
      fill: false,
      tension: 0.2,
    });
  }

  const isDarkMC = document.documentElement.dataset.theme !== 'light';
  const axisColor = isDarkMC ? '#a1a7c0' : '#4a5068';

  getOrCreateChart('mcChart', {
    type: 'line',
    data: { labels, datasets },
    options: chartDefaults({
      plugins: {
        legend: {
          display: true,
          labels: {
            filter: item => item.text && !item.text.startsWith('95% CI Lower')
          }
        }
      },
      scales: {
        x: { title: { display: true, text: 'Time Step', color: axisColor, font: { size: 12, weight: '500' } } },
        y: { title: { display: true, text: 'Price ($)', color: axisColor, font: { size: 12, weight: '500' } } },
      },
      animation: false,
    })
  });

  getOrCreateChart('convChart', {
    type: 'line',
    data: {
      labels: Array.from({ length: convergence.length }, (_, i) => i + 1),
      datasets: [{
        label: 'MC Price Convergence',
        data: convergence,
        borderColor: '#00e5a0',
        backgroundColor(ctx) {
          const chart = ctx.chart;
          const { ctx: cx, chartArea } = chart;
          return gradientFill(cx, chartArea, 0, 229, 160);
        },
        fill: true,
        tension: .25,
        pointRadius: 0,
        borderWidth: 2.5,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: '#00e5a0',
      }]
    },
    options: chartDefaults({
      scales: {
        x: { title: { display: true, text: 'Number of Paths', color: axisColor, font: { size: 12, weight: '500' } } },
        y: { title: { display: true, text: 'Estimated Price ($)', color: axisColor, font: { size: 12, weight: '500' } } },
      },
    })
  });
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
    spot:        pf('dlSpot', 100),
    strike:      pf('dlStrike', 100),
    maturity:    pf('dlMaturity', 1),
    rate:        pf('dlRate', 0.05),
    volatility:  pf('dlSigma', 0.2),
    option_type: $('dlType').value                  || 'call'
  };
  // Add news_text if provided
  const newsText = $('dlNewsText') ? $('dlNewsText').value.trim() : '';
  if (newsText) body.news_text = newsText;

  showLoading();
  try {
    const d = await api('/api/v1/dl/forecast', body);
    if (!d) return;

    $('dlResults').style.display = '';
    $('dlForecast').textContent = fmt(d.forecast_price);

    // Show LSTM prediction
    const lstm = d.lstm_prediction != null ? d.lstm_prediction : d.forecast_price;
    $('dlLSTM').textContent = fmt(lstm);

    // Show transformer sentiment (backend returns a string like "bullish"/"bearish"/"neutral")
    const sent = d.transformer_sentiment;
    if (sent != null && sent !== '') {
      const sentStr = String(sent).toLowerCase();
      const sentLabel = sentStr === 'bullish' ? 'Bullish' : sentStr === 'bearish' ? 'Bearish' : 'Neutral';
      const sentColor = sentStr === 'bullish' ? '#00e5a0' : sentStr === 'bearish' ? '#ff5c7c' : 'var(--text-secondary)';
      $('dlSentiment').textContent = sentLabel;
      $('dlSentiment').style.color = sentColor;
      $('dlSentLabel').textContent = 'Transformer NLP';
    } else {
      $('dlSentiment').textContent = '—';
      $('dlSentLabel').textContent = 'Transformer NLP';
    }

    // Show confidence
    $('dlConfidence').textContent = d.confidence != null ? (d.confidence * 100).toFixed(0) + '%' : '—';

    // Benchmarks (backend returns bs_price, mc_price in details dict)
    const bench = d.benchmarks || d.details || {};
    $('dlBS').textContent = bench.bs_price != null ? fmt(bench.bs_price) : (bench.bs != null ? fmt(bench.bs) : '—');
    $('dlMC').textContent = bench.mc_price != null ? fmt(bench.mc_price) : (bench.mc != null ? fmt(bench.mc) : '—');

    // Comparison chart
    $('compChartWrap').style.display = '';
    const bsVal = bench.bs_price != null ? bench.bs_price : bench.bs;
    const mcVal = bench.mc_price != null ? bench.mc_price : bench.mc;
    const chartData = [d.forecast_price, bsVal, mcVal].filter(v => v != null);
    const chartLabels = ['Deep Learning'];
    if (bsVal != null) chartLabels.push('Black-Scholes');
    if (mcVal != null) chartLabels.push('Monte Carlo');

    getOrCreateChart('compChart', {
      type: 'bar',
      data: {
        labels: chartLabels,
        datasets: [{
          label: 'Option Price ($)',
          data: chartData,
          backgroundColor: ['rgba(109,92,255,0.75)', 'rgba(0,229,160,0.75)', 'rgba(62,168,255,0.75)'],
          hoverBackgroundColor: ['#6d5cff', '#00e5a0', '#3ea8ff'],
          borderColor: ['#6d5cff', '#00e5a0', '#3ea8ff'],
          borderWidth: 1.5,
          borderRadius: 10,
          borderSkipped: false,
          barPercentage: 0.55,
          categoryPercentage: 0.7,
        }]
      },
      options: chartDefaults({
        plugins: { legend: { display: false } },
        scales: {
          y: {
            beginAtZero: true,
            title: { display: true, text: 'Price ($)', color: (document.documentElement.dataset.theme !== 'light') ? '#a1a7c0' : '#4a5068', font: { size: 12, weight: '500' } },
          },
        },
      })
    });

    toast('success', 'DL Forecast', `Price ≈ $${fmt(d.forecast_price, 2)}`);
  } catch (err) {
    toast('error', 'DL Forecast Failed', err.message);
  } finally {
    hideLoading();
  }
}

// ── 14a. DL Training (async background with progress polling) ──
$('dlTrainBtn').addEventListener('click', dlTrain);
let _dlTrainPollTimer = null;

async function dlTrain() {
  $('dlTrainBtn').disabled = true;
  const statusCard = $('dlTrainStatus');
  statusCard.style.display = '';
  $('dlTrainInfo').innerHTML = '<div style="display:flex;align-items:center;gap:0.6rem"><div class="spinner" style="width:18px;height:18px;border:2px solid rgba(99,102,241,.3);border-top-color:#6366f1;border-radius:50%;animation:spin .8s linear infinite"></div><span style="color:var(--text-secondary);font-size:0.85rem">Starting training…</span></div>';

  try {
    const startResp = await api('/api/v1/dl/train', { n_days: 500, spot: 100.0, volatility: 0.2, rate: 0.05, seed: 42 }, { timeout: 15000 });
    if (!startResp) { $('dlTrainBtn').disabled = false; return; }

    // Start polling for progress
    _dlTrainPollTimer = setInterval(() => _pollTrainingStatus(), 1500);
    _pollTrainingStatus(); // immediate first poll
  } catch (err) {
    $('dlTrainInfo').innerHTML = `<div style="color:#ff5c7c;padding:0.5rem">❌ Training failed to start: ${escapeHtml(err.message)}</div>`;
    toast('error', 'DL Training Failed', err.message);
    $('dlTrainBtn').disabled = false;
  }
}

async function _pollTrainingStatus() {
  try {
    const d = await apiGet('/api/v1/dl/training-status', { timeout: 10000 });
    if (!d) return;

    if (d.status === 'training' || d.status === 'queued') {
      const pct = d.progress != null ? d.progress.toFixed(0) : '0';
      const lastTrain = d.train_loss && d.train_loss.length ? fmt(d.train_loss[d.train_loss.length - 1], 6) : '—';
      const lastVal = d.val_loss && d.val_loss.length ? fmt(d.val_loss[d.val_loss.length - 1], 6) : '—';
      $('dlTrainInfo').innerHTML = `
        <div style="margin-bottom:0.5rem">
          <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:var(--text-secondary);margin-bottom:4px">
            <span>Epoch ${d.current_epoch || 0}/${d.total_epochs || 50}</span>
            <span>${pct}%</span>
          </div>
          <div style="width:100%;height:8px;background:rgba(99,102,241,.15);border-radius:4px;overflow:hidden">
            <div style="width:${pct}%;height:100%;background:linear-gradient(90deg,#6366f1,#818cf8);border-radius:4px;transition:width .3s"></div>
          </div>
        </div>
        <div class="metrics-row" style="margin:0">
          <div class="metric-card"><div class="metric-label">Status</div><div class="metric-value">⏳ Training</div></div>
          <div class="metric-card"><div class="metric-label">Train Loss</div><div class="metric-value">${lastTrain}</div></div>
          <div class="metric-card"><div class="metric-label">Val Loss</div><div class="metric-value">${lastVal}</div></div>
          <div class="metric-card"><div class="metric-label">Elapsed</div><div class="metric-value">${d.elapsed_seconds != null ? d.elapsed_seconds.toFixed(1) + 's' : '—'}</div></div>
        </div>`;
    } else if (d.status === 'completed' && d.result) {
      _stopTrainPoll();
      const r = d.result;
      $('dlTrainInfo').innerHTML = `
        <div class="metrics-row" style="margin:0">
          <div class="metric-card"><div class="metric-label">Status</div><div class="metric-value highlight">✅ Trained</div></div>
          <div class="metric-card"><div class="metric-label">LSTM RMSE</div><div class="metric-value">${r.lstm_rmse != null ? fmt(r.lstm_rmse, 6) : '—'}</div></div>
          <div class="metric-card"><div class="metric-label">Transformer</div><div class="metric-value">${r.transformer_accuracy != null ? (r.transformer_accuracy * 100).toFixed(0) + '%' : '—'}</div></div>
          <div class="metric-card"><div class="metric-label">Duration</div><div class="metric-value">${r.total_time_ms != null ? fmt(r.total_time_ms, 0) + 'ms' : '—'}</div></div>
        </div>`;
      toast('success', 'DL Training Complete', 'Models trained successfully');
      $('dlTrainBtn').disabled = false;
    } else if (d.status === 'failed') {
      _stopTrainPoll();
      $('dlTrainInfo').innerHTML = `<div style="color:#ff5c7c;padding:0.5rem">❌ Training failed: ${escapeHtml(d.error || 'Unknown error')}</div>`;
      toast('error', 'DL Training Failed', d.error || 'Unknown error');
      $('dlTrainBtn').disabled = false;
    }
  } catch (err) {
    // Polling error — don't stop, just log
    console.warn('Training poll error:', err.message);
  }
}

function _stopTrainPoll() {
  if (_dlTrainPollTimer) { clearInterval(_dlTrainPollTimer); _dlTrainPollTimer = null; }
}

// ── 14b. DL Status ─────────────────────────────────────────────
$('dlStatusBtn').addEventListener('click', dlStatus);
async function dlStatus() {
  try {
    const d = await apiGet('/api/v1/dl/status');
    if (!d) return;
    const statusCard = $('dlTrainStatus');
    statusCard.style.display = '';
    $('dlTrainInfo').innerHTML = `
      <div class="metrics-row" style="margin:0">
        <div class="metric-card"><div class="metric-label">LSTM</div><div class="metric-value">${d.lstm_trained ? '✅ Trained' : '⏳ Not Trained'}</div></div>
        <div class="metric-card"><div class="metric-label">Transformer</div><div class="metric-value">✅ Ready</div></div>
        <div class="metric-card"><div class="metric-label">Hidden Dim</div><div class="metric-value">${d.lstm_hidden_dim || '—'}</div></div>
        <div class="metric-card"><div class="metric-label">Attn Heads</div><div class="metric-value">${d.transformer_heads || '—'}</div></div>
      </div>`;
    toast('info', 'DL Status', `LSTM: ${d.lstm_trained ? 'Trained' : 'Not trained'} · Transformer: Ready`);
  } catch (err) {
    toast('error', 'Status Check Failed', err.message);
  }
}

// ── 15. ML Volatility Engine ───────────────────────────────────
let _volFeatureChart = null;

// ── 15a. Refresh Engine Status ─────────────────────────────────
async function volRefreshStatus() {
  try {
    const d = await apiGet('/api/v1/ml/vol/status');
    if (!d) return;
    $('engineStatusBadge').textContent = d.is_trained ? '✅ Trained' : '⏳ Not Trained';
    $('engineStatusBadge').style.color = d.is_trained ? 'var(--accent)' : 'var(--warning)';
    $('engineBestModel').textContent   = d.best_model  || '—';
    $('engineBestRMSE').textContent    = d.best_rmse != null ? fmt(d.best_rmse, 6) : '—';
    $('engineBestR2').textContent      = d.best_r2 != null ? fmt(d.best_r2, 4) : '—';
  } catch (e) {
    console.warn('vol status fetch failed', e);
  }
}

// ── 15b. Train Models ──────────────────────────────────────────
$('volTrainBtn').addEventListener('click', volTrain);
$('volStatusBtn').addEventListener('click', volRefreshStatus);
async function volTrain() {
  const checks = [...document.querySelectorAll('#volModelChecks input:checked')].map(c => c.value);
  if (checks.length === 0) { toast('warning', 'No Models', 'Select at least one model'); return; }

  const forwardWin = parseInt($('volForwardWin').value) || 20;
  const nDays      = parseInt($('volNDays').value) || 2520;
  const cvFolds    = parseInt($('volCVFolds').value) || 3;

  // ── Client-side validation ──
  if (nDays < 200)   { toast('warning', 'Invalid Input', 'Data Length must be at least 200 days'); return; }
  if (nDays > 10000) { toast('warning', 'Invalid Input', 'Data Length must be at most 10,000 days'); return; }
  if (forwardWin < 5)   { toast('warning', 'Invalid Input', 'Forward Window must be at least 5 days'); return; }
  if (forwardWin > 120) { toast('warning', 'Invalid Input', 'Forward Window must be at most 120 days'); return; }
  if (cvFolds < 1 || cvFolds > 10) { toast('warning', 'Invalid Input', 'CV Folds must be between 1 and 10'); return; }

  // Ensure enough data for features + targets + splits
  const minRequired = forwardWin + 120; // ~60 for feature warm-up + forward window + split headroom
  if (nDays < minRequired) {
    toast('warning', 'Insufficient Data', `With a ${forwardWin}-day forward window, you need at least ${minRequired} days of data`);
    return;
  }

  const body = {
    models:         checks,
    target:         $('volTarget').value,
    forward_window: forwardWin,
    n_days:         nDays,
    cv_folds:       cvFolds,
    seed:           42,
  };

  // Show progress
  $('volTrainProgress').style.display = '';
  $('volTrainBtn').disabled = true;
  $('volTrainMsg').textContent = `Training ${checks.length} model(s)... this may take a minute.`;

  try {
    const d = await api('/api/v1/ml/vol/train', body, { timeout: 180000 });
    if (!d) return;

    // ── Render Comparison Table ──
    $('volComparisonCard').style.display = '';
    const tbody = $('volCompBody');
    tbody.innerHTML = '';
    (d.comparisons || []).forEach(c => {
      const isBest = c.model_name === d.best_model;
      const t = c.test_metrics || {};
      const impCls = v => v > 0 ? 'improve-pos' : v < 0 ? 'improve-neg' : '';
      const impTxt = v => (v > 0 ? '+' : '') + fmt(v, 1);
      tbody.innerHTML += `
        <tr class="${isBest ? 'best-row' : ''}">
          <td>${isBest ? '🏆 ' : ''}${escapeHtml(c.model_name)}</td>
          <td>${fmt(t.rmse, 6)}</td>
          <td>${fmt(t.mae, 6)}</td>
          <td>${fmt(t.mape, 1)}</td>
          <td>${fmt(t.r_squared, 4)}</td>
          <td>${fmt(t.directional_accuracy, 1)}</td>
          <td class="${impCls(c.improvement_vs_historical)}">${impTxt(c.improvement_vs_historical)}</td>
          <td class="${impCls(c.improvement_vs_garch)}">${impTxt(c.improvement_vs_garch)}</td>
          <td class="${impCls(c.improvement_vs_ewma)}">${impTxt(c.improvement_vs_ewma)}</td>
          <td>${fmt(c.train_time_ms, 0)}</td>
        </tr>`;
    });

    // Baseline row
    const blRow = $('volBaselineRow');
    blRow.innerHTML = '';
    if (d.baseline_rmse) {
      Object.entries(d.baseline_rmse).forEach(([k, v]) => {
        blRow.innerHTML += `
          <div class="metric-card">
            <div class="metric-label">Baseline: ${k}</div>
            <div class="metric-value">${fmt(v, 6)}</div>
          </div>`;
      });
    }
    blRow.innerHTML += `
      <div class="metric-card">
        <div class="metric-label">Train / Val / Test</div>
        <div class="metric-value">${d.n_train} / ${d.n_val} / ${d.n_test}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Total Time</div>
        <div class="metric-value">${fmt(d.total_time_ms, 0)} ms</div>
      </div>`;

    // ── Feature Importance Chart ──
    if (d.top_features && d.top_features.length > 0) {
      $('volFeatureCard').style.display = '';
      const labels = d.top_features.map(f => f.name);
      const values = d.top_features.map(f => f.importance);
      if (_volFeatureChart) _volFeatureChart.destroy();
      const isDarkVF = document.documentElement.dataset.theme !== 'light';
      _volFeatureChart = new Chart($('volFeatureChart'), {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'Importance',
            data: values,
            backgroundColor: values.map((_, i) => {
              const opacity = 0.4 + 0.5 * ((values.length - i) / values.length);
              return `rgba(109,92,255,${opacity})`;
            }),
            hoverBackgroundColor: '#6d5cff',
            borderColor: '#6d5cff',
            borderWidth: 1,
            borderRadius: 6,
            borderSkipped: false,
          }],
        },
        options: chartDefaults({
          indexAxis: 'y',
          plugins: { legend: { display: false } },
          scales: {
            x: { title: { display: true, text: 'Importance', color: isDarkVF ? '#a1a7c0' : '#4a5068', font: { size: 12, weight: '500' } } },
            y: { ticks: { font: { size: 11, weight: '500' }, color: isDarkVF ? '#c8cce0' : '#4a5068' } },
          },
        }),
      });
    }

    // ── Update status banner ──
    volRefreshStatus();

    toast('success', 'Training Complete',
      `Best: ${d.best_model} · RMSE ${fmt(d.best_test_rmse, 6)} · R² ${fmt(d.best_test_r2, 4)} · ${fmt(d.total_time_ms, 0)} ms`);
  } catch (err) {
    toast('error', 'Training Failed', err.message);
  } finally {
    $('volTrainProgress').style.display = 'none';
    $('volTrainBtn').disabled = false;
  }
}

// ── 15c. IV Prediction ─────────────────────────────────────────
$('mlBtn').addEventListener('click', mlPredict);
async function mlPredict() {
  const body = {
    spot:          pf('mlSpot', 100),
    rate:          pf('mlRate', 0.05),
    maturity:      pf('mlMat', 0.5),
    realized_vol:  pf('mlRvol', 0.18),
    vix:           pf('mlVix', 20),
    skew:          pf('mlSkew', -0.15)
  };
  showLoading();
  try {
    const d = await api('/api/v1/ml/iv-predict', body);
    if (!d) return;
    $('mlResults').style.display = '';
    $('mlIV').textContent        = d.implied_vol != null ? fmtPct(d.implied_vol) : '—';
    $('mlRegime').textContent    = d.regime || '—';
    $('mlModelUsed').textContent  = d.model_used || 'analytical_fallback';
    $('mlConfidence').textContent = d.confidence != null ? fmt(d.confidence, 3) : '—';
    toast('success', 'IV Predicted', `IV = ${fmtPct(d.implied_vol)} · ${d.model_used || 'fallback'}`);
  } catch (err) {
    toast('error', 'ML Prediction Failed', err.message);
  } finally {
    hideLoading();
  }
}

// Load engine status on page load
volRefreshStatus();

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
  // Skip shortcuts when user is typing in an input/textarea
  const tag = e.target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
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
    else if (id === 'sec-sentiment') $('sentimentBtn').click();
    else if (id === 'sec-risk-analytics') $('varCalcBtn').click();
  }
});

// ── 18. Sidebar Collapse ──────────────────────────────────────
(function initSidebarCollapse() {
  const collapseBtn = $('sidebarCollapseBtn');
  if (!collapseBtn) return;

  // Restore saved state
  const saved = localStorage.getItem('oq-sidebar-collapsed');
  if (saved === 'true') {
    sidebarEl.classList.add('collapsed');
    collapseBtn.textContent = '›';
  }

  collapseBtn.addEventListener('click', () => {
    const isCollapsed = sidebarEl.classList.toggle('collapsed');
    collapseBtn.textContent = isCollapsed ? '›' : '‹';
    localStorage.setItem('oq-sidebar-collapsed', isCollapsed);
  });
})();

// ── 19. Dashboard Quick Actions ────────────────────────────────
document.querySelectorAll('.dash-action-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.goto;
    if (target) navigate(target);
  });
});

// Update dashboard health on load
async function updateDashboardHealth() {
  try {
    const ctrl = new AbortController();
    setTimeout(() => ctrl.abort(), 5000);
    const res = await fetch('/health', { signal: ctrl.signal });
    const ok = res.ok;
    const dashStatus = $('dashStatus');
    if (dashStatus) {
      dashStatus.textContent = ok ? '● Online' : '● Offline';
      dashStatus.style.color = ok ? '#00e5a0' : '#ff5c7c';
    }
    // Also update DL status
    try {
      const dlStatus = await apiGet('/api/v1/dl/status');
      const dashDL = $('dashDLStatus');
      if (dashDL && dlStatus) {
        const ready = dlStatus.lstm_trained;
        dashDL.textContent = ready ? '✅ Trained' : '⏳ Ready';
      }
    } catch {}
  } catch {
    const dashStatus = $('dashStatus');
    if (dashStatus) {
      dashStatus.textContent = '● Offline';
      dashStatus.style.color = '#ff5c7c';
    }
  }
}
updateDashboardHealth();

// ── 20. Market Sentiment ───────────────────────────────────────
$('sentimentBtn').addEventListener('click', analyzeSentiment);
$('sentimentClearBtn').addEventListener('click', () => {
  $('sentimentText').value = '';
  $('sentimentResults').style.display = 'none';
});

// Sentiment quick example chips
document.querySelectorAll('#sentimentChips .chip').forEach(chip => {
  chip.addEventListener('click', () => {
    $('sentimentText').value = chip.dataset.text;
    analyzeSentiment();
  });
});

async function analyzeSentiment() {
  const text = $('sentimentText').value.trim();
  if (!text) { toast('warning', 'No Text', 'Enter financial text to analyze'); return; }

  showLoading();
  try {
    const d = await api('/api/v1/dl/market-sentiment', { text: text });
    if (!d) return;

    $('sentimentResults').style.display = '';

    // Overall score (0-1 scale, 0.5 = neutral)
    const score = d.score != null ? d.score : 0.5;
    const sentimentLabel = d.sentiment || (score > 0.65 ? 'bullish' : score < 0.35 ? 'bearish' : 'neutral');
    const label = sentimentLabel.charAt(0).toUpperCase() + sentimentLabel.slice(1);
    const scoreColor = score > 0.65 ? '#00e5a0' : score < 0.35 ? '#ff5c7c' : '#ffc044';

    $('sentScore').textContent = (score * 100).toFixed(0) + '%';
    $('sentScore').style.color = scoreColor;
    $('sentLabel').textContent = label;

    $('sentConfidence').textContent = d.confidence != null ? (d.confidence * 100).toFixed(0) + '%' : '—';

    // Bullish / Bearish breakdown (derived from score)
    $('sentBullish').textContent = (score * 100).toFixed(0) + '%';
    $('sentBearish').textContent = ((1 - score) * 100).toFixed(0) + '%';

    // Sentiment gauge
    const gaugeFill = $('sentGaugeFill');
    const gaugeMarker = $('sentGaugeMarker');
    if (gaugeFill) gaugeFill.style.width = (score * 100) + '%';
    if (gaugeMarker) gaugeMarker.style.left = (score * 100) + '%';

    toast('success', 'Sentiment Analyzed', `${label} (${(score * 100).toFixed(0)}%)`);
  } catch (err) {
    toast('error', 'Sentiment Failed', err.message);
  } finally {
    hideLoading();
  }
}

// ── 21. Risk Analytics (VaR) ───────────────────────────────────
$('varCalcBtn').addEventListener('click', calculateVaR);
async function calculateVaR() {
  const params = {
    spot:        pf('varSpot', 100),
    strike:      pf('varStrike', 100),
    volatility:  pf('varSigma', 0.2),
    rate:        pf('varRate', 0.05),
    maturity:    pf('varMaturity', 1),
    option_type: $('varType').value                   || 'call'
  };
  const contracts = parseInt($('varContracts').value) || 10;
  const confLevel = parseFloat($('varConfidence').value) || 0.99;

  showLoading();
  try {
    // Get greeks for risk decomposition
    const greeks = await api('/api/v1/pricing/greeks', params);
    if (!greeks) return;

    // Get BS price for position value
    const bs = await api('/api/v1/pricing/bs', params);

    // Calculate Delta-Normal VaR
    const z = confLevel === 0.999 ? 3.09 : confLevel === 0.99 ? 2.326 : 1.645;
    const dailyVol = params.volatility / Math.sqrt(252);
    const deltaVaR = Math.abs(greeks.delta) * params.spot * dailyVol * z * contracts * 100;
    const positionValue = (bs ? bs.price : 0) * contracts * 100;
    const pctLoss = positionValue > 0 ? (deltaVaR / positionValue) * 100 : 0;

    $('varResults').style.display = '';
    $('varDeltaNormal').textContent = '$' + deltaVaR.toFixed(2);
    $('varPosition').textContent = '$' + positionValue.toFixed(2);
    $('varPctLoss').textContent = pctLoss.toFixed(1) + '%';
    $('varGreeksExp').textContent = fmt(greeks.delta * contracts * 100, 2);

    // Risk decomposition bars
    const deltaRisk = Math.abs(greeks.delta) * params.spot * dailyVol * z;
    const gammaRisk = 0.5 * Math.abs(greeks.gamma) * (params.spot * dailyVol * z) ** 2;
    const vegaRisk  = Math.abs(greeks.vega) * dailyVol * 100;
    const thetaRisk = Math.abs(greeks.theta) / 252;
    const maxRisk = Math.max(deltaRisk, gammaRisk, vegaRisk, thetaRisk, 0.001);

    $('riskBarDelta').style.width  = (deltaRisk / maxRisk * 100) + '%';
    $('riskBarGamma').style.width  = (gammaRisk / maxRisk * 100) + '%';
    $('riskBarVega').style.width   = (vegaRisk / maxRisk * 100) + '%';
    $('riskBarTheta').style.width  = (thetaRisk / maxRisk * 100) + '%';

    $('riskValDelta').textContent = '$' + (deltaRisk * contracts * 100).toFixed(2);
    $('riskValGamma').textContent = '$' + (gammaRisk * contracts * 100).toFixed(2);
    $('riskValVega').textContent  = '$' + (vegaRisk * contracts * 100).toFixed(2);
    $('riskValTheta').textContent = '$' + (thetaRisk * contracts * 100).toFixed(2);

    toast('success', 'VaR Calculated', `Delta-Normal VaR: $${deltaVaR.toFixed(2)} (${confLevel * 100}% confidence)`);
  } catch (err) {
    toast('error', 'VaR Failed', err.message);
  } finally {
    hideLoading();
  }
}

// ── 22. Initialization ────────────────────────────────────────
console.log('%c◈ OptionQuant v2.0 loaded', 'color:#6d5cff;font-size:14px;font-weight:700');

// ═══════════════════════════════════════════════════════════════
//  QUANT INTELLIGENCE ENGINE — New Sections
// ═══════════════════════════════════════════════════════════════

// ── 23. Market Intelligence ───────────────────────────────────
let _marketWs = null;
let _marketTicks = 0;

$('mktFetchBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const snap = await apiGet('/api/v1/market/snapshot/SPY');
    if (!snap) return;
    $('mktPrice').textContent = '$' + Number(snap.quote.price).toFixed(2);
    $('mktBidAsk').textContent = `Bid: $${Number(snap.quote.bid).toFixed(2)} / Ask: $${Number(snap.quote.ask).toFixed(2)}`;
    $('mktVix').textContent = Number(snap.vix).toFixed(1);

    // Option chain table
    const allContracts = [...(snap.chain.calls || []), ...(snap.chain.puts || [])];
    if (allContracts.length > 0) {
      const tbody = $('mktChainBody');
      tbody.innerHTML = '';
      allContracts.slice(0, 40).forEach(c => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${Number(c.strike).toFixed(1)}</td>
          <td>${c.option_type}</td>
          <td>$${Number(c.bid).toFixed(2)}</td>
          <td>$${Number(c.ask).toFixed(2)}</td>
          <td>$${Number(c.mid).toFixed(2)}</td>
          <td>${(Number(c.implied_vol) * 100).toFixed(1)}%</td>
          <td>${c.volume || 0}</td>
          <td>${c.open_interest || 0}</td>
          <td>${Number(c.moneyness || 1).toFixed(3)}</td>
        `;
        tbody.appendChild(tr);
      });
      $('mktChainTable').style.display = '';
      $('mktChainPlaceholder').style.display = 'none';
    }

    toast('success', 'Market Data', `SPY: $${Number(snap.quote.price).toFixed(2)}, VIX: ${Number(snap.vix).toFixed(1)}`);
  } catch (err) {
    toast('error', 'Market Fetch Failed', err.message);
  } finally {
    hideLoading();
  }
});

// Pipeline health
apiGet('/api/v1/market/health').then(h => {
  if (h) $('mktPipelineStatus').textContent = h.status || 'unknown';
}).catch(() => {});

$('mktStreamBtn')?.addEventListener('click', () => {
  if (_marketWs) return;
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${location.host}/ws/market/SPY?interval=1500&mispricing=true&regime=true`;
  _marketWs = new WebSocket(wsUrl);
  _marketTicks = 0;

  $('mktStreamBtn').disabled = true;
  $('mktStopStreamBtn').disabled = false;
  $('mktStreamLog').style.display = '';
  $('mktStreamStatus').textContent = 'Connecting…';

  _marketWs.onopen = () => {
    $('mktStreamStatus').textContent = 'Connected';
    $('mktStreamStatus').style.color = 'var(--success, #4ade80)';
    toast('info', 'Stream', 'WebSocket connected to SPY market stream');
  };

  _marketWs.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    _marketTicks++;
    $('mktStreamTicks').textContent = _marketTicks + ' ticks';
    $('mktPrice').textContent = '$' + Number(msg.price).toFixed(2);
    $('mktBidAsk').textContent = `Bid: $${Number(msg.bid).toFixed(2)} / Ask: $${Number(msg.ask).toFixed(2)}`;
    if (msg.vix) $('mktVix').textContent = Number(msg.vix).toFixed(1);

    // Log
    const log = $('mktStreamLog');
    const line = document.createElement('div');
    line.className = 'stream-line';
    let text = `[${msg.timestamp?.substring(11, 19) || 'T?'}] $${Number(msg.price).toFixed(2)} VIX:${Number(msg.vix||0).toFixed(1)}`;
    if (msg.regime) text += ` regime:${msg.regime.label}`;
    if (msg.mispricing) text += ` dev:${Number(msg.mispricing.deviation_pct).toFixed(1)}%`;
    if (msg.alert) text += ` ⚠ ALERT: ${msg.alert.title}`;
    line.textContent = text;
    log.prepend(line);
    if (log.children.length > 100) log.removeChild(log.lastChild);
  };

  _marketWs.onerror = () => {
    $('mktStreamStatus').textContent = 'Error';
    $('mktStreamStatus').style.color = 'var(--danger, #ff5c7c)';
  };

  _marketWs.onclose = () => {
    $('mktStreamStatus').textContent = 'Disconnected';
    $('mktStreamStatus').style.color = '';
    $('mktStreamBtn').disabled = false;
    $('mktStopStreamBtn').disabled = true;
    _marketWs = null;
  };
});

$('mktStopStreamBtn')?.addEventListener('click', () => {
  if (_marketWs) { _marketWs.close(); _marketWs = null; }
});


// ── 24. Mispricing Scanner ───────────────────────────────────
$('mispDetectBtn')?.addEventListener('click', async () => {
  const body = {
    spot:         pf('mispSpot', 100),
    strike:       pf('mispStrike', 100),
    maturity:     pf('mispMaturity', 0.25),
    rate:         pf('mispRate', 0.05),
    volatility:   pf('mispVol', 0.2),
    option_type:  $('mispType').value || 'call',
    market_price: pf('mispMarket', 10.5),
    pricing_model: $('mispModel').value || 'black_scholes',
    significance_threshold: 2.0,
    min_deviation_pct: 2.0,
  };

  showLoading();
  try {
    const d = await api('/api/v1/market/mispricing/detect', body);
    if (!d) return;
    $('mispResults').style.display = '';

    $('mispDirection').textContent = d.direction;
    $('mispDirection').style.color = d.direction === 'overpriced' ? 'var(--danger, #ff5c7c)' : 'var(--success, #4ade80)';
    $('mispModel_used').textContent = d.model_used || 'BS';
    $('mispDeviation').textContent = fmt(d.deviation_pct, 2) + '%';
    $('mispDevDollar').textContent = '$' + fmt(d.deviation_dollar, 4);
    $('mispZScore').textContent = fmt(d.z_score, 2);
    $('mispSignificant').textContent = d.is_significant ? '✅ Significant' : '❌ Not significant';
    $('mispStrength').textContent = fmt(d.signal_strength, 3);
    $('mispConfidence').textContent = 'Conf: ' + fmtPct(d.confidence);

    toast(d.is_significant ? 'warning' : 'info', 'Mispricing',
      `${d.direction} by ${fmt(d.deviation_pct, 1)}% (z=${fmt(d.z_score, 2)})`);
  } catch (err) {
    toast('error', 'Mispricing Error', err.message);
  } finally {
    hideLoading();
  }
});

$('mispScanBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const d = await api('/api/v1/market/mispricing/scan', {
      symbol: 'SPY',
      pricing_model: $('mispModel').value || 'black_scholes',
    }, { timeout: 60000 });
    if (!d) return;

    $('mispScanResults').style.display = '';
    const metrics = $('mispScanMetrics');
    metrics.innerHTML = `
      <div class="metric-card"><div class="metric-label">Contracts</div><div class="metric-value">${d.total_contracts || 0}</div></div>
      <div class="metric-card"><div class="metric-label">Mispriced</div><div class="metric-value highlight">${d.significant_signals || 0}</div></div>
      <div class="metric-card"><div class="metric-label">Arbitrage</div><div class="metric-value" style="color:var(--danger)">${d.arbitrage_opportunities || 0}</div></div>
      <div class="metric-card"><div class="metric-label">Time (ms)</div><div class="metric-value">${fmt(d.scan_time_ms, 0)}</div></div>
    `;

    const tbody = $('mispScanBody');
    tbody.innerHTML = '';
    (d.signals || []).slice(0, 50).forEach(s => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${Number(s.strike).toFixed(1)}</td>
        <td>${s.option_type}</td>
        <td class="${s.direction === 'overpriced' ? 'negative' : 'positive'}">${s.direction}</td>
        <td>${fmt(s.deviation_pct, 2)}%</td>
        <td>${fmt(s.z_score, 2)}</td>
        <td>${fmt(s.signal_strength, 3)}</td>
        <td>${s.model_used || 'BS'}</td>
        <td>${s.is_significant ? '✅' : '—'}</td>
      `;
      tbody.appendChild(tr);
    });

    toast('success', 'Scan Complete', `${d.significant_signals || 0} mispriced out of ${d.total_contracts || 0}`);
  } catch (err) {
    toast('error', 'Scan Failed', err.message);
  } finally {
    hideLoading();
  }
});


// ── 25. Regime Detection ─────────────────────────────────────
$('regimeDetectBtn')?.addEventListener('click', async () => {
  const raw = $('regimeReturns').value;
  const returns = raw.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
  if (returns.length < 10) { toast('error', 'Error', 'Need at least 10 return values'); return; }

  showLoading();
  try {
    const d = await api('/api/v1/market/regime/detect', {
      returns: returns,
      vix: pf('regimeVix', 20),
    });
    if (!d) return;

    $('regimeResults').style.display = '';
    $('regimeLabel').textContent = (d.label || 'unknown').toUpperCase();
    $('regimeLabel').className = 'metric-value highlight regime-badge ' + (d.label || '');
    $('regimeDuration').textContent = `Duration: ${d.duration_days || 0} days`;
    $('regimeProb').textContent = fmtPct(d.probability);
    $('regimeConf').textContent = 'Confidence: ' + fmtPct(d.confidence);
    $('regimeRisk').textContent = (d.risk_level || '—').toUpperCase();
    $('regimeRisk').style.color = d.risk_level === 'extreme' ? 'var(--danger)' :
      d.risk_level === 'high' ? '#fbbf24' : 'var(--success)';
    $('regimeModel').textContent = 'Model: ' + (d.recommended_model || '—');
    $('regimeVolAdj').textContent = fmt(d.vol_adjustment, 3) + '×';

    // Transition matrix
    const tp = d.transition_probs || {};
    const labels = ['bull', 'bear', 'high_vol', 'low_vol'];
    let html = '<div class="tm-header">&nbsp;</div>';
    labels.forEach(l => html += `<div class="tm-header">${l}</div>`);
    labels.forEach(from => {
      html += `<div class="tm-label">${from}</div>`;
      const row = tp[from] || {};
      labels.forEach(to => {
        const v = row[to] !== undefined ? row[to] : (from === to ? 0.7 : 0.1);
        const isHigh = v > 0.3;
        html += `<div class="tm-cell${isHigh ? ' high' : ''}">${(v * 100).toFixed(0)}%</div>`;
      });
    });
    $('regimeTransition').innerHTML = html;

    toast('success', 'Regime Detected', `${(d.label||'').toUpperCase()} (${fmtPct(d.probability)} confidence)`);
  } catch (err) {
    toast('error', 'Regime Error', err.message);
  } finally {
    hideLoading();
  }
});


// ── 26. SHAP Explainability ──────────────────────────────────
$('shapExplainBtn')?.addEventListener('click', async () => {
  const body = {
    spot:       pf('shapSpot', 100),
    strike:     pf('shapStrike', 105),
    maturity:   pf('shapMaturity', 1),
    rate:       pf('shapRate', 0.05),
    volatility: pf('shapVol', 0.2),
    option_type: $('shapType').value || 'call',
    pricing_model: 'black_scholes',
  };

  showLoading();
  try {
    const d = await api('/api/v1/market/explain/shap', body, { timeout: 30000 });
    if (!d) return;

    $('shapResults').style.display = '';
    $('shapBase').textContent = '$' + fmt(d.base_price, 4);
    $('shapPredicted').textContent = '$' + fmt(d.predicted_price, 4);
    $('shapModel').textContent = d.model || 'BS';

    // Waterfall chart
    const waterfall = $('shapWaterfall');
    waterfall.innerHTML = '';
    waterfall.className = 'shap-waterfall';
    const maxShap = Math.max(...(d.attributions || []).map(a => Math.abs(a.shap_value)), 0.01);

    (d.attributions || []).forEach(a => {
      const row = document.createElement('div');
      row.className = 'shap-bar-row';
      const pct = Math.abs(a.shap_value) / maxShap * 50;
      const isPos = a.shap_value >= 0;
      row.innerHTML = `
        <div class="shap-bar-label">${a.feature}</div>
        <div class="shap-bar-track">
          <div class="shap-bar-fill ${isPos ? 'positive' : 'negative'}" style="width:${pct}%"></div>
        </div>
        <div class="shap-bar-value" style="color:${isPos ? 'var(--success,#4ade80)' : 'var(--danger,#ff5c7c)'}">
          ${isPos ? '+' : ''}${fmt(a.shap_value, 4)}
        </div>
      `;
      waterfall.appendChild(row);
    });

    // Narrative
    $('shapNarrative').textContent = d.narrative || 'No narrative generated.';

    // Volatility sensitivity
    const vs = d.vol_sensitivity || {};
    const sensDiv = $('shapVolSens');
    sensDiv.className = 'sensitivity-grid';
    sensDiv.innerHTML = Object.entries(vs).map(([k, v]) =>
      `<div class="sens-item"><div class="sens-label">σ = ${k}</div><div class="sens-value">$${fmt(v, 4)}</div></div>`
    ).join('');

    // Time decay
    const td = d.time_decay_profile || {};
    const tdDiv = $('shapTimeDecay');
    tdDiv.className = 'sensitivity-grid';
    tdDiv.innerHTML = Object.entries(td).map(([k, v]) =>
      `<div class="sens-item"><div class="sens-label">T = ${k}</div><div class="sens-value">$${fmt(v, 4)}</div></div>`
    ).join('');

    toast('success', 'SHAP Explanation', `${(d.attributions || []).length} feature attributions computed`);
  } catch (err) {
    toast('error', 'SHAP Error', err.message);
  } finally {
    hideLoading();
  }
});


// ── 27. Performance Benchmark ────────────────────────────────
$('benchRunBtn')?.addEventListener('click', async () => {
  const body = {
    spot:       pf('benchSpot', 100),
    strike:     pf('benchStrike', 100),
    maturity:   pf('benchMaturity', 1),
    rate:       0.05,
    volatility: pf('benchVol', 0.2),
    option_type: 'call',
  };

  showLoading();
  try {
    const d = await api('/api/v1/market/benchmark', body, { timeout: 90000 });
    if (!d) return;

    $('benchResults').style.display = '';

    // Summary
    $('benchSummary').innerHTML = `
      <div class="metric-card"><div class="metric-label">Total Time</div><div class="metric-value highlight">${fmt(d.total_time_ms, 0)}ms</div></div>
      <div class="metric-card"><div class="metric-label">Memory Est.</div><div class="metric-value">${d.memory_estimate_mb} MB</div></div>
      <div class="metric-card"><div class="metric-label">Heston Price</div><div class="metric-value">${d.heston_benchmark?.price || '—'}</div></div>
      <div class="metric-card"><div class="metric-label">Heston Time</div><div class="metric-value">${fmt(d.heston_benchmark?.elapsed_ms, 1)}ms</div></div>
    `;

    // Path scaling
    const pathBody = $('benchPathBody');
    pathBody.innerHTML = '';
    (d.path_scaling || []).forEach(r => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${r.name}</td><td>${r.price}</td><td>${r.std_error}</td>
        <td>${fmt(r.elapsed_ms, 1)}</td><td>${Number(r.throughput_paths_per_sec).toLocaleString()}/s</td>
        <td>${r.error_vs_bs}</td>
      `;
      pathBody.appendChild(tr);
    });

    // Method comparison
    const methBody = $('benchMethodBody');
    methBody.innerHTML = '';
    (d.method_comparison || []).forEach(r => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${r.name}</td><td>${r.price}</td><td class="highlight-cell">${r.std_error}</td>
        <td>${fmt(r.elapsed_ms, 1)}</td><td>${Number(r.throughput_paths_per_sec).toLocaleString()}/s</td>
        <td>${r.error_vs_bs}</td>
      `;
      methBody.appendChild(tr);
    });

    // Latency
    const latBody = $('benchLatencyBody');
    latBody.innerHTML = '';
    (d.latency_profiles || []).forEach(p => {
      const tr = document.createElement('tr');
      const warn = p.p95_ms > 200;
      tr.innerHTML = `
        <td>${p.component}</td><td>${fmt(p.mean_ms, 2)}</td><td>${fmt(p.p50_ms, 2)}</td>
        <td class="${warn ? 'negative' : ''}">${fmt(p.p95_ms, 2)}</td>
        <td>${fmt(p.p99_ms, 2)}</td><td>${fmt(p.min_ms, 2)}</td><td>${fmt(p.max_ms, 2)}</td>
      `;
      latBody.appendChild(tr);
    });

    // Recommendations
    $('benchRecommendations').innerHTML = (d.recommendations || [])
      .map(r => `<div style="padding:0.3rem 0;${r.startsWith('WARN') ? 'color:var(--danger,#ff5c7c)' : ''}">${r.startsWith('WARN') ? '⚠️ ' : '✅ '}${r}</div>`)
      .join('');

    toast('success', 'Benchmark Done', `Completed in ${fmt(d.total_time_ms, 0)}ms`);
  } catch (err) {
    toast('error', 'Benchmark Failed', err.message);
  } finally {
    hideLoading();
  }
});


// ═══════════════════════════════════════════════════════════════
//  QUANT INTELLIGENCE ENGINE — Frontend Handlers
// ═══════════════════════════════════════════════════════════════

// ── 30. Quant Dashboard Status ─────────────────────────────────
$('quantStatusBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const d = await apiGet('/api/v1/quant/status');
    if (!d) return;
    $('quantActiveModules').textContent = d.active_modules;
    $('quantGPU').textContent = d.gpu_available ? '✅ CUDA' : '❌ CPU';
    $('quantGPU').className = `metric-value ${d.gpu_available ? 'positive' : ''}`;
    $('quantHealth').textContent = d.system_health.toUpperCase();
    $('quantHealth').className = `metric-value ${d.system_health === 'healthy' ? 'positive' : d.system_health === 'degraded' ? 'negative' : 'highlight'}`;
    $('quantTotal').textContent = d.total_modules;

    // Module grid
    const grid = $('quantModuleGrid');
    const icons = { pinns:'🔬', rl_hedging:'🤖', vol_surface_transformer:'🌊',
      jump_diffusion:'📈', arbitrage_engine:'🎯', uncertainty:'📊',
      gpu_monte_carlo:'🚀', portfolio_risk:'🛡️', explainer:'💡' };
    grid.innerHTML = Object.entries(d.modules || {}).map(([name, info]) => `
      <div class="capability-item">
        <div class="capability-icon">${icons[name] || '⚙️'}</div>
        <div>
          <div class="capability-title">${name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</div>
          <div class="capability-desc" style="color:${info.status === 'active' ? 'var(--accent, #22d3ee)' : 'var(--danger, #ff5c7c)'}">
            ${info.status === 'active' ? '● Active' : '○ Unavailable'}
            ${info.trained !== undefined ? (info.trained ? ' (trained)' : ' (untrained)') : ''}
            ${info.backend ? ` [${info.backend}]` : ''}
          </div>
        </div>
      </div>
    `).join('');

    toast('success', 'Quant Status', `${d.active_modules}/${d.total_modules} modules active`);
  } catch (err) {
    toast('error', 'Status Error', err.message);
  } finally {
    hideLoading();
  }
});


// ── 31. PINNs Pricing ──────────────────────────────────────────
$('pinnsTrainBtn')?.addEventListener('click', async () => {
  const nSamples = parseInt($('pinnsSamples').value) || 5000;
  const nEpochs = parseInt($('pinnsEpochs').value) || 200;
  if (nSamples < 500) { toast('warning', 'Invalid Input', 'PINNs samples must be at least 500'); return; }
  if (nEpochs < 10)   { toast('warning', 'Invalid Input', 'PINNs epochs must be at least 10'); return; }
  showLoading();
  try {
    const d = await api('/api/v1/quant/pinns/train', {
      n_samples: nSamples,
      epochs: nEpochs,
      spot_range: [50, 150],
      strike: pf('pinnsStrike', 100),
      rate: pf('pinnsRate', 0.05),
      volatility: pf('pinnsVol', 0.2),
    }, { timeout: 120000 });
    if (!d) return;

    $('pinnsResults').style.display = '';
    $('pinnsMetrics').innerHTML = `
      <div class="metric-card"><div class="metric-label">Final Loss</div><div class="metric-value highlight">${fmt(d.final_loss, 6)}</div></div>
      <div class="metric-card"><div class="metric-label">PDE Loss</div><div class="metric-value">${fmt(d.pde_loss, 6)}</div></div>
      <div class="metric-card"><div class="metric-label">Data Loss</div><div class="metric-value">${fmt(d.data_loss, 6)}</div></div>
      <div class="metric-card"><div class="metric-label">Training Time</div><div class="metric-value">${fmt(d.training_time_ms, 0)}ms</div></div>
    `;
    $('pinnsNarrative').textContent = `PINNs trained for ${d.epochs_trained} epochs. Final total loss: ${fmt(d.final_loss, 6)}. PDE residual loss: ${fmt(d.pde_loss, 6)}, ensuring Black-Scholes PDE is satisfied.`;
    toast('success', 'PINNs Trained', `Loss: ${fmt(d.final_loss, 6)} in ${fmt(d.training_time_ms, 0)}ms`);
  } catch (err) {
    toast('error', 'PINNs Train Error', err.message);
  } finally { hideLoading(); }
});

$('pinnsPredictBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const d = await api('/api/v1/quant/pinns/predict', {
      spot: pf('pinnsSpot', 100), strike: pf('pinnsStrike', 100),
      maturity: pf('pinnsMaturity', 1), rate: pf('pinnsRate', 0.05),
      volatility: pf('pinnsVol', 0.2), option_type: $('pinnsType').value,
    });
    if (!d) return;
    $('pinnsResults').style.display = '';
    $('pinnsMetrics').innerHTML = `
      <div class="metric-card"><div class="metric-label">PINNs Price</div><div class="metric-value highlight">$${fmt(d.pinns_price, 4)}</div></div>
      <div class="metric-card"><div class="metric-label">BS Price</div><div class="metric-value">$${fmt(d.bs_price, 4)}</div></div>
      <div class="metric-card"><div class="metric-label">Deviation</div><div class="metric-value ${Math.abs(d.deviation_pct) > 5 ? 'negative' : 'positive'}">${fmt(d.deviation_pct, 2)}%</div></div>
      <div class="metric-card"><div class="metric-label">PDE Residual</div><div class="metric-value">${fmt(d.pde_residual, 6)}</div></div>
    `;
    $('pinnsNarrative').textContent = `PINNs model predicted $${fmt(d.pinns_price, 4)} vs BS analytical $${fmt(d.bs_price, 4)} (${fmt(d.deviation_pct, 2)}% deviation). PDE residual: ${fmt(d.pde_residual, 6)}.`;
    toast('success', 'PINNs Predict', `$${fmt(d.pinns_price, 4)}`);
  } catch (err) {
    toast('error', 'PINNs Error', err.message);
  } finally { hideLoading(); }
});

$('pinnsGreeksBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const d = await api('/api/v1/quant/pinns/greeks', {
      spot: pf('pinnsSpot', 100), strike: pf('pinnsStrike', 100),
      maturity: pf('pinnsMaturity', 1), rate: pf('pinnsRate', 0.05),
      volatility: pf('pinnsVol', 0.2),
    });
    if (!d) return;
    $('pinnsResults').style.display = '';
    $('pinnsMetrics').innerHTML = `
      <div class="metric-card"><div class="metric-label">Delta (Δ)</div><div class="metric-value highlight">${fmt(d.delta, 6)}</div></div>
      <div class="metric-card"><div class="metric-label">Gamma (Γ)</div><div class="metric-value">${fmt(d.gamma, 6)}</div></div>
      <div class="metric-card"><div class="metric-label">Theta (Θ)</div><div class="metric-value">${fmt(d.theta, 6)}</div></div>
      <div class="metric-card"><div class="metric-label">Vega (ν)</div><div class="metric-value">${fmt(d.vega, 6)}</div></div>
    `;
    toast('success', 'PINNs Greeks', `Δ=${fmt(d.delta, 4)}`);
  } catch (err) {
    toast('error', 'PINNs Greeks Error', err.message);
  } finally { hideLoading(); }
});


// ── 32. RL Hedging ─────────────────────────────────────────────
$('hedgeTrainBtn')?.addEventListener('click', async () => {
  const episodes = parseInt($('hedgeEpisodes').value) || 500;
  if (episodes < 50) { toast('warning', 'Invalid Input', 'Episodes must be at least 50'); return; }
  showLoading();
  try {
    const d = await api('/api/v1/quant/hedging/train', {
      agent_type: $('hedgeAgent').value,
      episodes: episodes,
      spot: pf('hedgeSpot', 100), strike: pf('hedgeStrike', 100),
      maturity: pf('hedgeMaturity', 0.25), volatility: pf('hedgeVol', 0.2),
      rate: pf('hedgeRate', 0.05),
    }, { timeout: 300000 });
    if (!d) return;
    $('hedgeResults').style.display = '';
    $('hedgeMetrics').innerHTML = `
      <div class="metric-card"><div class="metric-label">Agent</div><div class="metric-value highlight">${d.agent_type.toUpperCase()}</div></div>
      <div class="metric-card"><div class="metric-label">Episodes</div><div class="metric-value">${d.episodes_trained}</div></div>
      <div class="metric-card"><div class="metric-label">Avg Reward (last 100)</div><div class="metric-value">${fmt(d.avg_reward_last_100, 4)}</div></div>
      <div class="metric-card"><div class="metric-label">Training Time</div><div class="metric-value">${fmt(d.training_time_ms, 0)}ms</div></div>
    `;
    $('hedgeNarrative').textContent = `${d.agent_type.toUpperCase()} agent trained for ${d.episodes_trained} episodes. Average reward (last 100): ${fmt(d.avg_reward_last_100, 4)}.`;
    toast('success', 'Agent Trained', `${d.agent_type.toUpperCase()} — ${d.episodes_trained} episodes`);
  } catch (err) {
    toast('error', 'Hedging Train Error', err.message);
  } finally { hideLoading(); }
});

$('hedgeBacktestBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const d = await api('/api/v1/quant/hedging/backtest', {
      agent_type: $('hedgeAgent').value,
      n_scenarios: parseInt($('hedgeScenarios').value) || 100,
      spot: pf('hedgeSpot', 100), strike: pf('hedgeStrike', 100),
      maturity: pf('hedgeMaturity', 0.25), volatility: pf('hedgeVol', 0.2),
      rate: pf('hedgeRate', 0.05),
    }, { timeout: 180000 });
    if (!d) return;
    $('hedgeResults').style.display = '';
    const improvClass = d.improvement_pct > 0 ? 'positive' : 'negative';
    $('hedgeMetrics').innerHTML = `
      <div class="metric-card"><div class="metric-label">RL P&L Mean</div><div class="metric-value highlight">${fmt(d.rl_pnl_mean, 4)}</div></div>
      <div class="metric-card"><div class="metric-label">BS P&L Mean</div><div class="metric-value">${fmt(d.bs_pnl_mean, 4)}</div></div>
      <div class="metric-card"><div class="metric-label">RL Sharpe</div><div class="metric-value">${fmt(d.rl_sharpe, 4)}</div></div>
      <div class="metric-card"><div class="metric-label">Improvement</div><div class="metric-value ${improvClass}">${fmt(d.improvement_pct, 1)}%</div></div>
    `;
    $('hedgeNarrative').textContent = `Backtest over ${d.n_scenarios} scenarios: RL agent P&L std ${fmt(d.rl_pnl_std, 4)} vs BS ${fmt(d.bs_pnl_std, 4)}. RL max drawdown: ${fmt(d.rl_max_drawdown, 4)}.`;
    toast('success', 'Backtest Complete', `Improvement: ${fmt(d.improvement_pct, 1)}%`);
  } catch (err) {
    toast('error', 'Backtest Error', err.message);
  } finally { hideLoading(); }
});

$('hedgeSuggestBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const d = await api('/api/v1/quant/hedging/suggest', {
      spot: pf('hedgeSpot', 100), strike: pf('hedgeStrike', 100),
      maturity: pf('hedgeMaturity', 0.25), volatility: pf('hedgeVol', 0.2),
      rate: pf('hedgeRate', 0.05), current_hedge_ratio: 0.5,
      current_pnl: 0, regime: 0,
    });
    if (!d) return;
    $('hedgeResults').style.display = '';
    $('hedgeMetrics').innerHTML = `
      <div class="metric-card"><div class="metric-label">Recommended Ratio</div><div class="metric-value highlight">${fmt(d.recommended_ratio, 4)}</div></div>
      <div class="metric-card"><div class="metric-label">BS Delta</div><div class="metric-value">${fmt(d.bs_delta, 4)}</div></div>
      <div class="metric-card"><div class="metric-label">Action</div><div class="metric-value">${d.action}</div></div>
      <div class="metric-card"><div class="metric-label">Regime</div><div class="metric-value">${d.regime}</div></div>
    `;
    $('hedgeNarrative').textContent = d.reasoning || `Suggested hedge ratio: ${fmt(d.recommended_ratio, 4)} (BS delta: ${fmt(d.bs_delta, 4)}). Action: ${d.action}.`;
    toast('success', 'Hedge Suggestion', d.action);
  } catch (err) {
    toast('error', 'Suggest Error', err.message);
  } finally { hideLoading(); }
});


// ── 33. Vol Surface ────────────────────────────────────────────
let vsChartInstance = null;

$('vsTrainBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const d = await api('/api/v1/quant/vol-surface/train', {
      n_samples: parseInt($('vsSamples').value) || 500,
      epochs: parseInt($('vsEpochs').value) || 100,
      regime: parseInt($('vsRegime').value) || 0,
    }, { timeout: 120000 });
    if (!d) return;
    $('vsResults').style.display = '';
    $('vsMetrics').innerHTML = `
      <div class="metric-card"><div class="metric-label">Final Loss</div><div class="metric-value highlight">${fmt(d.final_loss, 6)}</div></div>
      <div class="metric-card"><div class="metric-label">Smoothness</div><div class="metric-value">${fmt(d.smoothness_loss, 6)}</div></div>
      <div class="metric-card"><div class="metric-label">Arb Loss</div><div class="metric-value">${fmt(d.arbitrage_loss, 6)}</div></div>
      <div class="metric-card"><div class="metric-label">Time</div><div class="metric-value">${fmt(d.training_time_ms, 0)}ms</div></div>
    `;
    $('vsNarrative').textContent = `Transformer vol surface model trained for ${d.epochs_trained} epochs. Loss: ${fmt(d.final_loss, 6)}.`;
    toast('success', 'Vol Surface Trained', `Loss: ${fmt(d.final_loss, 6)}`);
  } catch (err) {
    toast('error', 'VS Train Error', err.message);
  } finally { hideLoading(); }
});

$('vsPredictBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const d = await api('/api/v1/quant/vol-surface/predict', {
      spot: pf('vsSpot', 100), rate: pf('vsRate', 0.05),
      base_vol: pf('vsBaseVol', 0.2), regime: parseInt($('vsRegime').value) || 0,
    });
    if (!d) return;
    $('vsResults').style.display = '';
    $('vsMetrics').innerHTML = `
      <div class="metric-card"><div class="metric-label">Regime</div><div class="metric-value highlight">${d.regime}</div></div>
      <div class="metric-card"><div class="metric-label">Strikes</div><div class="metric-value">${d.strikes.length}</div></div>
      <div class="metric-card"><div class="metric-label">Maturities</div><div class="metric-value">${d.maturities.length}</div></div>
      <div class="metric-card"><div class="metric-label">ATM Vol</div><div class="metric-value">${fmt(d.smile_atm[Math.floor(d.smile_atm.length/2)], 4)}</div></div>
    `;

    // Chart: ATM smile (vol vs maturity)
    if (vsChartInstance) vsChartInstance.destroy();
    const ctx = $('vsChart').getContext('2d');
    vsChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: d.maturities.map(m => m.toFixed(2) + 'Y'),
        datasets: [{
          label: 'ATM Vol Smile',
          data: d.smile_atm,
          borderColor: '#22d3ee',
          backgroundColor(ctxDS) {
            const chart = ctxDS.chart;
            const { ctx: cx, chartArea } = chart;
            return gradientFill(cx, chartArea, 34, 211, 238);
          },
          fill: true,
          tension: 0.35,
          borderWidth: 2.5,
          pointRadius: 3,
          pointHoverRadius: 6,
          pointBackgroundColor: '#22d3ee',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: '#22d3ee',
          pointHoverBorderWidth: 2.5,
        }, {
          label: 'Term Structure',
          data: d.term_structure,
          borderColor: '#a78bfa',
          backgroundColor(ctxDS) {
            const chart = ctxDS.chart;
            const { ctx: cx, chartArea } = chart;
            return gradientFill(cx, chartArea, 167, 139, 250);
          },
          fill: true,
          tension: 0.35,
          borderWidth: 2.5,
          pointRadius: 3,
          pointHoverRadius: 6,
          pointBackgroundColor: '#a78bfa',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: '#a78bfa',
          pointHoverBorderWidth: 2.5,
        }],
      },
      options: chartDefaults({
        scales: {
          y: { title: { display: true, text: 'Implied Vol', color: (document.documentElement.dataset.theme !== 'light') ? '#a1a7c0' : '#4a5068', font: { size: 12, weight: '500' } } },
        },
      }),
    });

    $('vsNarrative').textContent = `Vol surface predicted: ${d.strikes.length}×${d.maturities.length} grid in ${d.regime} regime.`;
    toast('success', 'Surface Predicted', `${d.strikes.length}×${d.maturities.length} surface`);
  } catch (err) {
    toast('error', 'Vol Surface Error', err.message);
  } finally { hideLoading(); }
});


// ── 34. Jump Diffusion ─────────────────────────────────────────
$('jdPriceBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const d = await api('/api/v1/quant/jump-diffusion/price', {
      spot: pf('jdSpot', 100), strike: pf('jdStrike', 100),
      maturity: pf('jdMaturity', 1), rate: pf('jdRate', 0.05),
      volatility: pf('jdVol', 0.2), option_type: $('jdType').value,
    });
    if (!d) return;
    $('jdResults').style.display = '';
    const premClass = d.jump_premium > 0 ? 'negative' : 'positive';
    $('jdMetrics').innerHTML = `
      <div class="metric-card"><div class="metric-label">JD Price</div><div class="metric-value highlight">$${fmt(d.price, 4)}</div></div>
      <div class="metric-card"><div class="metric-label">BS Price</div><div class="metric-value">$${fmt(d.bs_price, 4)}</div></div>
      <div class="metric-card"><div class="metric-label">Jump Premium</div><div class="metric-value ${premClass}">$${fmt(d.jump_premium, 4)}</div></div>
      <div class="metric-card"><div class="metric-label">Premium %</div><div class="metric-value">${fmt(d.jump_premium_pct, 2)}%</div></div>
    `;
    const regime = d.metadata?.regime || 'unknown';
    $('jdNarrative').textContent = `Merton Jump Diffusion price: $${fmt(d.price, 4)} vs BS $${fmt(d.bs_price, 4)}. Jump premium: $${fmt(d.jump_premium, 4)} (${fmt(d.jump_premium_pct, 2)}%). Current regime: ${regime}.`;
    toast('success', 'JD Price', `$${fmt(d.price, 4)} (+${fmt(d.jump_premium_pct, 1)}% jump premium)`);
  } catch (err) {
    toast('error', 'JD Price Error', err.message);
  } finally { hideLoading(); }
});

$('jdScenarioBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const d = await api('/api/v1/quant/jump-diffusion/scenario', {
      spot: pf('jdSpot', 100), strike: pf('jdStrike', 100),
      maturity: pf('jdMaturity', 1), rate: pf('jdRate', 0.05),
      base_vol: pf('jdVol', 0.2), option_type: $('jdType').value,
    });
    if (!d) return;
    $('jdResults').style.display = '';
    const scenarios = d.scenarios || {};
    let html = '';
    for (const [name, data] of Object.entries(scenarios)) {
      const price = typeof data === 'object' ? data.price : data;
      html += `<div class="metric-card"><div class="metric-label">${name}</div><div class="metric-value">$${fmt(price, 4)}</div></div>`;
    }
    $('jdMetrics').innerHTML = html;
    $('jdNarrative').textContent = d.regime_impact_summary || 'Scenario analysis complete.';
    toast('success', 'Scenario Analysis', `${Object.keys(scenarios).length} scenarios computed`);
  } catch (err) {
    toast('error', 'Scenario Error', err.message);
  } finally { hideLoading(); }
});


// ── 35. Arbitrage Scanner ──────────────────────────────────────
$('arbScanBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const d = await api('/api/v1/quant/arbitrage/scan', {
      spot: pf('arbSpot', 100), rate: pf('arbRate', 0.05),
      n_options: parseInt($('arbN').value) || 20,
      regime: parseInt($('arbRegime').value) || 0,
    });
    if (!d) return;
    $('arbResults').style.display = '';
    $('arbMetrics').innerHTML = `
      <div class="metric-card"><div class="metric-label">Total Signals</div><div class="metric-value highlight">${d.total_signals}</div></div>
      <div class="metric-card"><div class="metric-label">High Confidence</div><div class="metric-value positive">${d.high_confidence}</div></div>
      <div class="metric-card"><div class="metric-label">Medium</div><div class="metric-value">${d.medium_confidence}</div></div>
      <div class="metric-card"><div class="metric-label">Expected Profit</div><div class="metric-value">$${fmt(d.total_expected_profit, 2)}</div></div>
    `;
    const tbody = $('arbSignalBody');
    tbody.innerHTML = '';
    (d.signals || []).forEach(s => {
      const tr = document.createElement('tr');
      const strengthClass = s.strength > 0.7 ? 'positive' : s.strength > 0.3 ? '' : 'negative';
      tr.innerHTML = `
        <td>${s.type}</td>
        <td class="${strengthClass}">${fmt(s.strength, 3)}</td>
        <td>$${fmt(s.expected_profit, 2)}</td>
        <td>${fmt(s.risk_score, 2)}</td>
        <td style="font-size:0.75rem">${s.recommendation || '—'}</td>
      `;
      tbody.appendChild(tr);
    });
    toast('success', 'Arbitrage Scan', d.summary);
  } catch (err) {
    toast('error', 'Scan Error', err.message);
  } finally { hideLoading(); }
});


// ── 36. Uncertainty Quantification ─────────────────────────────
$('uqQuantifyBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const d = await api('/api/v1/quant/uncertainty/quantify', {
      spot: pf('uqSpot', 100), strike: pf('uqStrike', 100),
      maturity: pf('uqMaturity', 1), rate: pf('uqRate', 0.05),
      volatility: pf('uqVol', 0.2), option_type: $('uqType').value,
      n_samples: parseInt($('uqSamples').value) || 100,
    });
    if (!d) return;
    $('uqResults').style.display = '';
    const reliClass = d.reliability === 'high' ? 'positive' : d.reliability === 'low' ? 'negative' : '';
    $('uqMetrics').innerHTML = `
      <div class="metric-card"><div class="metric-label">Mean Price</div><div class="metric-value highlight">$${fmt(d.mean_price, 4)}</div></div>
      <div class="metric-card"><div class="metric-label">Std Dev</div><div class="metric-value">$${fmt(d.std_price, 4)}</div></div>
      <div class="metric-card"><div class="metric-label">95% CI</div><div class="metric-value">[$${fmt(d.ci_lower, 2)}, $${fmt(d.ci_upper, 2)}]</div></div>
      <div class="metric-card"><div class="metric-label">Reliability</div><div class="metric-value ${reliClass}">${d.reliability.toUpperCase()}</div></div>
    `;
    $('uqNarrative').textContent = `Mean price: $${fmt(d.mean_price, 4)} ± $${fmt(d.std_price, 4)}. Epistemic: ${fmt(d.epistemic_uncertainty, 4)}, Aleatoric: ${fmt(d.aleatoric_uncertainty, 4)}. Reliability: ${d.reliability}.`;
    toast('success', 'Uncertainty', `Reliability: ${d.reliability}`);
  } catch (err) {
    toast('error', 'UQ Error', err.message);
  } finally { hideLoading(); }
});


// ── 37. GPU Monte Carlo ────────────────────────────────────────
$('gmcPriceBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const d = await api('/api/v1/quant/gpu-mc/price', {
      spot: pf('gmcSpot', 100), strike: pf('gmcStrike', 100),
      maturity: pf('gmcMaturity', 1), rate: pf('gmcRate', 0.05),
      volatility: pf('gmcVol', 0.2), option_type: $('gmcType').value,
      n_paths: parseInt($('gmcPaths').value) || 100000,
      model: $('gmcModel').value,
      variance_reduction: $('gmcVR').value,
    }, { timeout: 60000 });
    if (!d) return;
    $('gmcResults').style.display = '';
    $('gmcMetrics').innerHTML = `
      <div class="metric-card"><div class="metric-label">Price</div><div class="metric-value highlight">$${fmt(d.price, 4)}</div></div>
      <div class="metric-card"><div class="metric-label">Std Error</div><div class="metric-value">${fmt(d.std_error, 6)}</div></div>
      <div class="metric-card"><div class="metric-label">95% CI</div><div class="metric-value">[$${fmt(d.ci_lower, 3)}, $${fmt(d.ci_upper, 3)}]</div></div>
      <div class="metric-card"><div class="metric-label">Backend</div><div class="metric-value">${d.backend.toUpperCase()}</div></div>
      <div class="metric-card"><div class="metric-label">Time</div><div class="metric-value">${fmt(d.elapsed_ms, 1)}ms</div></div>
      <div class="metric-card"><div class="metric-label">Paths</div><div class="metric-value">${Number(d.n_paths).toLocaleString()}</div></div>
    `;
    $('gmcNarrative').textContent = `GPU MC priced ${d.model.toUpperCase()} at $${fmt(d.price, 4)} ± ${fmt(d.std_error, 6)} using ${d.backend} backend with ${d.variance_reduction} variance reduction in ${fmt(d.elapsed_ms, 1)}ms.`;
    toast('success', 'GPU MC Price', `$${fmt(d.price, 4)} in ${fmt(d.elapsed_ms, 1)}ms`);
  } catch (err) {
    toast('error', 'GPU MC Error', err.message);
  } finally { hideLoading(); }
});

$('gmcBenchBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const d = await api('/api/v1/quant/gpu-mc/benchmark', {
      spot: pf('gmcSpot', 100), strike: pf('gmcStrike', 100),
      maturity: pf('gmcMaturity', 1), rate: pf('gmcRate', 0.05),
      volatility: pf('gmcVol', 0.2), option_type: $('gmcType').value,
      path_counts: [10000, 50000, 100000, 500000],
    }, { timeout: 120000 });
    if (!d) return;
    $('gmcResults').style.display = '';
    let html = '<div style="overflow-x:auto"><table class="data-table"><thead><tr><th>Paths</th><th>Price</th><th>Time (ms)</th><th>Backend</th></tr></thead><tbody>';
    (d.results || []).forEach(r => {
      html += `<tr><td>${Number(r.n_paths || 0).toLocaleString()}</td><td>$${fmt(r.price, 4)}</td><td>${fmt(r.elapsed_ms, 1)}</td><td>${r.backend || 'numpy'}</td></tr>`;
    });
    html += '</tbody></table></div>';
    $('gmcNarrative').innerHTML = html;
    $('gmcMetrics').innerHTML = `
      <div class="metric-card"><div class="metric-label">GPU Available</div><div class="metric-value ${d.gpu_available ? 'positive' : ''}">${d.gpu_available ? '✅ Yes' : '❌ No'}</div></div>
    `;
    toast('success', 'Benchmark Done', `${(d.results||[]).length} configurations tested`);
  } catch (err) {
    toast('error', 'Benchmark Error', err.message);
  } finally { hideLoading(); }
});


// ── 38. Portfolio Risk ─────────────────────────────────────────
$('pfAddPositionBtn')?.addEventListener('click', () => {
  const container = $('pfPositions');
  const idx = container.querySelectorAll('.pf-position').length;
  const div = document.createElement('div');
  div.className = 'form-grid pf-position';
  div.dataset.idx = idx;
  div.innerHTML = `
    <div class="field"><label>Spot ($)</label><input type="number" class="pfSpot" value="100" step="0.01" min="0.01" /></div>
    <div class="field"><label>Strike ($)</label><input type="number" class="pfStrike" value="105" step="0.01" min="0.01" /></div>
    <div class="field"><label>Maturity</label><input type="number" class="pfMaturity" value="0.5" step="0.01" min="0.01" /></div>
    <div class="field"><label>Vol (σ)</label><input type="number" class="pfVol" value="0.25" step="0.01" min="0.01" /></div>
    <div class="field"><label>Type</label><select class="pfType"><option value="call">Call</option><option value="put" selected>Put</option></select></div>
    <div class="field"><label>Quantity</label><input type="number" class="pfQty" value="-5" step="1" /></div>
  `;
  container.appendChild(div);
});

function getPortfolioPositions() {
  const positions = [];
  document.querySelectorAll('.pf-position').forEach(row => {
    positions.push({
      spot: parseFloat(row.querySelector('.pfSpot').value) || 100,
      strike: parseFloat(row.querySelector('.pfStrike').value) || 100,
      maturity: parseFloat(row.querySelector('.pfMaturity').value) || 1,
      rate: 0.05,
      volatility: parseFloat(row.querySelector('.pfVol').value) || 0.2,
      option_type: row.querySelector('.pfType').value,
      quantity: parseInt(row.querySelector('.pfQty').value) || 1,
      premium_paid: 0,
    });
  });
  return positions;
}

$('pfRiskReportBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const positions = getPortfolioPositions();
    const d = await api('/api/v1/quant/portfolio/risk-report', {
      positions, confidence_level: 0.95, horizon_days: 1, current_regime: 0,
    }, { timeout: 60000 });
    if (!d) return;
    $('pfResults').style.display = '';
    const ratingClass = d.risk_rating === 'LOW' ? 'positive' : d.risk_rating === 'CRITICAL' ? 'negative' : '';
    $('pfMetrics').innerHTML = `
      <div class="metric-card"><div class="metric-label">Portfolio Value</div><div class="metric-value highlight">$${fmt(d.total_value, 2)}</div></div>
      <div class="metric-card"><div class="metric-label">VaR (95%)</div><div class="metric-value negative">$${fmt(d.var_parametric, 2)}</div></div>
      <div class="metric-card"><div class="metric-label">Expected Shortfall</div><div class="metric-value">$${fmt(d.expected_shortfall, 2)}</div></div>
      <div class="metric-card"><div class="metric-label">Risk Rating</div><div class="metric-value ${ratingClass}">${d.risk_rating}</div></div>
    `;
    // Stress test table
    const tbody = $('pfStressBody');
    tbody.innerHTML = '';
    (d.stress_tests || []).forEach(s => {
      const tr = document.createElement('tr');
      const pnlClass = (s.pnl || 0) < 0 ? 'negative' : 'positive';
      tr.innerHTML = `<td>${s.scenario || '—'}</td><td class="${pnlClass}">$${fmt(s.pnl, 2)}</td><td>${fmt(s.pnl_pct, 1)}%</td><td>$${fmt(s.new_value, 2)}</td>`;
      tbody.appendChild(tr);
    });
    $('pfNarrative').innerHTML = (d.recommendations || []).map(r => `<div style="padding:0.2rem 0">• ${r}</div>`).join('');
    toast('success', 'Risk Report', `Rating: ${d.risk_rating}, VaR: $${fmt(d.var_parametric, 2)}`);
  } catch (err) {
    toast('error', 'Risk Report Error', err.message);
  } finally { hideLoading(); }
});

$('pfStressBtn')?.addEventListener('click', async () => {
  showLoading();
  try {
    const positions = getPortfolioPositions();
    const d = await api('/api/v1/quant/portfolio/stress-test', { positions }, { timeout: 60000 });
    if (!d) return;
    $('pfResults').style.display = '';
    $('pfMetrics').innerHTML = `
      <div class="metric-card"><div class="metric-label">Scenarios Run</div><div class="metric-value highlight">${(d.results || []).length}</div></div>
      <div class="metric-card"><div class="metric-label">Worst Case</div><div class="metric-value negative">${d.worst_case_scenario}</div></div>
      <div class="metric-card"><div class="metric-label">Worst Loss</div><div class="metric-value negative">$${fmt(d.worst_case_loss, 2)}</div></div>
    `;
    const tbody = $('pfStressBody');
    tbody.innerHTML = '';
    (d.results || []).forEach(s => {
      const tr = document.createElement('tr');
      const pnlClass = (s.pnl || 0) < 0 ? 'negative' : 'positive';
      tr.innerHTML = `<td>${s.scenario}</td><td class="${pnlClass}">$${fmt(s.pnl, 2)}</td><td>${fmt(s.pnl_pct, 1)}%</td><td>$${fmt(s.new_value, 2)}</td>`;
      tbody.appendChild(tr);
    });
    $('pfNarrative').textContent = d.summary;
    toast('success', 'Stress Test', d.summary);
  } catch (err) {
    toast('error', 'Stress Test Error', err.message);
  } finally { hideLoading(); }
});
