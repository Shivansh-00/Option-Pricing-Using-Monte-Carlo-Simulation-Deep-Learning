/* ═══════════════════════════════════════════════════════════════
   OptionQuant — Application Logic (Conference-Grade)
   ═══════════════════════════════════════════════════════════════ */

// ── 1. Auth Guard ──────────────────────────────────────────────
// Stores the ongoing refresh promise so API helpers can await it.
let _authReady = Promise.resolve();
let _tokenRefreshTimer = null;

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
      const res = await fetch('/api/v1/auth/refresh', {
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

async function api(url, body, { retries = 1, timeout = 30000 } = {}) {
  await _authReady;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const res = await fetch(url, {
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
      const res = await fetch(url, { headers: getAuthHeaders(), signal: controller.signal });
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
  explainability: { title: 'AI Explainability',     sub: 'RAG-powered Q&A engine' }
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
    const res = await fetch('/health', { signal: ctrl.signal });
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
  sessionStorage.removeItem('oq-chat-history');
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

    // Backend returns {model, price, metadata} — use actual std_error & CI from metadata
    const meta = mc.metadata || {};
    const se = meta.std_error != null ? meta.std_error : Math.abs(bs.price - mc.price) / 1.96;
    $('mcStd').textContent = se > 0.00005 ? fmt(se) : '< 0.0001';
    const ciLo = meta.ci_lower != null ? meta.ci_lower : mc.price - 1.96 * se;
    const ciHi = meta.ci_upper != null ? meta.ci_upper : mc.price + 1.96 * se;
    $('mcCI').textContent = se > 0.00005
      ? `[${fmt(ciLo, 2)}, ${fmt(ciHi, 2)}]`
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

// ── 14a. DL Training ──────────────────────────────────────────
$('dlTrainBtn').addEventListener('click', dlTrain);
async function dlTrain() {
  $('dlTrainBtn').disabled = true;
  const statusCard = $('dlTrainStatus');
  statusCard.style.display = '';
  $('dlTrainInfo').innerHTML = '<div style="display:flex;align-items:center;gap:0.6rem"><div class="spinner" style="width:18px;height:18px;border:2px solid rgba(99,102,241,.3);border-top-color:#6366f1;border-radius:50%;animation:spin .8s linear infinite"></div><span style="color:var(--text-secondary);font-size:0.85rem">Training LSTM & Transformer models…</span></div>';

  try {
    const d = await api('/api/v1/dl/train', { n_days: 500, spot: 100.0, volatility: 0.2, rate: 0.05, seed: 42 }, { timeout: 120000 });
    if (!d) return;
    $('dlTrainInfo').innerHTML = `
      <div class="metrics-row" style="margin:0">
        <div class="metric-card"><div class="metric-label">Status</div><div class="metric-value highlight">✅ Trained</div></div>
        <div class="metric-card"><div class="metric-label">LSTM RMSE</div><div class="metric-value">${d.lstm_rmse != null ? fmt(d.lstm_rmse, 6) : '—'}</div></div>
        <div class="metric-card"><div class="metric-label">Transformer</div><div class="metric-value">${d.transformer_accuracy != null ? (d.transformer_accuracy * 100).toFixed(0) + '%' : '—'}</div></div>
        <div class="metric-card"><div class="metric-label">Duration</div><div class="metric-value">${d.total_time_ms != null ? fmt(d.total_time_ms, 0) + 'ms' : d.lstm_elapsed_ms != null ? fmt(d.lstm_elapsed_ms, 0) + 'ms' : '—'}</div></div>
      </div>`;
    toast('success', 'DL Training Complete', 'Models trained successfully');
  } catch (err) {
    $('dlTrainInfo').innerHTML = `<div style="color:#ff5c7c;padding:0.5rem">❌ Training failed: ${escapeHtml(err.message)}</div>`;
    toast('error', 'DL Training Failed', err.message);
  } finally {
    $('dlTrainBtn').disabled = false;
  }
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
      _volFeatureChart = new Chart($('volFeatureChart'), {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'Importance',
            data: values,
            backgroundColor: 'rgba(109,92,255,0.55)',
            borderColor: '#6d5cff',
            borderWidth: 1,
            borderRadius: 4,
          }],
        },
        options: {
          ...chartDefaults(),
          indexAxis: 'y',
          plugins: { ...chartDefaults().plugins, legend: { display: false } },
          scales: {
            ...chartDefaults().scales,
            x: { ...chartDefaults().scales.x, title: { display: true, text: 'Importance', color: '#9ba1b7' } },
            y: { ...chartDefaults().scales.y, ticks: { font: { size: 11 }, color: '#c8cce0' } },
          },
        },
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
