/* OptionQuant — Ultra Polish behaviour
   Adds:
   1. Cursor-tracked radial highlight on buttons (--mx, --my)
   2. IntersectionObserver to mark .section as .is-visible for stagger reveal
   3. .live-pulse class auto-applied to numeric values that change
   4. Smooth nav transitions
   Self-contained, idempotent, no external deps.
*/
(function () {
  'use strict';
  if (window.__oqUltraPolish) return;
  window.__oqUltraPolish = true;

  // 1. Cursor-tracked button highlight ---------------------------------------
  const BTN_SEL = '.btn, .dash-action-btn, .action-btn, button.btn-primary, button.btn-secondary';
  document.addEventListener('pointermove', (e) => {
    const btn = e.target.closest(BTN_SEL);
    if (!btn) return;
    const r = btn.getBoundingClientRect();
    btn.style.setProperty('--mx', ((e.clientX - r.left) / r.width * 100).toFixed(1) + '%');
    btn.style.setProperty('--my', ((e.clientY - r.top)  / r.height * 100).toFixed(1) + '%');
  }, { passive: true });

  // 2. Section reveal observer -----------------------------------------------
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) en.target.classList.add('is-visible');
      });
    }, { rootMargin: '-10% 0px -10% 0px', threshold: 0.05 });
    document.querySelectorAll('.section, .card, .metric-card, .feature-card, .dashboard-card')
      .forEach((el) => io.observe(el));
  }

  // 3. Live-pulse on numeric updates -----------------------------------------
  // Mark elements that change with .live-pulse briefly.
  const PULSE_TARGETS = '.metric-value, [data-live], [data-metric]';
  const lastValues = new WeakMap();
  function pulseIfChanged(el) {
    const txt = el.textContent.trim();
    if (lastValues.get(el) !== undefined && lastValues.get(el) !== txt) {
      el.classList.remove('is-live');
      // restart animation
      // eslint-disable-next-line no-unused-expressions
      el.offsetWidth;
      el.classList.add('is-live');
      setTimeout(() => el.classList.remove('is-live'), 1300);
    }
    lastValues.set(el, txt);
  }
  if ('MutationObserver' in window) {
    const mo = new MutationObserver((muts) => {
      muts.forEach((m) => {
        if (m.target && m.target.matches && m.target.matches(PULSE_TARGETS)) {
          pulseIfChanged(m.target);
        } else if (m.target && m.target.parentElement && m.target.parentElement.matches &&
                   m.target.parentElement.matches(PULSE_TARGETS)) {
          pulseIfChanged(m.target.parentElement);
        }
      });
    });
    mo.observe(document.body, { childList: true, characterData: true, subtree: true });
  }

  // 4. Sidebar mobile toggle (graceful, only if not already wired) -----------
  const sidebar = document.getElementById('sidebar');
  const toggleBtn = document.querySelector('.mobile-toggle, [data-mobile-toggle]');
  const overlay = document.getElementById('sidebarOverlay');
  if (sidebar && toggleBtn && !toggleBtn.__oqWired) {
    toggleBtn.__oqWired = true;
    toggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('open');
      if (overlay) overlay.classList.toggle('show', sidebar.classList.contains('open'));
    });
    if (overlay) overlay.addEventListener('click', () => {
      sidebar.classList.remove('open');
      overlay.classList.remove('show');
    });
  }

  // 5. Smooth-scroll for any in-page anchors ---------------------------------
  document.addEventListener('click', (e) => {
    const a = e.target.closest('a[href^="#"]');
    if (!a) return;
    const id = a.getAttribute('href');
    if (id && id.length > 1) {
      const tgt = document.querySelector(id);
      if (tgt) {
        e.preventDefault();
        tgt.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  });
})();
