/* ═══════════════════════════════════════════════════════════════
   OptionQuant — Premium Motion Engine v4.0
   World-Class Micro-Interactions · GPU-Accelerated · 60fps
   ─────────────────────────────────────────────────────────────
   STRICTLY UI-ONLY: No business logic, no API calls, no data
   handling. Pure visual enhancement layer.
   ═══════════════════════════════════════════════════════════════ */

(function PremiumMotion() {
  'use strict';

  // ── Respect reduced motion preference ─────────────────────
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersReducedMotion) {
    console.log('%c✦ PremiumMotion: Reduced motion detected, skipping enhancements', 'color:#9299b3;font-size:11px');
    return;
  }

  // ── Detect touch device (disable 3D tilt & magnetic on touch) ─
  const isTouchDevice = ('ontouchstart' in window) || (navigator.maxTouchPoints > 0);

  // ── Utility: requestAnimationFrame throttle ───────────────
  function rafThrottle(fn) {
    let ticking = false;
    return function (...args) {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        fn.apply(this, args);
        ticking = false;
      });
    };
  }

  // ── 1. Cursor Spotlight on Cards ──────────────────────────
  // Injects a spotlight div and tracks mouse position via CSS vars
  function initCursorSpotlight() {
    const spotlightTargets = document.querySelectorAll(
      '.card, .metric-card, .greek-card, .chart-container, .rag-stats-bar, .rag-meta'
    );

    spotlightTargets.forEach(card => {
      // Create spotlight overlay
      const spotlight = document.createElement('div');
      spotlight.className = 'card-spotlight';
      spotlight.setAttribute('aria-hidden', 'true');
      card.appendChild(spotlight);

      card.addEventListener('mousemove', rafThrottle((e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        card.style.setProperty('--mouse-x', `${x}px`);
        card.style.setProperty('--mouse-y', `${y}px`);
        card.style.setProperty('--spotlight-active', '1');
      }));

      card.addEventListener('mouseleave', () => {
        card.style.setProperty('--spotlight-active', '0');
      });
    });
  }

  // ── 2. 3D Tilt Effect on Metric & Greek Cards ────────────
  function initCardTilt() {
    if (isTouchDevice) return; // Skip on touch devices

    const tiltCards = document.querySelectorAll('.metric-card, .greek-card');
    const maxTilt = 8; // degrees

    tiltCards.forEach(card => {
      card.style.transformStyle = 'preserve-3d';
      card.style.perspective = '800px';

      card.addEventListener('mousemove', rafThrottle((e) => {
        const rect = card.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const mouseX = e.clientX - centerX;
        const mouseY = e.clientY - centerY;

        const rotateX = -(mouseY / (rect.height / 2)) * maxTilt;
        const rotateY = (mouseX / (rect.width / 2)) * maxTilt;

        card.style.transform = `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-6px) scale(1.03)`;
        card.style.transition = 'transform 0.1s ease-out';
      }));

      card.addEventListener('mouseleave', () => {
        card.style.transform = 'perspective(800px) rotateX(0) rotateY(0) translateY(0) scale(1)';
        card.style.transition = 'transform 0.4s cubic-bezier(.34,1.56,.64,1)';
      });
    });
  }

  // ── 3. Magnetic Button Effect ─────────────────────────────
  function initMagneticButtons() {
    if (isTouchDevice) return;

    const buttons = document.querySelectorAll('.btn-primary, .btn-accent');
    const magnetStrength = 0.3; // How strongly the button follows cursor

    buttons.forEach(btn => {
      const parent = btn.parentElement || document.body;

      parent.addEventListener('mousemove', rafThrottle((e) => {
        const rect = btn.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const distX = e.clientX - centerX;
        const distY = e.clientY - centerY;
        const distance = Math.sqrt(distX * distX + distY * distY);

        // Only apply magnetic effect within 120px radius
        if (distance < 120) {
          const pullX = distX * magnetStrength;
          const pullY = distY * magnetStrength;
          btn.style.transform = `translate(${pullX}px, ${pullY}px) scale(1.02)`;
          btn.style.transition = 'transform 0.2s cubic-bezier(.34,1.56,.64,1)';
        }
      }));

      parent.addEventListener('mouseleave', () => {
        btn.style.transform = '';
        btn.style.transition = 'transform 0.4s cubic-bezier(.34,1.56,.64,1)';
      });

      // Reset on general mouse leave from button
      btn.addEventListener('mouseleave', () => {
        btn.style.transform = '';
        btn.style.transition = 'transform 0.4s cubic-bezier(.34,1.56,.64,1)';
      });
    });
  }

  // ── 4. Ripple Click Effect ────────────────────────────────
  function initRippleEffect() {
    const rippleTargets = document.querySelectorAll('.btn, .chip, .nav-item, .action-btn, .follow-up-chip');

    rippleTargets.forEach(el => {
      el.addEventListener('click', function (e) {
        // Don't add ripple if element already has a running one
        const existingRipple = el.querySelector('.ripple-effect');
        if (existingRipple) existingRipple.remove();

        const rect = el.getBoundingClientRect();
        const ripple = document.createElement('span');
        ripple.className = 'ripple-effect';

        const size = Math.max(rect.width, rect.height) * 2;
        const x = e.clientX - rect.left - size / 2;
        const y = e.clientY - rect.top - size / 2;

        ripple.style.cssText = `
          width: ${size}px;
          height: ${size}px;
          left: ${x}px;
          top: ${y}px;
        `;

        el.appendChild(ripple);

        // Clean up after animation
        ripple.addEventListener('animationend', () => ripple.remove());
      });
    });
  }

  // ── 5. Counter Animation for Metric Values ───────────────
  // Animates number from 0 to target with easing
  function animateCounter(element, targetValue, duration = 800) {
    if (!element || isNaN(targetValue)) return;

    const startTime = performance.now();
    const startValue = 0;

    // Determine decimal places from target
    const text = String(targetValue);
    const decimalPlaces = text.includes('.') ? text.split('.')[1].length : 0;

    function easeOutExpo(t) {
      return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
    }

    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easedProgress = easeOutExpo(progress);
      const currentValue = startValue + (targetValue - startValue) * easedProgress;

      element.textContent = currentValue.toFixed(decimalPlaces);

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        element.textContent = targetValue.toFixed(decimalPlaces);
        // Bounce effect at completion
        element.style.transform = 'scale(1.08)';
        element.style.transition = 'transform 0.2s cubic-bezier(.34,1.56,.64,1)';
        setTimeout(() => {
          element.style.transform = 'scale(1)';
        }, 200);
      }
    }

    requestAnimationFrame(update);
  }

  // Observe metric values for changes and animate them
  function initCounterAnimations() {
    const metricValues = document.querySelectorAll('.metric-value, .greek-value');
    let _counterAnimating = false;   // guard against infinite mutation loop

    const observer = new MutationObserver((mutations) => {
      if (_counterAnimating) return;  // skip self-triggered mutations

      mutations.forEach(mutation => {
        if (mutation.type === 'childList' || mutation.type === 'characterData') {
          const el = mutation.target.nodeType === Node.TEXT_NODE ? mutation.target.parentElement : mutation.target;
          if (!el) return;

          const text = el.textContent.trim();
          const num = parseFloat(text);
          if (!isNaN(num) && num !== 0 && text === String(num.toFixed(text.includes('.') ? text.split('.')[1]?.length || 0 : 0))) {
            const decimals = text.includes('.') ? text.split('.')[1].length : 0;
            _counterAnimating = true;
            el.textContent = (0).toFixed(decimals);
            animateCounter(el, num, 900);
            // Release guard after animation duration + buffer
            setTimeout(() => { _counterAnimating = false; }, 1000);
          }
        }
      });
    });

    metricValues.forEach(el => {
      observer.observe(el, { childList: true, characterData: true, subtree: true });
    });
  }

  // ── 6. Scroll-Triggered Reveal Animations ────────────────
  function initScrollReveals() {
    // Add scroll-reveal class to results that appear dynamically
    const revealTargets = document.querySelectorAll(
      '.metrics-row, .greeks-grid, .dl-results, .chart-container, .charts-grid, .rag-meta'
    );

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          // Don't unobserve - allow re-triggering when hidden/shown
        } else {
          entry.target.classList.remove('visible');
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: '0px 0px -30px 0px'
    });

    revealTargets.forEach(el => {
      el.classList.add('scroll-reveal');
      observer.observe(el);
    });

    // Stagger children for metrics rows
    const staggerTargets = document.querySelectorAll('.metrics-row, .greeks-grid, .dl-results');
    const staggerObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        } else {
          entry.target.classList.remove('visible');
        }
      });
    }, { threshold: 0.1 });

    staggerTargets.forEach(el => {
      el.classList.add('scroll-reveal-stagger');
      staggerObserver.observe(el);
    });
  }

  // ── 7. Navigation Mouse Position Tracking ────────────────
  // Tracks mouse position within nav items for radial hover gradient
  function initNavMouseTracking() {
    const navItems = document.querySelectorAll('.sidebar-nav .nav-item');

    navItems.forEach(item => {
      item.addEventListener('mousemove', rafThrottle((e) => {
        const rect = item.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        item.style.setProperty('--mouse-x', `${x}%`);
        item.style.setProperty('--mouse-y', `${y}%`);
      }));
    });
  }

  // ── 8. Smooth Section Transitions ─────────────────────────
  // Enhanced section switching with exit/enter choreography
  function initSectionTransitions() {
    const navItems = document.querySelectorAll('.sidebar-nav .nav-item');

    navItems.forEach(item => {
      item.addEventListener('click', () => {
        const sections = document.querySelectorAll('.section');
        const targetId = `sec-${item.dataset.section}`;

        sections.forEach(sec => {
          if (sec.classList.contains('active') && sec.id !== targetId) {
            // Exit animation
            sec.style.opacity = '0';
            sec.style.transform = 'translateY(-10px) scale(0.99)';
            sec.style.filter = 'blur(4px)';
            sec.style.transition = 'opacity 0.2s ease, transform 0.2s ease, filter 0.2s ease';

            setTimeout(() => {
              sec.style.opacity = '';
              sec.style.transform = '';
              sec.style.filter = '';
              sec.style.transition = '';
            }, 250);
          }
        });
      });
    });
  }

  // ── 9. Enhanced Sidebar Brand Interaction ─────────────────
  function initBrandInteraction() {
    const logo = document.querySelector('.sidebar-brand .logo');
    if (!logo) return;

    logo.addEventListener('mouseenter', () => {
      logo.style.animation = 'none';
      void logo.offsetHeight; // Trigger reflow
      logo.style.animation = '';
    });

    // Easter egg: Double-click logo for a subtle celebration
    logo.addEventListener('dblclick', () => {
      logo.style.transform = 'scale(1.15) rotate(360deg)';
      logo.style.transition = 'transform 0.6s cubic-bezier(.34,1.56,.64,1)';
      setTimeout(() => {
        logo.style.transform = '';
        logo.style.transition = 'transform 0.3s cubic-bezier(.34,1.56,.64,1)';
      }, 600);
    });
  }

  // ── 10. Animated Gradient Border on Active Card ──────────
  function initGradientBorders() {
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
      card.classList.add('gradient-border');
    });
  }

  // ── 11. Enhanced User Card Interaction ────────────────────
  function initUserCardInteraction() {
    const avatar = document.querySelector('.user-avatar');
    if (!avatar || isTouchDevice) return;

    let clickCount = 0;
    avatar.addEventListener('click', () => {
      clickCount++;
      if (clickCount >= 3) {
        // Fun little spin animation after triple-click on avatar
        avatar.style.transform = 'scale(1.2) rotate(360deg)';
        avatar.style.transition = 'transform 0.5s cubic-bezier(.34,1.56,.64,1)';
        setTimeout(() => {
          avatar.style.transform = '';
          avatar.style.transition = 'transform 0.3s cubic-bezier(.34,1.56,.64,1)';
        }, 500);
        clickCount = 0;
      }
    });
  }

  // ── 12. Smooth Page Title Transition ──────────────────────
  function initPageTitleTransition() {
    const pageTitle = document.getElementById('pageTitle');
    const pageSubtitle = document.getElementById('pageSubtitle');
    if (!pageTitle) return;

    // Observe text changes for smooth transition
    const observer = new MutationObserver(() => {
      pageTitle.style.animation = 'none';
      void pageTitle.offsetHeight;
      pageTitle.style.animation = 'textReveal 0.4s cubic-bezier(.16,1,.3,1) forwards';

      if (pageSubtitle) {
        pageSubtitle.style.animation = 'none';
        void pageSubtitle.offsetHeight;
        pageSubtitle.style.animation = 'textReveal 0.4s cubic-bezier(.16,1,.3,1) 0.05s forwards';
      }
    });

    observer.observe(pageTitle, { childList: true, characterData: true, subtree: true });
  }

  // ── 13. Input Focus Floating Labels ───────────────────────
  function initFloatingLabels() {
    // Add smooth label transitions on focus
    const fields = document.querySelectorAll('.field');

    fields.forEach(field => {
      const input = field.querySelector('input, select');
      const label = field.querySelector('label');
      if (!input || !label) return;

      // Add transition to label
      label.style.transition = 'color 0.25s cubic-bezier(.16,1,.3,1), transform 0.25s cubic-bezier(.16,1,.3,1)';

      input.addEventListener('focus', () => {
        label.style.color = '';  // Let CSS handle via :focus-within
        label.style.transform = 'translateX(2px)';
      });

      input.addEventListener('blur', () => {
        label.style.transform = '';
      });
    });
  }

  // ── 14. Smooth Theme Transition ───────────────────────────
  function initThemeTransition() {
    const themeToggle = document.getElementById('themeToggle');
    if (!themeToggle) return;

    themeToggle.addEventListener('click', () => {
      // Add a brief overlay flash for smooth theme transition
      const flash = document.createElement('div');
      flash.style.cssText = `
        position: fixed;
        inset: 0;
        background: var(--bg-body);
        z-index: 99999;
        opacity: 0.3;
        pointer-events: none;
        transition: opacity 0.4s ease;
      `;
      document.body.appendChild(flash);

      requestAnimationFrame(() => {
        flash.style.opacity = '0';
        flash.addEventListener('transitionend', () => flash.remove());
      });
    });
  }

  // ── 15. Parallax Background Orbs ──────────────────────────
  function initParallaxOrbs() {
    if (isTouchDevice) return;

    const orbs = document.querySelectorAll('.ambient-bg .orb');
    if (!orbs.length) return;

    const parallaxStrength = [0.02, 0.015, 0.025, 0.01]; // Different depth for each orb

    document.addEventListener('mousemove', rafThrottle((e) => {
      const centerX = window.innerWidth / 2;
      const centerY = window.innerHeight / 2;
      const deltaX = (e.clientX - centerX) / centerX;
      const deltaY = (e.clientY - centerY) / centerY;

      orbs.forEach((orb, i) => {
        const strength = parallaxStrength[i] || 0.02;
        const x = deltaX * strength * 100;
        const y = deltaY * strength * 100;
        orb.style.transform = `translate(${x}px, ${y}px)`;
      });
    }));
  }

  // ── 16. Enhanced Toast Interaction ────────────────────────
  function initEnhancedToasts() {
    const toastContainer = document.getElementById('toasts');
    if (!toastContainer) return;

    // Observe for new toasts and add enhanced entrance
    const observer = new MutationObserver((mutations) => {
      mutations.forEach(mutation => {
        mutation.addedNodes.forEach(node => {
          if (node.nodeType === Node.ELEMENT_NODE && node.classList.contains('toast')) {
            // Add stagger delay based on toast count
            const toasts = toastContainer.querySelectorAll('.toast');
            const index = Array.from(toasts).indexOf(node);
            node.style.transitionDelay = `${index * 50}ms`;

            // Add swipe-to-dismiss on touch
            if (isTouchDevice) {
              let startX = 0;
              let currentX = 0;

              node.addEventListener('touchstart', (e) => {
                startX = e.touches[0].clientX;
              }, { passive: true });

              node.addEventListener('touchmove', (e) => {
                currentX = e.touches[0].clientX;
                const diff = currentX - startX;
                if (diff > 0) {
                  node.style.transform = `translateX(${diff}px)`;
                  node.style.opacity = Math.max(0, 1 - diff / 200);
                }
              }, { passive: true });

              node.addEventListener('touchend', () => {
                const diff = currentX - startX;
                if (diff > 100) {
                  node.classList.remove('show');
                  setTimeout(() => node.remove(), 300);
                } else {
                  node.style.transform = '';
                  node.style.opacity = '';
                }
              });
            }
          }
        });
      });
    });

    observer.observe(toastContainer, { childList: true });
  }

  // ── 17. Smooth Scroll Progress Indicator ──────────────────
  function initScrollProgress() {
    const contentArea = document.querySelector('.content-area');
    if (!contentArea) return;

    // Create a subtle scroll progress line at top of main content
    const progressBar = document.createElement('div');
    progressBar.setAttribute('aria-hidden', 'true');
    const isMobile = window.innerWidth <= 1024;
    progressBar.style.cssText = `
      position: fixed;
      top: 0;
      left: ${isMobile ? '0' : 'var(--sidebar-w, 272px)'};
      right: 0;
      height: 2px;
      background: linear-gradient(90deg, var(--primary), var(--accent));
      transform-origin: left;
      transform: scaleX(0);
      z-index: 9999;
      pointer-events: none;
      transition: transform 0.1s linear;
      opacity: 0.7;
    `;
    document.body.appendChild(progressBar);

    // Update on resize
    window.addEventListener('resize', () => {
      progressBar.style.left = window.innerWidth <= 1024 ? '0' : 'var(--sidebar-w, 272px)';
    });

    const mainContent = document.querySelector('.main-content');
    if (!mainContent) return;

    mainContent.addEventListener('scroll', rafThrottle(() => {
      const scrollTop = mainContent.scrollTop;
      const scrollHeight = mainContent.scrollHeight - mainContent.clientHeight;
      const progress = scrollHeight > 0 ? scrollTop / scrollHeight : 0;
      progressBar.style.transform = `scaleX(${progress})`;
    }));

    // Also listen on window scroll
    window.addEventListener('scroll', rafThrottle(() => {
      const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
      const scrollHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const progress = scrollHeight > 0 ? scrollTop / scrollHeight : 0;
      progressBar.style.transform = `scaleX(${progress})`;
    }));
  }

  // ── 18. Card Shimmer on Hover ─────────────────────────────
  function initCardShimmer() {
    if (isTouchDevice) return;

    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
      card.addEventListener('mouseenter', () => {
        // Trigger the CSS shimmer animation
        const before = card.querySelector('::before');
        card.style.setProperty('--shimmer-active', '1');
      });
    });
  }

  // ── 19. Keyboard Navigation Enhancement ───────────────────
  function initKeyboardNav() {
    // Add focus ring animation on tab navigation
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        document.body.classList.add('keyboard-nav');
      }
    });

    document.addEventListener('mousedown', () => {
      document.body.classList.remove('keyboard-nav');
    });
  }

  // ── INITIALIZE ALL SYSTEMS ────────────────────────────────
  function init() {
    // Wait for DOM to be fully ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', bootstrap);
    } else {
      bootstrap();
    }
  }

  function bootstrap() {
    // Phase 1: Immediate (critical visual enhancements)
    initCursorSpotlight();
    initNavMouseTracking();
    initRippleEffect();
    initGradientBorders();
    initKeyboardNav();
    initFloatingLabels();

    // Phase 2: Deferred (non-critical, can wait a frame)
    requestAnimationFrame(() => {
      initCardTilt();
      initMagneticButtons();
      initCounterAnimations();
      initScrollReveals();
      initSectionTransitions();
      initPageTitleTransition();
      initBrandInteraction();
      initUserCardInteraction();
      initThemeTransition();
      initEnhancedToasts();
      initScrollProgress();

      // Phase 3: Low-priority (background effects)
      requestIdleCallback ? requestIdleCallback(() => {
        initParallaxOrbs();
      }) : setTimeout(() => {
        initParallaxOrbs();
      }, 500);
    });

    console.log(
      '%c✦ PremiumMotion v4.0 — All systems active',
      'color:#6d5cff;font-size:12px;font-weight:700;text-shadow:0 0 10px rgba(109,92,255,.5)'
    );
  }

  init();
})();
