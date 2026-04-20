"""Fix vendor prefix ordering and missing webkit prefixes in CSS files."""
from pathlib import Path

BASE = Path('d:/Option-Pricing-Using-Monte-Carlo-Simulation-Deep-Learning/frontend')

def fix(path, pairs):
    content = path.read_text(encoding='utf-8')
    for old, new in pairs:
        count = content.count(old)
        if count == 0:
            print(f'  NOT FOUND: {repr(old[:80])}')
        else:
            print(f'  [{count}x] {repr(old[:70])}')
        content = content.replace(old, new)
    path.write_text(content, encoding='utf-8')

# ─── styles.css ─────────────────────────────────────────────────────────────
print('\nstyles.css:')
fix(BASE / 'styles.css', [
    # SWAP mask-image (L271): webkit must come BEFORE standard
    (
        '  mask-image: radial-gradient(ellipse 65% 55% at 50% 50%, black 20%, transparent 75%);\n'
        '  -webkit-mask-image: radial-gradient(ellipse 65% 55% at 50% 50%, black 20%, transparent 75%);',
        '  -webkit-mask-image: radial-gradient(ellipse 65% 55% at 50% 50%, black 20%, transparent 75%);\n'
        '  mask-image: radial-gradient(ellipse 65% 55% at 50% 50%, black 20%, transparent 75%);',
    ),
    # SWAP backdrop-filter sidebar (L364): webkit must come BEFORE standard
    (
        '  backdrop-filter: blur(40px) saturate(1.4);\n  -webkit-backdrop-filter: blur(40px) saturate(1.4);',
        '  -webkit-backdrop-filter: blur(40px) saturate(1.4);\n  backdrop-filter: blur(40px) saturate(1.4);',
    ),
    # SWAP backdrop-filter topbar+card (L743, L860): same pattern twice — both get fixed
    (
        '  backdrop-filter: blur(20px) saturate(1.2);\n  -webkit-backdrop-filter: blur(20px) saturate(1.2);',
        '  -webkit-backdrop-filter: blur(20px) saturate(1.2);\n  backdrop-filter: blur(20px) saturate(1.2);',
    ),
    # ADD webkit — status-pill backdrop-filter (unique blur value 12px; ends rule)
    (
        '  transition: all var(--dur-fast) var(--ease-out);\n  backdrop-filter: blur(12px);\n}',
        '  transition: all var(--dur-fast) var(--ease-out);\n  -webkit-backdrop-filter: blur(12px);\n  backdrop-filter: blur(12px);\n}',
    ),
    # ADD webkit — sidebar-overlay backdrop-filter (unique: blur(4px) then z-index)
    (
        '  backdrop-filter: blur(4px);\n  z-index: calc(var(--z-sidebar) - 1);',
        '  -webkit-backdrop-filter: blur(4px);\n  backdrop-filter: blur(4px);\n  z-index: calc(var(--z-sidebar) - 1);',
    ),
    # ADD webkit — metric-card backdrop-filter (unique blur 16px; ends rule)
    (
        '  transition: all var(--dur-normal) var(--ease-out);\n  backdrop-filter: blur(16px);\n}',
        '  transition: all var(--dur-normal) var(--ease-out);\n  -webkit-backdrop-filter: blur(16px);\n  backdrop-filter: blur(16px);\n}',
    ),
    # ADD webkit — toast backdrop-filter (unique: blur 20px saturate 1.3 then box-shadow)
    (
        '  backdrop-filter: blur(20px) saturate(1.3);\n  box-shadow: var(--shadow-lg);',
        '  -webkit-backdrop-filter: blur(20px) saturate(1.3);\n  backdrop-filter: blur(20px) saturate(1.3);\n  box-shadow: var(--shadow-lg);',
    ),
    # ADD webkit — loading-overlay backdrop-filter (unique: blur 8px then display:flex)
    (
        '  backdrop-filter: blur(8px);\n  display: flex;',
        '  -webkit-backdrop-filter: blur(8px);\n  backdrop-filter: blur(8px);\n  display: flex;',
    ),
    # ADD -webkit-user-select before ALL user-select: none (5 occurrences, none have webkit)
    ('  user-select: none;', '  -webkit-user-select: none;\n  user-select: none;'),
])

# ─── premium.css ────────────────────────────────────────────────────────────
print('\npremium.css:')
fix(BASE / 'premium.css', [
    # SWAP mask (L43): webkit must come BEFORE standard
    (
        '  mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);\n'
        '  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);',
        '  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);\n'
        '  mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);',
    ),
    # SWAP mask-composite (L45): webkit must come BEFORE standard
    (
        '  mask-composite: exclude;\n  -webkit-mask-composite: xor;',
        '  -webkit-mask-composite: xor;\n  mask-composite: exclude;',
    ),
    # SWAP card-glass backdrop-filter (L265): webkit must come BEFORE standard
    (
        '  backdrop-filter: blur(30px) saturate(1.5);\n  -webkit-backdrop-filter: blur(30px) saturate(1.5);',
        '  -webkit-backdrop-filter: blur(30px) saturate(1.5);\n  backdrop-filter: blur(30px) saturate(1.5);',
    ),
    # ADD webkit — tooltip backdrop-filter (unique: blur(8px); ends rule with border before)
    (
        '  border: 1px solid var(--border);\n  backdrop-filter: blur(8px);\n}',
        '  border: 1px solid var(--border);\n  -webkit-backdrop-filter: blur(8px);\n  backdrop-filter: blur(8px);\n}',
    ),
    # ADD webkit user-select (tab-item, 1 occurrence)
    ('  user-select: none;', '  -webkit-user-select: none;\n  user-select: none;'),
])

print('\nAll done.')
