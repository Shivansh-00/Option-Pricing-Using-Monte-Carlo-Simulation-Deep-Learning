"""Convert all style.display manipulations in app.js to classList operations."""
from pathlib import Path

js_path = Path('d:/Option-Pricing-Using-Monte-Carlo-Simulation-Deep-Learning/frontend/app.js')
content = js_path.read_text(encoding='utf-8')
original = content

# Special-case conditionals FIRST (before simple patterns touch them)
content = content.replace(
    "$('grafanaFrame').style.display = ok ? 'block' : 'none';",
    "$('grafanaFrame').classList.toggle('hidden', !ok);"
)
content = content.replace(
    "cBadge.style.display = d.cached ? '' : 'none';",
    "cBadge.classList.toggle('hidden', !d.cached);"
)

# General patterns
content = content.replace(".style.display = '';", ".classList.remove('hidden');")
content = content.replace(".style.display = 'none';", ".classList.add('hidden');")

# Verify no style.display left
remaining = content.count('style.display')
print(f'Remaining style.display occurrences: {remaining}')
if remaining:
    for i, line in enumerate(content.split('\n'), 1):
        if 'style.display' in line:
            print(f'  L{i}: {line.strip()}')

js_path.write_text(content, encoding='utf-8')
print(f'Done. {len(original)} -> {len(content)} chars')
