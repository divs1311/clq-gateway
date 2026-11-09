"""Inline the simulator and the eval data into the dashboard page.

Writes dashboard/artifact.html (body-only, for publishing as a claude.ai Artifact) and
dashboard/index.html (a standalone page you can open locally).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "dashboard"
template = (D / "template.html").read_text()
sim = (D / "sim.js").read_text()
data = json.loads((D / "data.json").read_text())
blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
page = template.replace("/*__SIM_JS__*/", sim).replace("/*__DATA__*/", blob)
assert "/*__" not in page
(D / "artifact.html").write_text(page)
head, body = page.split('<div class="wrap">', 1)
(D / "index.html").write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                              '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                              + head + '</head>\n<body style="margin:0">\n<div class="wrap">' + body + "\n</body>\n</html>\n")
print(f"dashboard/artifact.html {len(page) / 1024:.0f} KB")
