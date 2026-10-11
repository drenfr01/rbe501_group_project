"""Render the Mermaid diagrams in docs/diagrams/ to PNG images.

HOW TO RUN (from the project folder):
    python scripts/render_diagrams.py

Each `docs/diagrams/<name>.mmd` becomes `docs/diagrams/<name>.png`. Rendering
uses the public mermaid.ink service, so it needs network access but no
Node.js. Edit the .mmd file, re-run this script, and commit both files.
"""

import base64
import time
import urllib.error
import urllib.request
from pathlib import Path

from rbe_501_so101.config import PROJECT_ROOT

DIAGRAMS_DIR = PROJECT_ROOT / "docs" / "diagrams"
RENDER_URL = "https://mermaid.ink/img/{payload}?type=png&bgColor=!white&width={width}"
WIDTH_PX = 1800
TIMEOUT_S = 60
MAX_ATTEMPTS = 3
RETRY_DELAY_S = 3.0
SYNTAX_ERROR_STATUS = 400


def render(source: Path) -> Path:
    """Fetch a PNG for one .mmd file and write it next to the source."""
    payload = base64.urlsafe_b64encode(source.read_bytes()).decode("ascii")
    url = RENDER_URL.format(payload=payload, width=WIDTH_PX)
    target = source.with_suffix(".png")
    target.write_bytes(_fetch_with_retry(url, source.name))
    return target


def _fetch_with_retry(url: str, name: str) -> bytes:
    """GET the image, retrying transient server errors; fail fast on syntax errors."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _fetch(url)
        except urllib.error.HTTPError as error:
            if error.code == SYNTAX_ERROR_STATUS or attempt == MAX_ATTEMPTS:
                detail = error.read().decode("utf-8", "replace")[:500]
                raise RuntimeError(f"{name}: HTTP {error.code}\n{detail}") from error
            time.sleep(RETRY_DELAY_S)
    raise AssertionError("unreachable")


def _fetch(url: str) -> bytes:
    """Single HTTP GET returning the response body."""
    request = urllib.request.Request(url, headers={"User-Agent": "render-diagrams"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return response.read()


def main() -> None:
    """Render every diagram in docs/diagrams/."""
    for source in sorted(DIAGRAMS_DIR.glob("*.mmd")):
        print(f"rendered {render(source).relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
