"""Load the PEP corpus and parse its RFC-822 headers.

The headers are the whole reason this corpus was chosen: they carry the authority
signal (``Status``), the version signal (``Python-Version``) and the supersession
edges (``Superseded-By`` / ``Replaces``). Everything the reranker will eventually
need is parsed here, in Phase 1, so no later phase has to touch the corpus again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# A PEP in one of these states must not be answered from as though it were current.
# This is the definition the superseded-citation-rate metric is built on, so it lives
# in one place rather than being re-spelled at each call site.
NON_AUTHORITATIVE = frozenset({"Rejected", "Withdrawn", "Superseded", "Deferred"})

# "Header-Name: value", with continuation lines indented (the Author field wraps).
_HEADER_RE = re.compile(r"^([A-Za-z][A-Za-z-]*):[ \t]*(.*)$")
_INT_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class Pep:
    number: int
    title: str
    status: str
    pep_type: str
    python_version: str | None
    superseded_by: int | None
    replaces: tuple[int, ...]
    body: str
    path: Path

    @property
    def is_authoritative(self) -> bool:
        return self.status not in NON_AUTHORITATIVE


def _parse_headers(lines: list[str]) -> tuple[dict[str, str], int]:
    """Return (headers, index of first body line).

    RFC-822 style: the block ends at the first blank line. A line starting with
    whitespace continues the previous header rather than beginning a new one.
    """
    headers: dict[str, str] = {}
    last_key: str | None = None
    body_start = 0

    for i, raw in enumerate(lines):
        if not raw.strip():
            body_start = i + 1
            break
        if raw[0] in " \t" and last_key is not None:
            headers[last_key] = f"{headers[last_key]} {raw.strip()}".strip()
            continue
        match = _HEADER_RE.match(raw)
        if match is None:
            # Not a header and not a continuation -- treat the block as finished.
            body_start = i
            break
        last_key = match.group(1)
        headers[last_key] = match.group(2).strip()
    else:
        body_start = len(lines)

    return headers, body_start


def _first_int(value: str | None) -> int | None:
    if not value:
        return None
    match = _INT_RE.search(value)
    return int(match.group()) if match else None


def _all_ints(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(n) for n in _INT_RE.findall(value))


def load_pep(path: Path) -> Pep | None:
    """Parse one ``pep-NNNN.rst`` file. Returns None if it has no usable PEP number."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    headers, body_start = _parse_headers(lines)

    number = _first_int(headers.get("PEP"))
    if number is None:
        return None

    return Pep(
        number=number,
        title=headers.get("Title", "").strip(),
        status=headers.get("Status", "").strip(),
        pep_type=headers.get("Type", "").strip(),
        python_version=(headers.get("Python-Version") or "").strip() or None,
        superseded_by=_first_int(headers.get("Superseded-By")),
        replaces=_all_ints(headers.get("Replaces")),
        body="\n".join(lines[body_start:]),
        path=path,
    )


def load_corpus(root: Path | str = "peps/peps") -> list[Pep]:
    """Load every PEP under ``root``, sorted by number.

    ``root`` defaults to the layout produced by
    ``git clone --depth 1 https://github.com/python/peps.git peps``.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"PEP corpus not found at {root!r}. Run scripts/fetch_corpus.sh first."
        )

    peps = [pep for path in sorted(root.glob("pep-*.rst")) if (pep := load_pep(path))]
    if not peps:
        raise RuntimeError(f"No PEPs parsed from {root!r} -- has the layout changed?")
    return sorted(peps, key=lambda p: p.number)
