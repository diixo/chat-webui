"""
Unit tests for the streaming <think>...</think> parser in server.py.

Run:
    python test_thinking.py

No model load, no FastAPI start — just the pure parser logic.
"""

from __future__ import annotations

import sys
from typing import Iterable

from server import _ThinkingParser


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

Parts = list[tuple[str, str]]


def feed_chunks(chunks: Iterable[str]) -> tuple[Parts, _ThinkingParser]:
    """Feed each chunk in order, then flush. Return (parts, parser)."""
    p = _ThinkingParser()
    out: Parts = []
    for c in chunks:
        out.extend(p.feed(c))
    out.extend(p.flush())
    return out, p


def merge_consecutive(parts: Parts) -> Parts:
    """Merge adjacent parts of the same kind. The parser may emit several
    contiguous chunks of the same kind (one per feed call) — for assertion
    purposes those should be equivalent to a single concatenated part."""
    merged: Parts = []
    for kind, text in parts:
        if merged and merged[-1][0] == kind:
            merged[-1] = (kind, merged[-1][1] + text)
        else:
            merged.append((kind, text))
    return merged


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# (name, chunks, expected_parts_after_merge)
CASES: list[tuple[str, list[str], Parts]] = [
    (
        "plain text, no tags",
        ["Hello world"],
        [("content", "Hello world")],
    ),
    (
        "single complete block in one chunk",
        ["<think>reasoning</think>answer"],
        [("reasoning", "reasoning"), ("content", "answer")],
    ),
    (
        "block split: open tag torn between chunks",
        ["<thi", "nk>thinking</think>answer"],
        [("reasoning", "thinking"), ("content", "answer")],
    ),
    (
        "block split: close tag torn between chunks",
        ["<think>thinking", "</thi", "nk>answer"],
        [("reasoning", "thinking"), ("content", "answer")],
    ),
    (
        "block split: every character is its own chunk",
        list("<think>foo</think>bar"),
        [("reasoning", "foo"), ("content", "bar")],
    ),
    (
        "open without close — flush emits remainder as reasoning",
        ["<think>still thinking..."],
        [("reasoning", "still thinking...")],
    ),
    (
        "false alarm: < followed by non-tag word",
        ["hello<world>!"],
        [("content", "hello<world>!")],
    ),
    (
        "partial-only stream that never grows into a tag",
        ["<", "t", "h", "i"],
        [("content", "<thi")],
    ),
    (
        "multiple consecutive blocks",
        ["<think>a</think>mid<think>b</think>end"],
        [
            ("reasoning", "a"),
            ("content", "mid"),
            ("reasoning", "b"),
            ("content", "end"),
        ],
    ),
    (
        "empty input",
        [""],
        [],
    ),
    (
        "content immediately followed by open tag at end of chunk",
        ["intro<think>", "thoughts</think>final"],
        [
            ("content", "intro"),
            ("reasoning", "thoughts"),
            ("content", "final"),
        ],
    ),
    (
        "reasoning that contains '<' but not the tag prefix",
        ["<think>a < b and c < d</think>done"],
        [("reasoning", "a < b and c < d"), ("content", "done")],
    ),
    (
        "content that almost-but-not-quite forms <think>",
        ["<think_other>not a tag"],
        [("content", "<think_other>not a tag")],
    ),
    (
        "unicode passes through unchanged",
        ["<think>thinking</think>ready"],
        [("reasoning", "thinking"), ("content", "ready")],
    ),
    (
        "open tag exactly at the chunk boundary",
        ["before", "<think>", "during", "</think>", "after"],
        [
            ("content", "before"),
            ("reasoning", "during"),
            ("content", "after"),
        ],
    ),
    (
        "stream ends mid-tag — partial open at very end is content on flush",
        ["all good <th"],
        [("content", "all good <th")],
    ),
]


# Behaviour-level checks beyond input/output, e.g. parser state mid-stream.
def behaviour_checks() -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    # 1. After partial open tag, parser should be holding it back, not emit.
    p = _ThinkingParser()
    out = p.feed("<thi")
    results.append((
        "partial open tag is held in buffer (not emitted)",
        out == [] and not p._in_thinking,
        f"got out={out!r}, in_thinking={p._in_thinking}",
    ))

    # 2. After full open tag, mode flips even with no following content.
    p = _ThinkingParser()
    out = p.feed("<think>")
    results.append((
        "open tag with no payload flips state to thinking, emits nothing",
        out == [] and p._in_thinking,
        f"got out={out!r}, in_thinking={p._in_thinking}",
    ))

    # 3. Open without close: state remains thinking after stream ends.
    p = _ThinkingParser()
    p.feed("<think>halfway")
    p.flush()
    results.append((
        "unclosed block leaves parser in thinking state after flush",
        p._in_thinking,
        f"in_thinking={p._in_thinking}",
    ))

    # 4. No empty parts are emitted.
    p = _ThinkingParser()
    parts = p.feed("<think></think>") + p.flush()
    results.append((
        "empty <think></think> emits no parts",
        parts == [],
        f"got {parts!r}",
    ))

    return results


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run() -> int:
    failures = 0
    total = 0

    print("== input/output cases ==")
    for name, chunks, expected in CASES:
        total += 1
        parts, _ = feed_chunks(chunks)
        merged = merge_consecutive(parts)
        if merged == expected:
            print(f"[PASS] {name}")
        else:
            failures += 1
            print(f"[FAIL] {name}")
            print(f"       expected: {expected}")
            print(f"       got:      {merged}")

    print()
    print("== behaviour checks ==")
    for name, ok, detail in behaviour_checks():
        total += 1
        if ok:
            print(f"[PASS] {name}")
        else:
            failures += 1
            print(f"[FAIL] {name} — {detail}")

    print()
    if failures:
        print(f"Results: {total - failures}/{total} passed — {failures} FAILED")
        return 1
    print(f"Results: {total}/{total} passed — all green")
    return 0


if __name__ == "__main__":
    sys.exit(run())
