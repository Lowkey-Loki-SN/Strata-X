"""moex (upstream f13a870d, Maya c25b1db): the prompt's token ids from the last request's, encoding only what changed.

An agent resends its whole conversation every turn.  Encoding ~880K tokens of it in Python took 8-12 s a request on
this box (synthetic agent text, 2026-11-04) - time the engine waits before it reads a single token.  The prompt is still
rendered in full; only its encode is shortened, and exactly:

- The tokenizer splits the text at its special tokens first and encodes each piece between them on its own
  (Tokenizer._encode_matching: a match that starts inside a `plain` span is text).  So the text up to the end of any
  special token it emitted - a CUT - encodes to the same ids alone as inside the whole text: scanning resumes there,
  and no regex looks back past it.
- A new prompt shares its first d characters with a remembered one.  Every special-token match that starts before the
  last cut c with c + (the longest special token) <= d sees the same characters in both, so both texts give the same
  matches - and the same ids - up to c.  The plain spans before c must be the same too (they are: they mark literals in
  that same text), and none may cross c.
- The rest, from c, is encoded piece by piece (each piece ends at a special token), which records the cuts for the next
  request; the first request of a conversation is encoded that way from 0.  Same ids as one encode of the whole text.

STRATA_PROMPT_MEMO: "1" (default) on, "0" off (one encode of the whole text, as before), "verify" on plus a full encode
of every prompt compared with it (a difference is printed and the full encode is used).
"""
from __future__ import annotations

import array
import bisect
import os
import threading


def common_prefix(a: str, b: str) -> int:
    """len of the longest common prefix of a and b: slices compared in C, halving the window (O(len) in all)."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n                      # a[:lo] == b[:lo]; a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def _spans_before(spans: tuple, pos: int) -> tuple:
    """The (sorted) spans that start before pos."""
    return spans[:bisect.bisect_left(spans, (pos, -1))]


def _crosses(spans: tuple, pos: int) -> bool:
    """A span that starts before pos ends after it."""
    return any(a < pos < b for a, b in _spans_before(spans, pos))


class _Entry:
    __slots__ = ("text", "spans", "ids", "cut_pos", "cut_ids")

    def __init__(self, text, spans, ids, cut_pos, cut_ids):
        self.text, self.spans, self.ids, self.cut_pos, self.cut_ids = text, spans, ids, cut_pos, cut_ids


class PromptMemo:
    """Remembers the last `keep` prompts (text, plain spans, ids, cuts) and encodes a new prompt from the one it shares the
    most with.  Thread-safe; the tokenizer is the caller's (its own caches as before)."""

    def __init__(self, tok, keep: int = 2, mode: str | None = None):
        self.tok = tok
        self.keep = max(1, keep)
        self.mode = (os.environ.get("STRATA_PROMPT_MEMO", "1") if mode is None else mode).strip().lower()
        self.pat = getattr(tok, "_special_re", None)
        specials = getattr(tok, "special_tokens", None) or {}
        self.lmax = max((len(t) for t in specials), default=0)
        self.lock = threading.Lock()
        self.entries: list[_Entry] = []     # most recent first
        self.reused = 0                     # tokens taken from a remembered prompt (the last request's)
        self.mismatches = 0                 # verify mode: prompts whose memo ids differed from a full encode

    def on(self) -> bool:
        return self.mode != "0" and self.pat is not None and self.lmax > 0

    def _pieces(self, text: str, spans: tuple, c: int):
        """text[c:] encoded piece by piece, each ending at a special token the tokenizer emits there -> (ids, the cuts'
        positions and ids counts, both relative to c).  c must be 0 or a cut."""
        tok, out, cut_pos, cut_ids = self.tok, array.array("i"), [], []
        i = bisect.bisect_left(spans, (c, -1))     # the spans from c on (none crosses c: checked by the caller)
        nspans, plain_end, pos = len(spans), 0, c
        seg_first = i                              # the first span of the piece being collected
        for m in self.pat.finditer(text, c):
            start = m.start()
            while i < nspans and spans[i][0] <= start:     # Tokenizer._encode_matching's rule, the same consumption
                plain_end = max(plain_end, spans[i][1])
                i += 1
            if start < plain_end:
                continue                           # a literal inside a plain span: text, inside the piece
            end = m.end()
            seg_spans = [(a - pos, b - pos) for a, b in spans[seg_first:i] if a < end]
            out.extend(tok.encode(text[pos:end], parse_special=True, plain=seg_spans))
            cut_pos.append(end - c)
            cut_ids.append(len(out))
            pos, seg_first = end, i
        if pos < len(text):
            seg_spans = [(a - pos, b - pos) for a, b in spans[seg_first:]]
            out.extend(tok.encode(text[pos:], parse_special=True, plain=seg_spans))
        return out, cut_pos, cut_ids

    def encode(self, text: str, plain=()) -> list[int]:
        """tok.encode(text, parse_special=True, plain=plain), from the remembered prompt that shares the most with it."""
        if not self.on():                          # the call encode_prompt made before (a plain-less one for mocks)
            return self.tok.encode(text, parse_special=True, plain=plain) if plain else self.tok.encode(text, parse_special=True)
        spans = tuple(sorted(plain)) if plain else ()
        with self.lock:
            entries = list(self.entries)
        best, best_d = None, 0
        for e in entries:
            d = common_prefix(e.text, text)
            if d > best_d:
                best, best_d = e, d
        ci = -1                                    # the cut to start from (an index into best's cuts)
        if best is not None:
            ci = bisect.bisect_right(best.cut_pos, best_d - self.lmax) - 1
            while ci > 0:
                c = best.cut_pos[ci]
                if (_spans_before(best.spans, c) == _spans_before(spans, c) and not _crosses(spans, c)
                        and not _crosses(best.spans, c)):
                    break
                ci -= 1
        if ci > 0:
            c, k = best.cut_pos[ci], best.cut_ids[ci]
            tail, tpos, tids = self._pieces(text, spans, c)
            ids = best.ids[:k]
            ids.extend(tail)
            cut_pos = best.cut_pos[:ci + 1] + [c + p for p in tpos]
            cut_ids = best.cut_ids[:ci + 1] + [k + q for q in tids]
        else:
            k = 0
            ids, tpos, tids = self._pieces(text, spans, 0)
            cut_pos, cut_ids = [0] + tpos, [0] + tids
        out = ids.tolist()
        if self.mode == "verify":
            full = self.tok.encode(text, parse_special=True, plain=plain)
            if full != out:
                self.mismatches += 1
                at = next((j for j, (x, y) in enumerate(zip(full, out)) if x != y), min(len(full), len(out)))
                print(f"[strata] prompt memo: the ids differ from a full encode at token {at} of {len(full)} "
                      f"(reused {k}); the full encode is used", flush=True)
                return full                        # nothing remembered from a prompt that went wrong
        entry = _Entry(text, spans, ids, cut_pos, cut_ids)
        # the prompt it continues is superseded by it; one it only starts like (another conversation, a side request, a
        # compacted history) stays remembered for that conversation's next turn
        superseded = best is not None and 2 * best_d >= len(best.text)
        with self.lock:
            self.reused = k
            rest = [e for e in self.entries if not (superseded and e is best)]
            self.entries = [entry] + rest[: self.keep - 1]
        return out


class LiteralMemo:
    """serve.frontend.mark_think_literals' per-string work (does a string hold a literal; the string with them marked),
    remembered for the strings of the last few requests: an agent's history is the same strings every turn, and testing
    each against ~1,280 control-token texts took 0.8 s a request at ~880K tokens.  Pure functions of (the tag set, the
    string), so the results are the same.  Bound to one tag set (`tags`, and its assistant variant); `gens`: the requests a
    string stays remembered unused."""

    def __init__(self, tags: dict, gens: int = 4):
        self.tags = tags
        self.gens = max(1, gens)
        self.lock = threading.Lock()
        self.gen = 0
        self.has: dict = {}                 # (variant, s) -> [bool, gen]
        self.marked: dict = {}              # (variant, s) -> [str, gen]

    def begin(self) -> None:
        """A request starts: entries unused for `gens` requests are dropped."""
        with self.lock:
            self.gen += 1
            old = self.gen - self.gens
            for d in (self.has, self.marked):
                for key in [key for key, v in d.items() if v[1] < old]:
                    del d[key]

    def has_tag(self, s: str, tags: dict, variant: str) -> bool:
        key = (variant, s)
        with self.lock:
            v = self.has.get(key)
            if v is not None:
                v[1] = self.gen
                return v[0]
        got = any(tag in s for tag in tags)
        with self.lock:
            self.has[key] = [got, self.gen]
        return got

    def mark(self, s: str, tags: dict, variant: str) -> str:
        key = (variant, s)
        with self.lock:
            v = self.marked.get(key)
            if v is not None:
                v[1] = self.gen
                return v[0]
        text = s
        for tag, mark in tags.items():
            text = text.replace(tag, mark)
        with self.lock:
            self.marked[key] = [text, self.gen]
        return text
