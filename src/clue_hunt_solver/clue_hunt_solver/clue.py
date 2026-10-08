"""Clue parsing and token-chain validation (final_plan.md §2.1, §4.3).

Chain rules (FINAL):
  root:   token == sha1("START")[:4]                 -> accept, anchor_id := id
  steady: id == anchor_id + k AND token == sha1(prev)[:4]  -> accept
  else:   id mismatch -> DECOY;  id ok, token bad -> LOOK-ALIKE
  malformed -> INVALID (never published)
Token comparison is exact (no re-casing).
"""
import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

_CLUE_RE = re.compile(r'^HUNT:(\d+):([0-9A-F]{4}):(.*)$')
VERBS = ('GOTO', 'PILLAR', 'BETWEEN', 'REL')
COLOURS = ('RED', 'GREEN', 'BLUE')


def token_for(prev_text: str) -> str:
    """sha1(prev)[:4].upper() - the chain token generator."""
    return hashlib.sha1(prev_text.encode()).hexdigest()[:4].upper()


START_TOKEN = token_for('START')          # == "7196" (asserted in tests)


class Verdict(Enum):
    VALID = 'valid'                # root or steady-state accept
    DECOY = 'decoy'                # id not next in chain (or pre-root non-root)
    LOOKALIKE = 'lookalike'        # right id, wrong token
    INVALID = 'invalid'            # malformed text


@dataclass(frozen=True)
class Clue:
    raw: str
    id: int
    token: str
    treasure: bool
    verb: str
    args: tuple


def _floats(parts, n):
    if len(parts) != n:
        return None
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        return None


def parse_clue(text: str) -> Optional[Clue]:
    """Parse one clue string. Returns None if malformed (Verdict.INVALID)."""
    raw = text.strip()
    m = _CLUE_RE.match(raw)
    if not m:
        return None
    clue_id, token, body = int(m.group(1)), m.group(2), m.group(3)

    treasure = False
    if body.startswith('TREASURE '):
        treasure = True
        body = body[len('TREASURE '):].strip()
    parts = body.split()
    if not parts or parts[0] not in VERBS:
        return None
    verb, rest = parts[0], parts[1:]

    if verb == 'GOTO':
        nums = _floats(rest, 2)
        if nums is None:
            return None
        args = nums
    elif verb == 'PILLAR':
        if len(rest) != 1 or not rest[0].isalpha():
            return None
        args = (rest[0].upper(),)
    elif verb == 'BETWEEN':
        if len(rest) != 3 or not rest[0].isalpha() or not rest[1].isalpha():
            return None
        try:
            f = float(rest[2])
        except ValueError:
            return None
        args = (rest[0].upper(), rest[1].upper(), f)
    elif verb == 'REL':
        nums = _floats(rest, 2)
        if nums is None:
            return None
        args = nums
    else:                                # pragma: no cover - VERBS guarded above
        return None

    if treasure and verb != 'REL':
        return None
    return Clue(raw=raw, id=clue_id, token=token, treasure=treasure,
                verb=verb, args=args)


class ChainValidator:
    """Stateful chain validator; rejected clues never advance the state."""

    def __init__(self):
        self.anchor_id: Optional[int] = None
        self.accepted = 0
        self.prev_raw: Optional[str] = None

    def expected_id(self) -> Optional[int]:
        """Next chain id, or None before the root is accepted."""
        if self.accepted == 0 or self.anchor_id is None:
            return None
        return self.anchor_id + self.accepted

    def validate(self, text: str) -> Tuple[Verdict, Optional[Clue]]:
        clue = parse_clue(text)
        if clue is None:
            return Verdict.INVALID, None

        if self.accepted == 0:
            if clue.token == START_TOKEN:
                self.anchor_id = clue.id
                self.accepted = 1
                self.prev_raw = clue.raw
                return Verdict.VALID, clue
            return Verdict.DECOY, clue

        expected_id = self.anchor_id + self.accepted
        if clue.id != expected_id:
            return Verdict.DECOY, clue
        if clue.token != token_for(self.prev_raw):
            return Verdict.LOOKALIKE, clue
        self.accepted += 1
        self.prev_raw = clue.raw
        return Verdict.VALID, clue
