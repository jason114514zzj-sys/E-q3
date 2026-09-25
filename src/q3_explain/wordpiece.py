"""WordPiece tokenisation and an **exact** subword -> word mapping (no approximation).

Why this module exists
----------------------
``evidence_time.py`` used to map evidence positions to words by proportional
index, because no BERT tokenizer was installed.  That made the *time* genuine
but the *word* approximate.  The review (Q3-07) correctly rejected that.

The fix does not need ``transformers``: the vocabulary of
``bert-base-uncased`` is a plain text file (30 522 lines) which is pinned in
``work/bert-base-uncased_vocab.txt``, and the WordPiece algorithm is 20 lines.
The mapping is then **verified against the ids that ship with the data**:
attachment 4's ``text_bert[0]`` carries the token id of every position, so the
subword sequence we reconstruct must reproduce those ids exactly.  If it does,
the position -> subword -> word mapping is exact, not approximate.

Position axis, as measured on attachment 4 (not assumed)
-------------------------------------------------------
For every one of the 20 clips::

    mask_len(text_bert[1]) == n_valid(audio) + 2 == n_valid(vision) + 2

and the content ids are the WordPiece pieces of ``raw_text`` in order, so::

    position 0            = [CLS]
    position 1 .. n       = the n subwords of the transcript
    position n+1 .. 48    = padding
    position 49           = [SEP]   (audio/vision also carry a structural zero row)

The subword count n is *not* the word count: "requirements" is two subwords
(``requirement`` + ``##s``) and "other's" is three (``other`` + ``'`` + ``s``),
which is why n / n_words ~= 1.2 on this data.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

VOCAB_NAME = "bert-base-uncased_vocab.txt"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
PAD_TOKEN = "[PAD]"
MAX_CHARS_PER_WORD = 100


def load_vocab(path: str | Path) -> dict[str, int]:
    """One token per line, line number = id (that is the BERT format)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    vocab = {tok: i for i, tok in enumerate(lines)}
    if len(vocab) != len(lines):
        raise ValueError("词表里有重复 token")
    return vocab


def _is_control(ch: str) -> bool:
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch) in ("Cc", "Cf")


def _is_whitespace(ch: str) -> bool:
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def _is_punctuation(ch: str) -> bool:
    """BERT's rule: ASCII punctuation ranges, or any Unicode P* category."""
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _clean_text(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if cp == 0 or cp == 0xFFFD or _is_control(ch):
            continue
        out.append(" " if _is_whitespace(ch) else ch)
    return "".join(out)


def _split_cjk(ch: str) -> str:
    """Pad CJK characters with spaces so they become their own tokens."""
    out = []
    for ch in ch:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
                or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F
                or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
                or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F):
            out.extend([" ", ch, " "])
        else:
            out.append(ch)
    return "".join(out)


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def _split_on_punc(text: str) -> list[str]:
    chars = list(text)
    out: list[list[str]] = []
    start_new = True
    for ch in chars:
        if _is_punctuation(ch):
            out.append([ch])
            start_new = True
        else:
            if start_new:
                out.append([])
            start_new = False
            out[-1].append(ch)
    return ["".join(x) for x in out]


def wordpiece(word: str, vocab: dict[str, int]) -> list[str]:
    """Greedy longest-match-first WordPiece for one already-cleaned word."""
    if len(word) > MAX_CHARS_PER_WORD:
        return [UNK_TOKEN]
    pieces: list[str] = []
    start = 0
    while start < len(word):
        end = len(word)
        cur = None
        while start < end:
            sub = word[start:end]
            if start > 0:
                sub = "##" + sub
            if sub in vocab:
                cur = sub
                break
            end -= 1
        if cur is None:
            return [UNK_TOKEN]
        pieces.append(cur)
        start = end
    return pieces


@dataclass
class TokenWordMap:
    """Subword sequence of one transcript, with the source word of each piece."""

    words: list[str]
    pieces: list[str]           # subword strings, in order
    piece_word: list[int]       # index into ``words`` for every piece
    ids: list[int]              # token ids of the pieces (no CLS/SEP)
    truncated_words: list[int]  # word indices with no surviving piece

    @property
    def n_pieces(self) -> int:
        return len(self.pieces)

    def to_dict(self) -> dict:
        return {
            "n_words": len(self.words),
            "n_pieces": len(self.pieces),
            "truncated_words": self.truncated_words,
            "piece_word": self.piece_word,
            "pieces": self.pieces,
        }


def map_words_to_pieces(words: list[str], vocab: dict[str, int],
                        max_pieces: int | None = None) -> TokenWordMap:
    """Tokenise every whitespace word and remember which word each piece came from.

    ``max_pieces`` mimics BERT truncation: only the first ``max_pieces`` pieces
    survive, so words entirely beyond the cut have no position.
    """
    pieces: list[str] = []
    piece_word: list[int] = []
    for wi, word in enumerate(words):
        cleaned = _clean_text(word)
        cleaned = _split_cjk(cleaned)
        for tok in cleaned.split():
            tok = _strip_accents(tok.lower())
            for piece in _split_on_punc(tok):
                if not piece:
                    continue
                for sub in wordpiece(piece, vocab):
                    pieces.append(sub)
                    piece_word.append(wi)
    if max_pieces is not None and len(pieces) > max_pieces:
        pieces = pieces[:max_pieces]
        piece_word = piece_word[:max_pieces]
    ids = [vocab.get(p, vocab[UNK_TOKEN]) for p in pieces]
    kept = set(piece_word)
    truncated = [i for i in range(len(words)) if i not in kept]
    return TokenWordMap(words=list(words), pieces=pieces, piece_word=piece_word,
                        ids=ids, truncated_words=truncated)


def default_vocab_path(project_root: Path) -> Path:
    return Path(project_root) / "work" / VOCAB_NAME
