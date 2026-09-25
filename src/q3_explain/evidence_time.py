"""Give attachment-4 evidence positions a defensible time in seconds.

Why the naive mapping is wrong
------------------------------
An earlier version converted a position to seconds as ``j * D / 50``, i.e. it
assumed the 50 slots are equal-length time bins.  Measured on the 20
attachment-4 samples that assumption fails:

    corr(valid audio positions, content token count) = 1.000
    mean  valid positions / content token count      = 1.000
    mean  valid positions / raw_text word count      = 1.200

The number of usable slots equals the *token* count exactly, and is 20% larger
than the word count.  So the 50-slot axis is a **word/token axis**, not a time
axis: slot j covers the span of the j-th word, and its duration varies with how
long that word is spoken.  Dividing the clip evenly misplaces evidence badly --
for sample 16 (11 tokens over 3.114 s) the naive map put token 9 at 0.56 s when
the token actually falls near 2.5 s, an error of more than 4x.

What this module does instead
-----------------------------
For each of the 20 clips:

1. read the real duration with ffprobe;
2. decode the audio track to a waveform (ffmpeg) and compute an RMS envelope;
3. align the words of ``raw_text`` to the audio with the problem-1 constrained
   monotonic aligner (``q1_alignment.monotonic_align``), which enforces full
   coverage, strict monotonicity and a plausible speaking rate;
4. store each word's ``[start, end]``.

A token slot is mapped to a word by **reconstructing the WordPiece sequence** of
``raw_text`` with the pinned ``bert-base-uncased`` vocabulary and checking it
against the token ids that ship with the data (``text_bert[0]``).  On attachment
4 that check passes for 20/20 clips position by position, and on 400 sampled
attachment-2 clips for 400/400, so position -> subword -> word is exact rather
than approximate.  The proportional estimate is kept only as a fallback for the
case where the vocabulary or the shipped ids are unavailable, and the JSON
records which of the two was used (``quality.token_map_verified``).

The *time* attached to a word comes from forced alignment, not from an even
split; both facts are reported per sample.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 让 q1_alignment 可被导入：优先用环境变量 MOSEI_SRC，否则取本文件所在 src 目录。
_SRC = os.environ.get("MOSEI_SRC") or str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from q1_alignment.monotonic_align import align_words_monotonic  # noqa: E402

from .wordpiece import map_words_to_pieces  # noqa: E402  (used via _token_map_for)

VOCAB_NAME = "bert-base-uncased_vocab.txt"


def _tool(name: str) -> str:
    """定位 ffmpeg / ffprobe。

    顺序：环境变量 → PATH → conda 前缀 → **当前解释器所在目录** → 报错。
    后两项很关键：本项目用 `envs/mosei/bin/python` 直接运行而不走 `conda activate`，
    此时 PATH 与 CONDA_PREFIX 都可能没有该二进制，但解释器自己的 bin 一定有。
    """
    env = os.environ.get(name.upper())
    if env and Path(env).exists():
        return env
    found = shutil.which(name)
    if found:
        return found
    candidates = []
    if os.environ.get("CONDA_PREFIX"):
        candidates.append(Path(os.environ["CONDA_PREFIX"]) / "bin" / name)
    if getattr(sys, "prefix", None):
        candidates.append(Path(sys.prefix) / "bin" / name)
    # 解释器所在目录本身就是 bin，不要再拼一层 "bin"
    candidates.append(Path(sys.executable).resolve().parent / name)
    for cand in candidates:
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        f"找不到 {name}：请把它加入 PATH，或设置环境变量 {name.upper()}")


FFPROBE = _tool("ffprobe")
FFMPEG = _tool("ffmpeg")
SAMPLE_RATE = 16000


@dataclass
class WordSpan:
    index: int
    word: str
    start: float
    end: float


@dataclass
class ClipTimeMap:
    sample_id: str
    duration: float
    words: list[WordSpan] = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    source: str = "forced_alignment"
    # Exact position -> word index, one entry per content position 1..n.
    # Built by reconstructing the WordPiece sequence of ``raw_text`` with the
    # pinned bert-base-uncased vocabulary and checking it against the token ids
    # that ship with the data (``text_bert[0]``).  Empty when that check could
    # not be made, in which case ``word_for_token`` falls back to a proportional
    # estimate and says so.
    token_word_index: list[int] = field(default_factory=list)
    pieces: list[str] = field(default_factory=list)
    piece_word: list[int] = field(default_factory=list)

    @property
    def exact_mapping(self) -> bool:
        return bool(self.quality.get("token_map_verified"))

    def word_for_token(self, token_index: int, n_tokens: int) -> int:
        """Subword slot -> word index.

        Exact when the position-by-position check against the shipped token ids
        passed; otherwise a proportional estimate (``(p-1)/n * n_words``), which
        is what this function did before the tokenizer was pinned.
        """
        n_words = len(self.words)
        if n_words == 0 or n_tokens <= 0:
            return 0
        if self.token_word_index:
            k = int(token_index) - 1
            if 0 <= k < len(self.token_word_index):
                return int(self.token_word_index[k])
        frac = (token_index - 1) / max(n_tokens, 1)
        idx = int(round(frac * n_words))
        return min(max(idx, 0), n_words - 1)

    def words_for_positions(self, positions) -> list[int]:
        """Word indices covered by a set of subword positions (sorted, unique)."""
        out: list[int] = []
        for p in positions:
            w = self.word_for_token(int(p), len(self.token_word_index))
            if w not in out:
                out.append(w)
        return sorted(out)

    def span(self, word_index: int) -> tuple[float, float]:
        if not self.words:
            return 0.0, 0.0
        w = self.words[min(max(word_index, 0), len(self.words) - 1)]
        return w.start, w.end

    def span_of_words(self, word_indices) -> tuple[float, float]:
        """Union of the spans of several words; (0,0) when there are none."""
        if not self.words or not word_indices:
            return 0.0, 0.0
        idx = [min(max(int(w), 0), len(self.words) - 1) for w in word_indices]
        return (min(self.words[i].start for i in idx),
                max(self.words[i].end for i in idx))

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "duration": round(self.duration, 4),
            "n_words": len(self.words),
            "words": [{"i": w.index, "w": w.word,
                       "start": round(w.start, 4), "end": round(w.end, 4)}
                      for w in self.words],
            "quality": self.quality,
            "source": self.source,
            "token_word_index": self.token_word_index,
            "pieces": self.pieces,
            "piece_word": self.piece_word,
        }


def _duration(mp4: Path) -> float:
    r = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(mp4)],
        capture_output=True, text=True)
    return float(r.stdout.strip())


def _rms_envelope(mp4: Path, hop: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Decode to mono PCM and return (rms, centres_seconds)."""
    r = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(mp4), "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"ffmpeg failed for {mp4}")
    wave = np.frombuffer(r.stdout, dtype=np.float32)
    if wave.size == 0:
        raise RuntimeError(f"empty audio for {mp4}")
    win = max(1, int(hop * SAMPLE_RATE))
    n = wave.size // win
    frames = wave[: n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    centres = (np.arange(n) + 0.5) * hop
    return rms, centres


def build_time_map(data_root: Path, out_path: Path,
                   verbose: bool = True) -> dict[str, ClipTimeMap]:
    """Align every attachment-4 clip's transcript to its audio.

    Also attaches the **exact** subword -> word mapping: the WordPiece sequence
    of ``raw_text`` is rebuilt with the pinned bert-base-uncased vocabulary and
    must reproduce ``text_bert[0]`` position by position.  That turns the old
    proportional estimate into a checked mapping.
    """
    root = Path(data_root) / "附件4-可解释专项视频样本与特征文件"
    base = None
    for c in root.rglob("*"):
        if c.is_dir() and (c / "对齐版本").exists():
            base = c
            break
    if base is None:
        raise FileNotFoundError(f"attachment 4 not found under {root}")

    import pickle

    vocab = None
    root = Path(data_root).parent
    cands = [os.environ.get("Q3_VOCAB"),
             Path(os.environ.get("MOSEI_ROOT", "")) / "work" / VOCAB_NAME,
             root / "work" / VOCAB_NAME,
             root / "vocab" / VOCAB_NAME,
             Path(__file__).resolve().parents[2] / "vocab" / VOCAB_NAME,
             Path(__file__).resolve().parents[2] / "work" / VOCAB_NAME]
    for cand in cands:
        if not cand:
            continue
        try:
            if Path(cand).exists():
                from .wordpiece import load_vocab
                vocab = load_vocab(cand)
                print(f"词表: {cand}（{len(vocab)} 个 token）")
                break
        except Exception as exc:      # pragma: no cover
            print(f"词表读取失败 {cand}: {exc}")
    if vocab is None:
        print("⚠ 未找到 bert-base-uncased 词表 → 子词→词映射退回比例近似")

    pkls = sorted((base / "对齐版本").glob("*.pkl"))
    vids = sorted((base / "对齐版本" / "videos").glob("*.mp4"))
    vid_by_stem = {v.stem: v for v in vids}

    maps: dict[str, ClipTimeMap] = {}
    if verbose:
        print(f"{'样本':<8}{'时长':>8}{'词数':>6}{'覆盖':>8}{'语速':>8}"
              f"{'首词':>9}{'末词起':>9}{'子词':>6}{'映射':>7}")

    for p in pkls:
        with p.open("rb") as fh:
            o = pickle.load(fh)
        sid = str(o.get("id", p.stem))
        words = str(o.get("raw_text", "")).split()
        mp4 = vid_by_stem.get(p.stem)
        if mp4 is None or not words:
            maps[sid] = ClipTimeMap(sid, 0.0, [],
                                    {"error": "missing video or transcript"})
            continue

        piece_word, pieces, tinfo = _token_map_for(
            words, o.get("text_bert"), vocab, sid)

        dur = _duration(mp4)
        rms, centres = _rms_envelope(mp4)
        timeline, quality = align_words_monotonic(
            words, dur, rms=rms, energy_centers=centres)

        # 词边界统一 round 到 4 位小数后再入对象。
        # 这是**复现性要求**，不是美化：`to_dict()` 写 JSON 时就是 round 4，
        # 若这里保留全精度，则「首次构建（内存全精度）」与「复用缓存（JSON 4 位）」
        # 两条路径会给出不同的 3 位小数证据秒（实测样本 11 曾出现 0.818 vs 0.817）。
        # 统一舍入后两条路径逐字节一致。
        spans = [WordSpan(i, str(t["word"]),
                          round(float(t["start"]), 4), round(float(t["end"]), 4))
                 for i, t in enumerate(timeline)]
        q = {"usable": bool(getattr(quality, "usable", True))}
        for attr in ("coverage", "rate", "words_per_second", "reason"):
            if hasattr(quality, attr):
                v = getattr(quality, attr)
                q[attr] = float(v) if isinstance(v, (int, float)) else v
        q.update(tinfo)

        cov = (spans[-1].end - spans[0].start) / dur if spans and dur else 0.0
        rate = len(spans) / dur if dur else 0.0
        maps[sid] = ClipTimeMap(sid, dur, spans, q,
                                token_word_index=piece_word,
                                pieces=pieces, piece_word=piece_word)
        if verbose:
            print(f"{sid:<8}{dur:>8.3f}{len(spans):>6}{cov:>8.3f}"
                  f"{rate:>8.2f}{spans[0].start:>9.3f}{spans[-1].start:>9.3f}"
                  f"{len(pieces):>6}{'精确' if tinfo.get('token_map_verified') else '近似':>7}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({k: v.to_dict() for k, v in maps.items()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        n_ok = sum(1 for v in maps.values() if v.quality.get("token_map_verified"))
        print(f"\n已写出 {out_path}")
        print(f"子词→词映射逐位核对通过：{n_ok}/{len(maps)} 条")
    return maps


def load_time_map(path: Path) -> dict[str, ClipTimeMap]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, ClipTimeMap] = {}
    for sid, d in raw.items():
        spans = [WordSpan(w["i"], w["w"], w["start"], w["end"])
                 for w in d.get("words", [])]
        out[sid] = ClipTimeMap(
            sid, float(d["duration"]), spans,
            d.get("quality", {}), d.get("source", ""),
            token_word_index=list(d.get("token_word_index", [])),
            pieces=list(d.get("pieces", [])),
            piece_word=list(d.get("piece_word", [])),
        )
    return out


def _token_map_for(words: list[str], text_bert, vocab, sample_id: str):
    """Reconstruct the subword sequence and verify it against the shipped ids.

    Returns ``(piece_word, pieces, info)``.  ``piece_word[k]`` is the word index
    of the k-th subword, which is also the word index of content position k+1
    (position 0 is ``[CLS]``), so the same list serves both roles.  When the ids
    cannot be reproduced the lists come back empty and ``info`` records why, so
    the caller falls back to the proportional estimate instead of pretending.
    """
    info: dict = {"token_map_verified": False}
    if vocab is None or text_bert is None:
        info["token_map_reason"] = "no vocabulary or no text_bert available"
        return [], [], info
    from .wordpiece import map_words_to_pieces

    tb = np.asarray(text_bert).astype(np.int64)
    mask = tb[1] > 0
    n_content = int(mask.sum()) - 2
    if n_content <= 0:
        info["token_map_reason"] = "no content positions"
        return [], [], info
    tm = map_words_to_pieces(words, vocab, max_pieces=n_content)
    want = tb[0, 1:1 + n_content]
    got = np.asarray(tm.ids, dtype=np.int64)
    info.update({
        "n_content_positions": int(n_content),
        "n_pieces": int(tm.n_pieces),
        "n_words": len(words),
        "ids_match": int(got.size == want.size and bool(np.array_equal(got, want))),
        "truncated_words": tm.truncated_words,
    })
    if got.size == want.size and np.array_equal(got, want):
        info["token_map_verified"] = True
        info["token_map_source"] = ("bert-base-uncased WordPiece reconstructed and "
                                    "checked against text_bert[0] position by position")
    else:
        bad = np.flatnonzero(got != want)[:5].tolist() if got.size == want.size else []
        info["token_map_reason"] = (f"id mismatch at {len(bad)} positions "
                                    f"{bad}; mapping falls back to proportional")
    return list(tm.piece_word), list(tm.pieces), info


def uniform_span(position: int, duration: float, positions: int = 50,
                 width: int = 3) -> tuple[float, float]:
    """The rejected naive mapping, kept so the report can quantify its error."""
    step = duration / positions
    lo = max(0, position - width // 2)
    return lo * step, min(duration, (lo + width) * step)


if __name__ == "__main__":
    root = Path(os.environ.get("MOSEI_ROOT") or Path(__file__).resolve().parents[2])
    build_time_map(root / "data",
                   root / "work" / "q3" / "att4_time_map.json")
