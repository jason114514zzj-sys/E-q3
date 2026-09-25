# -*- coding: utf-8 -*-
"""逐条校验 manifest.json：文件是否存在、字节数与 sha256 是否一致。

用法（在 q3/ 目录下执行）：
    python verify_manifest.py

退出码：0 = 全部一致；1 = 存在缺失或不符（或找不到 manifest.json）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    mf = ROOT / "manifest.json"
    if not mf.is_file():
        print(f"找不到 {mf}；请先执行 python package.py")
        return 1

    data = json.loads(mf.read_text(encoding="utf-8"))
    missing: list[str] = []
    mismatch: list[str] = []
    ok = 0

    for rel, meta in sorted(data.items()):
        p = ROOT / rel
        if not p.is_file():
            missing.append(rel)
            continue
        if p.stat().st_size != meta["bytes"] or sha256(p) != meta["sha256"]:
            mismatch.append(rel)
            continue
        ok += 1

    print(f"manifest 条目   : {len(data)}")
    print(f"一致           : {ok}")
    print(f"缺失           : {len(missing)}")
    print(f"字节/哈希不符   : {len(mismatch)}")
    for r in missing:
        print(f"  [缺失] {r}")
    for r in mismatch:
        print(f"  [不符] {r}")
    return 0 if not (missing or mismatch) else 1


if __name__ == "__main__":
    raise SystemExit(main())
