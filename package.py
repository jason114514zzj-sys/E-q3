"""把第三问的交付内容打成一个 zip，并做体积守门。

用法：
    python package.py            # 在 q3/ 目录下执行

产出：
    submission_q3.zip            含 core_code / results / models / docs / figures / 说明文档
    manifest.json                每个收录文件的 bytes 与 sha256（可逐文件核验）

设计取舍：
  * **不收录** `__pycache__`、`*.pyc`、`results/att4_time_map.json` 的重复副本等冗余内容；
  * 附件4 的 20 条原始 mp4 **不收录**（体积大且属赛题原始数据，读者自行放置）；
  * 附件2 的 `aligned_50.pkl` **不收录**（约 993 MB，属赛题原始数据）；
  * 收录体积上限断言 50 MB（竞赛对【整题全部附件】的限制，本包只是其中一部分）。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 50_000_000

# 收录规则：显式列出目录，避免把 __pycache__ / 中间产物卷进来
INCLUDE_DIRS = ["src", "models", "results", "figures", "docs"]
INCLUDE_FILES = ["README.md", "method.md", "config.yaml", "protocol.json",
                 "requirements.txt", "runtime_versions.json", "package.py",
                 "DELIVERABLES.md", "SERVER_ENV.md", "verify_manifest.py"]

SKIP_SUFFIX = {".pyc", ".pyo", ".log", ".tmp"}
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git"}

# ---------------- 段057 身份信息守门 ----------------
# 提交材料中严禁出现参赛单位、队员姓名、队伍编号等身份信息（违者按竞赛违规处理）。
# 分两级，避免自伤：
#   BLOCK —— 已知身份串，命中即中止打包（返回码 2）；
#   HINT  —— 结构性通用模式，仅提示人工确认。交付文档会**合法地引用禁令原文**
#            （例如方案校验报告写「严禁出现参赛单位…」），一律阻断会造成假阳性。
#
# ⚠️ 已知身份串**不写在本文件里**，而是从仓库根的 identity_blocklist.txt 读取。
#    原因：本文件自身也是提交材料 —— 若把身份串写在源码中，守门自己就成了泄漏源。
#    identity_blocklist.txt 既不在 INCLUDE_FILES 也不在 INCLUDE_DIRS 中，**永不入包**。
BLOCKLIST_FILE = "identity_blocklist.txt"
IDENTITY_HINT = [
    ("邮箱", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("IPv4 地址", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("家目录绝对路径", re.compile(r"/home/[A-Za-z0-9._-]+/")),
    ("账号@主机", re.compile(r"\b[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\.(?:com|cn|net|org|edu|io)\b")),
    ("身份字段标签", re.compile(r"队伍编号|参赛队号|参赛单位|队员姓名|学号")),
]


def load_blocklist() -> list[str]:
    """读取仓库根的 identity_blocklist.txt（每行一条，# 开头为注释）。"""
    p = ROOT / BLOCKLIST_FILE
    if not p.is_file():
        print(f"[身份信息·警告] 找不到 {BLOCKLIST_FILE}，已知身份串守门降级为「仅提示」。")
        return []
    tokens = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            tokens.append(s)
    return tokens
IDENTITY_TEXT_SUFFIX = {".md", ".py", ".json", ".yaml", ".yml", ".txt", ".csv", ".sh"}


def _read_text(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def assert_no_identity(files: list[Path]) -> None:
    """打包前的硬守门：命中已知身份串即中止（返回码 2）。"""
    blocklist = load_blocklist()
    blocked: list[str] = []
    hints: list[str] = []
    for p in files:
        if p.suffix.lower() not in IDENTITY_TEXT_SUFFIX:
            continue
        text = _read_text(p)
        if text is None:
            continue
        rel = p.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            for token in blocklist:
                if token.lower() in low:
                    blocked.append(f"{rel}:{lineno}  命中「{token}」  {line.strip()[:110]}")
            for label, pat in IDENTITY_HINT:
                m = pat.search(line)
                if m:
                    hints.append(f"{rel}:{lineno}  {label}「{m.group(0)[:40]}」")
    if hints:
        print(f"[身份信息·提示] 通用模式命中 {len(hints)} 处，需人工确认是否为合法引用：")
        for h in hints[:15]:
            print("    " + h)
    if blocked:
        print(f"\n[身份信息·阻断] 命中 {len(blocked)} 处已知身份串，拒绝打包：")
        for b in blocked:
            print("    " + b)
        raise SystemExit(2)
    print(f"[身份信息·通过] 已知身份串 {len(blocklist)} 条规则，0 处命中，允许打包。")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _has_any(names: set[str], parts: tuple[str, ...]) -> bool:
    return any(n in parts for n in names)


def collect() -> list[Path]:
    files: list[Path] = []
    for name in INCLUDE_FILES:
        p = ROOT / name
        if p.is_file():
            files.append(p)
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if _has_any(SKIP_DIRS, p.parts) or p.suffix in SKIP_SUFFIX:
                continue
            files.append(p)
    return files


def main() -> int:
    files = collect()
    assert_no_identity(files)
    # 路径一律用正斜杠：manifest.json 要能在 Linux/macOS 上照着重放，
    # 用 os.sep 会写出 "docs\qa\x.md" 这种在 POSIX 下无法定位的键。
    manifest = {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size,
                                                 "sha256": sha256(p)}
                for p in files}

    # manifest.json 自身也写盘（但不自我引用）
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    target = ROOT / "submission_q3.zip"
    with ZipFile(target, "w", ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(files) + [ROOT / "manifest.json"]:
            z.write(p, p.relative_to(ROOT).as_posix())
    with ZipFile(target) as z:
        assert z.testzip() is None, "zip 校验失败"

    size = target.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(f"第三问包过大: {size} 字节（>{MAX_BYTES}）")

    total_raw = sum(v["bytes"] for v in manifest.values())
    print(json.dumps({
        "package": target.name,
        "bytes": size,
        "decimal_MB": round(size / 1e6, 2),
        "files": len(manifest),
        "uncompressed_bytes": total_raw,
        "remaining_whole_submission_budget_bytes": MAX_BYTES - size,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
