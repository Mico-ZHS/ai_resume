"""简历解析：把 .docx / .pdf / .txt 统一转成纯文本 + 候选人姓名。

设计要点：
* 所有格式最终都归一化成一段干净文本，下游 Prompt 不关心来源格式；
* 编码探测按 UTF-8 → GB18030 → GBK → Big5 依次回退，覆盖国内简历常见的乱码场景；
* 解析失败不抛给调用方炸掉整批任务，而是返回带 warnings 的对象。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

#: 支持的扩展名。.doc（旧版二进制）不支持，会在 parse 时给出明确提示
SUPPORTED_SUFFIXES = {".docx", ".pdf", ".txt", ".md"}
TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk", "big5", "latin-1")

#: 姓名提取规则，按优先级依次尝试
_NAME_PATTERNS = (
    re.compile(r"姓\s*名\s*[:：]\s*([^\s,，、|/（(]{1,20})"),
    re.compile(r"^(?:姓名|名字)?\s*[:：]?\s*([一-龥]{2,4})\s*$", re.MULTILINE),
)

_WHITESPACE_RE = re.compile(r"[ \t　]+")
_BLANKLINE_RE = re.compile(r"\n{3,}")


class ResumeParseError(RuntimeError):
    """简历无法解析时抛出。"""


@dataclass
class Resume:
    """一份解析完成的简历。"""

    path: Path
    name: str
    text: str
    warnings: list[str] = field(default_factory=list)

    @property
    def source(self) -> str:
        return self.path.name

    @property
    def char_count(self) -> int:
        return len(self.text)

    def __str__(self) -> str:  # pragma: no cover - 仅用于调试
        return f"<Resume {self.name} ({self.char_count} 字) from {self.source}>"


def _normalize(text: str) -> str:
    """压缩空白、去掉多余空行，保持段落结构。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKLINE_RE.sub("\n\n", text).strip()


def _read_txt(path: Path) -> tuple[str, list[str]]:
    warnings: list[str] = []
    raw = path.read_bytes()
    for encoding in TEXT_ENCODINGS:
        try:
            return raw.decode(encoding), warnings
        except UnicodeDecodeError:
            continue
    warnings.append("所有编码尝试均失败，已用 UTF-8 忽略错误字符解码")
    return raw.decode("utf-8", errors="ignore"), warnings


def _read_docx(path: Path) -> tuple[str, list[str]]:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出可操作提示
        raise ResumeParseError(
            "解析 .docx 需要 python-docx，请执行：pip install python-docx"
        ) from exc

    document = Document(str(path))
    parts = [para.text for para in document.paragraphs]
    # 很多简历用表格排版，段落里是空的，必须把表格内容也抽出来
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            # 合并单元格会产生重复文本，去重后再拼
            deduped = list(dict.fromkeys(cell for cell in cells if cell))
            if deduped:
                parts.append(" | ".join(deduped))
    return "\n".join(parts), []


def _read_pdf(path: Path) -> tuple[str, list[str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ResumeParseError(
            "解析 .pdf 需要 pypdf，请执行：pip install pypdf"
        ) from exc

    warnings: list[str] = []
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n".join(pages)
    if len(text.strip()) < 20:
        warnings.append(
            "PDF 几乎提取不到文字，可能是扫描件/图片型 PDF，需要先做 OCR 才能参与筛选"
        )
    return text, warnings


_READERS = {
    ".txt": _read_txt,
    ".md": _read_txt,
    ".docx": _read_docx,
    ".pdf": _read_pdf,
}


def _guess_name(text: str, fallback: str) -> str:
    for pattern in _NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = match.group(1).strip()
            # 过滤掉「个人简历」这类被误命中的标题词
            if candidate and candidate not in {"个人简历", "求职简历", "简历"}:
                return candidate
    return fallback


def parse_resume(path: str | Path) -> Resume:
    """解析单份简历。

    Raises:
        FileNotFoundError: 文件不存在。
        ResumeParseError: 后缀不支持或内容解析失败。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"简历文件不存在：{path}")

    suffix = path.suffix.lower()
    if suffix == ".doc":
        raise ResumeParseError(
            f"{path.name}：不支持旧版 .doc 格式，请先另存为 .docx 或 PDF"
        )
    if suffix not in SUPPORTED_SUFFIXES:
        raise ResumeParseError(
            f"{path.name}：不支持的格式 {suffix}，仅支持 {'/'.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    text, warnings = _READERS[suffix](path)
    text = _normalize(text)

    if not text:
        raise ResumeParseError(f"{path.name}：解析后内容为空")

    return Resume(
        path=path,
        name=_guess_name(text, fallback=path.stem),
        text=text,
        warnings=warnings,
    )


def collect_resumes(directory: str | Path) -> list[Path]:
    """扫描目录下所有可解析的简历文件，按文件名排序（保证结果可复现）。"""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"简历目录不存在：{directory}")
    return sorted(
        (
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda p: p.name,
    )
