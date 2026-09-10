"""智能简历筛选系统 —— 主入口。

用法：
    python main.py                          # Mock 模式，无需 API Key，直接跑示例简历
    python main.py --real                   # 接入真实大模型（需 OPENAI_API_KEY）
    python main.py --real --model gpt-4o --base-url https://api.deepseek.com/v1
    python main.py --job my_job.json        # 用自定义岗位定义替换内置示例岗位
    python main.py --resumes D:/我的简历 --out-dir ./reports

产物：
    控制台报告 + screening_report.md + screening_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from job_requirement import SAMPLE_JOB, JobRequirement
from resume_parser import ResumeParseError, collect_resumes, parse_resume
from screening_engine import ScreeningEngine, ScreeningResult

#: 项目根目录，默认简历目录与报告输出目录都相对它定位
ROOT = Path(__file__).resolve().parent
DEFAULT_RESUME_DIR = ROOT / "resumes"

SEPARATOR = "=" * 72
THIN_SEPARATOR = "-" * 72


def _force_utf8_stdout() -> None:
    """Windows 控制台默认可能是 GBK，直接 print 中文会 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.encoding and stream.encoding.lower() not in {"utf-8", "utf8"}:
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - 非常规流
            pass


# --------------------------------------------------------------------------- #
# 报告渲染
# --------------------------------------------------------------------------- #


def _join(items: list[str], fallback: str = "—") -> str:
    return "；".join(items) if items else fallback


def render_console(results: list[ScreeningResult], job: JobRequirement, use_mock: bool) -> str:
    """渲染控制台报告。"""
    engine_label = "Mock 规则引擎（无需 API Key）" if use_mock else "真实大模型"
    lines = [
        SEPARATOR,
        "  智能简历筛选报告",
        f"  岗位：{job.company + ' · ' if job.company else ''}{job.title}",
        f"  引擎：{engine_label}",
        f"  简历数：{len(results)}    生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        SEPARATOR,
        "",
    ]

    for result in results:
        lines.append(
            f"第{result.rank}名 | {result.name} | 总分 {result.total:.2f} | {result.recommendation}"
        )
        dimensions = "  ".join(
            f"{label}{score:.2f}" for label, score, _ in result.dimension_items()
        )
        lines.append(f"  {dimensions}")
        if result.score.highlights:
            lines.append(f"  亮点：{_join(result.score.highlights)}")
        if result.score.concerns:
            lines.append(f"  风险：{_join(result.score.concerns)}")
        if result.evaluation.reason:
            lines.append(f"  评语：{result.evaluation.reason}")
        for warning in result.warnings:
            lines.append(f"  ⚠ {warning}")
        lines.append("")

    if results:
        counts: dict[str, int] = {}
        for result in results:
            counts[result.recommendation] = counts.get(result.recommendation, 0) + 1
        summary = "    ".join(f"{key} {value} 人" for key, value in counts.items())
        lines.extend([THIN_SEPARATOR, f"  汇总：{summary}", SEPARATOR])

    return "\n".join(lines)


def render_markdown(results: list[ScreeningResult], job: JobRequirement, use_mock: bool) -> str:
    """渲染 Markdown 报告，方便直接贴进飞书/钉钉/邮件。"""
    engine_label = "Mock 规则引擎" if use_mock else "真实大模型"
    lines = [
        f"# 简历筛选报告 · {job.title}",
        "",
        f"- **公司**：{job.company or '—'}",
        f"- **引擎**：{engine_label}",
        f"- **候选人数**：{len(results)}",
        f"- **生成时间**：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## 排名总览",
        "",
        "| 排名 | 姓名 | 总分 | 技能 | 学历 | 经验 | 项目 | 建议 |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for result in results:
        cells = {label: f"{score:.2f}" for label, score, _ in result.dimension_items()}
        lines.append(
            f"| {result.rank} | {result.name} | **{result.total:.2f}** "
            f"| {cells.get('技能', '—')} | {cells.get('学历', '—')} "
            f"| {cells.get('经验', '—')} | {cells.get('项目', '—')} "
            f"| {result.recommendation} |"
        )

    lines.extend(["", "## 逐人详情", ""])
    for result in results:
        lines.extend(
            [
                f"### 第{result.rank}名 · {result.name}（{result.total:.2f} 分 · {result.recommendation}）",
                "",
                f"*来源文件：`{result.resume.source}`*",
                "",
                "| 维度 | 得分 | 依据 |",
                "| :--- | ---: | :--- |",
            ]
        )
        for label, score, reason in result.dimension_items():
            lines.append(f"| {label} | {score:.2f} | {reason or '—'} |")
        lines.extend(
            [
                "",
                f"- **亮点**：{_join(result.score.highlights)}",
                f"- **风险**：{_join(result.score.concerns)}",
                f"- **评估结论**：{result.evaluation.reason or '—'}",
            ]
        )
        if result.warnings:
            lines.append(f"- **解析提示**：{_join(result.warnings)}")
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai_resume",
        description="基于 LangChain 的智能简历筛选系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--resumes", type=Path, default=DEFAULT_RESUME_DIR,
                        help=f"简历目录，默认 {DEFAULT_RESUME_DIR}")
    parser.add_argument("--out-dir", type=Path, default=ROOT,
                        help="报告输出目录，默认项目根目录")
    parser.add_argument("--job", type=Path, default=None,
                        help="自定义岗位定义 JSON 文件；不传则用内置示例岗位")
    parser.add_argument("--real", action="store_true",
                        help="使用真实大模型（默认用 Mock 规则引擎）")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                        help="模型名，默认取环境变量 MODEL_NAME 或 gpt-4o-mini")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"),
                        help="兼容 OpenAI 协议的 API 地址")
    parser.add_argument("--top", type=int, default=0,
                        help="只打印前 N 名（0 表示全部）")
    return parser


def load_job(path: Path | None) -> JobRequirement:
    if path is None:
        return SAMPLE_JOB
    if not path.exists():
        raise SystemExit(f"岗位定义文件不存在：{path}")
    try:
        return JobRequirement.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"岗位定义文件解析失败：{exc}") from exc


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    args = build_parser().parse_args(argv)

    job = load_job(args.job)
    use_mock = not args.real

    # ---- 1. 采集并解析简历 -------------------------------------------- #
    try:
        files = collect_resumes(args.resumes)
    except NotADirectoryError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    if not files:
        print(f"错误：{args.resumes} 下没有找到可解析的简历（支持 .docx/.pdf/.txt）", file=sys.stderr)
        return 2

    resumes, failed = [], []
    print(f"正在解析 {len(files)} 份简历 ...")
    for path in files:
        try:
            resumes.append(parse_resume(path))
        except (ResumeParseError, Exception) as exc:  # noqa: BLE001 - 单份失败不影响整批
            failed.append((path.name, str(exc)))
            print(f"  ✗ {path.name}：{exc}")

    if not resumes:
        print("错误：所有简历都解析失败，无法继续。", file=sys.stderr)
        return 2

    # ---- 2. 跑筛选链路 ------------------------------------------------- #
    print(f"正在筛选（引擎：{'Mock 规则引擎' if use_mock else args.model}） ...\n")
    try:
        engine = ScreeningEngine(
            job,
            use_mock=use_mock,
            model=args.model,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=args.base_url,
        )
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 3

    results = engine.screen_batch(resumes)

    # ---- 3. 输出报告 --------------------------------------------------- #
    print(render_console(results, job, use_mock))

    if failed:
        print(f"\n以下 {len(failed)} 份简历解析失败，未参与排名：")
        for name, reason in failed:
            print(f"  - {name}：{reason}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.out_dir / "screening_report.md"
    json_path = args.out_dir / "screening_report.json"
    md_path.write_text(render_markdown(results, job, use_mock), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "job": {"company": job.company, "title": job.title},
                "engine": "mock" if use_mock else args.model,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "failed": [{"file": name, "reason": reason} for name, reason in failed],
                "results": [result.to_dict() for result in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n报告已保存：\n  {md_path}\n  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
