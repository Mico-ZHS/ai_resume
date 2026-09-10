"""LangChain 筛选引擎：打分链 + 评估链。

编排形态（两条链结构同构，便于替换模型/解析器）：

    打分链  ChatPromptTemplate ─▶ LLM ─▶ JsonOutputParser ─▶ ScoreResult
    评估链  ChatPromptTemplate ─▶ LLM ─▶ JsonOutputParser ─▶ EvaluationResult
                    ▲                        ▲
                    └── 注入上一步的打分 JSON ─┘

`use_mock=True` 时 LLM 位置替换为 `MockChatModel`，链路完全一致，
因此从演示切到真机只需要改一个布尔值，不需要动任何编排代码。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from job_requirement import DIMENSION_KEYS, DIMENSION_LABELS, JobRequirement
from mock_llm import MockChatModel
from resume_parser import Resume

#: 录用建议枚举，按优先级排列（注意「推荐」是「强烈推荐」的子串，判断顺序不能反）
RECOMMENDATIONS = ("强烈推荐", "不推荐", "推荐", "待定")
DEFAULT_RECOMMENDATION = "待定"


# --------------------------------------------------------------------------- #
# 结构化输出模型
# --------------------------------------------------------------------------- #


class DimensionScore(BaseModel):
    """单个维度的得分与给分理由。"""

    score: float = Field(0.0, ge=0, le=100, description="该维度得分，0-100")
    reason: str = Field("", description="一句话说明为什么给这个分数")


class ScoreResult(BaseModel):
    """打分链的输出。"""

    # candidate_name 是重要字段，其余为可选
    candidate_name: str = Field("", description="候选人姓名")
    dimensions: dict[str, DimensionScore] = Field(
        default_factory=dict,
        description=f"四个维度的得分，key 必须是 {list(DIMENSION_KEYS)} 之一",
    )
    highlights: list[str] = Field(default_factory=list, description="简历亮点，每条不超过 20 字")
    concerns: list[str] = Field(default_factory=list, description="潜在风险或不足")


class EvaluationResult(BaseModel):
    """评估链的输出。"""

    recommendation: str = Field(
        DEFAULT_RECOMMENDATION,
        description=f"录用建议，必须是 {'/'.join(RECOMMENDATIONS)} 之一",
    )
    reason: str = Field("", description="给出该建议的理由，2-3 句")
    highlights: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

_SCORING_SYSTEM = """你是一位严谨的资深 HR 招聘专家，负责为岗位做初筛量化打分。
你的判断必须**只依据简历中真实出现的信息**，不得臆测、不得补全、不得因为候选人"看起来不错"就给高分。
缺少信息就如实扣分，并在 concerns 里写明缺什么。

打分维度固定为四项，每项 0-100 分：
- skills（技能）：必备技能的命中比例与深度、加分技能
- education（学历）：学历层次与岗位门槛的匹配度
- experience（经验）：工作/实习年限、行业相关度、职责含金量
- project（项目）：项目经历的数量、复杂度、与岗位的相关性

只输出 JSON，不要输出任何解释性文字或 Markdown 代码块。"""

_SCORING_HUMAN = """请为下面这位候选人打分。

【岗位需求】
{job_profile}

【候选人简历】
<resume>
{resume_text}
</resume>

{format_instructions}

额外要求：
1. dimensions 的 key 必须严格使用 "skills"、"education"、"experience"、"project"。
2. 每个维度都要在 reason 里写明给分依据，引用简历中的具体事实。
3. highlights 与 concerns 各输出 0-4 条，每条为独立短句。"""

_EVALUATION_SYSTEM = """你是一位用人部门负责人，正在对 HR 的初筛结果做二次确认。
你要跳出分数本身，综合判断这位候选人**是否值得进入面试流程**。

判断原则：
- 如果候选人触发了任何硬性门槛缺口（学历不达标、经验年限不足），最多只能给"待定"。
- 分数高但风险点致命（如明确的行业排斥、频繁跳槽），应当下调建议等级。
- 分数中等但亮点与岗位强相关，可以维持或上调到"推荐"。

只输出 JSON，不要输出任何解释性文字或 Markdown 代码块。"""

_EVALUATION_HUMAN = """请复核下面这位候选人的筛选结果并给出最终录用建议。

【岗位需求】
{job_profile}

【候选人简历】
<resume>
{resume_text}
</resume>

【HR 初筛打分结果】
<scoring>
{scoring_json}
</scoring>

{format_instructions}

额外要求：
1. recommendation 只能是"强烈推荐"、"推荐"、"待定"、"不推荐"四者之一。
2. reason 用 2-3 句话说明结论，必须提及具体分数或事实依据。"""


# --------------------------------------------------------------------------- #
# 结果聚合
# --------------------------------------------------------------------------- #


@dataclass
class ScreeningResult:
    """一位候选人的完整筛选结果。"""

    resume: Resume
    score: ScoreResult
    evaluation: EvaluationResult
    total: float
    rank: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.resume.name

    @property
    def recommendation(self) -> str:
        return self.evaluation.recommendation

    def dimension_items(self) -> list[tuple[str, float, str]]:
        """返回 [(维度中文名, 分数, 理由)]，按标准顺序排列。"""
        items = []
        for key in DIMENSION_KEYS:
            dim = self.score.dimensions.get(key)
            if dim is not None:
                items.append((DIMENSION_LABELS[key], dim.score, dim.reason))
        return items

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "name": self.name,
            "source": self.resume.source,
            "total": round(self.total, 2),
            "recommendation": self.recommendation,
            "dimensions": {
                label: {"score": round(score, 2), "reason": reason}
                for label, score, reason in self.dimension_items()
            },
            "highlights": self.score.highlights,
            "concerns": self.score.concerns,
            "evaluation_reason": self.evaluation.reason,
            "warnings": self.warnings,
        }


def compute_total(
    dimensions: dict[str, DimensionScore], weights: dict[str, float]
) -> float:
    """按归一化权重求加权总分。缺失的维度按 0 计。"""
    total = 0.0
    for key in DIMENSION_KEYS:
        dim = dimensions.get(key)
        if dim is not None:
            total += dim.score * weights.get(key, 0.0)
    return round(total, 2)


def normalize_recommendation(raw: str) -> str:
    """把模型可能输出的自由文本收敛到四个标准档位。"""
    raw = (raw or "").strip()
    for option in RECOMMENDATIONS:
        if option in raw:
            return option
    return DEFAULT_RECOMMENDATION


def _coerce_score_result(payload: dict[str, Any]) -> ScoreResult:
    """把解析出来的 dict 收敛成 ScoreResult，容忍模型的小幅跑偏。"""
    dimensions: dict[str, DimensionScore] = {}
    for key, value in (payload.get("dimensions") or {}).items():
        # 模型偶尔会返回中文 key，这里做一次映射兜底
        canonical = key if key in DIMENSION_KEYS else next(
            (k for k, label in DIMENSION_LABELS.items() if label == key), None
        )
        if canonical is None:
            continue
        if isinstance(value, (int, float)):
            dimensions[canonical] = DimensionScore(score=float(value))
        elif isinstance(value, dict):
            dimensions[canonical] = DimensionScore(
                score=float(value.get("score", 0) or 0),
                reason=str(value.get("reason", "")),
            )

    def _as_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        return [str(item) for item in (value or [])]

    return ScoreResult(
        candidate_name=str(payload.get("candidate_name", "")),
        dimensions=dimensions,
        highlights=_as_list(payload.get("highlights")),
        concerns=_as_list(payload.get("concerns")),
    )


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #


class ScreeningEngine:
    """把岗位需求 + LLM 封装成可复用的筛选引擎。"""

    def __init__(
        self,
        job: JobRequirement,
        *,
        use_mock: bool = True,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.job = job
        self.use_mock = use_mock
        self.weights = job.weights.normalized()

        scoring_parser = JsonOutputParser(pydantic_object=ScoreResult)
        evaluation_parser = JsonOutputParser(pydantic_object=EvaluationResult)

        self.scoring_chain = (
            ChatPromptTemplate.from_messages(
                [("system", _SCORING_SYSTEM), ("human", _SCORING_HUMAN)]
            ).partial(format_instructions=scoring_parser.get_format_instructions())
            | self._build_llm("scoring", model, api_key, base_url, temperature)
            | scoring_parser
        )

        self.evaluation_chain = (
            ChatPromptTemplate.from_messages(
                [("system", _EVALUATION_SYSTEM), ("human", _EVALUATION_HUMAN)]
            ).partial(format_instructions=evaluation_parser.get_format_instructions())
            | self._build_llm("evaluation", model, api_key, base_url, temperature)
            | evaluation_parser
        )

    # -- 模型构建 ---------------------------------------------------------- #

    def _build_llm(
        self,
        task: str,
        model: str,
        api_key: str | None,
        base_url: str | None,
        temperature: float,
    ):
        if self.use_mock:
            return MockChatModel(self.job, task)

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "接入真实模型需要 langchain-openai，请执行：\n"
                "    pip install langchain-openai\n"
                "或改用 mock 模式（不加 --real 参数）。"
            ) from exc

        kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        try:
            return ChatOpenAI(**kwargs)
        except Exception as exc:  # 缺 API Key 等配置问题
            raise RuntimeError(
                f"初始化模型 {model} 失败：{exc}\n"
                "请设置环境变量 OPENAI_API_KEY（或 --base-url 指向兼容 OpenAI 协议的服务）。"
            ) from exc

    # -- 单份筛选 ---------------------------------------------------------- #

    def _run_scoring(self, resume: Resume) -> tuple[ScoreResult, list[str]]:
        warnings: list[str] = []
        try:
            payload = self.scoring_chain.invoke({"job_profile": self.job.to_prompt(), "resume_text": resume.text})
            return _coerce_score_result(payload), warnings
        except Exception as exc:
            # 真实模型偶发返回非法 JSON 时，退化为规则引擎，保证批量任务不中断
            warnings.append(f"打分链失败，已回退到规则引擎：{type(exc).__name__}: {exc}")
            from mock_llm import score_resume

            return _coerce_score_result(score_resume(self.job, resume.text)), warnings

    def _run_evaluation(
        self, resume: Resume, score: ScoreResult
    ) -> tuple[EvaluationResult, list[str]]:
        warnings: list[str] = []
        scoring_json = json.dumps(score.model_dump(), ensure_ascii=False, indent=2)
        try:
            payload = self.evaluation_chain.invoke(
                {
                    "job_profile": self.job.to_prompt(),
                    "resume_text": resume.text,
                    "scoring_json": scoring_json,
                }
            )
            return (
                EvaluationResult(
                    recommendation=normalize_recommendation(payload.get("recommendation", "")),
                    reason=str(payload.get("reason", "")),
                    highlights=[str(x) for x in (payload.get("highlights") or [])],
                    concerns=[str(x) for x in (payload.get("concerns") or [])],
                ),
                warnings,
            )
        except Exception as exc:
            warnings.append(f"评估链失败，已回退到规则引擎：{type(exc).__name__}: {exc}")
            from mock_llm import evaluate_resume

            payload = evaluate_resume(self.job, resume.text, score.model_dump())
            return (
                EvaluationResult(
                    recommendation=normalize_recommendation(payload["recommendation"]),
                    reason=payload["reason"],
                    highlights=payload["highlights"],
                    concerns=payload["concerns"],
                ),
                warnings,
            )

    def screen(self, resume: Resume) -> ScreeningResult:
        """对单份简历跑完整条链路。"""
        score, w1 = self._run_scoring(resume)
        evaluation, w2 = self._run_evaluation(resume, score)
        total = compute_total(score.dimensions, self.weights)

        return ScreeningResult(
            resume=resume,
            score=score,
            evaluation=evaluation,
            total=total,
            warnings=resume.warnings + w1 + w2,
        )

    def screen_batch(self, resumes: list[Resume]) -> list[ScreeningResult]:
        """批量筛选并按总分降序排名（同分时按姓名稳定排序）。"""
        results = [self.screen(resume) for resume in resumes]
        results.sort(key=lambda r: (-r.total, r.name))
        for index, result in enumerate(results, start=1):
            result.rank = index
        return results
