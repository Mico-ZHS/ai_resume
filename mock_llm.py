"""Mock 大模型：用确定性规则引擎模拟 LLM 的输出。

存在的意义是让整条 LangChain 链路（Prompt → LLM → JsonOutputParser）
在**没有 API Key、没有网络**的情况下也能原样跑通。

它实现的是 `Runnable` 接口，因此在 `screening_engine` 里可以直接替换
`ChatOpenAI` 而不改动任何编排代码 —— 这正是 LangChain 抽象的价值所在。

打分口径（四维，0-100）：
    技能 = 必备技能命中率 * 100 + 加分技能奖励（上限 +10）
    学历 = 学历基准分，低于门槛打 6 折，高于门槛 +10
    经验 = 按年限折算，达标即高分
    项目 = 按项目条目数量给分
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from job_requirement import EducationLevel, JobRequirement

#: 学历基准分
_EDUCATION_BASE = {
    EducationLevel.DOCTOR: 100.0,
    EducationLevel.MASTER: 85.0,
    EducationLevel.BACHELOR: 70.0,
    EducationLevel.COLLEGE: 50.0,
    EducationLevel.HIGH_SCHOOL: 30.0,
}

#: 从 Prompt 里把简历正文抠出来（避免把岗位描述也算进关键词命中）
_RESUME_BLOCK_RE = re.compile(r"<resume>(.*?)</resume>", re.DOTALL)
_SCORING_BLOCK_RE = re.compile(r"<scoring>(.*?)</scoring>", re.DOTALL)

#: 显式年限，如「2 年工作经验」「3年以上相关经验」
_EXPLICIT_YEARS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*年(?:以上|多)?\s*(?:工作|相关|从业|行业)?\s*(?:经验|经历)"
)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
#: 只有带这些字样的行才被当作「工作经历」，防止把教育起止年份算成工龄
_WORK_CONTEXT_RE = re.compile(
    r"公司|集团|任职|在职|工作|实习|有限|事业部|银行|科技|事务所"
)

#: 项目小节标题 & 下一条大标题的判定
_PROJECT_HEADER_RE = re.compile(r"^\s*(?:[一二三四五六七八九十\d]+[、.．]\s*)?(项目经历|项目经验|项目实践|主要项目|科研项目)")
_SECTION_HEADER_RE = re.compile(
    r"^\s*(?:[一二三四五六七八九十\d]+[、.．]\s*)?(教育背景|教育经历|工作经历|实习经历|"
    r"校园经历|荣誉奖项|获奖情况|技能特长|专业技能|自我评价|个人信息)"
)
_BULLET_RE = re.compile(r"^\s*(?:[-•·*●○◆]|\d+[.、)]|[（(]\d+[）)])\s*")

#: 生成「亮点」时过滤掉的无信息片段
_TRAIT_STOPWORDS = {
    "能力", "力强", "良好", "较强", "优秀", "接受", "具备", "愿意",
    "意识", "精神", "素质", "经验", "压能",
}


def _extract(pattern: re.Pattern[str], text: str, default: str = "") -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else default


def _contains(text: str, keyword: str) -> bool:
    """大小写不敏感的子串匹配（对英文技能名友好）。"""
    return keyword.lower() in text.lower()


def _trait_hit(trait: str, text: str) -> bool:
    """判断软性素质是否在简历中体现。

    先精确匹配整句，再退化为 2-gram 关键词匹配，避免「接受基层轮岗」
    这种带动词前缀的表述因措辞差异被漏掉。
    """
    if _contains(text, trait):
        return True
    core = re.sub(r"^(?:接受|具备|有|良好的?|较强的?|优秀的?|愿意)\s*", "", trait)
    for segment in re.findall(r"[一-龥]{2,}", core):
        if len(segment) <= 3:
            grams = {segment}
        else:
            grams = {segment[i : i + 2] for i in range(len(segment) - 1)}
        if any(gram not in _TRAIT_STOPWORDS and _contains(text, gram) for gram in grams):
            return True
    return False


# --------------------------------------------------------------------------- #
# 各维度打分
# --------------------------------------------------------------------------- #


def _score_skills(job: JobRequirement, text: str) -> tuple[float, str, list[str], list[str]]:
    required = job.skills
    matched = [skill for skill in required if _contains(text, skill)]
    missed = [skill for skill in required if skill not in matched]

    base = 100.0 * len(matched) / len(required) if required else 70.0

    bonus_hits = [skill for skill in job.bonus_skills if _contains(text, skill)]
    base = min(100.0, base + min(10.0, 2.5 * len(bonus_hits)))

    reason = (
        f"必备技能命中 {len(matched)}/{len(required)}（{'、'.join(matched) or '无'}）"
        + (f"；加分项 {'、'.join(bonus_hits)}" if bonus_hits else "")
    )
    return round(base, 2), reason, matched, missed


def _score_education(job: JobRequirement, text: str) -> tuple[float, str, EducationLevel | None, bool]:
    found: EducationLevel | None = None
    for level in (EducationLevel.DOCTOR, EducationLevel.MASTER, EducationLevel.BACHELOR, EducationLevel.COLLEGE):
        if level.value in text or (level is EducationLevel.BACHELOR and "学士" in text):
            found = level
            break
    if found is None:
        return 45.0, "简历中未识别到明确的学历信息", None, False

    score = _EDUCATION_BASE[found]
    meets = found.rank >= job.education.rank
    if not meets:
        score *= 0.6
        reason = f"{found.value}，低于岗位门槛（{job.education.value}）"
    elif found.rank > job.education.rank:
        score = min(100.0, score + 10)
        reason = f"{found.value}，高于岗位门槛（{job.education.value}）"
    else:
        reason = f"{found.value}，满足岗位门槛"
    return round(score, 2), reason, found, meets


def _score_experience(job: JobRequirement, text: str) -> tuple[float, str, float]:
    explicit = [float(m) for m in _EXPLICIT_YEARS_RE.findall(text)]
    years = max(explicit) if explicit else 0.0

    if not explicit:
        # 退化方案：只在「工作经历」上下文中用年份跨度粗估，避免把大学起止年份当工龄
        years_found = {
            int(year)
            for line in text.split("\n")
            if _WORK_CONTEXT_RE.search(line)
            for year in _YEAR_RE.findall(line)
        }
        if len(years_found) >= 2:
            years = float(min(15, max(years_found) - min(years_found)))

    if job.min_years <= 0:
        score = min(100.0, 60.0 + min(years, 5.0) * 8.0)
        note = f"约 {years:g} 年经验，岗位不限年限"
    else:
        ratio = years / job.min_years
        score = min(100.0, 40.0 + 60.0 * ratio)
        note = f"约 {years:g} 年经验，岗位要求 {job.min_years:g} 年"
    return round(score, 2), note, years


def _count_projects(text: str) -> tuple[int, list[str]]:
    lines = text.split("\n")
    header_idx = next((i for i, line in enumerate(lines) if _PROJECT_HEADER_RE.match(line)), None)

    if header_idx is None:
        # 没有独立小节时，退化为统计「项目」关键词出现次数
        return min(4, text.count("项目")), []

    titles: list[str] = []
    for line in lines[header_idx + 1 : header_idx + 60]:
        if _SECTION_HEADER_RE.match(line):
            break
        if not line.strip():
            continue
        if _BULLET_RE.match(line):
            titles.append(_BULLET_RE.sub("", line).strip()[:40])
    return len(titles), titles


def _score_project(job: JobRequirement, text: str) -> tuple[float, str, int]:
    count, titles = _count_projects(text)
    if count == 0:
        return 40.0, "简历中未见项目经历", 0
    score = min(100.0, 50.0 + 15.0 * count)
    reason = f"识别到 {count} 段项目经历" + (f"：{'；'.join(titles[:3])}" if titles else "")
    return round(score, 2), reason, count


# --------------------------------------------------------------------------- #
# 两份「模型输出」
# --------------------------------------------------------------------------- #


def score_resume(job: JobRequirement, text: str) -> dict[str, Any]:
    """模拟打分链的输出（四维量化打分）。"""
    skills, skills_reason, _matched, missed = _score_skills(job, text)
    education, edu_reason, edu_level, edu_ok = _score_education(job, text)
    experience, exp_reason, years = _score_experience(job, text)
    project, proj_reason, proj_count = _score_project(job, text)

    highlights: list[str] = []
    for skill in job.bonus_skills:
        if _contains(text, skill):
            highlights.append(f"具备加分项：{skill}")
    for trait in job.traits:
        if _trait_hit(trait, text):
            highlights.append(f"简历体现「{trait}」")

    concerns: list[str] = []
    if not edu_ok:
        concerns.append(edu_reason)
    if years < job.min_years:
        concerns.append(exp_reason)
    if missed and len(missed) > len(job.skills) / 2:
        concerns.append(f"必备技能覆盖不足，缺少：{'、'.join(missed)}")
    if proj_count == 0:
        concerns.append("未见项目经历，难以评估实操能力")

    name_match = re.search(r"姓\s*名\s*[:：]\s*(\S+)", text)
    return {
        "candidate_name": name_match.group(1) if name_match else "",
        "dimensions": {
            "skills": {"score": skills, "reason": skills_reason},
            "education": {"score": education, "reason": edu_reason},
            "experience": {"score": experience, "reason": exp_reason},
            "project": {"score": project, "reason": proj_reason},
        },
        "highlights": highlights,
        "concerns": concerns,
    }


def evaluate_resume(job: JobRequirement, text: str, scoring: dict[str, Any]) -> dict[str, Any]:
    """模拟评估链的输出（基于初筛分数的二次确认）。"""
    weights = job.weights.normalized()
    dimensions = scoring.get("dimensions", {})
    total = sum(dimensions.get(key, {}).get("score", 0.0) * weight for key, weight in weights.items())

    edu_ok = job.education.value in text or any(
        level.value in text for level in EducationLevel if level.rank >= job.education.rank
    )
    years = _score_experience(job, text)[2]
    hard_fail = []
    if not edu_ok:
        hard_fail.append(f"学历未达 {job.education.value} 门槛")
    if years < job.min_years:
        hard_fail.append(f"经验不足 {job.min_years:g} 年")

    if total >= 80:
        recommendation = "强烈推荐"
    elif total >= 65:
        recommendation = "推荐"
    elif total >= 50:
        recommendation = "待定"
    else:
        recommendation = "不推荐"

    # 硬性门槛不达标时，不允许给出「推荐」以上结论
    if hard_fail and recommendation in {"强烈推荐", "推荐"}:
        recommendation = "待定"

    if hard_fail:
        reason = f"加权总分 {total:.2f}；但存在硬性缺口：{'、'.join(hard_fail)}，建议降级处理。"
    elif recommendation == "待定":
        reason = f"加权总分 {total:.2f}，处于可进可退区间，建议安排一轮电话初访再定。"
    else:
        reason = f"加权总分 {total:.2f}，核心维度匹配良好，建议尽快推进面试。"

    return {
        "recommendation": recommendation,
        "reason": reason,
        "highlights": scoring.get("highlights", []),
        "concerns": scoring.get("concerns", []),
    }


# --------------------------------------------------------------------------- #
# Runnable 封装：让规则引擎拥有和 ChatOpenAI 一样的编排接口
# --------------------------------------------------------------------------- #


class MockChatModel(Runnable):
    """模拟对话模型。`task` 决定模拟打分链还是评估链。"""

    def __init__(self, job: JobRequirement, task: str) -> None:
        if task not in {"scoring", "evaluation"}:
            raise ValueError(f"未知的 mock 任务类型：{task}")
        self.job = job
        self.task = task

    @property
    def name(self) -> str:
        return f"MockChatModel[{self.task}]"

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:  # noqa: A002
        prompt_text = input.to_string() if hasattr(input, "to_string") else str(input)

        if self.task == "scoring":
            payload = score_resume(self.job, _extract(_RESUME_BLOCK_RE, prompt_text))
        else:
            raw_scoring = _extract(_SCORING_BLOCK_RE, prompt_text, "{}")
            try:
                scoring = json.loads(raw_scoring)
            except json.JSONDecodeError:
                scoring = {}
            payload = evaluate_resume(
                self.job, _extract(_RESUME_BLOCK_RE, prompt_text), scoring
            )

        return AIMessage(content=json.dumps(payload, ensure_ascii=False))
