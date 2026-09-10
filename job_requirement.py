"""岗位需求的结构化模型。

把一份 JD（岗位描述）固化成 Pydantic 对象，后续所有 Prompt、打分权重、
硬性门槛都从这里派生，避免把岗位信息散落在各处字符串里。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class EducationLevel(str, Enum):
    """学历层级。值本身即人类可读文本，rank 用于比较高低。"""

    HIGH_SCHOOL = "高中及以下"
    COLLEGE = "大专"
    BACHELOR = "本科"
    MASTER = "硕士"
    DOCTOR = "博士"

    @property
    def rank(self) -> int:
        return _EDUCATION_RANK[self]


_EDUCATION_RANK = {
    EducationLevel.HIGH_SCHOOL: 1,
    EducationLevel.COLLEGE: 2,
    EducationLevel.BACHELOR: 3,
    EducationLevel.MASTER: 4,
    EducationLevel.DOCTOR: 5,
}

#: 四维打分的标准 key，与 Prompt 输出、权重配置保持一一对应
DIMENSION_KEYS = ("skills", "education", "experience", "project")

DIMENSION_LABELS = {
    "skills": "技能",
    "education": "学历",
    "experience": "经验",
    "project": "项目",
}


class DimensionWeights(BaseModel):
    """四个维度的加权系数，自动归一化到 1.0。"""

    skills: float = Field(0.4, ge=0)
    education: float = Field(0.2, ge=0)
    experience: float = Field(0.2, ge=0)
    project: float = Field(0.2, ge=0)

    @model_validator(mode="after")
    def _check_sum(self) -> "DimensionWeights":
        if sum(self.as_dict().values()) <= 0:
            raise ValueError("四个维度的权重不能全为 0")
        return self

    def as_dict(self) -> dict[str, float]:
        return {key: getattr(self, key) for key in DIMENSION_KEYS}

    def normalized(self) -> dict[str, float]:
        """返回和为 1.0 的权重，允许调用方随便填相对比例。"""
        raw = self.as_dict()
        total = sum(raw.values())
        return {key: value / total for key, value in raw.items()}


class JobRequirement(BaseModel):
    """一份结构化的岗位需求。"""

    title: str = Field(..., description="岗位名称，如「管培生」")
    company: str = Field("", description="公司名称，仅用于报告展示")
    education: EducationLevel = Field(
        EducationLevel.BACHELOR, description="最低学历门槛"
    )
    min_years: float = Field(0.0, ge=0, description="最低工作年限，0 表示不限")
    skills: list[str] = Field(
        default_factory=list, description="必备技能/关键词，参与技能维度打分"
    )
    bonus_skills: list[str] = Field(
        default_factory=list, description="加分技能，命中可额外提分"
    )
    responsibilities: list[str] = Field(
        default_factory=list, description="岗位职责，供 LLM 理解岗位语境"
    )
    traits: list[str] = Field(
        default_factory=list, description="软性素质要求，如「接受基层轮岗」"
    )
    weights: DimensionWeights = Field(default_factory=DimensionWeights)
    description: str = Field("", description="补充说明，会原样进入 Prompt")

    @field_validator("skills", "bonus_skills", "traits", mode="before")
    @classmethod
    def _coerce_str_list(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.replace("，", ",").split(",") if item.strip()]
        return [str(item).strip() for item in value if str(item).strip()]

    def to_prompt(self) -> str:
        """渲染成注入 Prompt 的岗位描述文本。"""
        lines = [f"【岗位名称】{self.title}"]
        if self.company:
            lines.insert(0, f"【公司】{self.company}")
        lines.append(f"【最低学历要求】{self.education.value}")
        lines.append(
            f"【最低工作年限】{'不限' if self.min_years <= 0 else f'{self.min_years:g} 年'}"
        )
        if self.skills:
            lines.append(f"【必备技能/关键词】{'、'.join(self.skills)}")
        if self.bonus_skills:
            lines.append(f"【加分技能】{'、'.join(self.bonus_skills)}")
        if self.responsibilities:
            lines.append("【岗位职责】")
            lines.extend(f"  - {item}" for item in self.responsibilities)
        if self.traits:
            lines.append(f"【软性素质要求】{'、'.join(self.traits)}")
        if self.description:
            lines.append(f"【补充说明】{self.description}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 预设岗位：直接 `python main.py` 即可用这份 JD 跑通全流程
# --------------------------------------------------------------------------- #

#: 顺丰管培生示例岗位
SAMPLE_JOB = JobRequirement(
    company="顺丰速运",
    title="管理培训生",
    education=EducationLevel.BACHELOR,
    min_years=0,
    skills=["数据分析", "Excel", "物流", "供应链", "沟通协调", "项目管理"],
    bonus_skills=["SQL", "Python", "英语六级", "中共党员", "学生干部"],
    responsibilities=[
        "参与集团核心业务轮岗，覆盖营运、市场、职能等多条线",
        "承接一线业务改善项目，输出可落地的流程优化方案",
        "协助区域负责人完成经营分析与团队管理",
    ],
    traits=["接受基层轮岗", "抗压能力强", "长期在物流行业发展"],
    weights=DimensionWeights(skills=0.4, education=0.2, experience=0.2, project=0.2),
    description="面向应届及 2 年内工作经验候选人，重视学习能力与一线下沉意愿。",
)
