"""SoulGrowth：一个人的成长内核。

分层（见 docs/02-系统架构）：L0 基质 / L1 身体 / L2 kernel / L3 世界 / L4 记忆 / L5 观察。
三条边界：`store` 是唯一写库处，`llm` 是唯一出网处，`safety` 是唯一让产出落地处。
"""

__version__ = "0.1.0"
