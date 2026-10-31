"""异常类型。刻意少而具体：出错时必须能一眼看出属于哪条边界。"""

from __future__ import annotations


class PersonaError(Exception):
    """所有本系统异常的基类。"""


class ConfigError(PersonaError):
    """配置缺失/非法。启动期应当直接失败，而不是带着默认值跑下去（docs/02 D10）。"""


class StoreError(PersonaError):
    """持久层错误（SQLite / 事件日志 / 快照）。"""


class TransportError(PersonaError):
    """core ↔ body 传输错误。"""


class CompileError(PersonaError):
    """动作/话语编译失败。注意：编译失败是**正常结果**，应转成拒绝码而非崩溃。"""


class SafetyViolation(PersonaError):
    """安全门拒绝一次产出。同样应转成事件，而不是崩溃。"""
