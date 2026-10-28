"""
core/agent — CRM LangGraph Agent 模块。

对外只暴露 run_agent_stream()，其余内部实现不导出。
"""
from .crm_agent import run_agent_stream

__all__ = ["run_agent_stream"]
