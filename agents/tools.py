"""
Tool Registry: Manages agent tools with structured definitions.
"""

import logging
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class ToolDefinition:
    """Structured tool definition."""
    name: str
    description: str
    parameters: Dict[str, str] = field(default_factory=dict)
    func: Optional[Callable] = None


class ToolRegistry:
    """Registry for agent tools with structured definitions."""

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}

    def register(self, name: str, func: Callable, description: str = "",
                 parameters: Dict[str, str] = None):
        """Register a tool with optional structured definition."""
        tool_def = ToolDefinition(
            name=name,
            description=description or f"Tool: {name}",
            parameters=parameters or {},
            func=func,
        )
        self.tools[name] = tool_def
        logger.info(f"Registered tool: {name}")

    def get(self, name: str) -> Callable:
        """Get a tool function by name."""
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found")
        return self.tools[name].func

    def execute(self, name: str, input_str: str) -> str:
        """Execute a tool by name with string input."""
        if name not in self.tools:
            return f"Error: Tool '{name}' not found. Available: {self.list_tools()}"
        try:
            result = self.tools[name].func(input_str)
            return str(result) if result is not None else "Done"
        except Exception as e:
            logger.error(f"Tool '{name}' execution failed: {e}")
            return f"Error executing {name}: {str(e)}"

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())

    def get_tools_prompt(self) -> str:
        """Generate tool descriptions for system prompt."""
        lines = []
        for name, tool_def in self.tools.items():
            params_str = ""
            if tool_def.parameters:
                params_str = ", ".join(f"{k}: {v}" for k, v in tool_def.parameters.items())
                params_str = f" (参数: {params_str})"
            lines.append(f"- {name}: {tool_def.description}{params_str}")
        return "\n".join(lines)
