"""Shared exception classes for the application"""

class LLMQueryError(Exception):
    """Exception raised when LLM query fails"""
    pass

class MCPConnectionError(Exception):
    """Exception raised when MCP connection fails"""
    pass

class MCPToolExecutionError(Exception):
    """Exception raised when MCP tool execution fails"""
    pass
