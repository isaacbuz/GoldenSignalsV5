"""
Base MCP Server
Provides foundation for Model Context Protocol servers
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import asyncio
from dataclasses import dataclass, field
from enum import Enum

from core.logging import get_logger

logger = get_logger(__name__)


class ToolType(str, Enum):
    """Types of MCP tools"""
    QUERY = "query"
    COMMAND = "command"
    ANALYSIS = "analysis"
    MONITOR = "monitor"


@dataclass
class MCPTool:
    """MCP Tool definition"""
    name: str
    description: str
    tool_type: ToolType
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    rate_limit: Optional[int] = None  # requests per minute
    requires_auth: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.tool_type.value,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "examples": self.examples,
            "rate_limit": self.rate_limit,
            "requires_auth": self.requires_auth
        }


@dataclass
class MCPResource:
    """MCP Resource definition"""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
            "metadata": self.metadata
        }


class BaseMCPServer(ABC):
    """
    Base class for MCP servers
    
    Features:
    - Tool registration and management
    - Resource management
    - Rate limiting
    - Caching
    - Error handling
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.is_running = False
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, float]] = {}  # tool -> {user: last_request_time}
        self.default_rate_limit = 60  # requests per minute
        
        # Caching
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes default
        
        logger.info(f"Initialized MCP server: {name} v{version}")
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the MCP server"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the MCP server"""
        pass
    
    def register_tool(self, tool: MCPTool) -> None:
        """Register a new tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def register_resource(self, resource: MCPResource) -> None:
        """Register a new resource"""
        self.resources[resource.uri] = resource
        logger.info(f"Registered resource: {resource.uri}")
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """List all available resources"""
        return [resource.to_dict() for resource in self.resources.values()]
    
    async def execute_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool with rate limiting and error handling
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            user_id: Optional user ID for rate limiting
            
        Returns:
            Tool execution result
        """
        # Check if tool exists
        if tool_name not in self.tools:
            return {
                "error": f"Tool '{tool_name}' not found",
                "available_tools": list(self.tools.keys())
            }
        
        tool = self.tools[tool_name]
        
        # Check rate limit
        if not await self._check_rate_limit(tool_name, user_id, tool.rate_limit):
            return {
                "error": "Rate limit exceeded",
                "retry_after": 60  # seconds
            }
        
        # Validate input
        if not self._validate_input(arguments, tool.input_schema):
            return {
                "error": "Invalid input",
                "expected_schema": tool.input_schema
            }
        
        # Check cache
        cache_key = self._get_cache_key(tool_name, arguments)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for {tool_name}")
            return cached_result
        
        try:
            # Execute tool-specific logic
            result = await self._execute_tool_logic(tool_name, arguments)
            
            # Cache result
            self._add_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}", exc_info=True)
            return {
                "error": f"Tool execution failed: {str(e)}",
                "tool": tool_name
            }
    
    @abstractmethod
    async def _execute_tool_logic(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the actual tool logic
        Must be implemented by subclasses
        """
        pass
    
    async def get_resource(self, uri: str) -> Optional[Dict[str, Any]]:
        """Get a resource by URI"""
        if uri not in self.resources:
            return None
        
        resource = self.resources[uri]
        
        try:
            # Get resource content (to be implemented by subclasses)
            content = await self._get_resource_content(uri)
            
            return {
                "uri": uri,
                "name": resource.name,
                "mimeType": resource.mime_type,
                "content": content,
                "metadata": resource.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get resource {uri}: {e}")
            return None
    
    @abstractmethod
    async def _get_resource_content(self, uri: str) -> Any:
        """Get the actual resource content"""
        pass
    
    async def _check_rate_limit(
        self, 
        tool_name: str, 
        user_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> bool:
        """Check if request is within rate limit"""
        if not limit:
            limit = self.default_rate_limit
        
        key = f"{tool_name}:{user_id or 'anonymous'}"
        now = datetime.utcnow().timestamp()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = {}
        
        # Clean old entries
        minute_ago = now - 60
        self.rate_limits[key] = {
            ts: ts for ts in self.rate_limits[key] 
            if ts > minute_ago
        }
        
        # Check limit
        if len(self.rate_limits[key]) >= limit:
            return False
        
        # Add current request
        self.rate_limits[key][now] = now
        return True
    
    def _validate_input(self, arguments: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate input against schema"""
        # Simple validation - in production use jsonschema
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        
        # Check required fields
        for field in required:
            if field not in arguments:
                return False
        
        # Check types (simplified)
        for field, value in arguments.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type:
                    if expected_type == "string" and not isinstance(value, str):
                        return False
                    elif expected_type == "integer" and not isinstance(value, int):
                        return False
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        return False
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        return False
                    elif expected_type == "array" and not isinstance(value, list):
                        return False
                    elif expected_type == "object" and not isinstance(value, dict):
                        return False
        
        return True
    
    def _get_cache_key(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Generate cache key"""
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            entry = self.cache[key]
            if datetime.utcnow().timestamp() - entry["timestamp"] < self.cache_ttl:
                return entry["value"]
            else:
                del self.cache[key]
        return None
    
    def _add_to_cache(self, key: str, value: Any) -> None:
        """Add value to cache"""
        self.cache[key] = {
            "value": value,
            "timestamp": datetime.utcnow().timestamp()
        }
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        return {
            "name": self.name,
            "version": self.version,
            "tools": len(self.tools),
            "resources": len(self.resources),
            "is_running": self.is_running,
            "capabilities": {
                "tools": True,
                "resources": True,
                "rate_limiting": True,
                "caching": True
            }
        }
    
    async def start(self) -> None:
        """Start the MCP server"""
        await self.initialize()
        self.is_running = True
        logger.info(f"MCP server {self.name} started")
    
    async def stop(self) -> None:
        """Stop the MCP server"""
        self.is_running = False
        await self.shutdown()
        logger.info(f"MCP server {self.name} stopped")