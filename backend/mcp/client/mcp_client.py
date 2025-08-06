"""
MCP Client
Client for interacting with MCP servers
"""

from typing import Any, Dict, List, Optional
import asyncio
from datetime import datetime

from core.logging import get_logger

logger = get_logger(__name__)


class MCPClient:
    """
    Client for interacting with MCP servers
    
    Features:
    - Tool discovery
    - Tool execution
    - Resource access
    - Error handling
    - Response caching
    """
    
    def __init__(self, server_name: str):
        self.server_name = server_name
        self.server = None
        self.tools_cache = {}
        self.resources_cache = {}
        
        logger.info(f"MCP Client initialized for server: {server_name}")
    
    async def connect(self, server_instance) -> None:
        """Connect to an MCP server instance"""
        self.server = server_instance
        await self.server.start()
        
        # Cache available tools and resources
        self.tools_cache = {tool.name: tool for tool in self.server.tools.values()}
        self.resources_cache = {res.uri: res for res in self.server.resources.values()}
        
        logger.info(f"Connected to MCP server: {self.server_name}")
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server"""
        if self.server:
            await self.server.stop()
            self.server = None
        logger.info(f"Disconnected from MCP server: {self.server_name}")
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        if not self.server:
            return []
        return self.server.list_tools()
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources"""
        if not self.server:
            return []
        return self.server.list_resources()
    
    async def execute_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool on the MCP server
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            user_id: Optional user ID for tracking
            
        Returns:
            Tool execution result
        """
        if not self.server:
            return {
                "error": "Not connected to MCP server",
                "server": self.server_name
            }
        
        try:
            result = await self.server.execute_tool(tool_name, arguments, user_id)
            
            # Log successful execution
            if "error" not in result:
                logger.debug(f"Tool {tool_name} executed successfully")
            else:
                logger.warning(f"Tool {tool_name} returned error: {result['error']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}", exc_info=True)
            return {
                "error": f"Tool execution failed: {str(e)}",
                "tool": tool_name,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_resource(self, uri: str) -> Optional[Dict[str, Any]]:
        """
        Get a resource from the MCP server
        
        Args:
            uri: Resource URI
            
        Returns:
            Resource content or None
        """
        if not self.server:
            logger.error("Not connected to MCP server")
            return None
        
        try:
            resource = await self.server.get_resource(uri)
            
            if resource:
                logger.debug(f"Resource {uri} retrieved successfully")
            else:
                logger.warning(f"Resource {uri} not found")
            
            return resource
            
        except Exception as e:
            logger.error(f"Failed to get resource {uri}: {e}", exc_info=True)
            return None
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool"""
        if tool_name in self.tools_cache:
            return self.tools_cache[tool_name].to_dict()
        return None
    
    def get_resource_info(self, uri: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific resource"""
        if uri in self.resources_cache:
            return self.resources_cache[uri].to_dict()
        return None
    
    async def execute_batch(
        self, 
        requests: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tool requests in batch
        
        Args:
            requests: List of tool requests with format:
                      [{"tool": "tool_name", "arguments": {...}}, ...]
            user_id: Optional user ID
            
        Returns:
            List of results in same order as requests
        """
        tasks = []
        
        for request in requests:
            tool_name = request.get("tool")
            arguments = request.get("arguments", {})
            
            if not tool_name:
                tasks.append(asyncio.create_task(
                    asyncio.coroutine(lambda: {"error": "Tool name required"})()
                ))
            else:
                tasks.append(asyncio.create_task(
                    self.execute_tool(tool_name, arguments, user_id)
                ))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    "error": f"Request failed: {str(result)}",
                    "request_index": i
                })
            else:
                final_results.append(result)
        
        return final_results
    
    def validate_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate arguments for a tool
        
        Returns:
            Dict with "valid" boolean and optional "errors" list
        """
        if tool_name not in self.tools_cache:
            return {
                "valid": False,
                "errors": [f"Unknown tool: {tool_name}"]
            }
        
        tool = self.tools_cache[tool_name]
        schema = tool.input_schema
        
        errors = []
        
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in arguments:
                errors.append(f"Missing required field: {field}")
        
        # Check field types (basic validation)
        properties = schema.get("properties", {})
        for field, value in arguments.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type:
                    if not self._check_type(value, expected_type):
                        errors.append(
                            f"Invalid type for {field}: expected {expected_type}, "
                            f"got {type(value).__name__}"
                        )
        
        return {
            "valid": len(errors) == 0,
            "errors": errors if errors else None
        }
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        return True
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get information about the connected server"""
        if not self.server:
            return {
                "connected": False,
                "server_name": self.server_name
            }
        
        info = self.server.get_server_info()
        info["connected"] = True
        return info