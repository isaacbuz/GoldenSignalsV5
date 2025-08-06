"""
MCP API Routes V2
Enhanced endpoints for Model Context Protocol interactions
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime

from mcp.servers.market_data import MarketDataMCPServer
from mcp.client import MCPClient
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/mcp", tags=["mcp"])

# Global MCP servers and clients
mcp_servers = {}
mcp_clients = {}


class ToolExecutionRequest(BaseModel):
    """Request model for tool execution"""
    tool: str
    arguments: Dict[str, Any]
    user_id: Optional[str] = None


class BatchExecutionRequest(BaseModel):
    """Request model for batch tool execution"""
    requests: List[ToolExecutionRequest]
    user_id: Optional[str] = None


@router.on_event("startup")
async def startup_event():
    """Initialize MCP servers on startup"""
    # Initialize Market Data MCP Server
    market_server = MarketDataMCPServer()
    mcp_servers["market-data"] = market_server
    
    # Create client for market data server
    market_client = MCPClient("market-data")
    await market_client.connect(market_server)
    mcp_clients["market-data"] = market_client
    
    logger.info("MCP servers initialized")


@router.on_event("shutdown")
async def shutdown_event():
    """Shutdown MCP servers"""
    for client in mcp_clients.values():
        await client.disconnect()
    logger.info("MCP servers shutdown")


@router.get("/servers")
async def list_servers() -> Dict[str, Any]:
    """List available MCP servers"""
    servers = []
    
    for name, client in mcp_clients.items():
        info = client.get_server_info()
        servers.append({
            "name": name,
            "info": info
        })
    
    return {
        "count": len(servers),
        "servers": servers
    }


@router.get("/servers/{server_name}/tools")
async def list_server_tools(server_name: str) -> Dict[str, Any]:
    """List tools available on a specific server"""
    if server_name not in mcp_clients:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")
    
    client = mcp_clients[server_name]
    tools = client.list_tools()
    
    return {
        "server": server_name,
        "count": len(tools),
        "tools": tools
    }


@router.get("/servers/{server_name}/resources")
async def list_server_resources(server_name: str) -> Dict[str, Any]:
    """List resources available on a specific server"""
    if server_name not in mcp_clients:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")
    
    client = mcp_clients[server_name]
    resources = client.list_resources()
    
    return {
        "server": server_name,
        "count": len(resources),
        "resources": resources
    }


@router.post("/servers/{server_name}/execute")
async def execute_tool(
    server_name: str,
    request: ToolExecutionRequest
) -> Dict[str, Any]:
    """Execute a tool on a specific server"""
    if server_name not in mcp_clients:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")
    
    client = mcp_clients[server_name]
    
    # Validate arguments
    validation = client.validate_arguments(request.tool, request.arguments)
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid arguments",
                "validation_errors": validation["errors"]
            }
        )
    
    # Execute tool
    try:
        result = await client.execute_tool(
            request.tool,
            request.arguments,
            request.user_id
        )
        
        return {
            "server": server_name,
            "tool": request.tool,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Tool execution failed: {request.tool}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Tool execution failed: {str(e)}"
        )


@router.post("/servers/{server_name}/execute-batch")
async def execute_batch(
    server_name: str,
    request: BatchExecutionRequest
) -> Dict[str, Any]:
    """Execute multiple tools in batch"""
    if server_name not in mcp_clients:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")
    
    client = mcp_clients[server_name]
    
    # Convert requests to format expected by client
    batch_requests = [
        {
            "tool": req.tool,
            "arguments": req.arguments
        }
        for req in request.requests
    ]
    
    try:
        results = await client.execute_batch(batch_requests, request.user_id)
        
        return {
            "server": server_name,
            "count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error("Batch execution failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch execution failed: {str(e)}"
        )


@router.get("/servers/{server_name}/resources/{resource_uri:path}")
async def get_resource(server_name: str, resource_uri: str) -> Dict[str, Any]:
    """Get a specific resource from a server"""
    if server_name not in mcp_clients:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")
    
    client = mcp_clients[server_name]
    
    try:
        # Add protocol prefix if not present
        if not resource_uri.startswith("market://"):
            resource_uri = f"market://{resource_uri}"
        
        resource = await client.get_resource(resource_uri)
        
        if not resource:
            raise HTTPException(
                status_code=404,
                detail=f"Resource '{resource_uri}' not found"
            )
        
        return {
            "server": server_name,
            "resource": resource
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get resource: {resource_uri}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get resource: {str(e)}"
        )


@router.get("/tools")
async def list_all_tools() -> Dict[str, Any]:
    """List all available tools across all servers"""
    all_tools = []
    
    for server_name, client in mcp_clients.items():
        tools = client.list_tools()
        for tool in tools:
            tool["server"] = server_name
            all_tools.append(tool)
    
    return {
        "count": len(all_tools),
        "tools": all_tools
    }


@router.get("/tools/{tool_name}")
async def get_tool_info(tool_name: str) -> Dict[str, Any]:
    """Get information about a specific tool"""
    # Search across all servers
    for server_name, client in mcp_clients.items():
        tool_info = client.get_tool_info(tool_name)
        if tool_info:
            return {
                "server": server_name,
                "tool": tool_info
            }
    
    raise HTTPException(
        status_code=404,
        detail=f"Tool '{tool_name}' not found on any server"
    )


# Convenience endpoints for common operations

@router.get("/quote/{symbol}")
async def get_quote(symbol: str) -> Dict[str, Any]:
    """Get real-time quote for a symbol (convenience endpoint)"""
    client = mcp_clients.get("market-data")
    if not client:
        raise HTTPException(status_code=503, detail="Market data server not available")
    
    result = await client.execute_tool("get_quote", {"symbol": symbol})
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.get("/market-summary")
async def get_market_summary() -> Dict[str, Any]:
    """Get market summary (convenience endpoint)"""
    client = mcp_clients.get("market-data")
    if not client:
        raise HTTPException(status_code=503, detail="Market data server not available")
    
    result = await client.execute_tool("get_market_summary", {"include_sectors": True})
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/compare-symbols")
async def compare_symbols(symbols: List[str]) -> Dict[str, Any]:
    """Compare multiple symbols (convenience endpoint)"""
    client = mcp_clients.get("market-data")
    if not client:
        raise HTTPException(status_code=503, detail="Market data server not available")
    
    result = await client.execute_tool("compare_symbols", {"symbols": symbols})
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result