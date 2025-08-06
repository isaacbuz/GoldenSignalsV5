"""
Plugin Architecture Base
Provides extensibility through a plugin system
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type, TypeVar
from enum import Enum
import importlib
import inspect
from pathlib import Path

from pydantic import BaseModel, Field

from core.logging import get_logger
from core.events.bus import event_bus, EventTypes

logger = get_logger(__name__)

T = TypeVar('T')


class PluginStatus(Enum):
    """Plugin status states"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class PluginType(Enum):
    """Types of plugins"""
    AGENT = "agent"
    DATA_PROVIDER = "data_provider"
    STRATEGY = "strategy"
    INDICATOR = "indicator"
    RISK_MANAGER = "risk_manager"
    EXECUTOR = "executor"
    ANALYZER = "analyzer"
    STORAGE = "storage"
    NOTIFICATION = "notification"
    AUTHENTICATION = "authentication"


@dataclass
class PluginMetadata:
    """Plugin metadata"""
    name: str
    version: str
    type: PluginType
    author: str = "Unknown"
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "type": self.type.value,
            "author": self.author,
            "description": self.description,
            "dependencies": self.dependencies,
            "tags": self.tags
        }


class PluginConfig(BaseModel):
    """Base plugin configuration"""
    enabled: bool = True
    auto_start: bool = True
    priority: int = 0
    settings: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"


class PluginContext:
    """Context provided to plugins"""
    
    def __init__(
        self,
        config: PluginConfig,
        services: Dict[str, Any],
        event_bus: Any = None
    ):
        self.config = config
        self.services = services
        self.event_bus = event_bus or globals()['event_bus']
        self._state: Dict[str, Any] = {}
    
    def get_service(self, service_type: Type[T]) -> Optional[T]:
        """Get a service by type"""
        return self.services.get(service_type.__name__)
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get plugin state"""
        return self._state.get(key, default)
    
    def set_state(self, key: str, value: Any) -> None:
        """Set plugin state"""
        self._state[key] = value


class IPlugin(ABC):
    """Base plugin interface"""
    
    def __init__(self):
        self._status = PluginStatus.UNLOADED
        self._context: Optional[PluginContext] = None
        self._error: Optional[str] = None
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        pass
    
    @property
    def status(self) -> PluginStatus:
        """Get plugin status"""
        return self._status
    
    @property
    def context(self) -> Optional[PluginContext]:
        """Get plugin context"""
        return self._context
    
    @abstractmethod
    async def initialize(self, context: PluginContext) -> None:
        """
        Initialize the plugin
        
        Args:
            context: Plugin context with config and services
        """
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the plugin"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the plugin"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown and cleanup"""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check
        
        Returns:
            Health status dictionary
        """
        return {
            "status": self._status.value,
            "healthy": self._status == PluginStatus.RUNNING,
            "error": self._error
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate plugin configuration
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid
        """
        if self.metadata.config_schema:
            # Validate against schema
            # This would use a schema validation library
            pass
        return True


class DataProviderPlugin(IPlugin):
    """Plugin for market data providers"""
    
    @abstractmethod
    async def get_provider(self) -> Any:
        """Get the data provider instance"""
        pass
    
    @abstractmethod
    async def fetch_data(self, symbols: List[str], **kwargs) -> Any:
        """Fetch market data"""
        pass


class AgentPlugin(IPlugin):
    """Plugin for trading agents"""
    
    @abstractmethod
    async def get_agent(self) -> Any:
        """Get the agent instance"""
        pass
    
    @abstractmethod
    async def analyze(self, market_data: Dict[str, Any]) -> Any:
        """Perform analysis"""
        pass


class StrategyPlugin(IPlugin):
    """Plugin for trading strategies"""
    
    @abstractmethod
    async def get_strategy(self) -> Any:
        """Get the strategy instance"""
        pass
    
    @abstractmethod
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Any]:
        """Generate trading signals"""
        pass


class PluginManager:
    """
    Manages plugin lifecycle and registry
    """
    
    def __init__(self):
        self._plugins: Dict[str, IPlugin] = {}
        self._plugin_types: Dict[PluginType, List[str]] = {
            plugin_type: [] for plugin_type in PluginType
        }
        self._plugin_paths: List[Path] = []
        self._services: Dict[str, Any] = {}
        
        logger.info("Plugin manager initialized")
    
    def add_plugin_path(self, path: str) -> None:
        """Add a path to search for plugins"""
        plugin_path = Path(path)
        if plugin_path.exists() and plugin_path.is_dir():
            self._plugin_paths.append(plugin_path)
            logger.info(f"Added plugin path: {path}")
        else:
            logger.warning(f"Invalid plugin path: {path}")
    
    def register_service(self, service_type: Type, instance: Any) -> None:
        """Register a service for plugins to use"""
        self._services[service_type.__name__] = instance
        logger.debug(f"Registered service: {service_type.__name__}")
    
    async def load_plugin(self, plugin_class: Type[IPlugin], config: PluginConfig = None) -> str:
        """
        Load a plugin
        
        Args:
            plugin_class: Plugin class to instantiate
            config: Plugin configuration
            
        Returns:
            Plugin name
        """
        try:
            # Create plugin instance
            plugin = plugin_class()
            
            # Get metadata
            metadata = plugin.metadata
            plugin_name = metadata.name
            
            # Check dependencies
            for dep in metadata.dependencies:
                if dep not in self._plugins:
                    raise ValueError(f"Missing dependency: {dep}")
            
            # Set status
            plugin._status = PluginStatus.LOADING
            
            # Create context
            context = PluginContext(
                config=config or PluginConfig(),
                services=self._services,
                event_bus=event_bus
            )
            
            # Initialize plugin
            await plugin.initialize(context)
            plugin._context = context
            plugin._status = PluginStatus.INITIALIZED
            
            # Register plugin
            self._plugins[plugin_name] = plugin
            self._plugin_types[metadata.type].append(plugin_name)
            
            # Auto-start if configured
            if config and config.auto_start:
                await self.start_plugin(plugin_name)
            
            # Publish event
            await event_bus.publish(
                "plugin.loaded",
                data={"plugin": plugin_name, "metadata": metadata.to_dict()}
            )
            
            logger.info(f"Loaded plugin: {plugin_name} v{metadata.version}")
            return plugin_name
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_class.__name__}: {str(e)}")
            raise
    
    async def load_from_module(self, module_path: str, config: PluginConfig = None) -> List[str]:
        """
        Load plugins from a Python module
        
        Args:
            module_path: Module path (e.g., 'plugins.my_plugin')
            config: Plugin configuration
            
        Returns:
            List of loaded plugin names
        """
        try:
            # Import module
            module = importlib.import_module(module_path)
            
            # Find plugin classes
            loaded = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, IPlugin) and 
                    obj != IPlugin and
                    not inspect.isabstract(obj)):
                    
                    plugin_name = await self.load_plugin(obj, config)
                    loaded.append(plugin_name)
            
            return loaded
            
        except Exception as e:
            logger.error(f"Failed to load module {module_path}: {str(e)}")
            raise
    
    async def discover_plugins(self) -> List[str]:
        """
        Discover and load plugins from configured paths
        
        Returns:
            List of discovered plugin names
        """
        discovered = []
        
        for plugin_path in self._plugin_paths:
            # Find Python files
            for file_path in plugin_path.glob("**/*.py"):
                if file_path.name.startswith("_"):
                    continue
                
                # Convert to module path
                relative_path = file_path.relative_to(plugin_path.parent)
                module_path = str(relative_path.with_suffix("")).replace(os.sep, ".")
                
                try:
                    loaded = await self.load_from_module(module_path)
                    discovered.extend(loaded)
                except Exception as e:
                    logger.warning(f"Failed to load plugin from {file_path}: {str(e)}")
        
        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered
    
    async def start_plugin(self, plugin_name: str) -> None:
        """Start a plugin"""
        plugin = self._plugins.get(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        if plugin.status == PluginStatus.RUNNING:
            logger.warning(f"Plugin already running: {plugin_name}")
            return
        
        try:
            await plugin.start()
            plugin._status = PluginStatus.RUNNING
            
            await event_bus.publish(
                "plugin.started",
                data={"plugin": plugin_name}
            )
            
            logger.info(f"Started plugin: {plugin_name}")
            
        except Exception as e:
            plugin._status = PluginStatus.ERROR
            plugin._error = str(e)
            logger.error(f"Failed to start plugin {plugin_name}: {str(e)}")
            raise
    
    async def stop_plugin(self, plugin_name: str) -> None:
        """Stop a plugin"""
        plugin = self._plugins.get(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        if plugin.status != PluginStatus.RUNNING:
            logger.warning(f"Plugin not running: {plugin_name}")
            return
        
        try:
            await plugin.stop()
            plugin._status = PluginStatus.STOPPED
            
            await event_bus.publish(
                "plugin.stopped",
                data={"plugin": plugin_name}
            )
            
            logger.info(f"Stopped plugin: {plugin_name}")
            
        except Exception as e:
            plugin._status = PluginStatus.ERROR
            plugin._error = str(e)
            logger.error(f"Failed to stop plugin {plugin_name}: {str(e)}")
            raise
    
    async def unload_plugin(self, plugin_name: str) -> None:
        """Unload a plugin"""
        plugin = self._plugins.get(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        # Stop if running
        if plugin.status == PluginStatus.RUNNING:
            await self.stop_plugin(plugin_name)
        
        # Shutdown plugin
        await plugin.shutdown()
        
        # Remove from registry
        del self._plugins[plugin_name]
        
        # Remove from type list
        for plugin_type, names in self._plugin_types.items():
            if plugin_name in names:
                names.remove(plugin_name)
        
        await event_bus.publish(
            "plugin.unloaded",
            data={"plugin": plugin_name}
        )
        
        logger.info(f"Unloaded plugin: {plugin_name}")
    
    def get_plugin(self, plugin_name: str) -> Optional[IPlugin]:
        """Get a plugin by name"""
        return self._plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[IPlugin]:
        """Get all plugins of a specific type"""
        plugin_names = self._plugin_types.get(plugin_type, [])
        return [self._plugins[name] for name in plugin_names if name in self._plugins]
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all plugins with their status"""
        return {
            name: {
                "metadata": plugin.metadata.to_dict(),
                "status": plugin.status.value,
                "error": plugin._error
            }
            for name, plugin in self._plugins.items()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all plugins"""
        health = {}
        
        for name, plugin in self._plugins.items():
            health[name] = await plugin.health_check()
        
        return {
            "plugins": health,
            "total": len(self._plugins),
            "running": sum(1 for p in self._plugins.values() if p.status == PluginStatus.RUNNING),
            "errors": sum(1 for p in self._plugins.values() if p.status == PluginStatus.ERROR)
        }
    
    async def start_all(self) -> None:
        """Start all plugins"""
        for plugin_name in self._plugins:
            if self._plugins[plugin_name].context.config.auto_start:
                await self.start_plugin(plugin_name)
    
    async def stop_all(self) -> None:
        """Stop all plugins"""
        for plugin_name in list(self._plugins.keys()):
            await self.stop_plugin(plugin_name)
    
    async def shutdown(self) -> None:
        """Shutdown plugin manager"""
        logger.info("Shutting down plugin manager...")
        
        # Stop all plugins
        await self.stop_all()
        
        # Unload all plugins
        for plugin_name in list(self._plugins.keys()):
            await self.unload_plugin(plugin_name)
        
        logger.info("Plugin manager shutdown complete")


# Global plugin manager instance
plugin_manager = PluginManager()