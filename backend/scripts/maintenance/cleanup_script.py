#!/usr/bin/env python3
"""
Deep Cleanup Script for GoldenSignalsAI Backend
Removes legacy code, fixes circular dependencies, and standardizes architecture
"""

import shutil
import re
from pathlib import Path
from typing import List, Dict, Set

class CodebaseCleanup:
    """Comprehensive codebase cleanup utility"""
    
    def __init__(self, backend_path: str):
        self.backend_path = Path(backend_path)
        self.removed_files = []
        self.updated_files = []
        self.errors = []
        
    def run_full_cleanup(self):
        """Execute complete cleanup process"""
        print("🧹 Starting Deep Cleanup of GoldenSignalsAI Backend")
        print("=" * 60)
        
        # Phase 1: Remove legacy files
        self.remove_legacy_files()
        
        # Phase 2: Consolidate duplicate services
        self.consolidate_services()
        
        # Phase 3: Fix circular dependencies
        self.fix_circular_dependencies()
        
        # Phase 4: Clean up API routes
        self.cleanup_api_routes()
        
        # Phase 5: Standardize imports
        self.standardize_imports()
        
        # Phase 6: Update main application
        self.update_main_app()
        
        # Phase 7: Clean unused files
        self.remove_unused_files()
        
        # Summary
        self.print_summary()
    
    def remove_legacy_files(self):
        """Remove legacy and duplicate files"""
        print("📂 Phase 1: Removing legacy files...")
        
        files_to_remove = [
            # Legacy orchestrators
            "agents/orchestrator.py",
            "services/orchestrator.py", 
            "agents/meta_signal_orchestrator.py",
            "services/websocket_orchestrator.py",
            
            # Duplicate market data services
            "services/market_data_service.py",
            "services/universal_market_data.py",
            "services/market_data_aggregator.py",
            
            # Old RAG implementations
            "services/rag/rag_system.py",
            
            # Legacy agents directory
            "agents/legacy/",
            
            # Duplicate test files
            "tests/test_meta_orchestrator.py",
            "tests/test_websocket_orchestration.py",
            
            # Old API routes
            "api/routes/market_data.py",  # Keep v2
            "api/routes/mcp.py",  # Keep v2
            
            # Unused WebSocket implementations
            "api/websocket/orchestrated_ws.py",
        ]
        
        for file_path in files_to_remove:
            full_path = self.backend_path / file_path
            if full_path.exists():
                if full_path.is_dir():
                    shutil.rmtree(full_path)
                    print(f"  ✅ Removed directory: {file_path}")
                else:
                    full_path.unlink()
                    print(f"  ✅ Removed file: {file_path}")
                self.removed_files.append(str(file_path))
            else:
                print(f"  ⚠️  File not found: {file_path}")
    
    def consolidate_services(self):
        """Consolidate duplicate services"""
        print("\n🔧 Phase 2: Consolidating services...")
        
        # Rename v2 files to primary versions
        renames = [
            ("api/routes/market_data_v2.py", "api/routes/market_data.py"),
            ("api/routes/mcp_v2.py", "api/routes/mcp.py"),
        ]
        
        for old_path, new_path in renames:
            old_file = self.backend_path / old_path
            new_file = self.backend_path / new_path
            
            if old_file.exists():
                # Remove new file if it exists
                if new_file.exists():
                    new_file.unlink()
                
                # Rename old to new
                old_file.rename(new_file)
                print(f"  ✅ Renamed: {old_path} → {new_path}")
                self.updated_files.append(new_path)
    
    def fix_circular_dependencies(self):
        """Fix circular import dependencies"""
        print("\n🔄 Phase 3: Fixing circular dependencies...")
        
        # Create interfaces directory
        interfaces_dir = self.backend_path / "core" / "interfaces"
        interfaces_dir.mkdir(exist_ok=True)
        
        # Create base interfaces
        self.create_base_interfaces()
        
        # Update imports in key files
        self.update_dependency_imports()
    
    def create_base_interfaces(self):
        """Create base interfaces to break circular dependencies"""
        interfaces_content = '''"""
Base interfaces to prevent circular dependencies
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

class IMarketDataService(ABC):
    """Interface for market data services"""
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        pass

class IOrchestrator(ABC):
    """Interface for orchestrator services"""
    
    @abstractmethod
    async def analyze(self, symbol: str, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        pass

class IRAGService(ABC):
    """Interface for RAG services"""
    
    @abstractmethod
    async def query(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def ingest_documents(self, documents: List[Dict[str, Any]]) -> int:
        pass

class IWebSocketManager(ABC):
    """Interface for WebSocket management"""
    
    @abstractmethod
    async def broadcast_signal(self, data: Dict[str, Any]):
        pass
    
    @abstractmethod
    async def broadcast_price_update(self, symbol: str, price: float, volume: int):
        pass
'''
        
        interfaces_file = self.backend_path / "core" / "interfaces" / "__init__.py"
        with open(interfaces_file, 'w') as f:
            f.write(interfaces_content)
        
        print("  ✅ Created base interfaces")
    
    def update_dependency_imports(self):
        """Update imports to use consolidated services"""
        print("  🔍 Updating dependency imports...")
        
        # Files that need import updates
        files_to_update = [
            "app.py",
            "api/routes/agents.py", 
            "api/routes/signals.py",
            "api/routes/ai_analysis.py",
            "core/dependencies.py"
        ]
        
        import_mappings = {
            # Old orchestrator imports
            "from agents.orchestrator import AgentOrchestrator": "from core.orchestrator import unified_orchestrator as AgentOrchestrator",
            "from services.orchestrator import TradingOrchestrator": "from core.orchestrator import unified_orchestrator as TradingOrchestrator", 
            "from agents.meta_signal_orchestrator import MetaSignalOrchestrator": "from core.orchestrator import unified_orchestrator as MetaSignalOrchestrator",
            "from services.websocket_orchestrator import websocket_orchestrator": "from core.orchestrator import unified_orchestrator",
            
            # Market data service imports
            "from services.market_data_service import MarketDataService": "from services.market_data_unified import unified_market_service as MarketDataService",
            "from services.universal_market_data import UniversalMarketDataService": "from services.market_data_unified import unified_market_service as UniversalMarketDataService",
            
            # RAG service imports
            "from services.rag.rag_system import RAGSystem": "from services.rag_service import rag_service as RAGSystem",
        }
        
        for file_path in files_to_update:
            full_path = self.backend_path / file_path
            if full_path.exists():
                self.update_imports_in_file(full_path, import_mappings)
    
    def update_imports_in_file(self, file_path: Path, mappings: Dict[str, str]):
        """Update imports in a specific file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            
            for old_import, new_import in mappings.items():
                content = content.replace(old_import, new_import)
            
            if content != original_content:
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f"  ✅ Updated imports in: {file_path.relative_to(self.backend_path)}")
                self.updated_files.append(str(file_path.relative_to(self.backend_path)))
        
        except Exception as e:
            print(f"  ❌ Error updating {file_path}: {e}")
            self.errors.append(f"Import update failed: {file_path} - {e}")
    
    def cleanup_api_routes(self):
        """Clean up API routes and remove unused endpoints"""
        print("\n🛣️  Phase 4: Cleaning up API routes...")
        
        # Update main routes file to remove references to deleted routes
        routes_init = self.backend_path / "api" / "routes" / "__init__.py"
        if routes_init.exists():
            with open(routes_init, 'r') as f:
                content = f.read()
            
            # Remove imports for deleted routes
            lines_to_remove = [
                "from . import orchestrator_v1",
                "from . import websocket_orchestration", 
                "from . import meta_orchestrator"
            ]
            
            for line in lines_to_remove:
                content = content.replace(line, "")
            
            # Clean up empty lines
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            
            with open(routes_init, 'w') as f:
                f.write(content)
            
            print("  ✅ Updated routes __init__.py")
    
    def standardize_imports(self):
        """Standardize import patterns across the codebase"""
        print("\n📋 Phase 5: Standardizing imports...")
        
        # Find all Python files
        py_files = list(self.backend_path.rglob("*.py"))
        
        standardized_count = 0
        
        for py_file in py_files:
            if self.standardize_file_imports(py_file):
                standardized_count += 1
        
        print(f"  ✅ Standardized imports in {standardized_count} files")
    
    def standardize_file_imports(self, file_path: Path) -> bool:
        """Standardize imports in a single file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # Remove unused imports (common patterns)
            unused_patterns = [
                r'import\s+os\s*\n(?!.*os\.)',  # os not used
                r'import\s+sys\s*\n(?!.*sys\.)',  # sys not used  
                r'from\s+typing\s+import.*# unused.*\n',  # marked unused
            ]
            
            for pattern in unused_patterns:
                content = re.sub(pattern, '', content, flags=re.MULTILINE)
            
            # Sort imports (basic sorting)
            lines = content.split('\n')
            import_lines = []
            other_lines = []
            in_imports = True
            
            for line in lines:
                if in_imports and (line.startswith('import ') or line.startswith('from ')):
                    import_lines.append(line)
                elif line.strip() == '':
                    if in_imports:
                        import_lines.append(line)
                    else:
                        other_lines.append(line)
                else:
                    in_imports = False
                    other_lines.append(line)
            
            # Sort import lines
            import_lines.sort()
            
            content = '\n'.join(import_lines + other_lines)
            
            if content != original_content:
                with open(file_path, 'w') as f:
                    f.write(content)
                return True
            
            return False
        
        except Exception as e:
            return False
    
    def update_main_app(self):
        """Update main application file"""
        print("\n🚀 Phase 6: Updating main application...")
        
        app_file = self.backend_path / "app.py"
        if not app_file.exists():
            print("  ⚠️  app.py not found")
            return
        
        with open(app_file, 'r') as f:
            content = f.read()
        
        # Update imports to use consolidated services
        updates = [
            # Remove old orchestrator imports
            ("from services.websocket_orchestrator import websocket_orchestrator", 
             "from core.orchestrator import unified_orchestrator"),
            
            # Update initialization calls
            ("await websocket_orchestrator.initialize()", 
             "await unified_orchestrator.initialize()"),
             
            ("await websocket_orchestrator.stop()", 
             "# Unified orchestrator handles cleanup automatically"),
        ]
        
        for old, new in updates:
            content = content.replace(old, new)
        
        with open(app_file, 'w') as f:
            f.write(content)
        
        print("  ✅ Updated app.py")
    
    def remove_unused_files(self):
        """Remove unused files and empty directories"""
        print("\n🗑️  Phase 7: Removing unused files...")
        
        # Remove empty __pycache__ directories
        pycache_dirs = list(self.backend_path.rglob("__pycache__"))
        for cache_dir in pycache_dirs:
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir)
        
        print(f"  ✅ Removed {len(pycache_dirs)} __pycache__ directories")
        
        # Remove .pyc files
        pyc_files = list(self.backend_path.rglob("*.pyc"))
        for pyc_file in pyc_files:
            pyc_file.unlink()
        
        print(f"  ✅ Removed {len(pyc_files)} .pyc files")
        
        # Remove empty directories
        empty_dirs = []
        for root, dirs, files in os.walk(self.backend_path, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):  # Directory is empty
                        dir_path.rmdir()
                        empty_dirs.append(dir_path)
                except OSError:
                    pass  # Directory not empty or permission issue
        
        print(f"  ✅ Removed {len(empty_dirs)} empty directories")
    
    def print_summary(self):
        """Print cleanup summary"""
        print("\n" + "=" * 60)
        print("🎉 CLEANUP COMPLETED!")
        print("=" * 60)
        
        print(f"\n📊 SUMMARY:")
        print(f"  • Files removed: {len(self.removed_files)}")
        print(f"  • Files updated: {len(self.updated_files)}")
        print(f"  • Errors encountered: {len(self.errors)}")
        
        if self.removed_files:
            print(f"\n🗑️  REMOVED FILES:")
            for file in self.removed_files[:10]:  # Show first 10
                print(f"  - {file}")
            if len(self.removed_files) > 10:
                print(f"  ... and {len(self.removed_files) - 10} more")
        
        if self.updated_files:
            print(f"\n🔧 UPDATED FILES:")
            for file in self.updated_files[:10]:  # Show first 10  
                print(f"  - {file}")
            if len(self.updated_files) > 10:
                print(f"  ... and {len(self.updated_files) - 10} more")
        
        if self.errors:
            print(f"\n❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")
        
        print(f"\n✅ Codebase cleanup completed successfully!")
        print(f"   Next steps: Run tests to ensure everything works correctly.")

if __name__ == "__main__":
    backend_path = "/Users/isaacbuz/Documents/Projects/Signals/GoldenSignalsV5/backend"
    cleanup = CodebaseCleanup(backend_path)
    cleanup.run_full_cleanup()