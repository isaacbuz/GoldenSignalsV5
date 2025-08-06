# Claude Instructions

## Project Overview
This is the GoldenSignalsAI project (Version 5). Please help with software engineering tasks including:
- Code analysis and debugging
- File editing and refactoring
- Running bash commands
- Git operations
- General development assistance

## CRITICAL DESIGN REQUIREMENTS (DO NOT DEVIATE)

### Layout System - Source of Truth
- **ALWAYS use the ProfessionalLayout component** located at `/frontend/src/components/layout/ProfessionalLayout.tsx`
- This layout is the single source of truth for app structure
- DO NOT create alternative layouts or deviate from this design
- All pages MUST use this layout for consistency
- Layout modifications require explicit user permission

### Chart System - Single Robust Implementation
- **USE ONLY the AIHybridChart component** located at `/frontend/src/components/charts/AIHybridChart.tsx`
- This is the consolidated, robust chart combining best practices from:
  - D3.js for advanced visualizations and overlays
  - Lightweight Charts for performance
  - Custom AI indicators and pattern recognition
- DO NOT create multiple chart variations
- All chart features must be integrated into this single component

### Design Principles
1. **Consistency First**: Every page follows the same layout structure
2. **Professional Dark Theme**: Black background (#0a0a0a) with golden accents (#FFD700)
3. **Clean Typography**: Inter font family throughout
4. **Responsive Grid**: 12-column grid system with consistent spacing
5. **Status Bar**: Always visible at top with system health
6. **Side Navigation**: Fixed sidebar with main navigation
7. **Content Area**: Main content area with consistent padding

## Project-Specific Instructions
- Follow existing code conventions and patterns
- Check for existing libraries before suggesting new ones
- Maintain code quality with proper linting and type checking
- Focus on defensive security practices

## Common Commands
- Build: `npm run build`
- Dev: `npm run dev`
- Backend: `python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000`
- Frontend: `npm run dev` (port 3000)

## Key Directories
- `/frontend/src/components/layout` - Layout components (USE ProfessionalLayout.tsx)
- `/frontend/src/components/charts` - Chart component (USE AIHybridChart.tsx ONLY)
- `/frontend/src/pages` - Page components
- `/backend` - Backend API

## Notes
- Layout and chart architecture are finalized - do not create alternatives
- All UI changes must maintain the professional dark/gold theme
- Performance is critical - optimize all chart renderings

## INCOMPLETE IMPLEMENTATIONS - COMPREHENSIVE REVIEW

### 🚨 CRITICAL SECURITY ISSUE
- **Real API keys exposed** in `.env.example` instead of placeholders - IMMEDIATE SECURITY VULNERABILITY

### 🏗️ CORE ARCHITECTURE GAPS
1. **Missing AIHybridChart.tsx** - The main chart component specified above doesn't exist
2. **Incomplete user authentication** - User relationships commented out in models/signal.py
3. **RAG system broken** - Circular dependencies and orchestrator placeholder in api/routes/rag.py

### 🤖 AGENT IMPLEMENTATION ISSUES
- Multiple agents have `pass` statements instead of actual implementations
- Base agent abstract methods are empty in agents/base.py (lines 181, 191, 411, 415, 419)
- Interface definitions are all placeholder `pass` statements in core/interfaces/__init__.py

### 🔧 SERVICE LAYER PROBLEMS
- MCP server returns mock data in services/mcp/mcp_server.py (lines 68, 94, 121)
- Social sentiment analyzer throws `NotImplementedError` for search_posts and get_trending_topics
- WebSocket manager has empty exception handlers
- Enhanced data aggregator uses placeholder blockchain metrics (lines 590-595)

### 📊 SYSTEM COMPLETENESS ASSESSMENT
- **Frontend**: ~30% complete (basic layout done, major components missing)
- **Backend API**: ~60% complete (structure exists, many endpoints incomplete)
- **Agents**: ~40% complete (framework solid, implementations partial)
- **Database**: ~70% complete (models exist, relationships incomplete)
- **Testing**: ~15% complete (basic tests only, major coverage gaps)
- **Security**: ~50% complete (framework exists, implementation gaps)
- **Overall**: ~45-50% complete

### 🔧 HIGH PRIORITY FIXES NEEDED
1. Replace exposed API keys with placeholders immediately
2. Create missing AIHybridChart.tsx component
3. Complete user authentication system
4. Fix RAG system circular dependencies
5. Implement actual MCP functionality (remove mock data)
6. Complete social sentiment analyzer methods
7. Remove all 'pass' statements from agent implementations
8. Implement interface methods
9. Fix empty exception handlers throughout codebase
10. Expand test coverage significantly

The system has solid architectural foundations but requires significant work to be production-ready.