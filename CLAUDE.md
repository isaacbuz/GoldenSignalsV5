# Claude Instructions

## Project Overview
This is the GoldenSignalsAI_Clean project. Please help with software engineering tasks including:
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