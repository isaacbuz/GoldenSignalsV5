#!/bin/bash
# Enhanced startup script for GoldenSignalsAI with service checks

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🚀 Starting GoldenSignalsAI..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=0
    
    echo -n "Waiting for $name to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "200\|404"; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done
    echo -e " ${RED}✗${NC}"
    return 1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

if ! command_exists node; then
    echo -e "${RED}Error: Node.js is not installed${NC}"
    exit 1
fi

if ! command_exists redis-server; then
    echo -e "${YELLOW}Warning: Redis is not installed. Some features may not work.${NC}"
    echo "To install Redis: brew install redis (macOS) or sudo apt-get install redis-server (Ubuntu)"
fi

# Check if Redis is running
if command_exists redis-cli; then
    if ! redis-cli ping >/dev/null 2>&1; then
        echo -e "${YELLOW}Starting Redis...${NC}"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew services start redis >/dev/null 2>&1 || true
        else
            sudo service redis-server start >/dev/null 2>&1 || true
        fi
        sleep 2
    fi
fi

# Create logs directory
mkdir -p logs

# Start backend
echo -e "\n${GREEN}Starting Backend...${NC}"
cd backend

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install/update dependencies
echo "Installing backend dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found in backend directory${NC}"
    echo "Creating .env from example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
    else
        echo -e "${RED}Error: No .env.example file found${NC}"
        exit 1
    fi
fi

# Start backend server
echo "Starting backend server..."
uvicorn app:app --reload --host 0.0.0.0 --port 8000 > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
if ! wait_for_service "http://localhost:8000/api/v1/health" "Backend"; then
    echo -e "${RED}Backend failed to start. Check logs/backend.log for details${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start frontend
echo -e "\n${GREEN}Starting Frontend...${NC}"
cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start frontend development server
echo "Starting frontend server..."
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to be ready
if ! wait_for_service "http://localhost:3000" "Frontend"; then
    echo -e "${RED}Frontend failed to start. Check logs/frontend.log for details${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down GoldenSignalsAI...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}Shutdown complete${NC}"
}

# Set up cleanup trap
trap cleanup EXIT INT TERM

# Display success message
echo -e "\n${GREEN}✅ GoldenSignalsAI is running!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "   Frontend:    ${GREEN}http://localhost:3000${NC}"
echo -e "   Backend API: ${GREEN}http://localhost:8000${NC}"
echo -e "   API Docs:    ${GREEN}http://localhost:8000/docs${NC}"
echo -e "   Health:      ${GREEN}http://localhost:8000/api/v1/health/detailed${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "\n${YELLOW}Press Ctrl+C to stop all services${NC}\n"

# Check services health
echo "Checking services health..."
curl -s http://localhost:8000/api/v1/health/detailed | python3 -m json.tool || true

# Keep script running
wait $BACKEND_PID $FRONTEND_PID