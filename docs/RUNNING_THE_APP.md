# Running GoldenSignalsAI

This guide explains how to run the GoldenSignalsAI trading platform in various configurations.

## Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- PostgreSQL 14+ (for production)
- Redis 7+ (for caching)

## Quick Start

The simplest way to run the application:

```bash
./start.sh
```

This will:
1. Create/activate Python virtual environment
2. Install Python dependencies
3. Install frontend dependencies
4. Start the backend on port 8000
5. Start the frontend on port 3000

## Startup Options

### Development Mode (Default)
```bash
./start.sh
```
- Backend with hot reload
- Frontend with hot reload
- SQLite database
- Debug logging enabled

### Production Mode
```bash
./start.sh --prod
```
- Backend with multiple workers
- Frontend optimized build
- PostgreSQL database
- Production logging

### Custom Ports
```bash
# Change backend port
./start.sh --port 8080

# Change frontend port
./start.sh --frontend-port 3001

# Both
./start.sh --port 8080 --frontend-port 3001
```

## Manual Startup

### Backend Only

1. Create virtual environment:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export DATABASE_URL=sqlite:///./goldensignals.db
export SECRET_KEY=your-secret-key
export OPENAI_API_KEY=your-api-key  # If using OpenAI
```

4. Run the backend:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Only

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Set environment variables:
```bash
export VITE_API_URL=http://localhost:8000
```

3. Run the frontend:
```bash
npm run dev
```

## Environment Configuration

### Required Environment Variables

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/goldensignals
# For development: DATABASE_URL=sqlite:///./goldensignals.db

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-very-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Keys (Optional)
OPENAI_API_KEY=sk-...
POLYGON_API_KEY=your-polygon-key
ALPACA_API_KEY=your-alpaca-key
ALPACA_SECRET_KEY=your-alpaca-secret

# Trading Configuration
MAX_POSITION_SIZE=0.1  # 10% of portfolio
DEFAULT_STOP_LOSS=0.02  # 2%

# Environment
ENVIRONMENT=development  # or staging, production
```

### Frontend Environment Variables

Create `.env` in the frontend directory:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
VITE_ENABLE_MOCK_DATA=false
```

## Database Setup

### PostgreSQL (Production)

1. Install PostgreSQL:
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql
```

2. Create database:
```sql
CREATE DATABASE goldensignals;
CREATE USER goldensignals_user WITH PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE goldensignals TO goldensignals_user;
```

3. Run migrations:
```bash
cd backend
alembic upgrade head
```

### Redis Setup

1. Install Redis:
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis
```

2. Start Redis:
```bash
redis-server
```

## Stopping the Application

### Using the Script
```bash
./stop.sh
```

### Manual Stop
- Backend: `Ctrl+C` in the terminal running uvicorn
- Frontend: `Ctrl+C` in the terminal running npm
- Or find and kill the processes:
  ```bash
  lsof -ti:8000 | xargs kill -9  # Backend
  lsof -ti:3000 | xargs kill -9  # Frontend
  ```

## Accessing the Application

Once running, you can access:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### Dependencies Not Found
```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### Database Connection Error
- Check PostgreSQL is running: `sudo service postgresql status`
- Verify connection string in `.env`
- Check database exists: `psql -U postgres -c "\l"`

### Redis Connection Error
- Check Redis is running: `redis-cli ping`
- Verify Redis URL in `.env`

## Development Tools

### Running Tests
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Code Quality
```bash
# Backend
black .
isort .
flake8

# Frontend
npm run lint
npm run format
```

### Database Migrations
```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Production Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for production deployment instructions using Docker and Kubernetes.