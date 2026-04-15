#!/bin/bash

# Medical RAG Application Launcher
# Usage: ./start_app.sh

echo "🚀 Starting Medical RAG Application..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Check if dependencies are installed
echo "📚 Checking dependencies..."
pip list | grep fastapi > /dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r app/requirements.txt
fi

# Check if PostgreSQL is running
echo "🗄️  Checking PostgreSQL connection..."
psql -h localhost -U leahashebir -d medrag -c "SELECT 1" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Cannot connect to PostgreSQL database 'medrag'"
    echo "   Please ensure PostgreSQL is running and the database exists."
    echo "   You can create it with: createdb medrag"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Display startup info
echo ""
echo "✅ Starting FastAPI server..."
echo ""
echo "📍 Application URLs:"
echo "   Main Interface:  http://localhost:8000"
echo "   API Docs:        http://localhost:8000/docs"
echo "   Health Check:    http://localhost:8000/health"
echo ""
echo "Press CTRL+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
