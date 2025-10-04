#!/bin/bash

# KCET Recommendation System Startup Script

echo "🚀 Starting KCET College Recommendation System..."

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ and try again."
    exit 1
fi

# Check if model artifacts exist
if [ ! -f "artifacts/model_l2r.txt" ]; then
    echo "❌ Model artifacts not found. Please train the model first:"
    echo "   cd scripts"
    echo "   python kcet_l2r_train_infer.py train --data ../KCET_cleaned_with_2025.csv"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Install backend dependencies
echo "📦 Installing backend dependencies..."
cd backend
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✅ Created Python virtual environment"
fi

# Activate virtual environment (adjust for your OS)
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

pip install -r requirements.txt
echo "✅ Backend dependencies installed"

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd ../frontend
if [ ! -d "node_modules" ]; then
    npm install
    echo "✅ Frontend dependencies installed"
fi

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created frontend environment file"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the application:"
echo "1. Start backend:  cd backend && python main.py"
echo "2. Start frontend: cd frontend && npm run dev"
echo ""
echo "Then open http://localhost:3000 in your browser"