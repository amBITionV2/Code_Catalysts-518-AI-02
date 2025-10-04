@echo off
REM KCET Recommendation System Startup Script for Windows

echo 🚀 Starting KCET College Recommendation System...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js 16+ and try again.
    pause
    exit /b 1
)

REM Check if model artifacts exist
if not exist "artifacts\model_l2r.txt" (
    echo ❌ Model artifacts not found. Please train the model first:
    echo    cd scripts
    echo    python kcet_l2r_train_infer.py train --data ..\KCET_cleaned_with_2025.csv
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed

REM Install backend dependencies
echo 📦 Installing backend dependencies...
cd backend
if not exist "venv" (
    python -m venv venv
    echo ✅ Created Python virtual environment
)

call venv\Scripts\activate.bat
pip install -r requirements.txt
echo ✅ Backend dependencies installed

REM Install frontend dependencies
echo 📦 Installing frontend dependencies...
cd ..\frontend
if not exist "node_modules" (
    npm install
    echo ✅ Frontend dependencies installed
)

REM Create environment file if it doesn't exist
if not exist ".env" (
    copy .env.example .env
    echo ✅ Created frontend environment file
)

echo.
echo 🎉 Setup complete!
echo.
echo To start the application:
echo 1. Start backend:  cd backend ^&^& python main.py
echo 2. Start frontend: cd frontend ^&^& npm run dev
echo.
echo Then open http://localhost:3000 in your browser
pause