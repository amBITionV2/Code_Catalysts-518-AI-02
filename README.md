# KCET College Recommendation System

A modern web application that provides AI-powered college recommendations for KCET (Karnataka Common Entrance Test) students based on their rank, category, and location preferences.

## 🚀 Features

- **AI-Powered Predictions**: Uses LightGBM LambdaMART model trained on 10+ years of historical data
- **User-Friendly Interface**: Clean, intuitive React frontend with responsive design
- **Personalized Recommendations**: Get top 10 college suggestions with admission probabilities
- **Detailed Analytics**: View top 3 recommendations with comprehensive metrics
- **Location Filtering**: Filter colleges by preferred cities/regions
- **Safety Categories**: Colleges categorized as Safe, Good, Match, or Reach
- **Real-time Predictions**: Fast API responses with confidence scores

## 🏗️ Architecture

- **Frontend**: React + Vite + Tailwind CSS
- **Backend**: FastAPI + Python
- **ML Model**: LightGBM LambdaMART (Learning-to-Rank)
- **Data**: 10+ years of KCET admission data (2015-2025)

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd ambition
```

### 2. Set up Backend
```bash
cd backend
pip install -r requirements.txt
```

### 3. Set up Frontend
```bash
cd ../frontend
npm install
```

## 🚀 Running the Application

### 1. Start the Backend (Terminal 1)
```bash
cd backend
python main.py
```
The API will be available at `http://localhost:8000`

### 2. Start the Frontend (Terminal 2)
```bash
cd frontend
npm run dev
```
The web app will be available at `http://localhost:3000`

## 📊 Training the Model

If you need to retrain the model with new data:

```bash
cd scripts
python kcet_l2r_train_infer.py train --data ../KCET_cleaned_with_2025.csv
```

This will create the necessary model artifacts in the `artifacts/` directory.

## 🌐 API Endpoints

- `GET /health` - Check system health
- `GET /categories` - Get available categories
- `GET /locations` - Get available locations
- `POST /recommend` - Get college recommendations
- `GET /stats` - Get dataset statistics

## 🎯 Usage

1. Open the web application in your browser
2. Enter your KCET rank (1-200,000)
3. Select your category (GM, 1G, 2AK, etc.)
4. Optionally select a preferred location
5. Choose number of recommendations (5-20)
6. Click "Get Recommendations"

The system will show:
- **Top 3 detailed recommendations** with comprehensive metrics
- **Full list** of all recommendations in a sortable table
- **Admission probabilities** and safety categories
- **Confidence scores** based on historical data

## 📁 Project Structure

```
ambition/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main API application
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API service layer
│   │   └── App.jsx         # Main app component
│   ├── package.json        # Node.js dependencies
│   └── vite.config.js      # Vite configuration
├── scripts/                # Original ML scripts
│   ├── kcet_l2r_train_infer.py
│   ├── merging-data.py
│   └── append_2025.py
├── artifacts/              # Model artifacts
│   ├── model_l2r.txt       # Trained model
│   ├── items.parquet       # Processed data
│   └── feature_meta.json   # Model metadata
└── datasets/               # Raw data files
```

## 🚢 Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Manual Deployment
1. Deploy backend to any Python hosting service (Railway, Render, etc.)
2. Deploy frontend to static hosting (Vercel, Netlify, etc.)
3. Update API URL in frontend environment variables

## 🔧 Environment Variables

### Frontend (.env)
```
VITE_API_URL=http://localhost:8000
```

### Backend
```
PORT=8000
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

## 📈 Model Performance

- **NDCG@5**: 87.6% (Test set)
- **NDCG@10**: 86.1% (Test set)
- **Training Data**: 144K+ queries
- **Validation Data**: 24K+ queries
- **Test Data**: 53K+ queries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

If you encounter any issues:
1. Check that the backend is running on port 8000
2. Ensure the model artifacts exist in the `artifacts/` directory
3. Verify that the frontend can connect to the backend
4. Check browser console for any errors

For additional support, please open an issue on GitHub.

## 🙏 Acknowledgments

- Karnataka Examinations Authority for KCET data
- LightGBM team for the excellent ML framework
- React and FastAPI communities for great documentation