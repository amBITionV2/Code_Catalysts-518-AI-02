import React, { useState, useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import Header from './components/Header';
import RecommendationForm from './components/RecommendationForm';
import RecommendationResults from './components/RecommendationResults';
import LoadingSpinner from './components/LoadingSpinner';
import { apiService } from './services/api';
import toast from 'react-hot-toast';

function App() {
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(false);
  const [categories, setCategories] = useState([]);
  const [locations, setLocations] = useState([]);
  const [appReady, setAppReady] = useState(false);

  // Load initial data on component mount
  useEffect(() => {
    const initializeApp = async () => {
      try {
        setLoading(true);
        
        // Check if backend is healthy
        const health = await apiService.checkHealth();
        if (!health.model_loaded || !health.data_loaded) {
          toast.error('Backend is not ready. Please ensure the model is trained and data is loaded.');
          return;
        }

        // Load categories and locations
        const [categoriesData, locationsData] = await Promise.all([
          apiService.getCategories(),
          apiService.getLocations()
        ]);

        setCategories(categoriesData);
        setLocations(locationsData);
        setAppReady(true);
        
        toast.success('App initialized successfully!');
      } catch (error) {
        console.error('Failed to initialize app:', error);
        toast.error('Failed to connect to backend. Please ensure the API server is running.');
      } finally {
        setLoading(false);
      }
    };

    initializeApp();
  }, []);

  const handleGetRecommendations = async (formData) => {
    try {
      setLoading(true);
      setRecommendations(null);

      const result = await apiService.getRecommendations(formData);
      setRecommendations(result);

      if (result.recommendations.length === 0) {
        toast.error(result.message || 'No recommendations found for your criteria.');
      } else {
        toast.success(`Found ${result.recommendations.length} recommendations!`);
      }
    } catch (error) {
      console.error('Failed to get recommendations:', error);
      toast.error(
        error.response?.data?.detail || 
        'Failed to get recommendations. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setRecommendations(null);
  };

  if (!appReady && loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner size="large" />
          <p className="mt-4 text-gray-600 text-lg">Initializing KCET Recommendation System...</p>
        </div>
      </div>
    );
  }

  if (!appReady) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-red-50 to-red-100 flex items-center justify-center">
        <div className="text-center p-8 bg-white rounded-xl shadow-lg max-w-md">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">System Not Ready</h2>
          <p className="text-gray-600 mb-4">
            The backend system is not available. Please ensure:
          </p>
          <ul className="text-left text-sm text-gray-600 mb-4">
            <li>• Backend server is running on port 8000</li>
            <li>• Model has been trained</li>
            <li>• Data files are available</li>
          </ul>
          <button 
            onClick={() => window.location.reload()} 
            className="btn-primary"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            duration: 3000,
            iconTheme: {
              primary: '#22c55e',
              secondary: '#fff',
            },
          },
          error: {
            duration: 5000,
            iconTheme: {
              primary: '#ef4444',
              secondary: '#fff',
            },
          },
        }}
      />
      
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto">
          {/* Introduction Section */}
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-800 mb-4 animate-fade-in">
              Find Your Perfect Engineering College
            </h1>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto animate-slide-up">
              Get AI-powered college recommendations based on your KCET rank, category, and location preferences. 
              Our system analyzes 10+ years of admission data to predict your best options.
            </p>
          </div>

          {/* Stats Row */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-white rounded-lg shadow-md p-6 text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">{categories.length}</div>
              <div className="text-gray-600">Categories Available</div>
            </div>
            <div className="bg-white rounded-lg shadow-md p-6 text-center">
              <div className="text-3xl font-bold text-green-600 mb-2">{locations.length}</div>
              <div className="text-gray-600">Cities Covered</div>
            </div>
            <div className="bg-white rounded-lg shadow-md p-6 text-center">
              <div className="text-3xl font-bold text-purple-600 mb-2">10+</div>
              <div className="text-gray-600">Years of Data</div>
            </div>
          </div>

          {/* Main Content */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Form Section */}
            <div className="lg:col-span-1">
              <RecommendationForm
                categories={categories}
                locations={locations}
                onSubmit={handleGetRecommendations}
                onReset={handleReset}
                loading={loading}
              />
            </div>

            {/* Results Section */}
            <div className="lg:col-span-2">
              {loading ? (
                <div className="bg-white rounded-xl shadow-lg p-8 text-center">
                  <LoadingSpinner size="large" />
                  <p className="mt-4 text-gray-600 text-lg">
                    Analyzing your preferences and generating recommendations...
                  </p>
                </div>
              ) : recommendations ? (
                <RecommendationResults data={recommendations} />
              ) : (
                <div className="bg-white rounded-xl shadow-lg p-8 text-center">
                  <div className="text-gray-400 text-6xl mb-4">🎓</div>
                  <h3 className="text-xl font-semibold text-gray-700 mb-2">
                    Ready to Find Your College?
                  </h3>
                  <p className="text-gray-600">
                    Fill out the form on the left to get personalized college recommendations 
                    based on your KCET rank and preferences.
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Footer */}
          <footer className="mt-12 text-center text-gray-600 text-sm">
            <p>
              © 2025 KCET College Recommendation System. 
              Built with AI to help students make informed decisions.
            </p>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;