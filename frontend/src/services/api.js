import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API functions
export const apiService = {
  // Health check
  checkHealth: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  // Get available categories
  getCategories: async () => {
    const response = await api.get('/categories');
    return response.data;
  },

  // Get available locations
  getLocations: async () => {
    const response = await api.get('/locations');
    return response.data;
  },

  // Get recommendations
  getRecommendations: async (requestData) => {
    const response = await api.post('/recommend', requestData);
    return response.data;
  },

  // Get statistics
  getStatistics: async () => {
    const response = await api.get('/stats');
    return response.data;
  },
};

export default api;