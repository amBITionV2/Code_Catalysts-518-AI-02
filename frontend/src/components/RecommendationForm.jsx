import React, { useState } from 'react';
import { Search, MapPin, Users, RotateCcw, AlertCircle } from 'lucide-react';
import LoadingSpinner from './LoadingSpinner';

const RecommendationForm = ({ categories, locations, onSubmit, onReset, loading }) => {
  const [formData, setFormData] = useState({
    rank: '',
    category: '',
    location: '',
    top_n: 10
  });

  const [errors, setErrors] = useState({});

  const validateForm = () => {
    const newErrors = {};

    // Validate rank
    const rank = parseInt(formData.rank);
    if (!formData.rank) {
      newErrors.rank = 'KCET rank is required';
    } else if (isNaN(rank) || rank < 1 || rank > 200000) {
      newErrors.rank = 'Please enter a valid rank between 1 and 200,000';
    }

    // Validate category
    if (!formData.category) {
      newErrors.category = 'Category is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateForm()) {
      onSubmit({
        ...formData,
        rank: parseInt(formData.rank),
        location: formData.location || undefined
      });
    }
  };

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  const handleReset = () => {
    setFormData({
      rank: '',
      category: '',
      location: '',
      top_n: 10
    });
    setErrors({});
    onReset();
  };

  return (
    <div className="card animate-slide-up">
      <div className="card-header">
        <h2 className="text-xl font-semibold text-gray-800 flex items-center">
          <Search className="h-6 w-6 mr-2 text-blue-600" />
          Get Recommendations
        </h2>
      </div>
      
      <div className="card-body">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* KCET Rank Input */}
          <div>
            <label htmlFor="rank" className="block text-sm font-medium text-gray-700 mb-2">
              Your KCET Rank *
            </label>
            <input
              type="number"
              id="rank"
              min="1"
              max="200000"
              placeholder="e.g., 42000"
              value={formData.rank}
              onChange={(e) => handleInputChange('rank', e.target.value)}
              className={`input-field ${errors.rank ? 'border-red-500 focus:ring-red-500' : ''}`}
              disabled={loading}
            />
            {errors.rank && (
              <div className="flex items-center mt-1 text-sm text-red-600">
                <AlertCircle className="h-4 w-4 mr-1" />
                {errors.rank}
              </div>
            )}
            <p className="text-xs text-gray-500 mt-1">
              Enter your KCET rank (between 1 and 200,000)
            </p>
          </div>

          {/* Category Selection */}
          <div>
            <label htmlFor="category" className="block text-sm font-medium text-gray-700 mb-2">
              <Users className="h-4 w-4 inline mr-1" />
              Category *
            </label>
            <select
              id="category"
              value={formData.category}
              onChange={(e) => handleInputChange('category', e.target.value)}
              className={`input-field ${errors.category ? 'border-red-500 focus:ring-red-500' : ''}`}
              disabled={loading}
            >
              <option value="">Select your category</option>
              {categories.map((cat) => (
                <option key={cat} value={cat}>
                  {cat}
                </option>
              ))}
            </select>
            {errors.category && (
              <div className="flex items-center mt-1 text-sm text-red-600">
                <AlertCircle className="h-4 w-4 mr-1" />
                {errors.category}
              </div>
            )}
            <p className="text-xs text-gray-500 mt-1">
              Select your reservation category
            </p>
          </div>

          {/* Location Preference */}
          <div>
            <label htmlFor="location" className="block text-sm font-medium text-gray-700 mb-2">
              <MapPin className="h-4 w-4 inline mr-1" />
              Location Preference (Optional)
            </label>
            <select
              id="location"
              value={formData.location}
              onChange={(e) => handleInputChange('location', e.target.value)}
              className="input-field"
              disabled={loading}
            >
              <option value="">All locations</option>
              {locations.map((loc) => (
                <option key={loc} value={loc}>
                  {loc}
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Filter colleges by preferred city/region
            </p>
          </div>

          {/* Number of Recommendations */}
          <div>
            <label htmlFor="top_n" className="block text-sm font-medium text-gray-700 mb-2">
              Number of Recommendations
            </label>
            <select
              id="top_n"
              value={formData.top_n}
              onChange={(e) => handleInputChange('top_n', parseInt(e.target.value))}
              className="input-field"
              disabled={loading}
            >
              <option value={5}>Top 5</option>
              <option value={10}>Top 10</option>
              <option value={15}>Top 15</option>
              <option value={20}>Top 20</option>
            </select>
          </div>

          {/* Action Buttons */}
          <div className="flex space-x-3">
            <button
              type="submit"
              disabled={loading}
              className="flex-1 btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {loading ? (
                <>
                  <LoadingSpinner size="small" color="white" />
                  <span className="ml-2">Getting Recommendations...</span>
                </>
              ) : (
                <>
                  <Search className="h-5 w-5 mr-2" />
                  Get Recommendations
                </>
              )}
            </button>
            
            <button
              type="button"
              onClick={handleReset}
              disabled={loading}
              className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
            >
              <RotateCcw className="h-4 w-4 mr-1" />
              Reset
            </button>
          </div>
        </form>

        {/* Help Text */}
        <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <h4 className="text-sm font-medium text-blue-800 mb-2">💡 How it works:</h4>
          <ul className="text-sm text-blue-700 space-y-1">
            <li>• Our AI analyzes 10+ years of KCET admission data</li>
            <li>• Predictions are based on historical cutoffs and trends</li>
            <li>• Results show admission probability and safety levels</li>
            <li>• Location filter helps find colleges near you</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default RecommendationForm;