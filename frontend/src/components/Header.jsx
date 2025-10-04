import React from 'react';
import { GraduationCap, Target, TrendingUp } from 'lucide-react';

const Header = () => {
  return (
    <header className="bg-white shadow-lg border-b border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-blue-600 p-2 rounded-lg">
              <GraduationCap className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-800">KCET Advisor</h1>
              <p className="text-sm text-gray-600">College Recommendation System</p>
            </div>
          </div>
          
          <div className="hidden md:flex items-center space-x-6">
            <div className="flex items-center space-x-2 text-gray-600">
              <Target className="h-5 w-5 text-blue-500" />
              <span className="text-sm font-medium">AI-Powered Predictions</span>
            </div>
            <div className="flex items-center space-x-2 text-gray-600">
              <TrendingUp className="h-5 w-5 text-green-500" />
              <span className="text-sm font-medium">10+ Years Data</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;