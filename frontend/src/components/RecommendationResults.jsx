import React from 'react';
import { 
  Award, 
  MapPin, 
  TrendingUp, 
  Users, 
  CheckCircle, 
  AlertTriangle, 
  XCircle,
  Info,
  Star,
  Target
} from 'lucide-react';

const RecommendationResults = ({ data }) => {
  if (!data || !data.recommendations || data.recommendations.length === 0) {
    return (
      <div className="card animate-fade-in">
        <div className="card-body text-center">
          <XCircle className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-700 mb-2">No Recommendations Found</h3>
          <p className="text-gray-600">
            {data?.message || "No colleges match your criteria. Try adjusting your preferences."}
          </p>
        </div>
      </div>
    );
  }

  const getChanceBadgeStyle = (chance) => {
    switch (chance.toLowerCase()) {
      case 'safe':
        return 'badge badge-success';
      case 'good':
        return 'badge badge-primary';
      case 'match':
        return 'badge badge-warning';
      case 'reach':
        return 'badge badge-danger';
      default:
        return 'badge';
    }
  };

  const getChanceIcon = (chance) => {
    switch (chance.toLowerCase()) {
      case 'safe':
        return <CheckCircle className="h-4 w-4" />;
      case 'good':
        return <TrendingUp className="h-4 w-4" />;
      case 'match':
        return <Target className="h-4 w-4" />;
      case 'reach':
        return <AlertTriangle className="h-4 w-4" />;
      default:
        return <Info className="h-4 w-4" />;
    }
  };

  const getAdmissionColor = (probability) => {
    if (probability >= 80) return 'text-green-600';
    if (probability >= 60) return 'text-blue-600';
    if (probability >= 40) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header with User Info */}
      <div className="card">
        <div className="card-body">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold text-gray-800">
              Your College Recommendations
            </h2>
            <div className="text-sm text-gray-600">
              Found {data.recommendations.length} matches
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{data.user_info.rank}</div>
              <div className="text-sm text-gray-600">Your Rank</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-semibold text-gray-700">{data.user_info.category}</div>
              <div className="text-sm text-gray-600">Category</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-semibold text-gray-700">{data.user_info.location}</div>
              <div className="text-sm text-gray-600">Location</div>
            </div>
          </div>
        </div>
      </div>

      {/* Top 3 Detailed Recommendations */}
      {data.top_3_details && data.top_3_details.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="text-xl font-semibold text-gray-800 flex items-center">
              <Star className="h-6 w-6 mr-2 text-yellow-500" />
              Top 3 Recommendations
            </h3>
          </div>
          <div className="card-body">
            <div className="space-y-4">
              {data.top_3_details.map((rec, index) => (
                <div
                  key={`top-${index}`}
                  className="p-6 border border-gray-200 rounded-lg bg-gradient-to-r from-blue-50 to-indigo-50 hover:shadow-md transition-shadow"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <div className="flex items-center mb-2">
                        <div className="bg-blue-600 text-white rounded-full w-8 h-8 flex items-center justify-center text-sm font-bold mr-3">
                          {index + 1}
                        </div>
                        <h4 className="text-lg font-semibold text-gray-800 line-clamp-2">
                          {rec.college}
                        </h4>
                      </div>
                      <p className="text-md text-gray-700 font-medium mb-2">{rec.branch}</p>
                    </div>
                    <div className={getChanceBadgeStyle(rec.chance)}>
                      {getChanceIcon(rec.chance)}
                      <span className="ml-1">{rec.chance}</span>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div className="text-center p-3 bg-white rounded-lg">
                      <div className={`text-xl font-bold ${getAdmissionColor(rec.admission_probability)}`}>
                        {rec.admission_probability.toFixed(1)}%
                      </div>
                      <div className="text-gray-600">Admission Chance</div>
                    </div>
                    <div className="text-center p-3 bg-white rounded-lg">
                      <div className="text-xl font-bold text-gray-700">
                        {rec.last_year_cutoff.toLocaleString()}
                      </div>
                      <div className="text-gray-600">Last Year Cutoff</div>
                    </div>
                    <div className="text-center p-3 bg-white rounded-lg">
                      <div className={`text-xl font-bold ${rec.rank_advantage_percent > 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {rec.rank_advantage_percent > 0 ? '+' : ''}{rec.rank_advantage_percent.toFixed(1)}%
                      </div>
                      <div className="text-gray-600">Rank Advantage</div>
                    </div>
                    <div className="text-center p-3 bg-white rounded-lg">
                      <div className="text-xl font-bold text-blue-600">
                        {rec.accuracy.toFixed(1)}%
                      </div>
                      <div className="text-gray-600">Confidence</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* All Recommendations Table */}
      <div className="card">
        <div className="card-header">
          <h3 className="text-xl font-semibold text-gray-800 flex items-center">
            <Award className="h-6 w-6 mr-2 text-blue-600" />
            All Recommendations
          </h3>
        </div>
        <div className="card-body p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">#</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">College</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Branch</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-700">Admission %</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-700">Last Cutoff</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-700">Advantage</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-700">Type</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-700">Data Points</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {data.recommendations.map((rec, index) => (
                  <tr 
                    key={index} 
                    className="hover:bg-gray-50 transition-colors"
                  >
                    <td className="py-3 px-4 text-sm font-medium text-gray-900">
                      {index + 1}
                    </td>
                    <td className="py-3 px-4">
                      <div className="text-sm font-medium text-gray-900 max-w-xs truncate" title={rec.college}>
                        {rec.college}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="text-sm text-gray-700 max-w-xs truncate" title={rec.branch}>
                        {rec.branch}
                      </div>
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span className={`text-sm font-semibold ${getAdmissionColor(rec.admission_probability)}`}>
                        {rec.admission_probability.toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3 px-4 text-center text-sm text-gray-700">
                      {rec.last_year_cutoff.toLocaleString()}
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span className={`text-sm font-medium ${rec.rank_advantage_percent > 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {rec.rank_advantage_percent > 0 ? '+' : ''}{rec.rank_advantage_percent.toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span className={getChanceBadgeStyle(rec.chance)}>
                        {rec.chance}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-center text-sm text-gray-600">
                      {rec.data_points}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="card">
        <div className="card-body">
          <h4 className="text-lg font-semibold text-gray-800 mb-4">Understanding Your Results</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h5 className="font-medium text-gray-700 mb-2">Admission Categories:</h5>
              <div className="space-y-2 text-sm">
                <div className="flex items-center">
                  <span className="badge badge-success mr-2">Safe</span>
                  <span className="text-gray-600">Very high chance of admission (80%+)</span>
                </div>
                <div className="flex items-center">
                  <span className="badge badge-primary mr-2">Good</span>
                  <span className="text-gray-600">Good chance of admission (60-80%)</span>
                </div>
                <div className="flex items-center">
                  <span className="badge badge-warning mr-2">Match</span>
                  <span className="text-gray-600">Moderate chance of admission (40-60%)</span>
                </div>
                <div className="flex items-center">
                  <span className="badge badge-danger mr-2">Reach</span>
                  <span className="text-gray-600">Lower chance, but worth trying (&lt;40%)</span>
                </div>
              </div>
            </div>
            <div>
              <h5 className="font-medium text-gray-700 mb-2">Key Metrics:</h5>
              <div className="space-y-2 text-sm text-gray-600">
                <div><strong>Admission %:</strong> Predicted probability of getting admission</div>
                <div><strong>Last Cutoff:</strong> Previous year's closing rank for this course</div>
                <div><strong>Rank Advantage:</strong> How much better/worse your rank is vs last cutoff</div>
                <div><strong>Data Points:</strong> Number of historical records used for prediction</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecommendationResults;