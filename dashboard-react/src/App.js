import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Settings, TrendingUp, AlertCircle, CheckCircle, Activity } from 'lucide-react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [mixParams, setMixParams] = useState({
    cement: 300,
    blast_furnace_slag: 50,
    fly_ash: 0,
    water: 180,
    superplasticizer: 5,
    coarse_aggregate: 1000,
    fine_aggregate: 750,
    age: 28
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');
  const [ageStrengthData, setAgeStrengthData] = useState([]);
  const [wcRatioData, setWcRatioData] = useState([]);

  // Check API status on mount
  useEffect(() => {
    checkAPIStatus();
  }, []);

  // Update prediction when params change
  useEffect(() => {
    const debounce = setTimeout(() => {
      updatePrediction();
      updateCharts();
    }, 300);

    return () => clearTimeout(debounce);
  }, [mixParams]);

  const checkAPIStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/health`);
      if (response.status === 200) {
        setApiStatus('online');
      }
    } catch (error) {
      setApiStatus('offline');
      console.error('API is offline:', error);
    }
  };

  const updatePrediction = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/predict`, mixParams);
      setPrediction(response.data);
    } catch (error) {
      console.error('Prediction error:', error);
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const updateCharts = async () => {
    // Age vs Strength data
    const ages = [1, 3, 7, 14, 28, 56, 90, 180, 365];
    const ageData = [];

    for (const age of ages) {
      try {
        const response = await axios.post(`${API_URL}/predict`, { ...mixParams, age });
        ageData.push({
          age,
          strength: parseFloat(response.data.predicted_strength)
        });
      } catch (error) {
        console.error('Error fetching age data:', error);
      }
    }
    setAgeStrengthData(ageData);

    // W/C Ratio data
    const ratios = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7];
    const wcData = [];

    for (const ratio of ratios) {
      try {
        const response = await axios.post(`${API_URL}/predict`, {
          ...mixParams,
          water: mixParams.cement * ratio
        });
        wcData.push({
          ratio: ratio.toFixed(2),
          strength: parseFloat(response.data.predicted_strength)
        });
      } catch (error) {
        console.error('Error fetching W/C data:', error);
      }
    }
    setWcRatioData(wcData);
  };

  const handleSliderChange = (param, value) => {
    setMixParams(prev => ({
      ...prev,
      [param]: parseFloat(value)
    }));
  };

  const resetToDefault = () => {
    setMixParams({
      cement: 300,
      blast_furnace_slag: 50,
      fly_ash: 0,
      water: 180,
      superplasticizer: 5,
      coarse_aggregate: 1000,
      fine_aggregate: 750,
      age: 28
    });
  };

  const sliders = [
    { label: 'Cement', key: 'cement', min: 100, max: 500, step: 10, unit: 'kg/m³' },
    { label: 'Blast Furnace Slag', key: 'blast_furnace_slag', min: 0, max: 200, step: 10, unit: 'kg/m³' },
    { label: 'Fly Ash', key: 'fly_ash', min: 0, max: 150, step: 10, unit: 'kg/m³' },
    { label: 'Water', key: 'water', min: 120, max: 250, step: 5, unit: 'kg/m³' },
    { label: 'Superplasticizer', key: 'superplasticizer', min: 0, max: 15, step: 0.5, unit: 'kg/m³' },
    { label: 'Coarse Aggregate', key: 'coarse_aggregate', min: 800, max: 1200, step: 20, unit: 'kg/m³' },
    { label: 'Fine Aggregate', key: 'fine_aggregate', min: 600, max: 900, step: 20, unit: 'kg/m³' },
    { label: 'Curing Age', key: 'age', min: 1, max: 365, step: 1, unit: 'days' }
  ];

  const radarData = [
    { property: 'Cement', value: (mixParams.cement / 500) * 100, fullMark: 100 },
    { property: 'W/C Ratio', value: ((1 - (mixParams.water / mixParams.cement)) / 0.7) * 100, fullMark: 100 },
    { property: 'Admixtures', value: (mixParams.superplasticizer / 15) * 100, fullMark: 100 },
    { property: 'SCM', value: ((mixParams.blast_furnace_slag + mixParams.fly_ash) / 200) * 100, fullMark: 100 },
    { property: 'Age Factor', value: (Math.min(mixParams.age, 90) / 90) * 100, fullMark: 100 },
  ];

  const componentData = [
    { component: 'Cement', value: mixParams.cement, optimal: 350 },
    { component: 'Slag', value: mixParams.blast_furnace_slag, optimal: 80 },
    { component: 'Fly Ash', value: mixParams.fly_ash, optimal: 40 },
    { component: 'Water', value: mixParams.water, optimal: 160 },
    { component: 'SP', value: mixParams.superplasticizer, optimal: 8 },
  ];

  const getStatusColor = () => {
    switch (apiStatus) {
      case 'online': return 'text-green-500';
      case 'offline': return 'text-red-500';
      default: return 'text-yellow-500';
    }
  };

  const getStatusText = () => {
    switch (apiStatus) {
      case 'online': return '● API Online';
      case 'offline': return '● API Offline';
      default: return '⏳ Checking...';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-800 flex items-center gap-3">
                <Activity className="text-blue-600" size={36} />
                Système de prédiction de la qualité du béton
              </h1>
              <p className="text-gray-600 mt-2">Tableau de bord interactif de conception de mélange et de prédiction de force</p>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500">Modèle: XGBoost v1.0</div>
              <div className="text-sm text-gray-500">Accuracy: R² = 0.92</div>
              <div className={`text-sm font-semibold mt-2 ${getStatusColor()}`}>
                {getStatusText()}
              </div>
            </div>
          </div>
        </div>

        {/* Main Prediction Card */}
        {prediction && !loading && (
          <div className="bg-gradient-to-r from-blue-500 to-blue-600 rounded-xl shadow-xl p-8 mb-6 text-white">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-sm opacity-90 mb-2">Résistance à la compression prévue</div>
                <div className="text-5xl font-bold">{prediction.predicted_strength}</div>
                <div className="text-xl mt-1">MPa</div>
                <div className="text-sm opacity-75 mt-2">
                  Range: {prediction.confidence_interval_lower} - {prediction.confidence_interval_upper} MPa
                </div>
              </div>
              
              <div className="text-center border-l border-r border-blue-400 px-4">
                <div className="text-sm opacity-90 mb-2">Classification de qualité</div>
                <div className="text-3xl font-bold mt-4">{prediction.quality_assessment.split('-')[0].trim()}</div>
                <div className="flex items-center justify-center mt-3">
                  {parseFloat(prediction.predicted_strength) >= 30 ? (
                    <CheckCircle size={24} className="mr-2" />
                  ) : (
                    <AlertCircle size={24} className="mr-2" />
                  )}
                  <span className="text-sm">
                    {parseFloat(prediction.predicted_strength) >= 30 ? 'Structural Grade' : 'Non-Structural'}
                  </span>
                </div>
              </div>
              
              <div className="text-center">
                <div className="text-sm opacity-90 mb-2">Rapport eau-ciment</div>
                <div className="text-4xl font-bold mt-4">
                  {(mixParams.water / mixParams.cement).toFixed(3)}
                </div>
                <div className="text-sm opacity-75 mt-3">
                  {(mixParams.water / mixParams.cement) <= 0.5 ? '✓ Optimal' : '⚠ Review recommended'}
                </div>
              </div>
            </div>
          </div>
        )}

        {loading && (
          <div className="bg-blue-500 rounded-xl shadow-xl p-8 mb-6 text-white text-center">
            <div className="text-2xl">⏳ Calcul de la prédiction...</div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Mix Design Controls */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Settings size={24} className="text-blue-600" />
              Paramètres de conception du mélange
            </h2>
            <div className="space-y-4">
              {sliders.map(slider => (
                <div key={slider.key}>
                  <div className="flex justify-between mb-1">
                    <label className="text-sm font-medium text-gray-700">{slider.label}</label>
                    <span className="text-sm font-semibold text-blue-600">
                      {mixParams[slider.key]} {slider.unit}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={slider.min}
                    max={slider.max}
                    step={slider.step}
                    value={mixParams[slider.key]}
                    onChange={(e) => handleSliderChange(slider.key, e.target.value)}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>{slider.min}</span>
                    <span>{slider.max}</span>
                  </div>
                </div>
              ))}
            </div>
            <button
              onClick={resetToDefault}
              className="mt-4 w-full bg-gray-600 text-white py-2 px-4 rounded-lg hover:bg-gray-700 transition"
            >
              Réinitialiser au mixage par défaut
            </button>
          </div>

          {/* Mix Composition Radar */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Analyse de la composition du mélange</h2>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#e5e7eb" />
                <PolarAngleAxis dataKey="property" tick={{ fill: '#6b7280', fontSize: 12 }} />
                <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#6b7280' }} />
                <Radar name="Current Mix" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
              </RadarChart>
            </ResponsiveContainer>
            <div className="mt-4 text-sm text-gray-600 text-center">
              Conception de mélange équilibré sur des paramètres clés
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Age vs Strength Chart */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <TrendingUp size={24} className="text-green-600" />
              Développement de la force au fil du temps
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={ageStrengthData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis 
                  dataKey="age" 
                  label={{ value: 'Age (days)', position: 'insideBottom', offset: -5 }}
                  tick={{ fill: '#6b7280' }}
                />
                <YAxis 
                  label={{ value: 'Strength (MPa)', angle: -90, position: 'insideLeft' }}
                  tick={{ fill: '#6b7280' }}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb' }}
                  labelFormatter={(value) => `Age: ${value} days`}
                />
                <Line 
                  type="monotone" 
                  dataKey="strength" 
                  stroke="#3b82f6" 
                  strokeWidth={3}
                  dot={{ fill: '#3b82f6', r: 4 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Water-Cement Ratio Impact */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Impact du rapport eau-ciment</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={wcRatioData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis 
                  dataKey="ratio" 
                  label={{ value: 'W/C Ratio', position: 'insideBottom', offset: -5 }}
                  tick={{ fill: '#6b7280' }}
                />
                <YAxis 
                  label={{ value: 'Strength (MPa)', angle: -90, position: 'insideLeft' }}
                  tick={{ fill: '#6b7280' }}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="strength" 
                  stroke="#ef4444" 
                  strokeWidth={3}
                  dot={{ fill: '#ef4444', r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
            <div className="mt-4 text-sm text-gray-600 text-center">
              Un rapport E/C plus faible produit généralement une résistance plus élevée
            </div>
          </div>

          {/* Component Comparison */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Composants mixtes vs Optimal</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={componentData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="component" tick={{ fill: '#6b7280' }} />
                <YAxis tick={{ fill: '#6b7280' }} />
                <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb' }} />
                <Legend />
                <Bar dataKey="value" fill="#3b82f6" name="Current" />
                <Bar dataKey="optimal" fill="#10b981" name="Optimal" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Quick Stats */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Statistiques de mixage</h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
                <span className="text-gray-700">Total Cimentaire</span>
                <span className="font-bold text-blue-600">
                  {(mixParams.cement + mixParams.blast_furnace_slag + mixParams.fly_ash).toFixed(0)} kg/m³
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
                <span className="text-gray-700">Total des granulats</span>
                <span className="font-bold text-green-600">
                  {(mixParams.coarse_aggregate + mixParams.fine_aggregate).toFixed(0)} kg/m³
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-purple-50 rounded-lg">
                <span className="text-gray-700">Remplacement du SCM</span>
                <span className="font-bold text-purple-600">
                  {(((mixParams.blast_furnace_slag + mixParams.fly_ash) / (mixParams.cement + mixParams.blast_furnace_slag + mixParams.fly_ash)) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-orange-50 rounded-lg">
                <span className="text-gray-700">Rapport fin/grossier</span>
                <span className="font-bold text-orange-600">
                  {(mixParams.fine_aggregate / mixParams.coarse_aggregate).toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-6 text-center text-sm text-gray-600">
          <p>🇲🇦 Développé pour le secteur de la construction marocain | Propulsé par Azure ML et XGBoost</p>
          <p className="mt-1">© 2024 NAAIMA BAKRIM. Tous droits réservés.</p>
        </div>
      </div>
    </div>
  );
}

export default App;