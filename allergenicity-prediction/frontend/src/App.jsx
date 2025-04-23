import { useState } from 'react';
import './App.css';

function App() {
  const [sequence, setSequence] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction failed');
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1>Allergenicity Predictor</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={sequence}
          onChange={e => setSequence(e.target.value)}
          placeholder="Enter protein sequence (ACDEFGHIK...)"
          rows={6}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </form>
      {error && <div className="error">{error}</div>}
      {result && (
        <div className="result">
          <p>Prediction: <strong>{result.prediction}</strong></p>
          <p>Confidence: {result.confidence}%</p>
          {result.allergenic_probability !== null && (
            <p>Allergenic Probability: {result.allergenic_probability}%</p>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
