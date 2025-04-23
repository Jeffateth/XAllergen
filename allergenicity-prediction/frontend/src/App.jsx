// App.jsx
import { useState } from 'react';
import './App.css';

function App() {
  const [sequence, setSequence] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [exampleSequences] = useState([
    { name: 'Example 1: Beta-lactoglobulin (Known allergen)', 
      seq: 'MKCLLLALALTCGAQALIVTQTMKGLDIQKVAGTWYSLAMAASDISLLDAQSAPLRVYVEELKPTPEGDLEILLQKWENGECAQKKIIAEKTKIPAVFKIDALNENKVLVLDTDYKKYLLFCMENSAEPEQSLACQCLVRTPEVDDEALEKFDKALKALPMHIRLSFNPTQLEEQCHI' },
    { name: 'Example 2: Hemoglobin subunit alpha (Non-allergen)', 
      seq: 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR' }
  ]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!sequence.trim()) {
      setError('Please enter a protein sequence');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sequence }),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to make prediction');
      }
      
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (exampleSeq) => {
    setSequence(exampleSeq);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Protein Allergenicity Prediction</h1>
        <p className="subtitle">Powered by ESM-2 Transformer Model</p>
      </header>
      
      <main className="main-content">
        <section className="about-section">
          <h2>About this Tool</h2>
          <p>
            This web application uses a fine-tuned ESM-2 Transformer model to predict whether a protein sequence 
            is likely to be allergenic. ESM-2 is a state-of-the-art protein language model developed by Meta AI Research.
          </p>
          <p>
            Submit your protein sequence below to receive an allergenicity prediction.
          </p>
        </section>
        
        <section className="prediction-section">
          <h2>Prediction Tool</h2>
          <form onSubmit={handleSubmit} className="prediction-form">
            <div className="form-group">
              <label htmlFor="sequence">Protein Sequence (FASTA format or amino acid sequence):</label>
              <textarea
                id="sequence"
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                placeholder="Enter your protein sequence here (amino acid sequence using single-letter codes)"
                rows="6"
                required
              />
            </div>
            
            <div className="examples">
              <p>Or try one of our examples:</p>
              <div className="example-buttons">
                {exampleSequences.map((ex, index) => (
                  <button 
                    key={index} 
                    type="button" 
                    onClick={() => handleExampleClick(ex.seq)}
                    className="example-button"
                  >
                    {ex.name}
                  </button>
                ))}
              </div>
            </div>
            
            <button type="submit" className="submit-button" disabled={loading}>
              {loading ? 'Processing...' : 'Predict Allergenicity'}
            </button>
          </form>
          
          {error && <div className="error-message">{error}</div>}
          
          {result && (
            <div className={`result-card ${result.prediction.toLowerCase().includes('allergen') ? 'allergenic' : 'non-allergenic'}`}>
              <h3>Prediction Result</h3>
              <div className="result-content">
                <div className="result-item">
                  <span className="result-label">Prediction:</span>
                  <span className="result-value">{result.prediction}</span>
                </div>
                <div className="result-item">
                  <span className="result-label">Confidence:</span>
                  <span className="result-value">{result.confidence}%</span>
                </div>
                {result.allergenic_probability !== null && (
                  <div className="result-item">
                    <span className="result-label">Allergenic Probability:</span>
                    <span className="result-value">{result.allergenic_probability}%</span>
                  </div>
                )}
              </div>
              <div className="probability-bar">
                <div 
                  className="probability-fill" 
                  style={{ width: `${result.allergenic_probability || 0}%` }}
                ></div>
              </div>
              <div className="probability-labels">
                <span>0% (Non-allergenic)</span>
                <span>100% (Allergenic)</span>
              </div>
            </div>
          )}
        </section>
      </main>
      
      <footer className="footer">
        <p>© 2025 ESM-2 Allergenicity Prediction Tool. All rights reserved.</p>
        <p>
          This tool is provided for research purposes only. The predictions should not be used as the sole basis 
          for making decisions about allergenicity without appropriate expert review.
        </p>
      </footer>
    </div>
  );
}

export default App;