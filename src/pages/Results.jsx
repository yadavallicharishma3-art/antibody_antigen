import { Link, useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import InteractionMatrix from "../components/InteractionMatrix";

function Results() {
  const navigate = useNavigate();

  const stored = sessionStorage.getItem("analysisResult");

  if (!stored) {
    navigate("/analysis");
    return null;
  }

  const result = JSON.parse(stored);

  const percentage = Math.round(result.probability * 100);

  return (
    <div className="app">
      <Navbar />

      <main className="results-page">
        <div className="results-header">
          <div>
            <p className="eyebrow">ANALYSIS COMPLETE</p>
            <h1>Analysis Results</h1>
            <p>
              Structural interaction analysis using the project demonstration
              pipeline.
            </p>
          </div>

          <div className="demo-badge">DEMO MODE</div>
        </div>

        {/* PREDICTION */}
        <section className="prediction-card">
          <div className="prediction-icon">✓</div>

          <div className="prediction-content">
            <span>MODEL PREDICTION</span>
            <h2>Antibody–Antigen Interaction Detected</h2>
            <p>
              The demonstration model predicts a pattern associated with an
              antibody–antigen interaction.
            </p>
          </div>

          <div className="confidence">
            <span>Probability</span>
            <strong>{percentage}%</strong>
          </div>
        </section>

        {/* INFO */}
        <section className="result-grid">
          <div className="info-card">
            <span>PDB ID</span>
            <strong>{result.pdb_id}</strong>
          </div>

          <div className="info-card">
            <span>ANTIBODY CHAIN</span>
            <strong>{result.antibody_chain}</strong>
          </div>

          <div className="info-card">
            <span>ANTIGEN CHAIN</span>
            <strong>{result.antigen_chain}</strong>
          </div>

          <div className="info-card">
            <span>CDR RESIDUES</span>
            <strong>{result.cdr_residues.join(", ")}</strong>
          </div>
        </section>

        {/* MATRIX */}
        <section className="matrix-section">
          <div className="section-heading left">
            <p className="eyebrow">STRUCTURAL FEATURES</p>
            <h2>20 × 20 Interaction Matrix</h2>
            <p>
              A numerical representation of amino-acid interaction information
              at the binding interface.
            </p>
          </div>

          <InteractionMatrix matrix={result.matrix} />
        </section>

        {/* EXPLANATION */}
        <section className="explanation">
          <div className="explanation-number">?</div>

          <div>
            <p className="eyebrow">INTERPRETING THE MATRIX</p>

            <h2>What does this matrix represent?</h2>

            <p>
              The matrix represents interaction information between amino acids
              at the antibody–antigen binding interface. Rows and columns
              correspond to the 20 standard amino acids.
            </p>

            <p>
              The CNN receives this numerical representation and learns
              patterns associated with antibody–antigen interactions.
            </p>
          </div>
        </section>

        <div className="results-actions">
          <Link to="/analysis" className="primary-button">
            Analyze Another Structure →
          </Link>
        </div>
      </main>

      <footer>
        <p>Antibody ML • Project School Research Project</p>
      </footer>
    </div>
  );
}

export default Results;