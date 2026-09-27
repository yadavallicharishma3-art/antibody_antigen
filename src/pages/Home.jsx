import { Link } from "react-router-dom";
import Navbar from "../components/Navbar";

function Home() {
  return (
    <div className="app">
      <Navbar />

      <main>
        {/* HERO */}
        <section className="hero">
          <div className="hero-text">
            <p className="eyebrow">MACHINE LEARNING × STRUCTURAL BIOLOGY</p>

            <h1>
              Antibody–Antigen
              <span> Interaction Analyzer</span>
            </h1>

            <p className="hero-description">
              Explore structural patterns at antibody–antigen binding
              interfaces using machine learning.
            </p>

            <div className="hero-buttons">
              <Link to="/analysis" className="primary-button">
                Analyze Structure →
              </Link>

              <a href="#how-it-works" className="secondary-button">
                How It Works
              </a>
            </div>
          </div>

          <div className="hero-visual">
            <div className="molecule">
              <div className="molecule-center">Ab</div>
              <span className="dot dot1"></span>
              <span className="dot dot2"></span>
              <span className="dot dot3"></span>
              <span className="dot dot4"></span>
              <span className="dot dot5"></span>
            </div>
            <p>Structural Interaction Analysis</p>
          </div>
        </section>

        {/* PIPELINE */}
        <section className="pipeline-section" id="how-it-works">
          <div className="section-heading">
            <p className="eyebrow">THE PIPELINE</p>
            <h2>From structure to prediction</h2>
          </div>

          <div className="pipeline">
            <div className="pipeline-card">
              <div className="pipeline-number">01</div>
              <div className="pipeline-icon">🧬</div>
              <h3>Structure</h3>
              <p>Input an antibody–antigen protein structure.</p>
            </div>

            <div className="arrow">→</div>

            <div className="pipeline-card">
              <div className="pipeline-number">02</div>
              <div className="pipeline-icon">🔗</div>
              <h3>Interface</h3>
              <p>Identify interacting residues at the binding interface.</p>
            </div>

            <div className="arrow">→</div>

            <div className="pipeline-card">
              <div className="pipeline-number">03</div>
              <div className="pipeline-icon">▦</div>
              <h3>Matrix</h3>
              <p>Represent amino-acid interactions as a 20×20 matrix.</p>
            </div>

            <div className="arrow">→</div>

            <div className="pipeline-card">
              <div className="pipeline-number">04</div>
              <div className="pipeline-icon">◈</div>
              <h3>Prediction</h3>
              <p>A CNN learns patterns from the interaction matrix.</p>
            </div>
          </div>
        </section>

        {/* WHY */}
        <section className="why-section">
          <div>
            <p className="eyebrow">WHY THIS MATTERS</p>
            <h2>Understanding molecular recognition</h2>
          </div>

          <div className="why-content">
            <p>
              Antibodies recognize specific molecular structures on antigens.
              This recognition occurs through interactions at a molecular
              binding interface.
            </p>

            <p>
              By converting these interactions into numerical representations,
              machine learning models can learn patterns associated with
              antibody–antigen interactions.
            </p>
          </div>
        </section>

        {/* RESEARCH */}
        <section className="research-section">
          <div className="section-heading">
            <p className="eyebrow">RESEARCH FOCUS</p>
            <h2>Key concepts</h2>
          </div>

          <div className="concept-grid">
            <div className="concept-card">
              <span>01</span>
              <h3>Antibody</h3>
              <p>Protein capable of recognizing specific molecular structures.</p>
            </div>

            <div className="concept-card">
              <span>02</span>
              <h3>Antigen</h3>
              <p>Molecular structure recognized by the immune system.</p>
            </div>

            <div className="concept-card">
              <span>03</span>
              <h3>CDR</h3>
              <p>Complementarity-Determining Regions involved in recognition.</p>
            </div>

            <div className="concept-card">
              <span>04</span>
              <h3>CNN</h3>
              <p>Machine learning architecture used to learn matrix patterns.</p>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="cta">
          <p className="eyebrow">READY TO EXPLORE?</p>
          <h2>Analyze an antibody–antigen structure.</h2>
          <Link to="/analysis" className="primary-button">
            Start Analysis →
          </Link>
        </section>
      </main>

      <footer>
        <p>Antibody ML • Project School Research Project</p>
      </footer>
    </div>
  );
}

export default Home;