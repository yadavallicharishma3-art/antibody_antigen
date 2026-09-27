import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import { mockData } from "../data/mockData";

function Analysis() {
  const navigate = useNavigate();

  const [pdbId, setPdbId] = useState("");
  const [file, setFile] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFile = (e) => {
    const selectedFile = e.target.files[0];

    if (!selectedFile) return;

    const validTypes = [".pdb", ".cif", ".mmcif"];
    const extension = "." + selectedFile.name.split(".").pop().toLowerCase();

    if (!validTypes.includes(extension)) {
      setError("Please upload a .pdb, .cif or .mmcif file.");
      return;
    }

    setError("");
    setFile(selectedFile);
    setPdbId("");
  };

  const analyze = () => {
    setError("");

    if (!pdbId.trim() && !file) {
      setError("Please enter a PDB ID or upload a structure file.");
      return;
    }

    setLoading(true);

    // Demo mode
    setTimeout(() => {
      const result = {
        ...mockData,
        pdb_id: pdbId.trim() || "Uploaded Structure",
      };

      sessionStorage.setItem("analysisResult", JSON.stringify(result));
      navigate("/results");
    }, 1200);
  };

  return (
    <div className="app">
      <Navbar />

      <main className="analysis-page">
        <div className="analysis-header">
          <p className="eyebrow">STRUCTURE ANALYSIS</p>
          <h1>Analyze a Protein Structure</h1>
          <p>
            Provide a PDB identifier or upload a protein structure to explore
            antibody–antigen interaction patterns.
          </p>
        </div>

        <div className="demo-warning">
          <span>●</span>
          Demo Mode — results currently use sample data.
        </div>

        <div className="input-grid">
          {/* PDB */}
          <div className="input-card">
            <div className="card-number">01</div>
            <h2>Enter PDB ID</h2>
            <p>
              Enter the identifier of a protein structure from the Protein Data
              Bank.
            </p>

            <label>PDB ID</label>

            <input
              type="text"
              value={pdbId}
              onChange={(e) => {
                setPdbId(e.target.value);
                setFile(null);
              }}
              placeholder="e.g. 1HZH"
            />

            <small>Example: 1HZH</small>
          </div>

          {/* FILE */}
          <div className="input-card">
            <div className="card-number">02</div>
            <h2>Upload Structure</h2>
            <p>Upload a PDB or mmCIF structure file from your computer.</p>

            <label className="upload-box">
              <input
                type="file"
                accept=".pdb,.cif,.mmcif"
                onChange={handleFile}
              />

              <div className="upload-icon">↑</div>

              {file ? (
                <>
                  <strong>{file.name}</strong>
                  <span>{(file.size / 1024).toFixed(1)} KB</span>
                </>
              ) : (
                <>
                  <strong>Choose a structure file</strong>
                  <span>.pdb • .cif • .mmcif</span>
                </>
              )}
            </label>

            {file && (
              <button
                className="remove-file"
                onClick={() => setFile(null)}
              >
                Remove file
              </button>
            )}
          </div>
        </div>

        {error && <div className="error-box">{error}</div>}

        <div className="analyze-area">
          {loading ? (
            <div className="loading">
              <div className="spinner"></div>
              <h3>Analyzing Structure</h3>
              <p>Processing your input...</p>
            </div>
          ) : (
            <button className="analyze-button" onClick={analyze}>
              Analyze Structure →
            </button>
          )}
        </div>

        <div className="science-note">
          <strong>What happens next?</strong>
          <p>
            The research pipeline identifies interaction information at the
            antibody–antigen interface and represents it as a numerical
            interaction matrix for the CNN.
          </p>
        </div>
      </main>
    </div>
  );
}

export default Analysis;