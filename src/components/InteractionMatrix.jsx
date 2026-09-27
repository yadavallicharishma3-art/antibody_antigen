import { aminoAcids } from "../data/mockData";

function InteractionMatrix({ matrix }) {
  if (!matrix || matrix.length !== 20) {
    return <p>No interaction matrix available.</p>;
  }

  return (
    <div className="matrix-wrapper">
      <div className="matrix-container">
        <div className="matrix-y-label">AMINO ACIDS</div>

        <div>
          <div className="matrix">
            <div className="matrix-corner"></div>

            {aminoAcids.map((aa) => (
              <div className="matrix-label" key={`top-${aa}`}>
                {aa}
              </div>
            ))}

            {matrix.map((row, i) => (
              <div className="matrix-row" key={i}>
                <div className="matrix-label">{aminoAcids[i]}</div>

                {row.map((value, j) => (
                  <div
                    className="matrix-cell"
                    key={`${i}-${j}`}
                    title={`${aminoAcids[i]} ↔ ${aminoAcids[j]} : ${value}`}
                    style={{
                      opacity: 0.25 + value * 0.75,
                    }}
                  >
                    {value.toFixed(1)}
                  </div>
                ))}
              </div>
            ))}
          </div>

          <div className="matrix-x-label">AMINO ACIDS</div>

          <div className="matrix-legend">
            <span>Low interaction</span>

            <div className="legend-gradient">
              <span></span>
              <span></span>
              <span></span>
              <span></span>
              <span></span>
            </div>

            <span>High interaction</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default InteractionMatrix;