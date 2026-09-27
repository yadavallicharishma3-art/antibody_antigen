const aminoAcids = [
  "A", "C", "D", "E", "F",
  "G", "H", "I", "K", "L",
  "M", "N", "P", "Q", "R",
  "S", "T", "V", "W", "Y"
];

// Demo 20 × 20 amino-acid interaction matrix
const matrix = aminoAcids.map((row, i) =>
  aminoAcids.map((col, j) => {
    const value = ((i * 7 + j * 11 + i * j) % 100) / 100;
    return Number(value.toFixed(2));
  })
);

export const mockData = {
  prediction: "antibody-antigen",
  probability: 0.97,
  pdb_id: "1HZH",
  antibody_chain: "H",
  antigen_chain: "A",
  cdr_residues: ["H31", "H32", "H33"],
  matrix: matrix
};

export { aminoAcids };