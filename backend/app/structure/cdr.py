"""Antibody Complementarity-Determining Region (CDR) identification and structure mapping."""

import ast
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from Bio.Align import PairwiseAligner

from .constants import IMGT_CDR_RANGES, STANDARD_AA_3TO1
from .models import ComplexData, CDRData, ResidueData
from .chains import find_sabdab_record


def parse_numbering_list(raw_value: Any) -> List[Tuple[str, int, str]]:
    """Parse SAbDab IMGT numbering list representation.

    Format: [["Q", [1, " "]], ["V", [2, " "]], ...]
    Returns: List of tuples (one_letter_code, imgt_number, insertion_code).
    """
    if pd.isna(raw_value):
        return []
    if isinstance(raw_value, list):
        items = raw_value
    else:
        try:
            items = ast.literal_eval(str(raw_value).strip())
        except Exception:
            return []

    parsed = []
    for item in items:
        aa = str(item[0]).strip().upper()
        num_spec = item[1]
        imgt_num = int(num_spec[0])
        ins_code = str(num_spec[1]) if len(num_spec) > 1 else " "
        parsed.append((aa, imgt_num, ins_code))
    return parsed


def align_and_map_chain_to_imgt(
    standard_residues: List[ResidueData],
    numbering_list: List[Tuple[str, int, str]]
) -> Tuple[Dict[int, Tuple[int, str]], float]:
    """Align structure residues with SAbDab numbering list to get 1-to-1 IMGT mapping.

    Args:
        standard_residues: Ordered list of standard ResidueData objects from the structure chain.
        numbering_list: Ordered list of (aa, imgt_num, ins_code) from SAbDab metadata.

    Returns:
        Tuple of (mapping dict from structure residue index -> (imgt_number, ins_code), alignment_score).
    """
    if not standard_residues or not numbering_list:
        return {}, 0.0

    struct_seq = "".join(r.one_letter_code for r in standard_residues)
    meta_seq = "".join(item[0] for item in numbering_list)

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(meta_seq, struct_seq)
    if not alignments:
        return {}, 0.0

    best_alignment = alignments[0]
    struct_to_imgt: Dict[int, Tuple[int, str]] = {}

    for (meta_start, meta_end), (struct_start, struct_end) in zip(
        best_alignment.aligned[0], best_alignment.aligned[1]
    ):
        seg_len = min(meta_end - meta_start, struct_end - struct_start)
        for offset in range(seg_len):
            m_idx = meta_start + offset
            s_idx = struct_start + offset
            imgt_num = numbering_list[m_idx][1]
            ins_code = numbering_list[m_idx][2]
            struct_to_imgt[s_idx] = (imgt_num, ins_code)

    return struct_to_imgt, best_alignment.score


def identify_antibody_cdrs(
    complex_data: ComplexData,
    sabdab_record: Optional[pd.Series] = None
) -> ComplexData:
    """Identify CDRs (H1, H2, H3, L1, L2, L3) and map to structural residues.

    Requires validated Heavy and Light chains and SAbDab numbering metadata.

    Args:
        complex_data: ComplexData containing identified heavy and light chains.
        sabdab_record: Optional pre-fetched SAbDab metadata row.

    Returns:
        Modified ComplexData with populated complex_data.antibody.cdrs.
    """
    record = sabdab_record if sabdab_record is not None else find_sabdab_record(complex_data.pdb_id)
    if record is None:
        complex_data.warnings.append("Cannot identify CDRs: SAbDab metadata missing.")
        return complex_data

    # Map Heavy Chain CDRs (H1, H2, H3)
    h_chain_id = complex_data.antibody.heavy_chain_id
    if h_chain_id and h_chain_id in complex_data.chains and not pd.isna(record.get("VH_numbering_list")):
        h_chain = complex_data.chains[h_chain_id]
        h_std_res = h_chain.standard_residues
        h_num_list = parse_numbering_list(record["VH_numbering_list"])
        h_map, score = align_and_map_chain_to_imgt(h_std_res, h_num_list)

        for cdr_num in ["1", "2", "3"]:
            cdr_name = f"H{cdr_num}"
            rng = IMGT_CDR_RANGES[cdr_num]
            expected_seq = str(record.get(f"CDR{cdr_name}", "")).strip()

            mapped_res: List[ResidueData] = []
            for s_idx, r in enumerate(h_std_res):
                if s_idx in h_map:
                    imgt_num, ins_code = h_map[s_idx]
                    if rng[0] <= imgt_num <= rng[1]:
                        mapped_res.append(r)

            mapped_seq = "".join(r.one_letter_code for r in mapped_res)
            is_exact = (mapped_seq == expected_seq) if expected_seq else bool(mapped_seq)

            complex_data.antibody.cdrs[cdr_name] = CDRData(
                name=cdr_name,
                chain_id=h_chain_id,
                sequence=mapped_seq,
                expected_sequence=expected_seq if expected_seq else None,
                residues=mapped_res,
                mapping_status="MATCH" if is_exact else ("PARTIAL" if mapped_seq else "FAILED"),
                notes=f"Alignment score: {score:.1f}. IMGT range {rng[0]}-{rng[1]}."
            )
    elif h_chain_id:
        complex_data.warnings.append(f"Heavy chain {h_chain_id} missing numbering list metadata.")

    # Map Light Chain CDRs (L1, L2, L3)
    l_chain_id = complex_data.antibody.light_chain_id
    if l_chain_id and l_chain_id in complex_data.chains and not pd.isna(record.get("VL_numbering_list")):
        l_chain = complex_data.chains[l_chain_id]
        l_std_res = l_chain.standard_residues
        l_num_list = parse_numbering_list(record["VL_numbering_list"])
        l_map, score = align_and_map_chain_to_imgt(l_std_res, l_num_list)

        for cdr_num in ["1", "2", "3"]:
            cdr_name = f"L{cdr_num}"
            rng = IMGT_CDR_RANGES[cdr_num]
            expected_seq = str(record.get(f"CDR{cdr_name}", "")).strip()

            mapped_res: List[ResidueData] = []
            for s_idx, r in enumerate(l_std_res):
                if s_idx in l_map:
                    imgt_num, ins_code = l_map[s_idx]
                    if rng[0] <= imgt_num <= rng[1]:
                        mapped_res.append(r)

            mapped_seq = "".join(r.one_letter_code for r in mapped_res)
            is_exact = (mapped_seq == expected_seq) if expected_seq else bool(mapped_seq)

            complex_data.antibody.cdrs[cdr_name] = CDRData(
                name=cdr_name,
                chain_id=l_chain_id,
                sequence=mapped_seq,
                expected_sequence=expected_seq if expected_seq else None,
                residues=mapped_res,
                mapping_status="MATCH" if is_exact else ("PARTIAL" if mapped_seq else "FAILED"),
                notes=f"Alignment score: {score:.1f}. IMGT range {rng[0]}-{rng[1]}."
            )
    elif l_chain_id:
        complex_data.warnings.append(f"Light chain {l_chain_id} missing numbering list metadata.")

    return complex_data
