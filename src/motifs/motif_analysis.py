import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# --- Core Motif Position Weight Matrices (PWMs) ---
# Each PWM represents the probability distribution of nucleotides at each position.
CORE_MOTIFS = {
    "TATA-box": {
        # Standard consensus: T-A-T-A-A-A-A-H (where H is A, C, or T)
        "matrix": {
            "A": [0.05, 0.90, 0.05, 0.90, 0.85, 0.85, 0.80, 0.30],
            "C": [0.05, 0.02, 0.05, 0.02, 0.05, 0.05, 0.05, 0.30],
            "G": [0.05, 0.03, 0.05, 0.03, 0.05, 0.05, 0.05, 0.10],
            "T": [0.85, 0.05, 0.85, 0.05, 0.05, 0.05, 0.10, 0.30]
        },
        "description": "Core promoter element that initiates transcription by binding TATA-binding protein (TBP)."
    },
    "GC-box": {
        # Standard consensus: G-G-G-C-G-G
        "matrix": {
            "A": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            "C": [0.05, 0.05, 0.05, 0.80, 0.05, 0.05],
            "G": [0.85, 0.85, 0.85, 0.10, 0.85, 0.85],
            "T": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        },
        "description": "Common promoter element bound by Sp1 transcription factor."
    },
    "CCAAT-box": {
        # Standard consensus: G-G-C-C-A-A-T
        "matrix": {
            "A": [0.05, 0.05, 0.05, 0.05, 0.85, 0.85, 0.05],
            "C": [0.05, 0.05, 0.85, 0.85, 0.05, 0.05, 0.05],
            "G": [0.85, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05],
            "T": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.85]
        },
        "description": "Promoter proximal element bound by NF-Y transcription factor."
    },
    "CTCF-binding-site": {
        # CTCF consensus core: C-C-A-C-N-N-G-G-N-G-G-C-G-C
        "matrix": {
            "A": [0.05, 0.05, 0.80, 0.05, 0.25, 0.25, 0.05, 0.05, 0.25, 0.05, 0.05, 0.05, 0.05, 0.05],
            "C": [0.80, 0.80, 0.05, 0.80, 0.25, 0.25, 0.05, 0.05, 0.25, 0.05, 0.05, 0.80, 0.05, 0.80],
            "G": [0.10, 0.10, 0.10, 0.10, 0.25, 0.25, 0.85, 0.85, 0.25, 0.85, 0.85, 0.10, 0.85, 0.10],
            "T": [0.05, 0.05, 0.05, 0.05, 0.25, 0.25, 0.05, 0.05, 0.25, 0.05, 0.05, 0.05, 0.05, 0.05]
        },
        "description": "Insulator protein binding site involved in 3D chromatin conformation."
    },
    "NF-kB-motif": {
        # Consensus: G-G-G-R-N-N-Y-Y-C-C (where R is A/G, Y is C/T)
        "matrix": {
            "A": [0.05, 0.05, 0.05, 0.45, 0.25, 0.25, 0.05, 0.05, 0.05, 0.05],
            "C": [0.05, 0.05, 0.05, 0.05, 0.25, 0.25, 0.45, 0.45, 0.85, 0.85],
            "G": [0.85, 0.85, 0.85, 0.45, 0.25, 0.25, 0.05, 0.05, 0.05, 0.05],
            "T": [0.05, 0.05, 0.05, 0.05, 0.25, 0.25, 0.45, 0.45, 0.05, 0.05]
        },
        "description": "Transcription factor binding site activated by cellular stress and immune response."
    }
}

def score_window(window: str, pwm: Dict[str, List[float]]) -> float:
    """
    Computes the Position Weight Matrix similarity score for a given DNA window.
    Args:
        window: String of nucleotides (e.g. 'TATAAGAA') of length equal to pwm length.
        pwm: Position weight matrix dictionary with keys 'A', 'C', 'G', 'T'.
    Returns:
        mean_score: Average probability of the nucleotides across the window.
    """
    if len(window) != len(pwm["A"]):
        return 0.0
        
    score_sum = 0.0
    for pos, nucleotide in enumerate(window):
        # Fallback to background frequency if nucleotide is 'N'
        if nucleotide == "N":
            score_sum += 0.25
        elif nucleotide in pwm:
            score_sum += pwm[nucleotide][pos]
        else:
            score_sum += 0.05  # Penalty for non-DNA characters
            
    return score_sum / len(window)

def scan_sequence_for_motifs(
    sequence: str, 
    threshold: float = 0.75
) -> List[Dict[str, Any]]:
    """
    Slides a window across the DNA sequence to scan for core biological promoter
    and transcription factor binding site motifs.
    
    Returns a list of identified motif matches.
    """
    sequence = sequence.upper()
    matches = []
    
    for name, data in CORE_MOTIFS.items():
        pwm = data["matrix"]
        motif_len = len(pwm["A"])
        
        # Slide window across sequence
        for start_pos in range(len(sequence) - motif_len + 1):
            end_pos = start_pos + motif_len
            window = sequence[start_pos:end_pos]
            
            score = score_window(window, pwm)
            
            if score >= threshold:
                matches.append({
                    "matched": True,
                    "motif_name": name,
                    "sequence": window,
                    "start": start_pos,
                    "end": end_pos,
                    "score": float(score),
                    "threshold": threshold,
                    "description": data["description"]
                })
                
    # Sort matches by score descending
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches
