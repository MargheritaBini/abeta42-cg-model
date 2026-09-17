"""
Coarse-Grained Backbone Angle Distribution Analysis

This script analyzes backbone angle distributions from a coarse-grained (CG) molecular
dynamics trajectory. Each residue is represented by a single bead (typically at the
Cα position), and the script computes the angle formed by consecutive triplets of beads.

For a protein with 42 residues, there are 40 possible angles to analyze (from residue
2 to residue 41, as each angle requires 3 consecutive beads).

Author: [Your name]
Date: 2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# =============================================================================
# PROTEIN SEQUENCE DEFINITION
# =============================================================================

# Define the sequence from residue 1 to 42
# This is the amyloid-beta 1-42 peptide sequence
sequence = [
    "Asp", "Ala", "Glu", "Phe", "Arg", "His", "Asp", "Ser", "Gly", "Tyr",  # 1-10
    "Glu", "Val", "His", "His", "Gln", "Lys", "Leu", "Val", "Phe", "Phe",  # 11-20
    "Ala", "Glu", "Asp", "Val", "Gly", "Ser", "Asn", "Lys", "Gly", "Ala",  # 21-30
    "Ile", "Ile", "Gly", "Leu", "Met", "Val", "Gly", "Gly", "Val", "Val",  # 31-40
    "Ile", "Ala"                                                             # 41-42
]

# Verify sequence length
assert len(sequence) == 42, "Sequence must have exactly 42 residues"


# =============================================================================
# GEOMETRIC CALCULATIONS
# =============================================================================

def compute_angle(p1, p2, p3):
    """
    Compute the angle formed by three consecutive beads (p1-p2-p3).
    
    Parameters:
    -----------
    p1, p2, p3 : numpy.ndarray
        3D coordinates [x, y, z] of three consecutive beads
        p2 is the vertex of the angle
    
    Returns:
    --------
    float : Angle in degrees (range: 0-180°)
    
    Mathematical Formula:
    ---------------------
    The angle θ is computed using the dot product formula:
    
        cos(θ) = (v1 · v2) / (|v1| × |v2|)
    
    where:
        v1 = p1 - p2  (vector from p2 to p1)
        v2 = p3 - p2  (vector from p2 to p3)
    
    Physical Interpretation:
    ------------------------
    In a coarse-grained model:
    - ~180°: Extended/linear conformation (β-strand-like)
    - ~90-120°: Bent conformation (turn-like)
    - ~60-90°: Highly bent (α-helix-like or tight turn)
    
    Note: CG angles typically span a wider range than all-atom Cα-Cα-Cα angles
    because they don't include the constraint of peptide bond geometry.
    """
    # Compute vectors from central bead (p2) to neighbors
    v1 = p1 - p2  # Vector pointing from p2 to p1
    v2 = p3 - p2  # Vector pointing from p2 to p3
    
    # Compute the dot product and vector magnitudes
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Compute cosine of angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    
    # Clip to [-1, 1] to avoid numerical errors in arccos
    # (Sometimes floating point arithmetic gives values slightly outside [-1, 1])
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Compute angle in radians, then convert to degrees
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees


# =============================================================================
# ANGLE EXTRACTION FROM TRAJECTORY
# =============================================================================

def compute_angles_per_triplet(frames):
    """
    Compute angle distributions for each triplet across all trajectory frames.
    
    Parameters:
    -----------
    frames : list of numpy.ndarray
        Each element is a frame containing 42 beads with shape (42, 3)
        where each row is [x, y, z] coordinates
    
    Returns:
    --------
    dict : Dictionary mapping residue index to list of angles
        Keys: residue indices from 1 to 41 (central bead in each triplet)
        Values: list of angles in degrees
    
    Note on Indexing:
    -----------------
    - Residue 1 cannot be a central bead (no predecessor)
    - Residue 42 cannot be a central bead (no successor)
    - Therefore, we compute angles for residues 2-41 (40 angles total)
    
    For each central residue i:
        - Triplet is: residue[i-1] - residue[i] - residue[i+1]
        - The angle is at the vertex of residue[i]
    
    Example:
    --------
    For residue 5 (Arg):
        Triplet: Phe(4) - Arg(5) - His(6)
        Angle at Arg(5)
    """
    # Initialize dictionary to store angles for each central residue
    # Keys: 1 to 41 (but residue 1 and 42 will remain empty)
    all_angles = {i: [] for i in range(1, 42)}
    
    # Loop through all frames in the trajectory
    for frame in frames:
        # Verify that each frame has exactly 42 beads
        if len(frame) != 42:
            print(f"WARNING: Frame has {len(frame)} beads, expected 42. Skipping frame.")
            continue
        
        # Loop through possible central residues (indices 1 to 40, corresponding to residues 2-41)
        # We use range(1, len(frame) - 1) = range(1, 41)
        for i in range(1, len(frame) - 1):
            # Extract the three consecutive beads
            p1 = frame[i - 1]  # Predecessor bead
            p2 = frame[i]      # Central bead (vertex of angle)
            p3 = frame[i + 1]  # Successor bead
            
            # Compute the angle
            angle = compute_angle(p1, p2, p3)
            
            # Apply a physical filter: only keep angles between 60° and 180°
            # Angles < 60° are extremely unlikely in CG models and may indicate errors
            if 60 <= angle <= 180:
                all_angles[i].append(angle)
    
    # Report statistics
    print(f"\nAngle Statistics:")
    print(f"Total frames analyzed: {len(frames)}")
    for i in range(1, 41):
        n_angles = len(all_angles[i])
        if n_angles > 0:
            mean_angle = np.mean(all_angles[i])
            std_angle = np.std(all_angles[i])
            print(f"  Residue {i+1:2d} ({sequence[i]:3s}): "
                  f"{n_angles:5d} angles, mean = {mean_angle:6.2f}°, std = {std_angle:5.2f}°")
    
    return all_angles


# =============================================================================
# VISUALIZATION AND OUTPUT
# =============================================================================

def plot_angle_distributions(all_angles, save_plots=False):
    """
    Generate and save angle distribution plots for each triplet.
    
    Parameters:
    -----------
    all_angles : dict
        Dictionary mapping residue index to list of angles (from compute_angles_per_triplet)
    save_plots : bool
        If True, save plots as PNG files; if False, only save data files
    
    Outputs:
    --------
    For each triplet:
    - Text file: angle_distributions/RES1-RES2-RES3.txt
      Contains two columns: angle (degrees) and probability density
    - Plot file (optional): angle_distributions/RES1-RES2-RES3.png
    
    Note on Kernel Density Estimation (KDE):
    -----------------------------------------
    We use KDE to convert discrete angle samples into smooth probability distributions.
    - bw_method='silverman': Uses Silverman's rule for bandwidth selection
    - This is appropriate for smooth, continuous distributions
    - Alternative: bw_method='scott' (slightly wider bandwidth)
    """
    # Create output directory
    output_dir = "angle_distributions"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving distributions to: {output_dir}/")
    
    # Loop through central residues (2 to 41, corresponding to indices 1 to 40)
    for i in range(1, 41):
        angles = np.array(all_angles[i])
        
        # Skip if insufficient data for KDE
        if len(angles) < 2:
            print(f"  WARNING: Residue {i+1} has insufficient data ({len(angles)} angles). Skipping.")
            continue
        
        # Perform Kernel Density Estimation
        kde = gaussian_kde(angles, bw_method='silverman')
        
        # Generate smooth angle range for evaluation
        angle_range = np.linspace(60, 180, 1000)
        kde_values = kde(angle_range)
        
        # Create triplet name for file naming
        # Format: RES1-RES2-RES3 where RES2 is the central residue
        triplet_name = f"{sequence[i-1].upper()}-{sequence[i].upper()}-{sequence[i+1].upper()}"
        
        # Save distribution data to text file
        output_file = os.path.join(output_dir, f"{triplet_name}.txt")
        np.savetxt(
            output_file,
            np.column_stack((angle_range, kde_values)),
            fmt='%.6f',
            comments=''
        )
        
        # Optionally create and save plot
        if save_plots:
            plt.figure(figsize=(8, 6))
            plt.plot(angle_range, kde_values, color='blue', linewidth=2)
            plt.title(f"Angle Distribution: {triplet_name}\n(Residue {i+1}: {sequence[i]})",
                     fontsize=18)
            plt.xlabel("Angle (degrees)", fontsize=16)
            plt.ylabel("Probability Density", fontsize=16)
            plt.tick_params(axis='both', labelsize=14)
            plt.grid(alpha=0.3)
            plt.xlim(60, 180)
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(output_dir, f"{triplet_name}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory
    
    print(f"  Saved {len([a for a in all_angles.values() if len(a) >= 2])} distributions.")


# =============================================================================
# FILE I/O
# =============================================================================

def read_pdb_trajectory(file_path):
    """
    Read a multi-frame PDB trajectory file containing CG beads.
    
    Parameters:
    -----------
    file_path : str
        Path to PDB file containing trajectory frames
    
    Returns:
    --------
    list of numpy.ndarray :
        List of frames, each frame is a (42, 3) array of [x, y, z] coordinates
    
    Expected PDB Format:
    --------------------
    ATOM      1  CA  ALA     1      x.xxx   y.yyy   z.zzz  1.00  0.00
    ATOM      2  CA  GLU     2      x.xxx   y.yyy   z.zzz  1.00  0.00
    ...
    ATOM     42  CA  ALA    42      x.xxx   y.yyy   z.zzz  1.00  0.00
    END
    ATOM      1  CA  ALA     1      x.xxx   y.yyy   z.zzz  1.00  0.00
    ...
    
    Note:
    -----
    - Each frame should contain exactly 42 ATOM lines
    - Frames are separated by END or ENDMDL keywords
    - Only coordinates (columns 31-54) are extracted
    """
    frames = []
    current_frame = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                # Extract x, y, z coordinates from PDB format
                # PDB format: columns 31-38 (x), 39-46 (y), 47-54 (z)
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                current_frame.append([x, y, z])
                
            elif line.startswith("END") or line.startswith("ENDMDL"):
                # End of current frame
                if current_frame:
                    if len(current_frame) == 42:
                        frames.append(np.array(current_frame))
                    else:
                        print(f"WARNING: Frame has {len(current_frame)} atoms, expected 42. Skipping.")
                    current_frame = []
    
    # Don't forget the last frame if file doesn't end with END
    if current_frame:
        if len(current_frame) == 42:
            frames.append(np.array(current_frame))
        else:
            print(f"WARNING: Last frame has {len(current_frame)} atoms, expected 42. Skipping.")
    
    print(f"\nRead {len(frames)} frames from {file_path}")
    
    return frames


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main analysis workflow.
    
    Steps:
    1. Read CG trajectory from PDB file
    2. Compute angle distributions for all triplets
    3. Generate and save distribution plots
    """
    print("="*70)
    print(" COARSE-GRAINED BACKBONE ANGLE DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Input file path
    file_path = 'trajectory.pdb'
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"\nERROR: File '{file_path}' not found!")
        print("Please ensure the trajectory file is in the current directory.")
        return
    
    # Step 1: Read trajectory
    frames = read_pdb_trajectory(file_path)
    
    if len(frames) == 0:
        print("\nERROR: No valid frames found in trajectory!")
        return
    
    # Step 2: Compute angle distributions
    all_angles = compute_angles_per_triplet(frames)
    
    # Step 3: Generate and save distributions
    plot_angle_distributions(all_angles, save_plots=False)  # Set to True to save PNG plots
    
    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved in: angle_distributions/")
    print("Each file contains: angle (degrees) | probability density")


if __name__ == "__main__":
    main()
