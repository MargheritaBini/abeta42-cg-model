"""
Coarse-Grained Dihedral Angle Distribution Analysis

This script analyzes backbone dihedral (torsion) angle distributions from a
coarse-grained molecular dynamics trajectory. Each residue is represented by
a single bead, and the script computes the dihedral angle formed by consecutive
quadruplets of beads.

For a protein with 42 residues, there are 39 possible dihedral angles to analyze
(from residue 1-2-3-4 to residue 39-40-41-42).

Dihedral angles range from -180° to +180° and describe the rotation around the
central bond in a four-bead sequence.

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

# Define the sequence from residue 1 to 42 (Amyloid-beta 1-42)
sequence = [
    "Asp", "Ala", "Glu", "Phe", "Arg", "His", "Asp", "Ser", "Gly", "Tyr",  # 1-10
    "Glu", "Val", "His", "His", "Gln", "Lys", "Leu", "Val", "Phe", "Phe",  # 11-20 (NOTE: Fixed "Elu" → "Glu")
    "Ala", "Glu", "Asp", "Val", "Gly", "Ser", "Asn", "Lys", "Gly", "Ala",  # 21-30
    "Ile", "Ile", "Gly", "Leu", "Met", "Val", "Gly", "Gly", "Val", "Val",  # 31-40
    "Ile", "Ala"                                                             # 41-42
]

# Verify sequence length
assert len(sequence) == 42, "Sequence must have exactly 42 residues"

# Output directory for dihedral distributions
output_folder = "dihedral_distribution"
os.makedirs(output_folder, exist_ok=True)


# =============================================================================
# GEOMETRIC CALCULATIONS
# =============================================================================

def compute_dihedral_angle(coord1, coord2, coord3, coord4):
    """
    Compute the dihedral (torsion) angle formed by four consecutive beads.
    
    Parameters:
    -----------
    coord1, coord2, coord3, coord4 : numpy.ndarray
        3D coordinates [x, y, z] of four consecutive beads
    
    Returns:
    --------
    float : Dihedral angle in degrees (range: -180° to +180°)
    
    Mathematical Formula:
    ---------------------
    The dihedral angle is the angle between two planes:
    - Plane 1: defined by beads 1, 2, 3
    - Plane 2: defined by beads 2, 3, 4
    
    Algorithm (using cross products):
    1. Define bond vectors:
       b1 = coord2 - coord1
       b2 = coord3 - coord2  (central bond, axis of rotation)
       b3 = coord4 - coord3
    
    2. Compute plane normal vectors:
       v = b1 × b2  (normal to plane 1)
       w = b2 × b3  (normal to plane 2)
    
    3. Compute dihedral angle:
       cos(φ) = v · w / (|v| |w|)
       sin(φ) = (b2/|b2| × v) · w / |w|
       φ = atan2(sin(φ), cos(φ))
    
    Physical Interpretation in CG:
    ------------------------------
    In coarse-grained models, dihedral angles represent the overall backbone
    twist and are less constrained than in all-atom models:
    - All-atom: Strong preferences (φ, ψ Ramachandran regions)
    - CG: Broader distributions, less structure-specific
    
    Typical ranges:
    - Extended conformations: φ ~ ±180° (trans)
    - Helical conformations: φ ~ -60° to -120°
    - Turn/bend conformations: Wide variety
    
    Special Cases Handled:
    ----------------------
    - Collinear atoms (no plane defined): returns 0.0
    - Very small vectors (numerical instability): returns 0.0
    """
    # Compute bond vectors
    b1 = coord2 - coord1  # Vector along bond 1-2
    b2 = coord3 - coord2  # Vector along bond 2-3 (rotation axis)
    b3 = coord4 - coord3  # Vector along bond 3-4
    
    # Normalize b2 (the central bond)
    norm_b2 = np.linalg.norm(b2)
    if norm_b2 < 1e-6:
        # Atoms 2 and 3 are too close - undefined dihedral
        return 0.0
    b2_normalized = b2 / norm_b2
    
    # Compute normal vectors to the two planes
    v = np.cross(b1, b2)  # Normal to plane 1-2-3
    w = np.cross(b2, b3)  # Normal to plane 2-3-4
    
    # Check for degenerate cases (collinear atoms)
    if np.linalg.norm(v) < 1e-6 or np.linalg.norm(w) < 1e-6:
        # Atoms are collinear - dihedral is undefined
        return 0.0
    
    # Normalize the normal vectors
    v = v / np.linalg.norm(v)
    w = w / np.linalg.norm(w)
    
    # Compute the dihedral angle using atan2 for proper quadrant
    x = np.dot(v, w)  # cos(φ)
    y = np.dot(np.cross(b2_normalized, v), w)  # sin(φ)
    
    # Return angle in degrees (-180 to +180)
    return np.degrees(np.arctan2(y, x))


# =============================================================================
# FILE I/O
# =============================================================================

def read_pdb_frames(file_path):
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
    CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1           1
    ATOM      1  CA  ALA     1      x.xxx   y.yyy   z.zzz  1.00  0.00
    ATOM      2  CA  GLU     2      x.xxx   y.yyy   z.zzz  1.00  0.00
    ...
    ATOM     42  CA  ALA    42      x.xxx   y.yyy   z.zzz  1.00  0.00
    END
    ATOM      1  CA  ALA     1      x.xxx   y.yyy   z.zzz  1.00  0.00
    ...
    
    Notes:
    ------
    - CRYST1 lines (box dimensions) are skipped
    - Each frame should contain exactly 42 ATOM lines
    - Frames are separated by END or ENDMDL keywords
    """
    frames = []
    current_frame = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("CRYST1"):
                # Skip periodic box size information
                continue
            elif line.startswith("ATOM"):
                # Extract x, y, z coordinates from PDB format
                # Columns: 31-38 (x), 39-46 (y), 47-54 (z)
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coords = np.array([x, y, z])
                current_frame.append(coords)
            elif line.startswith("END") or line.startswith("ENDMDL"):
                # End of current frame
                if current_frame:
                    if len(current_frame) == 42:
                        frames.append(np.array(current_frame))
                    else:
                        print(f"WARNING: Frame has {len(current_frame)} atoms, expected 42. Skipping.")
                    current_frame = []
    
    # Handle last frame if file doesn't end with END
    if current_frame:
        if len(current_frame) == 42:
            frames.append(np.array(current_frame))
        else:
            print(f"WARNING: Last frame has {len(current_frame)} atoms, expected 42. Skipping.")
    
    print(f"\nRead {len(frames)} frames from {file_path}")
    
    return frames


# =============================================================================
# DIHEDRAL EXTRACTION FROM TRAJECTORY
# =============================================================================

def process_pdb_file(file_path):
    """
    Process PDB file to extract dihedral angle distributions.
    
    Parameters:
    -----------
    file_path : str
        Path to PDB trajectory file
    
    Returns:
    --------
    list of lists :
        Outer list: frames
        Inner list: 39 dihedral angles per frame (one for each quadruplet)
    
    Note on Dihedral Indexing:
    --------------------------
    For a 42-residue protein, there are 39 possible dihedrals:
    - Dihedral 0: residues 1-2-3-4
    - Dihedral 1: residues 2-3-4-5
    - ...
    - Dihedral 38: residues 39-40-41-42
    
    Each dihedral is named by its constituent residues.
    For example, dihedral 0 for Aβ42: Asp-Ala-Glu-Phe
    """
    # Read all frames from trajectory
    frames = read_pdb_frames(file_path)
    
    # Storage for dihedrals: list of frames, each containing list of dihedrals
    dihedrals_by_frame = []
    
    for frame_idx, coordinates in enumerate(frames):
        num_atoms = len(coordinates)
        
        # Verify frame has correct number of beads
        if num_atoms != 42:
            print(f"WARNING: Frame {frame_idx} has {num_atoms} atoms, expected 42. Skipping.")
            continue
        
        frame_dihedrals = []
        
        # Compute all dihedrals for this frame
        # Loop through all possible quadruplets (39 total)
        for j in range(num_atoms - 3):
            # Extract four consecutive beads
            c1 = coordinates[j]
            c2 = coordinates[j + 1]
            c3 = coordinates[j + 2]
            c4 = coordinates[j + 3]
            
            # Compute dihedral angle
            dihedral = compute_dihedral_angle(c1, c2, c3, c4)
            frame_dihedrals.append(dihedral)
        
        dihedrals_by_frame.append(frame_dihedrals)
    
    # Print statistics
    print(f"\nDihedral Statistics:")
    print(f"Total frames analyzed: {len(dihedrals_by_frame)}")
    print(f"Dihedrals per frame: {len(dihedrals_by_frame[0]) if dihedrals_by_frame else 0}")
    
    return dihedrals_by_frame


# =============================================================================
# VISUALIZATION AND OUTPUT
# =============================================================================

def plot_distributions(dihedrals, save_plots=False):
    """
    Generate and save dihedral distribution plots.
    
    Parameters:
    -----------
    dihedrals : list of lists
        Dihedral angles organized by frame and position
    save_plots : bool
        If True, save plots as PNG files (default: False)
    
    Outputs:
    --------
    For each quadruplet (39 total):
    - Text file: dihedral_distribution/res1-res2-res3-res4.txt
      Contains two columns: angle (degrees, -180 to 180) and probability density
    - Plot (optional): Distribution curve
    
    Note on Kernel Density Estimation:
    -----------------------------------
    KDE converts discrete angle samples into smooth probability distributions.
    For dihedral angles:
    - Periodic boundary conditions are NOT enforced (would require circular KDE)
    - This is acceptable for broad distributions
    - For narrow distributions near ±180°, peaks may be artificially split
    """
    # Determine number of dihedrals (should be 39)
    num_dihedrals = len(dihedrals[0]) if dihedrals else 0
    
    if num_dihedrals == 0:
        print("ERROR: No dihedral data to plot!")
        return
    
    print(f"\nGenerating distributions for {num_dihedrals} dihedrals...")
    
    # Loop through each dihedral position
    for i in range(num_dihedrals):
        # Extract all values for this dihedral across all frames
        dihedral_values = [frame[i] for frame in dihedrals if len(frame) > i]
        
        if not dihedral_values:
            print(f"  WARNING: No data for dihedral {i+1}. Skipping.")
            continue
        
        # Create quadruplet name for file naming
        # Format: res1-res2-res3-res4 (lowercase)
        quadruplet_name = "-".join(sequence[i:i+4]).lower()
        
        # Statistics for this dihedral
        mean_dihedral = np.mean(dihedral_values)
        std_dihedral = np.std(dihedral_values)
        print(f"  Dihedral {i+1} ({quadruplet_name}): "
              f"{len(dihedral_values)} samples, "
              f"mean = {mean_dihedral:7.2f}°, std = {std_dihedral:6.2f}°")
        
        # Perform Kernel Density Estimation
        kde = gaussian_kde(dihedral_values, bw_method='silverman')
        
        # Generate smooth angle range from -180 to +180
        alpha_range = np.linspace(-180, 180, 1000)
        density = kde(alpha_range)
        
        # Save distribution data to file (NO HEADER for compatibility)
        filename = os.path.join(output_folder, f"{quadruplet_name}.txt")
        np.savetxt(filename,
                  np.column_stack((alpha_range, density)),
                  fmt='%.6f')
        
        # Optionally create plot
        if save_plots:
            plt.figure(figsize=(8, 6))
            
            # Plot distribution
            plt.plot(alpha_range, density, color='blue', linewidth=2)
            plt.title(f"Dihedral: {quadruplet_name.upper()}\n"
                     f"(Residues {i+1}-{i+2}-{i+3}-{i+4})", fontsize=16)
            plt.xlabel(r"Dihedral Angle $\phi$ (degrees)", fontsize=14)
            plt.ylabel(r"Probability Density $P(\phi)$", fontsize=14)
            plt.xlim(-180, 180)
            plt.grid(alpha=0.3)
            plt.tick_params(axis='both', labelsize=12)
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(output_folder, f"{quadruplet_name}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"\nSaved {num_dihedrals} distributions to: {output_folder}/")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function.
    
    Workflow:
    1. Read CG trajectory from PDB file
    2. Compute dihedral angles for all quadruplets
    3. Generate and save distribution plots
    """
    print("="*70)
    print(" COARSE-GRAINED DIHEDRAL ANGLE DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Input file path
    pdb_file = 'trajectory.pdb'
    
    # Check if file exists
    if not os.path.exists(pdb_file):
        print(f"\nERROR: File '{pdb_file}' not found!")
        print("Please ensure the trajectory file is in the current directory.")
        return
    
    # Step 1: Read and process trajectory
    print(f"\nProcessing trajectory: {pdb_file}")
    dihedrals = process_pdb_file(pdb_file)
    
    if not dihedrals:
        print("\nERROR: No valid frames found!")
        return
    
    # Step 2: Generate distributions
    # Set save_plots=True to save PNG images of distributions
    plot_distributions(dihedrals, save_plots=False)
    
    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved in: {output_folder}/")
    print("Each file contains: angle (degrees) | probability density")
    print("Angle range: -180° to +180°")


if __name__ == "__main__":
    main()
