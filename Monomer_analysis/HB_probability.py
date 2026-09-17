import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Function to calculate the normal vector of a plane formed by three atoms
def calculate_plane_normal(atom1, atom2, atom3):
    vec1 = atom2 - atom1
    vec2 = atom3 - atom2
    normal = np.cross(vec1, vec2)
    return normal / np.linalg.norm(normal)  # Normalize the vector

# Function to check if two vectors are parallel or antiparallel (within a threshold angle)
def are_vectors_parallel(vec1, vec2, angle_threshold=20):
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle < angle_threshold or angle > 180.0 - angle_threshold  # parallel or antiparallel

# Read multiple frames from a single PDB file
def read_pdb_frames(file):
    frames = []
    current_frame = []

    with open(file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_index = int(line[6:11].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                current_frame.append((atom_index, np.array([x, y, z])))
            elif line.startswith("END"):
                if current_frame:  # Store the completed frame
                    frames.append(current_frame)
                    current_frame = []  # Reset for the next frame

    # Ensure the last frame is stored in case the file doesn't end with "END"
    if current_frame:
        frames.append(current_frame)

    return frames

# Hydrogen bond detection for each frame
def detect_hydrogen_bonds(atoms, cutoff_distance=7.5, angle_threshold=20):
    num_atoms = len(atoms)
    hydrogen_bonds = np.zeros((num_atoms, num_atoms))  # Matrix to store hydrogen bond presence
    
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            # Ensure hydrogen bonds are only considered for residues at least 4 apart
            if abs(j - i) < 4:
                continue
            if i + 1 >= num_atoms or j + 1 >= num_atoms or i-1 <0 or j-1<0:
                continue

            vec1 = calculate_plane_normal(atoms[i-1][1], atoms[i][1], atoms[i+1][1])
            vec2 = calculate_plane_normal(atoms[j-1][1], atoms[j][1], atoms[j+1][1])

            
            # Check if the two vectors are parallel
            if are_vectors_parallel(vec1, vec2, angle_threshold):
                dist = np.linalg.norm(atoms[i][1] - atoms[j][1])
                if dist < cutoff_distance:
                    hydrogen_bonds[i][j] += 1
                    hydrogen_bonds[j][i] += 1  # Symmetric matrix
    
    return hydrogen_bonds

# Function to plot hydrogen bond probability matrix
def plot_hydrogen_bond_matrix(hydrogen_bonds, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(hydrogen_bonds, cmap='Oranges', interpolation='nearest', vmin=0.0, vmax=0.07)
    plt.colorbar(label='Hydrogen Bond Probability')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.xlabel('Atom Index')
    plt.ylabel('Atom Index')
    plt.show()

# Main function
def main():
    pdb_file = "trajectory.pdb"  # Path to your trajectory PDB file
    frames = read_pdb_frames(pdb_file)  # Read all frames from the PDB file
    
    total_frames = len(frames)
    num_atoms = len(frames[0])  # Assume all frames have the same number of atoms
    
    hydrogen_bonds_cumulative = np.zeros((num_atoms, num_atoms))  # Cumulative matrix
    
    for i, atoms in enumerate(frames):
        hydrogen_bonds = detect_hydrogen_bonds(atoms)  # Detect hydrogen bonds for this frame
        hydrogen_bonds_cumulative += hydrogen_bonds  # Accumulate across frames
        print(f"Processed frame {i+1}/{total_frames}")
    
    # Compute probability matrix
    hydrogen_bond_probabilities = hydrogen_bonds_cumulative / total_frames
    
    np.savetxt("hydrogen_bond_matrix.txt", hydrogen_bond_probabilities, fmt="%.4f")
    plot_hydrogen_bond_matrix(hydrogen_bond_probabilities, "Hydrogen Bond Probability Matrix")

if __name__ == "__main__":
    main()
