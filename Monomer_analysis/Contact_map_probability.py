import numpy as np
import matplotlib.pyplot as plt

# Read multiple frames from a single PDB file
def read_pdb_frames(file):
    frames = []
    current_frame = []

    with open(file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                current_frame.append(np.array([x, y, z]))
            elif line.startswith("END"):
                if current_frame:
                    frames.append(current_frame)
                    current_frame = []
    if current_frame:
        frames.append(current_frame)

    return np.array(frames)  # Shape: (num_frames, num_atoms, 3)

# Compute contact matrix: fraction of frames where distance < threshold
def compute_contact_probability(frames, cutoff=8.2):
    num_frames, num_atoms, _ = frames.shape
    contact_counts = np.zeros((num_atoms, num_atoms))

    for frame_idx in range(num_frames):
        coords = frames[frame_idx]
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if abs(i - j) <= 4:
                    continue
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < cutoff:
                    contact_counts[i, j] += 1
                    contact_counts[j, i] += 1

    contact_probabilities = contact_counts / num_frames
    return contact_probabilities

# Plot the contact probability matrix — clean, no title/axis labels/colorbar label
def plot_contact_matrix(matrix, vmax=1):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(matrix, cmap='Reds', interpolation='nearest', vmin=0, vmax=vmax)

    # Residue ticks: 1-based labels
    ticks = np.arange(0, 42, 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks + 1, fontsize=9)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks + 1, fontsize=9)

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.invert_yaxis()

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('')
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig("contact_map_CG.png", dpi=200, bbox_inches='tight')
    plt.show()

# Compute mean contact probability over a rectangular sub-region
# res_i and res_j are 1-based inclusive ranges
def regional_contact_probability(matrix, res_i_range, res_j_range):
    i0, i1 = res_i_range[0] - 1, res_i_range[1] - 1  # convert to 0-based
    j0, j1 = res_j_range[0] - 1, res_j_range[1] - 1

    # Exclude diagonal-adjacent entries (|i-j| <= 4) from the average
    values = []
    for ii in range(i0, i1+1):
        for jj in range(j0, j1+1):
            if abs(ii - jj) > 4:
                values.append(matrix[ii, jj])

    if len(values) == 0:
        return float('nan')
    return np.mean(values)

def main():
    pdb_file = "trajectory.pdb"
    frames = read_pdb_frames(pdb_file)

    if len(frames) == 0:
        print("No frames found in the PDB file.")
        return

    num_atoms = len(frames[0])
    if num_atoms != 42:
        print(f"Warning: Expected 42 atoms, but found {num_atoms}.")

    contact_probs = compute_contact_probability(frames, cutoff=8.2)

    plot_contact_matrix(contact_probs, vmax=0.8)
    np.savetxt("distance_map_CG.txt", contact_probs, fmt="%.4f")

    # --- Regional contact probabilities ---
    p1 = regional_contact_probability(contact_probs, (1, 10),  (5, 15))
    p2 = regional_contact_probability(contact_probs, (30, 35), (36, 42))
    p3 = regional_contact_probability(contact_probs, (12, 25), (25, 35))

    print("\n--- Regional Contact Probabilities ---")
    print(f"P_contact( res  1-10  with  5-15 ) = {p1:.4f}")
    print(f"P_contact( res 30-35  with 36-42 ) = {p2:.4f}")
    print(f"P_contact( res 12-25  with 25-35 ) = {p3:.4f}")

if __name__ == "__main__":
    main()
