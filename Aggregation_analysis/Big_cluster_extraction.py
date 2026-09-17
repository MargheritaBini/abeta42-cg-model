"""
1. Cluster proteins in the LAST frame using distance cutoff (8.2 Å)
2. Find the biggest cluster → save its protein indices
3. Extract THOSE SAME proteins from ALL frames of the trajectory
4. Save as a PDB trajectory

Usage: python cluster_extract_v3.py
"""

import numpy as np
from scipy.spatial.distance import cdist
import os

# ─── Config ───────────────────────────────────────────────────────────────────
INPUT_PDB  = "trajectory_pbc.pdb"
CUTOFF     = 8.2
N_ATOMS    = 42
OUTPUT_DIR = "."
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Parse all frames — keep original lines AND coordinates ──────────────────
def parse_frames(path, n_atoms):
    frames_coords = []
    frames_lines  = []
    headers       = []
    current_coords = []
    current_lines  = []
    current_header = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                current_coords.append([x, y, z])
                current_lines.append(line.rstrip('\n'))
            elif line.startswith("END"):
                if current_coords:
                    n_prot = len(current_coords) // n_atoms
                    if n_prot > 0:
                        coords = np.array(current_coords[:n_prot * n_atoms])
                        lines  = current_lines[:n_prot * n_atoms]
                        frames_coords.append(coords.reshape(n_prot, n_atoms, 3))
                        frames_lines.append(
                            [lines[i*n_atoms:(i+1)*n_atoms] for i in range(n_prot)]
                        )
                        headers.append(current_header[:])
                current_coords = []
                current_lines  = []
                current_header = []
            else:
                current_header.append(line.rstrip('\n'))
    return frames_coords, frames_lines, headers

print("Parsing trajectory...")
frames_coords, frames_lines, headers = parse_frames(INPUT_PDB, N_ATOMS)
n_frames   = len(frames_coords)
n_proteins = frames_coords[0].shape[0]
print(f"  {n_frames} frames, {n_proteins} proteins/frame, {N_ATOMS} atoms/protein")

# ─── Union-Find ───────────────────────────────────────────────────────────────
def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

def union(parent, rank, a, b):
    ra, rb = find(parent, a), find(parent, b)
    if ra == rb: return
    if rank[ra] < rank[rb]: ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]: rank[ra] += 1

def cluster_frame(coords_frame, cutoff):
    N = coords_frame.shape[0]
    parent = list(range(N))
    rank   = [0] * N
    for i in range(N):
        for j in range(i+1, N):
            if cdist(coords_frame[i], coords_frame[j]).min() < cutoff:
                union(parent, rank, i, j)
    labels = [find(parent, i) for i in range(N)]
    roots  = {}
    out    = []
    for l in labels:
        if l not in roots: roots[l] = len(roots)
        out.append(roots[l])
    return out

# ─── Cluster LAST frame only ──────────────────────────────────────────────────
print(f"\nClustering last frame (frame {n_frames}) with cutoff {CUTOFF} Å...")
last_coords = frames_coords[-1]
labels = cluster_frame(last_coords, CUTOFF)

sizes = {}
for l in labels:
    sizes[l] = sizes.get(l, 0) + 1

sorted_clusters = sorted(sizes.items(), key=lambda x: -x[1])
print(f"  {len(sizes)} clusters found in last frame")
print(f"  Top cluster sizes: {[s for _,s in sorted_clusters[:6]]}")

# Biggest cluster
biggest_id   = sorted_clusters[0][0]
biggest_size = sorted_clusters[0][1]
protein_indices = [i for i, l in enumerate(labels) if l == biggest_id]

print(f"\nBiggest cluster: {biggest_size} proteins")
print(f"Protein indices (0-based): {protein_indices}")

# Second biggest
if len(sorted_clusters) > 1:
    second_id      = sorted_clusters[1][0]
    second_size    = sorted_clusters[1][1]
    protein_indices_2 = [i for i, l in enumerate(labels) if l == second_id]
    print(f"Second cluster:  {second_size} proteins")
    print(f"Protein indices (0-based): {protein_indices_2}")

# ─── Write output — same protein indices across ALL frames ────────────────────
def write_pdb_trajectory(protein_idx_list, frames_lines, headers, out_path):
    with open(out_path, 'w') as f:
        for fi in range(len(frames_lines)):
            for h in headers[fi]:
                f.write(h + '\n')
            atom_idx = 1
            for pi in protein_idx_list:
                for line in frames_lines[fi][pi]:
                    new_line = f"ATOM  {atom_idx:5d}" + line[11:]
                    f.write(new_line + '\n')
                    atom_idx += 1
            f.write("END\n")

print(f"\nWriting trajectories for all {n_frames} frames...")

out1 = os.path.join(OUTPUT_DIR, "cluster1_trajectory.pdb")
write_pdb_trajectory(protein_indices, frames_lines, headers, out1)
print(f"  Saved cluster 1 ({biggest_size} proteins × {n_frames} frames): {out1}")

if len(sorted_clusters) > 1:
    out2 = os.path.join(OUTPUT_DIR, "cluster2_trajectory.pdb")
    write_pdb_trajectory(protein_indices_2, frames_lines, headers, out2)
    print(f"  Saved cluster 2 ({second_size} proteins × {n_frames} frames): {out2}")

print("\nDone!")
