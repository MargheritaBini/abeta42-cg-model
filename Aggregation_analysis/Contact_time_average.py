import numpy as np
import matplotlib.pyplot as plt

def parse_pdb_frames(file_path):
    """
    Parse a PDB file with multiple frames separated by 'END' lines.
    Extract atomic coordinates for each frame.
    """
    frames = []
    coordinates = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                parts = line.split()
                x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
                coordinates.append([x, y, z])
            elif line.startswith("END"):
                # Store the frame coordinates and reset for the next frame
                frames.append(np.array(coordinates))
                coordinates = []

    # In case there is no 'END' at the end of the file
    if coordinates:
        frames.append(np.array(coordinates))

    return frames

def compute_distances(coords1, coords2):
    """
    Compute pairwise distances between two sets of coordinates.
    """
    diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    return distances


def count_protein_contacts(coords, num_atoms_per_protein, cutoff):
    """
    Count unique contacts between proteins based on distances less than the cutoff.
    """
    num_proteins = len(coords) // num_atoms_per_protein
    unique_contacts = set()

    for i in range(num_proteins):
        for j in range(i + 1, num_proteins):  # Only check each pair once
            start_i, end_i = i * num_atoms_per_protein, (i + 1) * num_atoms_per_protein
            start_j, end_j = j * num_atoms_per_protein, (j + 1) * num_atoms_per_protein
            distances = compute_distances(coords[start_i:end_i], coords[start_j:end_j])
            if np.any(distances < cutoff):
                unique_contacts.add((i, j))

    return len(unique_contacts)

# Main script
pdb_file_path = 'traj.pdb'
num_atoms_per_protein = 42
cutoff_distance = 8.2

# Parse the PDB file into frames
frames = parse_pdb_frames(pdb_file_path)

# Calculate contacts for each frame and store the results
contacts_over_time = []
frame = []

# Open the text file for writing contact data
output_file_path = 'contacts_over_time.txt'
with open(output_file_path, 'w') as output_file:
    output_file.write("Frame\tContacts\n")  # Header

    for frame_idx, frame_coords in enumerate(frames):
        contacts_count = count_protein_contacts(frame_coords, num_atoms_per_protein, cutoff_distance)
        contacts_over_time.append(contacts_count)
        frame.append(frame_idx+1)
        print(f"Frame {frame_idx + 1}: {contacts_count} contacts")

        # Write frame and contact count to the file
        output_file.write(f"{frame_idx + 1}\t{contacts_count}\n")


frame = np.array(frame) * 0.05# Plot the contacts over time
plt.figure(figsize=(12, 6))
plt.plot(frame, contacts_over_time, marker='o', color='b')
plt.xlabel("Time (ns)", fontsize = 20)
plt.ylabel("Number of Contacts", fontsize = 20 )
plt.tick_params(axis='both', labelsize=22)
#plt.title("Protein Contacts Over Time")
#plt.grid(True)
plt.show()


