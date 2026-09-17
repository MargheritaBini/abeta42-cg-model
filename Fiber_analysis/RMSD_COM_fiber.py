import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── parametri ─────────────────────────────────────────────────────────────────
PDB_FILE       = "traj.pdb"
N_STRANDS      = 5
N_RES          = 42
NS_PER_FRAME   = 0.05
STRAND_ORDER   = [3, 4, 0, 1, 2]

TEMP_WINDOWS = [
    (0.0,  0.5,  50,  '#cce5ff'),
    (0.5,  1.0, 100,  '#b3d9ff'),
    (1.0,  1.5, 150,  '#ffd6a5'),
    (1.5,  2.0, 200,  '#ffb347'),
    (2.0,  2.5, 250,  '#ff8c42'),
    (2.5,  3.0, 300,  '#a8d5a2'),
    (3.0, 999.0, 300,  '#a8d5a2'),
]
PRODUCTION_LABEL = "300 K prod"
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdb_frames(filename):
    frames, current = [], []
    with open(filename) as f:
        for line in f:
            if line.startswith("ATOM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                current.append([x, y, z])
            elif line.startswith("END") and current:
                frames.append(np.array(current))
                current = []
    if current:
        frames.append(np.array(current))
    return frames

def rmsd(ref, mob):
    return np.sqrt(np.mean(np.sum((mob - ref)**2, axis=1)))

def com(coords):
    return coords.mean(axis=0)

def get_strand(frame, s):
    idx = STRAND_ORDER[s]
    return frame[idx * N_RES:(idx + 1) * N_RES]

def add_temp_shading(ax, temp_windows, time_array):
    tmax = time_array[-1]
    for i, (t0, t1, temp, color) in enumerate(temp_windows):
        t0c = min(t0, tmax)
        t1c = min(t1, tmax)
        if t0c >= t1c:
            continue
        ax.axvspan(t0c, t1c, color=color, alpha=0.30, zorder=0)

# ── parsing ───────────────────────────────────────────────────────────────────
frames   = parse_pdb_frames(PDB_FILE)
n_frames = len(frames)
n_atoms  = frames[0].shape[0]

assert n_atoms == N_STRANDS * N_RES, \
    f"Errore: {n_atoms} beads != {N_STRANDS*N_RES}"

ref     = frames[0]
time_ns = np.arange(n_frames) * NS_PER_FRAME

# ── calcoli ───────────────────────────────────────────────────────────────────
rmsd_data = np.zeros((n_frames, N_STRANDS))
com_dists = np.zeros((n_frames, N_STRANDS - 1))

for fi, frame in enumerate(frames):
    for s in range(N_STRANDS):
        rmsd_data[fi, s] = rmsd(get_strand(ref, s), get_strand(frame, s))
    for s in range(N_STRANDS - 1):
        c1 = com(get_strand(frame, s))
        c2 = com(get_strand(frame, s + 1))
        com_dists[fi, s] = np.linalg.norm(c2 - c1)

# ── plot ──────────────────────────────────────────────────────────────────────
colors_strand = ['#e63946', '#f4a261', '#2a9d8f', '#457b9d', '#9b2226']
colors_pair   = ['#e63946', '#f4a261', '#2a9d8f', '#457b9d']

fig = plt.figure(figsize=(15, 6), facecolor='white')
gs  = gridspec.GridSpec(2, 1, hspace=0.0)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

TICK_SIZE  = 26
LABEL_SIZE = 29

for ax in [ax1, ax2]:
    ax.set_facecolor('white')
    ax.tick_params(colors='black', labelsize=TICK_SIZE)
    for spine in ax.spines.values():
        spine.set_edgecolor('#aaaaaa')
    ax.grid(False)
    ax.set_xlim(time_ns[0], time_ns[-1])

ax2.set_xlabel("Time (ns)", color='black', fontsize=LABEL_SIZE)
ax1.tick_params(labelbottom=False, bottom=False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['bottom'].set_edgecolor('#555555')
ax2.spines['top'].set_visible(False)

# ── RMSD ─────────────────────────────────────────────────────────────────────
for s in range(N_STRANDS):
    label = f"Strand {s+1}" + (" (bordo)" if s in [0, N_STRANDS-1] else "")
    lw    = 2.2 if s in [0, N_STRANDS - 1] else 1.4
    ax1.plot(time_ns, rmsd_data[:, s], color=colors_strand[s],
             lw=lw, label=label, marker='o', ms=3)

ax1.set_ylabel("RMSD (Å)", color='black', fontsize=LABEL_SIZE)
add_temp_shading(ax1, TEMP_WINDOWS, time_ns)
ax1.set_ylim(0, 30)

# ── COM distances ─────────────────────────────────────────────────────────────
for s in range(N_STRANDS - 1):
    ax2.plot(time_ns, com_dists[:, s], color=colors_pair[s],
             lw=1.6, label=f"Strand {s+1}–{s+2}", marker='s', ms=3)

ax2.set_ylabel("Dist. COM (Å)", color='black', fontsize=LABEL_SIZE)
add_temp_shading(ax2, TEMP_WINDOWS, time_ns)
ax2.set_ylim(3, 18)

plt.show()
