# Atomistic Analysis

## `Angles_computation_param.py`

This script computes the CG bond-angle potential parameters from atomistic simulations.
The pipeline proceeds in four steps:

1. **Angle computation** — for each cluster (from both OPLS and a99SB force fields),
   computes the bond angles for all consecutive C$\alpha$ triplets across the ensemble.

2. **Weighted average** — combines the per-cluster angle distributions into a single
   distribution, weighted by the cluster populations.

3. **Boltzmann inversion** — converts the averaged distribution $P(\theta)$ into a
   potential energy profile via
   $U(\theta) = -k_{\mathrm{B}}T \ln \left[ P(\theta) / \sin\theta \right]$.

4. **Potential fitting** — fits the resulting $U(\theta)$ with the target functional
   form to extract the force field parameters.
   
   
## `Dih_computation_param.py`

This script computes the CG bond-dihedral potential parameters from atomistic simulations.
The pipeline proceeds in four steps:

1. **Dihedral computation** — for each cluster (from both OPLS and a99SB force fields),
   computes the dihedral angles for all consecutive C$\alpha$ quadruplets across the ensemble.

2. **Weighted average** — combines the per-cluster dihedral distributions into a single
   distribution, weighted by the cluster populations.

3. **Boltzmann inversion** — converts the averaged distribution $P(\alpha)$ into a
   potential energy profile via
   $U(\alpha) = -k_{\mathrm{B}}T \ln \left[ P(\alpha) \right]$.

4. **Potential fitting** — fits the resulting $U(\alpha)$ with the target functional
   form to extract the force field parameters.


## `HB_probability_computation.py`

This script computes a weighted hydrogen bond probability matrix from
atomistic simulation trajectories.

1. **Per-frame contact assignment** — for each pair of residues $(i, j)$
with $|i-j| > \Delta$, it evaluates the geometric HB criterion: the
C$_\alpha$--C$_\alpha$ distance must be below $d_\text{cut}$ and the angle
between the local backbone normal vectors $\hat{\mathbf{n}}_i$ and
$\hat{\mathbf{n}}_j$ must satisfy $\theta < \alpha$ or
$\theta > 180° - \alpha$, capturing both parallel and antiparallel
$\beta$-sheet geometries. The result is a binary contact matrix
$C_{ij}^{(k)} \in \{0,1\}$ for each cluster representative $k$.

2. **Cluster-weighted average** — the binary matrices are combined using
the cluster populations $N_k$ as weights, yielding the final $42 \times 42$
probability matrix

$$P_{ij} = \sum_k \frac{N_k}{N_\text{tot}}\, C_{ij}^{(k)}$$

which represents the probability of a backbone HB-like contact between
residues $i$ and $j$ across the conformational ensemble.


# CG Monomer Analysis 

This collection of Python scripts performs structural analysis of coarse-grained (CG) molecular dynamics trajectories of the Aβ42 peptide. Each residue is represented by a single bead at the Cα position. All scripts read a multi-frame PDB trajectory file (`trajectory.pdb`) as input.


### `Angles_distribution.py`

Computes the backbone **angle probability distributions** for each consecutive triplet of CG beads, following the same methodology used for atomistic simulations.

For a 42-residue protein, 40 triplet angles are computed (residues 2–41 as central beads). The angle at each central bead *i* is defined by the triplet (i−1, i, i+1) using the standard dot-product formula:

$$\cos\theta = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{|\mathbf{v}_1||\mathbf{v}_2|}$$

 Probability distributions are estimated via Kernel Density Estimation (KDE, Silverman bandwidth) and saved as two-column text files (`angle (deg) | probability density`). 


### `Dihedral_distribution.py`

Computes the backbone **dihedral (torsion) angle probability distributions** for each consecutive quadruplet of CG beads, following the same methodology used for atomistic simulations.

For a 42-residue protein, 39 dihedral angles are computed (quadruplets 1–2–3–4 through 39–40–41–42). The dihedral angle is calculated from the two plane normals defined by the quadruplet using the standard `atan2`-based formula, yielding values in the range [−180°, +180°]. Distributions are estimated via KDE (Silverman bandwidth). 

### `Contact_map_probability.py`

Constructs a **42×42 residue contact probability matrix** from the CG trajectory.

Two residues *i* and *j* are considered in contact if their inter-bead distance is below a cutoff of **8.2 Å**. Pairs with sequence separation |i − j| ≤ 4 are excluded to avoid trivial local contacts. The matrix element (i, j) reports the fraction of trajectory frames in which the contact is formed. The resulting matrix is saved as a plain-text file and visualized as a heatmap.

### `HB_probability.py`

Constructs a **42×42 residue hydrogen bond probability matrix** from the CG trajectory.

Hydrogen bond formation between residues *i* and *j* is assessed using two geometric criteria applied to the CG bead triplets centered on each residue:

1. **Distance criterion:** inter-bead distance < **7.5 Å**
2. **Orientation criterion:** the local plane normals at residues *i* and *j* — computed from the triplets (i−1, i, i+1) and (j−1, j, j+1) — are quasi-parallel or anti-parallel within an angular threshold of **20°**

As in `Contact_map_probability.py`, pairs with |i − j| < 4 are excluded. The matrix element (i, j) reports the fraction of frames in which a hydrogen bond is detected. The matrix is saved as a text file and visualized as a heatmap.


# CG Fiber Analysis

This Python script analyzes coarse-grained (CG) molecular dynamics trajectories of amyloid
fibrils, tracking structural stability and inter-strand organization over time.

## `RMSD_COM_fiber.py`

The script reads a multi-frame PDB trajectory and computes two structural observables for a
fibril composed of `N_STRANDS` β-strands, each with `N_RES` residues (beads in the CG
representation).

**RMSD per strand** — the root mean square deviation of each strand's bead positions with
respect to the first frame (reference), computed as a function of simulation time. 

**Inter-strand COM distances** — the distance between the centers of mass of consecutive
strand pairs (strand *i* and strand *i+1*), computed as a function of time. 

Both observables are plotted in a shared two-panel figure with temperature-window shading,
allowing direct visual correlation between structural changes and the thermal ramp protocol.


# Aggregation Analysis

## `Contact_time_average.py`


The script reads a multi-frame PDB trajectory of a multi-protein CG system and computes,
as a function of simulation time, the number of inter-protein contacts. Two proteins are
considered to be in contact if at least one pair of beads — one from each protein — is
within a cutoff distance.

The results are written to a plain-text output file and plotted as a time series.
