# Active-Learning-Accelerated Structure Discovery

A compact, production-oriented active-learning workflow for atomistic structure discovery. The workflow represents unrelaxed Co–O–H structures with SOAP descriptors, learns formation energies with a Gaussian-process model, selects diverse low-confidence candidates using GP-LCB and K-means clustering, and obtains new labels through ASE-driven VASP calculations.

This repository contains the real DFT-labeling workflow only. It does not use a complete energy table, a CSV oracle, or a virtual VASP service to replace DFT calculations.

## Workflow

1. Load every unrelaxed candidate from `initial_structures/<ID>_POSCAR`.
2. Read the first labeled batch from a CSV file containing `ID` and `E_f_per_atom`.
3. Compute SOAP descriptors and reduce them with variance filtering, standardization, and PCA.
4. Fit a Gaussian-process regressor to the currently labeled structures.
5. Rank unlabeled structures by the lower confidence bound

   ```text
   LCB(x) = mu(x) - kappa * sigma(x)
   ```

6. Preselect low-LCB candidates and retain batch diversity with K-means clustering.
7. Copy the external `INCAR`, `POTCAR`, and `KPOINTS`, write `POSCAR` with ASE, and run the supplied blocking VASP command.
8. Accept only electronically and ionically converged calculations, return the new energies to the active-learning loop, and checkpoint the completed round.

## Repository contents

| File | Purpose |
| --- | --- |
| `active_learning.py` | Structure loading, SOAP/PCA representation, GP-LCB acquisition, diverse batch selection, campaign orchestration, and command-line interface. |
| `vasp_label_backend.py` | ASE/VASP input preparation, blocking job execution, convergence validation, output collection, formation-energy calculation, and safe result reuse. |
| `cache.py` | Atomic checkpoint writing and strict validation of resumable campaign state. |

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

VASP is not included. A working VASP installation or scheduler wrapper must be available on the target machine.

## Input layout

The project directory supplied with `--project-root` should contain:

```text
project/
├── initial_structures/
│   ├── structure-001_POSCAR
│   ├── structure-002_POSCAR
│   └── ...
└── initial_labels.csv
```

`initial_labels.csv` contains only the first DFT-labeled batch:

```csv
ID,E_f_per_atom
structure-001,-1.234
structure-002,-1.087
```

The IDs must exactly match the `<ID>` part of the structure filenames. All initial labels must use the same DFT settings and energy convention as subsequent VASP calculations.

## Running a campaign

```bash
python active_learning.py \
  --project-root /path/to/project \
  --output-dir /path/to/project/results/active_learning \
  --initial-labels-csv /path/to/project/initial_labels.csv \
  --incar-path /path/to/INCAR \
  --potcar-path /path/to/POTCAR \
  --kpoints-path /path/to/KPOINTS \
  --vasp-command "mpirun -np 16 vasp_std" \
  --vasp-workers 1 \
  --iterations 100 \
  --batch-size 2 \
  --kappa 2.5 \
  --seed 35
```

`--vasp-command` must block until the calculation finishes and must propagate a nonzero exit code on failure. A scheduler submission command is suitable only when its wrapper waits for job completion.

Increase `--vasp-workers` only when the allocation supports that many concurrent VASP calculations. Run only one campaign controller for a given output directory.

## Structural representation and model

The current implementation uses:

- SOAP species: Co, O, and H
- cutoff radius: 5.0 Å
- radial basis functions: `n_max = 8`
- angular degree: `l_max = 6`
- periodic descriptors with outer averaging
- removal of exactly constant descriptor columns
- standardization followed by full-SVD PCA retaining 99% cumulative variance
- Gaussian process: constant kernel × Matérn-3/2 kernel + white-noise kernel
- acquisition: `mu - kappa * sigma`
- acquisition preselection factor: 5
- K-means diversity selection with 10 initializations

The same unrelaxed structure pool is used for SOAP throughout the campaign. Relaxed structures are retained as `CONTCAR` files but do not replace the original SOAP inputs.

## Energy convention

The returned label is the formation energy per atom:

```text
E_f = (E_total - N_Co * E_Co - N_H * E_H) / N_atoms
```

The default reference energies are:

- `E_Co = -6.355317795 eV`
- `E_H = -11.029648055 eV`

There is no oxygen reference term in this inherited convention. Override the reference energies through the command-line options only when the complete dataset uses the same alternative convention.

## VASP templates and POTCAR order

The same external `INCAR`, `POTCAR`, and `KPOINTS` files are copied byte-for-byte into every calculation directory. The code does not generate or modify these files.

ASE writes each `POSCAR` in the species order expected by the shared `POTCAR`. Every candidate composition must match the corresponding prefix of that POTCAR. Confirm this explicitly when a pool mixes binary Co–O and ternary Co–O–H structures. Composition-dependent INCAR arrays must also be valid for every structure.

## Convergence, checkpoints, and restart behavior

- The final electronic SCF step and the ionic optimization must both be converged.
- ASE reads the final zero-smearing energy from `OUTCAR` and the relaxed structure from `CONTCAR`.
- `OUTCAR` and `CONTCAR` compositions, cells, and positions are cross-checked.
- Every round is checkpointed before submission and after acceptance of the complete batch.
- Completed VASP results are reused only after signatures, immutable input hashes, output hashes, convergence, composition, geometry, and reconstructed energies are revalidated.
- If the controller stops after VASP finishes, a valid completed output can be collected on restart without rerunning VASP.
- Running, failed-after-launch, or unverifiable calculations are not submitted again automatically.

Repeat the same command with a larger `--iterations` value to extend a completed campaign. Keep the VASP calculation directories available because earlier completed rounds are revalidated during resume.

Campaign checkpoints use Python pickle and must be loaded only from trusted local sources.

## Outputs

The output directory contains:

```text
results/active_learning/
├── campaign_results.pkl
├── campaign_summary.json
└── vasp_calculations/
    └── <request-id>/<structure-id>/
        ├── INCAR
        ├── KPOINTS
        ├── POTCAR
        ├── POSCAR
        ├── OUTCAR
        ├── CONTCAR
        ├── request.json
        ├── execution.json
        ├── result.json
        ├── vasp.stdout
        └── vasp.stderr
```

## Scope

This repository intentionally excludes plotting, publication post-processing, complete precomputed energy databases, and virtual DFT interfaces. Those components are not required to run the production active-learning/VASP labeling loop.

## License

No license has been selected yet. Add a license file before redistributing or accepting external contributions.
