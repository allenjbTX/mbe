# Many-Body Expansion (MBE) driver for ORCA

Small command-line tool to compute the traditional Many-Body Expansion (MBE)
energy and gradients of a molecular cluster using ORCA as the electronic
structure back end.

This repository contains a single main script, `mbe.py`, which automates
fragment generation (for water clusters or from a JSON mask), runs ORCA jobs
for every subsystem up to a requested MBE order, and assembles non-redundant
MBE energies and gradients.

## Key features

- Automatic water fragmentation (O + 2 H grouping) or user-supplied fragments
- Arbitrary MBE order (1..N fragments)
- Parallel execution via Python `concurrent.futures`
- Robust restart: existing ORCA outputs in the scratch dir are reused
- Produces both energy breakdown (`.mbe`) and per-order cumulative gradients
  (`.mbegrad`)

## Prerequisites

- Python 3.8 or newer
- numpy
- ORCA (4.2+), either available on `PATH` as `orca` or pointed to with
  the `--orca-path` argument or `ORCA_PATH` environment variable

Install the minimal Python dependency with pip:

```bash
pip install numpy
```

## Usage

Basic invocation:

```bash
python mbe.py cluster.xyz --order 3 --method "ri-mp2 def2-svp" \
    --charge 0 --multiplicity 1 --nprocs 8 --scratch _mbe_tmp \
    --orca-path /path/to/orca
```

Options of note:
- `xyz` (positional): input cluster `.xyz` file
- `--order` / `-n`: MBE order (default: 3)
- `--method`: ORCA method and basis line (default: `HF 6-31G*`)
- `--charge`, `--multiplicity`: whole-cluster charge and multiplicity
- `--pointcharges`: optional point-charge file (plain text: first line =
  integer count; remaining lines = `q x y z`); the filename is forwarded to
  ORCA
- `--fragments`: JSON file with fragment definitions; allowed formats:
  - `list[list[int]]` (each fragment = list of atom indices)
  - `list[{"atoms": [...], "charge": q}]` (per-fragment charge)
- `--nprocs`: number of parallel workers
- `--scratch`: scratch directory where per-subsystem folders and ORCA outputs
  are created (default: `_mbe_tmp`)

Example: run a 2nd-order MBE from a manually defined fragments JSON

```bash
python mbe.py cluster.xyz --order 2 --fragments frags.json
```

## Output files

- `cluster.mbe` — plain-text energy breakdown per order and cumulative totals
- `cluster.mbegrad` — cumulative gradients for each order in Eh/bohr; formatted
  with atom index, symbol, and vector components
- `_mbe_tmp/` (or the directory supplied with `--scratch`) — scratch
  subdirectories for each subsystem (e.g. `0_1_2/`) containing the ORCA
  `.inp`, `.out`, and `.engrad` files

The script re-uses any existing ORCA `.out` files found in the scratch
subdirectories; if an `.out` exists but doesn't contain a final energy, the
subsystem will be recomputed.

## Fragment JSON examples

Simple index-based fragments (zero charges):

```json
[ [0,1,2], [3,4,5], [6,7,8] ]
```

Fragments with per-fragment charges:

```json
[
  {"atoms": [0,1,2], "charge": 0},
  {"atoms": [3,4,5], "charge": -1}
]
```

## Notes and limitations

- The shipped water detector (`detect_water_fragments`) assumes each oxygen
  has exactly two H within 1.2 Å. It's intended for O–H2 style water clusters
  and will raise an error otherwise.
- ORCA must be installed and callable from the provided `--orca-path` or the
  `ORCA_PATH` environment variable.
- The program writes ORCA's stdout/stderr into the `.out` files in the
  scratch directory; inspect those files for ORCA runtime errors.

## Tests / examples

The `tests/` folder contains example clusters and a populated `_mbe_tmp/`
tree produced by a prior run which can be used as a reference for expected
output layout.

## License & contributions

Refer to the repository owner for license and contribution policies.

---

For troubleshooting or to request a feature (parallelism, different
fragmentation heuristics, better error handling), open an issue or send a
description of the desired behavior with an example input.
