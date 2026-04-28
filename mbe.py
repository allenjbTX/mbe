"""
Many-Body Expansion (MBE) wrapper for ORCA
==========================================

A command-line tool to compute the traditional many-body expansion (MBE)
energy of a molecular cluster to arbitrary order *n* using ORCA for the
underlying electronic-structure calculations.

Key features
------------
* **Automatic fragmentation** - for water clusters (O···H₂ pattern) or user-supplied
  fragment masks.
* **Arbitrary MBE order** - choose *n*=1 (monomers) up to the number of
  fragments.
* **Massively parallel** - each subsystem is an embarrassingly parallel ORCA job;
  executed via *concurrent.futures*.
* **Robust restart** - existing ORCA outputs are reused; failed jobs are logged
  and skipped.
* **Plain-text summary** - total MBE energy and per-order breakdown (ΔE¹, ΔE², …).

Prerequisites
-------------
* Python ≥ 3.8 with *numpy*.
* ORCA ≥ 4.2 reachable via the command found in ``$ORCA_PATH`` or ``orca`` on
  *PATH*.

Usage
-----
>>> python mbe.py cluster.xyz --order 3 --method "ri-mp2 def2-svp" \
        --charge 0 --multiplicity 1 --nprocs 8 --scratch /tmp/orca_scratch \
        --orca-path /path/to/orca

See ``python mbe.py -h`` for full help.
"""
import argparse
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional
from modules import counterpoise

import numpy as np

###############################################################################
#                         Geometry & Fragmentation                            #
###############################################################################

def read_xyz(path: Path) -> Tuple[List[str], np.ndarray]:
    """Return (symbols, coordinates[Å]) from a standard XYZ file."""
    with path.open() as f:
        try:
            natm = int(f.readline())
        except ValueError:
            raise RuntimeError("First line must contain number of atoms")
        _ = f.readline()  # comment
        symbols, xyz = [], []
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                raise RuntimeError("Malformed XYZ line: " + line)
            symbols.append(parts[0])
            xyz.append([float(x) for x in parts[1:4]])
        if len(symbols) != natm:
            raise RuntimeError(f"Expected {natm} atoms, found {len(symbols)}")
        return symbols, np.array(xyz)


def detect_water_fragments(symbols: List[str], xyz: np.ndarray) -> List[List[int]]:
    """Naïve O-H₂ grouping: each O with its two nearest H within 1.2 Å."""
    oxygen_idx = [i for i, s in enumerate(symbols) if s.upper() == "O"]
    hydrogens = [i for i, s in enumerate(symbols) if s.upper() == "H"]
    taken = set()
    frags = []
    for o in oxygen_idx:
        d2 = np.linalg.norm(xyz[hydrogens] - xyz[o], axis=1)
        nearest = np.argsort(d2)[:2]
        hs = [hydrogens[i] for i in nearest if d2[i] < 1.2]
        # Ensure deterministic ordering of hydrogens within fragment matching
        # the global atom index order. Without this, gradients can appear with
        # H atoms swapped relative to the original XYZ ordering.
        hs = sorted(hs)
        if len(hs) != 2:
            raise RuntimeError("Failed to assign two H atoms to oxygen {}".format(o))
        if any(h in taken for h in hs):
            raise RuntimeError("Hydrogen assigned to multiple waters; check geometry")
        taken.update(hs)
        frags.append([o] + hs)
    return frags


def generate_combinations(n_frag: int, order: int) -> List[Tuple[int, ...]]:
    combos: List[Tuple[int, ...]] = []
    for k in range(1, order + 1):
        combos.extend(itertools.combinations(range(n_frag), k))
    return combos


###############################################################################
#                          ORCA Job Management                                #
###############################################################################

ORCA_CMD = os.environ.get("ORCA_PATH", "orca")


def write_orca_input(sym, xyz, sel, method, charge, mult, path,
                     pointcharge_file: Optional[Path] = None,
                     xtb_accuracy: Optional[int] = None,
                     cp_fragments: Optional[List[List[int]]] = None):
    """Create ORCA *.inp file for selected fragment indices."""
    with path.open("w") as fh:
        fh.write(f"!{method} EnGrad NoAutoStart\n")
        fh.write("%pal nprocs 1 end\n")
        fh.write("%scf maxiter 800 end\n")
        # fh.write("%maxcore 1500\n")
        if pointcharge_file:
            fh.write(f"%pointcharges \"{pointcharge_file.name}\"\n")
        if method.lower() in ['xtb0', 'xtb1', 'xtb2', 'gfn0-xtb', 'gfn1-xtb', 'gfn2-xtb']:
            fh.write("%xtb\n")
            fh.write("     etemp 0\n")
            if xtb_accuracy:
                fh.write(f'     accuracy {xtb_accuracy}\n')
            fh.write('     xtbinputstring "--iterations 1000"\n')
            fh.write("end\n")
        fh.write(f"*xyz {charge} {mult}\n")
        if cp_fragments:
            counterpoise.write_cp_atoms(sym, xyz, fh, cp_fragments)
        else:
            for i in sel:
                fh.write(f"{sym[i]} {xyz[i,0]:.8f} {xyz[i,1]:.8f} {xyz[i,2]:.8f}\n")
        fh.write("*\n")

def run_orca(inp: Path) -> float:
    """Run ORCA and return FINAL SINGLE POINT ENERGY (Hartree)."""
    out = inp.with_suffix(".out")
    if out.exists():
        try:
            return parse_energy(out)
        except RuntimeError:
            print(f"[recompute] Energy not found in {out}")
    cmd = [ORCA_CMD, inp.name]
    # redirect ORCA stdout into the .out file so parse_energy can find it
    with out.open("w") as fh:
        with subprocess.Popen(cmd, cwd=inp.parent, stdout=fh,
                          stderr=subprocess.STDOUT, universal_newlines=True) as proc:
            proc.communicate()
    return parse_energy(out)


def parse_energy(out_path: Path) -> float:
    """Extract energy (Ha) from ORCA output."""
    with out_path.open() as f:
        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                return float(line.split()[-1])
    raise RuntimeError("Energy line not found in {}".format(out_path))


def parse_engrad_file(path: Path) -> np.ndarray:
    """Extract gradient (Eh/bohr) from ORCA .engrad file as array shape (n_atoms,3)."""
    with path.open() as f:
        natm = None
        gradients = []
        # first find number of atoms
        for line in f:
            if "Number of atoms" in line:
                # next non-comment numeric line is natm
                while True:
                    line2 = next(f)
                    if line2.strip() and not line2.strip().startswith('#'):
                        natm = int(line2.strip())
                        break
                break
        if natm is None:
            raise RuntimeError(f"Cannot find atom count in {path}")
        # find gradient block
        for line in f:
            if "current gradient" in line:
                # read 3*natm float lines
                count = 3 * natm
                while len(gradients) < count:
                    line3 = next(f)
                    if line3.strip() and not line3.strip().startswith('#'):
                        gradients.append(float(line3.strip()))
                break
        if len(gradients) != 3 * natm:
            raise RuntimeError(f"Unexpected gradient entries in {path}")
        return np.array(gradients).reshape(natm, 3)

def write_modified_pc_file(original_pc_file: Path, modified_pc_file: Path, exclude_fragments: List[List[int]]):
    """
    Create modified point-charge file excluding charges from atoms of selected fragment indices.
    """
    exclude_atoms = set()
    for frag in exclude_fragments:
        exclude_atoms.update(frag)
    with original_pc_file.open() as fin, modified_pc_file.open("w") as fout:
        lines = fin.readlines()
        # First line is the original number of charges
        n_charges = int(lines[0].strip())
        filtered_lines = []
        for idx, line in enumerate(lines[1:]):  # skip first line
            parts = line.strip().split()
            if len(parts) != 4:
                continue  # skip malformed lines
            q, x, y, z = parts
            atom_index = idx  # charges are listed in order of atom indices
            if atom_index in exclude_atoms:
                continue  # skip this charge
            filtered_lines.append(line)
        # Write the new number of charges
        fout.write(f"{len(filtered_lines)}\n")
        # Write the remaining charges
        for line in filtered_lines:
            fout.write(line)

###############################################################################
#                             MBE bookkeeping                                 #
###############################################################################

def recursive_delta(energies: Dict[Tuple[int, ...], float], order: int) -> Dict[Tuple[int, ...], float]:
    """Compute non-redundant ΔE for all subsets up to *order*."""
    delta: Dict[Tuple[int, ...], float] = {}
    for k in range(1, order + 1):
        for combo in itertools.combinations(range(max(itertools.chain(*energies.keys())) + 1), k):
            if combo not in energies:
                continue
            subtotal = sum(delta[sub] for sub in proper_subsets(combo) if sub in delta)
            delta[combo] = energies[combo] - subtotal
    return delta


def recursive_delta_vector(gradients: Dict[Tuple[int, ...], np.ndarray], order: int) -> Dict[Tuple[int, ...], np.ndarray]:
    """Compute non-redundant Δ gradient arrays for all subsets up to order."""
    delta_g: Dict[Tuple[int, ...], np.ndarray] = {}
    max_idx = max(idx for combo in gradients for idx in combo)
    for k in range(1, order + 1):
        for combo in itertools.combinations(range(max_idx + 1), k):
            if combo not in gradients:
                continue
            subtotal = sum(delta_g[sub] for sub in proper_subsets(combo) if sub in delta_g)
            delta_g[combo] = gradients[combo] - subtotal
    return delta_g


def proper_subsets(t: Tuple[int, ...]):
    for k in range(1, len(t)):
        yield from itertools.combinations(t, k)


###############################################################################
#                             Main workflow                                   #
###############################################################################

def main(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Many-Body Expansion driver (ORCA)")
    p.add_argument("xyz", type=Path, help="Input cluster .xyz file")
    p.add_argument("--order", "-n", type=int, default=3, help="MBE order (default=3)")
    p.add_argument("--method", type=str, default="HF 6-31G*", help="ORCA method/basis line")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--multiplicity", type=int, default=1)
    p.add_argument("--pointcharges", type=Path,
                   help=("External point-charge file; first line = integer "
                         "count, remaining lines = q x y z.  "
                         "File name will be forwarded to ORCA via "
                         '`% pointcharges \"file.pc\"`.'))
    p.add_argument("--nprocs", type=int, default=os.cpu_count() or 1, help="Parallel workers")
    p.add_argument("--scratch", type=Path, default=Path("_mbe_tmp"), help="Scratch directory")
    p.add_argument("--fragments", type=Path,
                   help=("JSON file with fragment definitions.  "
                         "Allowed formats:\n"
                         "  • list[list[int]]                       -> charge 0 for every fragment\n"
                         "  • list[{\"atoms\": [...], \"charge\": q}] -> arbitrary charges"))
    p.add_argument("--orca-path", type=str, default=os.environ.get("ORCA_PATH", "orca"), help="Path to the ORCA executable")
    p.add_argument("--ee", type=Path, help="Point-charge file of supersystem with which to perform electrostatic embedding")
    p.add_argument("--xtb-accuracy", type=int, default=None,
                   help="Accuracy setting for XTB calculations (only if using an XTB method)")
    p.add_argument("--cp", type=bool, default=False,
                   help="Set True to use Boys-Bernardi counterpoise correction. WARNING: breaks gradients.")

    args = p.parse_args(argv)

    # Override the global ORCA_CMD with the user‐supplied path
    global ORCA_CMD
    ORCA_CMD = args.orca_path

    sym, xyz = read_xyz(args.xyz)

    # ------------------------------------------------------------------
    # 0.  Optional external point-charge array
    # ------------------------------------------------------------------
    pointcharge_file: Optional[Path] = None
    ee_pointcharge_file: Optional[Path] = None
    if args.pointcharges:
        # quick sanity-check: header must be an int
        first = args.pointcharges.read_text().splitlines()[0].strip()
        if not first.isdigit():
            raise RuntimeError(f"{args.pointcharges}: first line must be an integer (#charges)")
        pointcharge_file = args.pointcharges
    if args.ee:
        # quick sanity-check: header must be an int
        first = args.ee.read_text().splitlines()[0].strip()
        if not first.isdigit():
            raise RuntimeError(f"{args.ee}: first line must be an integer (#charges)")
        ee_pointcharge_file = args.ee

    # ---------------------------------------------------------------------
    # 1.  Fragment definition & charge parsing
    # ---------------------------------------------------------------------
    frag_charges: List[int]
    if args.fragments:
        raw = json.loads(args.fragments.read_text())
        # a) list[dict]  → {'atoms':[...], 'charge':q}
        if (isinstance(raw, list)
                and len(raw) > 0
                and all(isinstance(f, dict) for f in raw)):
            frags       = [f["atoms"]  for f in raw]
            frag_charges = [int(f.get("charge", 0)) for f in raw]
        # b) simple list[list[int]]
        elif isinstance(raw, list) and all(isinstance(f, list) for f in raw):
            frags        = raw
            frag_charges = [0]*len(frags)
        else:
            raise RuntimeError("--fragments JSON must be a list of lists "
                        "or a list of {'atoms','charge'} objects")
    else:
        frags        = detect_water_fragments(sym, xyz)
        frag_charges = [0]*len(frags)
    n_frag = len(frags)
    if args.order > n_frag:
        raise RuntimeError("MBE order cannot exceed number of fragments")

    workdir = args.scratch.resolve()
    workdir.mkdir(exist_ok=True, parents=True)

    energies: Dict[Tuple[int, ...], float] = {}
    grads: Dict[Tuple[int, ...], np.ndarray] = {}
    combos = generate_combinations(n_frag, args.order)

    def _submit(combo):
        subdir = workdir / ("_".join(map(str, combo)))
        subdir.mkdir(exist_ok=True)
        inp = subdir / f"frag.inp"
        sel_atoms  = [idx for frag_idx in combo for idx in frags[frag_idx]]
        sub_pc_file = None
        # use a local variable to avoid shadowing/assigning the outer pointcharge_file
        pc_to_use = pointcharge_file
        if ee_pointcharge_file:
            sub_pc_file = subdir / "frag.pc"
            write_modified_pc_file(ee_pointcharge_file, sub_pc_file, [frags[frag_idx] for frag_idx in combo])
            pc_to_use = sub_pc_file

        # -----------------------------------------------------------------
        # 2.   Sub-system charge = sum of the constituent fragment charges
        # -----------------------------------------------------------------
        sub_charge = sum(frag_charges[frag_idx] for frag_idx in combo)

        if args.cp:
            write_orca_input(sym, xyz, sel_atoms,
                            args.method, sub_charge, args.multiplicity,
                            inp, pc_to_use,
                            xtb_accuracy=args.xtb_accuracy,
                            cp_fragments=[frags[frag_idx] for frag_idx in combo])
        else:
            write_orca_input(sym, xyz, sel_atoms,
                            args.method, sub_charge, args.multiplicity,
                            inp, pc_to_use,
                            xtb_accuracy=args.xtb_accuracy)
        E = run_orca(inp)
        # parse fragment gradient
        eng = subdir / "frag.engrad"
        Gfrag = parse_engrad_file(eng)
        # map to full array
        Gfull = np.zeros((len(sym), 3))
        for i, a in enumerate(sel_atoms):
            Gfull[a] = Gfrag[i]
        return combo, E, Gfull

    with ThreadPoolExecutor(max_workers=args.nprocs) as pool:
        futures = {pool.submit(_submit, c): c for c in combos}
        for fut in as_completed(futures):
            combo, E, G = fut.result()
            energies[combo] = E
            grads[combo] = G
            combo_str = "(" + ",".join(str(i) for i in combo) + ")"
            print(f"done {combo_str:>15s}  {E: .8f} Ha")

    delta = recursive_delta(energies, args.order)
    E_total = sum(delta.values())
    # compute MBE gradient (non-redundant Δg for each subsystem)
    delta_g = recursive_delta_vector(grads, args.order)
    # cumulative total gradient at final order
    total_grad = sum(delta_g.values())
    # Pre-compute cumulative gradients for each intermediate order k
    cumulative_grad_by_order = {}
    for k in range(1, args.order + 1):
        cumulative_grad_by_order[k] = sum(g for c, g in delta_g.items() if len(c) <= k)

    # write MBE energies to a “.mbe” file
    out_path = args.xyz.with_suffix(".mbe")
    with out_path.open("w") as mbef:
        mbef.write("MBE ENERGY BREAKDOWN\n")
        for k in range(1, args.order + 1):
            term = sum(delta[c] for c in delta if len(c) == k)
            mbef.write(f"ΔE{k} = {term:.10f} Ha   ({term * 627.509474:.4f} kcal/mol)\n")
        mbef.write("-" * 60 + "\n")
        for i in range(1, args.order + 1):
            cum = sum(term for c, term in delta.items() if len(c) <= i)
            mbef.write(f"TOTAL E(MBE{i}) = {cum:.10f} Ha   ({cum * 627.509474:.4f} kcal/mol)\n")
        #mbef.write(f"TOTAL E(MBE{args.order}) = {E_total:.10f} Ha   ({E_total * 627.509474:.4f} kcal/mol)\n")
        print("\nMBE ENERGY BREAKDOWN")

    for k in range(1, args.order + 1):
        term = sum(delta[c] for c in delta if len(c) == k)
        print(f"ΔE{k} = {term: .10f} Ha   ({term*627.509474:.4f} kcal/mol)")
    print("-"*60)
    print(f"TOTAL E(MBE{args.order}) = {E_total: .10f} Ha   ({E_total*627.509474:.4f} kcal/mol)\n")

    print(f"Results written to {out_path}")
    # write MBE gradients for every order up to the requested one
    # File format: blocks labeled ORDER k with per-atom cumulative gradient up to that order
    gpath = args.xyz.with_suffix('.mbegrad')
    with gpath.open('w') as gf:
        gf.write('# Many-Body Expansion Gradients\n')
        gf.write(f'# Source XYZ: {args.xyz.name}\n')
        gf.write(f'# Method: {args.method}\n')
        gf.write(f'# Orders: 1..{args.order}\n')
        gf.write('# Each block is the cumulative gradient up to (and including) order k.\n')
        for k in range(1, args.order + 1):
            gf.write(f'\n# ---- ORDER {k} (cumulative) ----\n')
            gf.write('# Atom_index Symbol Grad_x(Eh/bohr) Grad_y Grad_z\n')
            Gk = cumulative_grad_by_order[k]
            for i, symb in enumerate(sym):
                gx, gy, gz = Gk[i]
                gf.write(f"{i} {symb} {gx:.10f} {gy:.10f} {gz:.10f}\n")
    print(f"MBE(1..{args.order}) gradients written to {gpath}")

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
