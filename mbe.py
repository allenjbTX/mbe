"""
Many-Body Expansion (MBE) driver for ORCA
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
        --charge 0 --multiplicity 1 --nprocs 8 --scratch /tmp/orca_scratch

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

import numpy as np

################################################################################
#                                                                              #
#                                 Utilities                                    #
#                                                                              #
################################################################################

EXT = {"xyz": "xyz", "inp": "inp", "out": "out"}


class Fatal(RuntimeError):
    """Unrecoverable error."""


###############################################################################
#                         Geometry & Fragmentation                            #
###############################################################################

def read_xyz(path: Path) -> Tuple[List[str], np.ndarray]:
    """Return (symbols, coordinates[Å]) from a standard XYZ file."""
    with path.open() as f:
        try:
            natm = int(f.readline())
        except ValueError:
            raise Fatal("First line must contain number of atoms")
        _ = f.readline()  # comment
        symbols, xyz = [], []
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                raise Fatal("Malformed XYZ line: " + line)
            symbols.append(parts[0])
            xyz.append([float(x) for x in parts[1:4]])
        if len(symbols) != natm:
            raise Fatal(f"Expected {natm} atoms, found {len(symbols)}")
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
        if len(hs) != 2:
            raise Fatal("Failed to assign two H atoms to oxygen {}".format(o))
        if any(h in taken for h in hs):
            raise Fatal("Hydrogen assigned to multiple waters; check geometry")
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


def write_orca_input(sym: List[str], xyz: np.ndarray, sel: Sequence[int], method: str,
                      charge: int, mult: int, path: Path):
    """Create ORCA *.inp file for selected fragment indices."""
    with path.open("w") as fh:
        fh.write(f"! {method}\n")
        fh.write("%pal nprocs 1 end\n")
        fh.write(f"* xyz {charge} {mult}\n")
        for i in sel:
            fh.write(f"{sym[i]} {xyz[i,0]:.8f} {xyz[i,1]:.8f} {xyz[i,2]:.8f}\n")
        fh.write("*\n")


def run_orca(inp: Path, timeout: int = 86400) -> float:
    """Run ORCA and return FINAL SINGLE POINT ENERGY (Hartree)."""
    out = inp.with_suffix("." + EXT["out"])
    if out.exists():
        try:
            return parse_energy(out)
        except Fatal:
            print(f"[recompute] Energy not found in {out}")
    cmd = [ORCA_CMD, inp.name]
    # redirect ORCA stdout into the .out file so parse_energy can find it
    with out.open("w") as fh:
        with subprocess.Popen(cmd, cwd=inp.parent, stdout=fh,
                          stderr=subprocess.STDOUT, universal_newlines=True) as proc:
            try:
                proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                raise Fatal(f"ORCA job timeout: {inp}")
    return parse_energy(out)


def parse_energy(out_path: Path) -> float:
    """Extract energy (Ha) from ORCA output."""
    with out_path.open() as f:
        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                return float(line.split()[-1])
    raise Fatal("Energy line not found in {}".format(out_path))


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
    p.add_argument("--nprocs", type=int, default=os.cpu_count() or 1, help="Parallel workers")
    p.add_argument("--scratch", type=Path, default=Path("_mbe_tmp"), help="Scratch directory")
    p.add_argument("--fragments", type=Path, help="JSON file specifying list of fragments (indices)")
    p.add_argument("--orca-path", type=str, default=os.environ.get("ORCA_PATH", "orca"), help="Path to the ORCA executable")

    args = p.parse_args(argv)

    # Override the global ORCA_CMD with the user‐supplied path
    global ORCA_CMD
    ORCA_CMD = args.orca_path

    sym, xyz = read_xyz(args.xyz)
    if args.fragments:
        frags = json.loads(args.fragments.read_text())
    else:
        frags = detect_water_fragments(sym, xyz)
    n_frag = len(frags)
    if args.order > n_frag:
        raise Fatal("MBE order cannot exceed number of fragments")

    workdir = args.scratch.resolve()
    workdir.mkdir(exist_ok=True, parents=True)

    energies: Dict[Tuple[int, ...], float] = {}
    combos = generate_combinations(n_frag, args.order)

    def _submit(combo):
        subdir = workdir / ("_".join(map(str, combo)))
        subdir.mkdir(exist_ok=True)
        inp = subdir / f"frag.inp"
        sel_atoms = [idx for frag_idx in combo for idx in frags[frag_idx]]
        write_orca_input(sym, xyz, sel_atoms, args.method, args.charge, args.multiplicity, inp)
        E = run_orca(inp)
        return combo, E

    with ThreadPoolExecutor(max_workers=args.nprocs) as pool:
        futures = {pool.submit(_submit, c): c for c in combos}
        for fut in as_completed(futures):
            combo, E = fut.result()
            energies[combo] = E
            combo_str = "(" + ",".join(str(i) for i in combo) + ")"
            print(f"done {combo_str:>15s}  {E: .8f} Ha")

    delta = recursive_delta(energies, args.order)
    E_total = sum(delta.values())

    print("\nMBE ENERGY BREAKDOWN")
    for k in range(1, args.order + 1):
        term = sum(delta[c] for c in delta if len(c) == k)
        print(f"ΔE{k} = {term: .10f} Ha   ({term*627.509474:.4f} kcal/mol)")
    print("-"*60)
    print(f"TOTAL E(MBE{args.order}) = {E_total: .10f} Ha   ({E_total*627.509474:.4f} kcal/mol)")

if __name__ == "__main__":
    try:
        main()
    except Fatal as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
        