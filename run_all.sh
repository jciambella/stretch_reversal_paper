#!/usr/bin/env bash
# Regenerates every figure and table of the paper in the current directory.
set -e
python3 make_figures.py                                                    # Figs 1, 2, 5
python3 biaxial_equilibrium_mapper.py --selftest
python3 biaxial_equilibrium_mapper.py --material constant   --alpha 0.70 --plot   # Fig. 3
python3 biaxial_equilibrium_mapper.py --material mooney1000 --alpha 0.90 --plot   # Fig. 4
python3 exact_elimination.py                                               # Sect. 7
python3 stability_margins.py --selftest
python3 stability_margins.py --table                                       # Table 1
