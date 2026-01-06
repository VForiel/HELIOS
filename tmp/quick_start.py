#!/usr/bin/env python
"""
QUICK START: Sin/Sout Parameters for HELIOS MMI

This script demonstrates the new Sin/Sout parameters for single-mode waveguide coupling.
"""

import sys
sys.path.insert(0, r'd:\HELIOS\src')

import numpy as np
from helios.sim.mmi import simulate

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    HELIOS MMI - Sin/Sout Quick Start                       ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

print("Exemple 1: Simulation simple SANS Sin/Sout (comportement original)")
print("-" * 70)

result1 = simulate(
    N=2, M=2,
    L=100e-6,
    W=10.0e-6,
    wavelength=1.55e-6,
    input_amplitudes=np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2)]),
    num_modes=50,
    verbose=False,
    # Sin et Sout non fournis → utilise défaut
)

print(f"Amplitudes de sortie: {result1}")
print(f"Intensités: {np.abs(result1)**2}")
print(f"Intensité totale: {np.sum(np.abs(result1)**2):.4f}")
print()

print("Exemple 2: AVEC Sin/Sout = 2.5 µm (gaines étroites)")
print("-" * 70)

result2 = simulate(
    N=2, M=2,
    L=100e-6,
    W=10.0e-6,
    wavelength=1.55e-6,
    input_amplitudes=np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2)]),
    num_modes=50,
    verbose=False,
    Sin=2.5e-6,   # Gaine d'entrée: 2.5 µm
    Sout=2.5e-6,  # Gaine de sortie: 2.5 µm
)

print(f"Amplitudes de sortie: {result2}")
print(f"Intensités: {np.abs(result2)**2}")
print(f"Intensité totale: {np.sum(np.abs(result2)**2):.4f}")
print()

print("Exemple 3: AVEC Sin=2.5 µm, Sout=5.0 µm (sortie large)")
print("-" * 70)

result3 = simulate(
    N=2, M=2,
    L=100e-6,
    W=10.0e-6,
    wavelength=1.55e-6,
    input_amplitudes=np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2)]),
    num_modes=50,
    verbose=False,
    Sin=2.5e-6,   # Entrée étroite
    Sout=5.0e-6,  # Sortie large (plus de couplage)
)

print(f"Amplitudes de sortie: {result3}")
print(f"Intensités: {np.abs(result3)**2}")
print(f"Intensité totale: {np.sum(np.abs(result3)**2):.4f}")
print()

print("Comparaison des résultats:")
print("-" * 70)
print(f"Défaut (W/N/4):        I_total = {np.sum(np.abs(result1)**2):.4f}")
print(f"Sout=2.5 µm:           I_total = {np.sum(np.abs(result2)**2):.4f}")
print(f"Sout=5.0 µm (large):   I_total = {np.sum(np.abs(result3)**2):.4f}")
print(f"\nRatio (large/narrow) = {np.sum(np.abs(result3)**2) / np.sum(np.abs(result2)**2):.2f}x")
print()

print("Observations physiques:")
print("-" * 70)
print("""
✓ Plus Sout est LARGE, plus d'énergie est COUPLÉE en sortie
✓ C'est l'effet de l'intégrale de recouvrement (overlap integral)
✓ Mode étroit (1 µm) = détecteur sélectif
✓ Mode large (5 µm) = détecteur sensible

Application pratique:
- Photonique intégrée (1-2 µm): gaines monomodes étroites
- Fibre optique (5-8 µm): gaines monomodes larges
""")

print("\nPour un usage interactif, consulter: examples/mmi.ipynb")
print("Pour tester, exécuter: tmp/test_sin_sout.py")
