"""
Démonstration de la propagation de Fresnel (Champ Proche) avec HELIOS.

Ce script illustre l'évolution du profil d'intensité d'un front d'onde
passant à travers une ouverture circulaire, en utilisant la méthode
du spectre angulaire (Angular Spectrum Method).

Nous observons la transition du champ proche (figures géométriques, ombres nettes)
vers le champ lointain (tache d'Airy), caractérisée par le nombre de Fresnel.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

# Ajout du chemin vers les sources si nécessaire
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import helios

def demo_fresnel():
    # 1. Configuration
    wavelength = 633 * u.nm  # Laser HeNe rouge
    size = 10 * u.mm         # Taille totale de la grille
    npix = 512               # Résolution
    
    # Rayon de l'ouverture
    radius = 0.5 * u.mm
    
    print(f"Configuration :")
    print(f"  Longueur d'onde : {wavelength}")
    print(f"  Taille grille   : {size} ({npix}x{npix})")
    print(f"  Rayon ouverture : {radius}")
    
    # 2. Création du Front d'onde initial
    wf = helios.Wavefront(wavelength=wavelength, size=size, npix=npix)
    
    # Création d'une pupille circulaire simple
    pupil = helios.Pupil(diameter=size)
    pupil.add_disk(radius=radius)
    
    # Application de la pupille
    wf[:] = pupil.get_array(npix)
    
    # 3. Définition des distances de propagation
    # Nombre de Fresnel N_F = a^2 / (L * lambda)
    # L = a^2 / (N_F * lambda)
    # Pour N_F = 1 (transition), L ~ (0.5mm)^2 / 633nm ~ 0.4 m
    
    distances = [0 * u.cm, 5 * u.cm, 20 * u.cm, 40 * u.cm, 100 * u.cm]
    fresnel_numbers = [(radius**2 / (d * wavelength)).decompose() if d.value > 0 else np.inf for d in distances]
    
    print("\nPropagation...")
    
    # 4. Propagation et Visualisation
    fig, axes = plt.subplots(1, len(distances), figsize=(15, 4))
    
    # Zoom pour mieux voir la figure centrale (on affiche +/- 2mm)
    zoom_val = 2 * u.mm
    
    for i, d in enumerate(distances):
        print(f"  Distance : {d} (N_F ~ {fresnel_numbers[i]:.2f})")
        
        if d.value == 0:
            wf_prop = wf
        else:
            # Propagation de Fresnel (ASM)
            wf_prop = wf.propagate_fresnel(distance=d)
            
        # Récupération de l'intensité
        intensity = wf_prop.intensity.value
        
        # Si on a une dimension source (1, H, W), on prend le premier élément
        if intensity.ndim == 3:
            intensity = intensity[0]
        
        # Affichage
        ax = axes[i]
        
        # On utilise les coordonnées physiques pour l'affichage
        extent, xlabel, ylabel = helios.core.simulation.get_smart_extent(wf_prop.shape, wf_prop.pixel_scale)
        
        im = ax.imshow(intensity, extent=extent, cmap='inferno', origin='lower')
        
        ax.set_title(f"z = {d}\n$N_F \\approx {fresnel_numbers[i]:.1f}$")
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_yticks([])
            
        # Zoom sur la zone centrale
        ax.set_xlim(-zoom_val.to(u.mm).value, zoom_val.to(u.mm).value)
        ax.set_ylim(-zoom_val.to(u.mm).value, zoom_val.to(u.mm).value)

    plt.suptitle(f"Diffraction de Fresnel (Ouverture circulaire r={radius})", fontsize=14)
    plt.tight_layout()
    
    if os.environ.get("HELIOS_SAVE_PLOTS") == "true":
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated/examples'))
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(__file__).replace('.py', '_1.png')
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()

    # --- Comparaison au Plan Focal (f = 1 m) ---
    print("\n--- Comparaison au Plan Focal (f = 1 m) ---")
    focal_length = 1 * u.m
    
    # 1. Méthode Fraunhofer (FFT)
    print("1. Propagation Fraunhofer (FFT)")
    wf_fft = wf.copy()
    # propagate() utilise une FFT et redimensionne la grille pour le plan focal
    wf_fft = wf_fft.propagate(distance=focal_length)
    
    # 2. Méthode Fresnel (Lentille + ASM)
    print("2. Propagation Fresnel (Lentille + ASM)")
    wf_fresnel = wf.copy()
    lens = helios.Lens(focal_length=focal_length)
    wf_fresnel = lens.process(wf_fresnel) # Application de la phase quadratique
    wf_fresnel = wf_fresnel.propagate_fresnel(distance=focal_length) # Propagation ASM
    
    # Visualisation comparative
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Fraunhofer
    ext_fft, xl_fft, yl_fft = helios.core.simulation.get_smart_extent(wf_fft.shape, wf_fft.pixel_scale)
    im1 = axes[0].imshow(wf_fft.intensity.value[0] if wf_fft.ndim==3 else wf_fft.intensity.value, 
                   extent=ext_fft, cmap='inferno', origin='lower')
    axes[0].set_title("Fraunhofer (FFT)\n(Grille redimensionnée)")
    axes[0].set_xlabel(xl_fft)
    axes[0].set_ylabel(yl_fft)
    plt.colorbar(im1, ax=axes[0], label='Intensité')
    
    # Plot Fresnel
    ext_fres, xl_fres, yl_fres = helios.core.simulation.get_smart_extent(wf_fresnel.shape, wf_fresnel.pixel_scale)
    im2 = axes[1].imshow(wf_fresnel.intensity.value[0] if wf_fresnel.ndim==3 else wf_fresnel.intensity.value, 
                   extent=ext_fres, cmap='inferno', origin='lower')
    axes[1].set_title("Fresnel (Lentille + ASM)\n(Grille fixe)")
    axes[1].set_xlabel(xl_fres)
    axes[1].set_ylabel(yl_fres)
    plt.colorbar(im2, ax=axes[1], label='Intensité')
    
    # Zoom commun pour comparer (si possible)
    # La méthode ASM a une résolution limitée par la grille pupille (pixel ~20um)
    # La tache de diffraction fait ~63um. On devrait la voir, mais pixelisée.
    zoom_focal = 0.5 * u.mm
    
    # Appliquer le zoom si les unités sont compatibles (m/mm)
    try:
        limit = zoom_focal.to(u.mm).value
        # Pour Fraunhofer (souvent en mm au plan focal)
        if 'mm' in xl_fft or 'm' in xl_fft:
             axes[0].set_xlim(-limit, limit)
             axes[0].set_ylim(-limit, limit)
        
        # Pour Fresnel (toujours en mm/m)
        axes[1].set_xlim(-limit, limit)
        axes[1].set_ylim(-limit, limit)
    except:
        pass

    plt.suptitle(f"Comparaison au foyer f={focal_length}\nNotez la différence de résolution (FFT vs ASM fixe)", fontsize=14)
    plt.tight_layout()
    
    if os.environ.get("HELIOS_SAVE_PLOTS") == "true":
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../generated/examples'))
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(__file__).replace('.py', '_2.png')
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    demo_fresnel()
