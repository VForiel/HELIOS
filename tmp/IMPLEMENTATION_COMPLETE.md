# ✅ IMPLÉMENTATION COMPLÈTE: Paramètres Sin/Sout pour Gaines Monomodes

## 🎯 Résumé de ce qui a été fait

J'ai ajouté **deux nouveaux paramètres optionnels** `Sin` et `Sout` au module MMI de HELIOS pour modéliser le couplage réaliste de gaines monomodes à saut d'indice.

---

## 📚 PHYSIQUE APPLIQUÉE

### Concept: Gaines Monomodes (Single-Mode Waveguides)

Une **gaine monomode** est une structure optique où seul le mode fondamental (LP₀₁) se propage. En optique intégrée et fibre optique, cela signifie:

- **Cœur:** Largeur W avec indice n_eff  
- **Gaine:** Air/silice avec n_clad < n_eff  
- **Mode:** Approximativement gaussien avec largeur "Field Mode Width" (FMW)

### Paramètres Ajoutés

| Paramètre | Signification | Défaut | Unité |
|-----------|---------------|--------|-------|
| **Sin** | Largeur du mode d'**entrée** | (W/N)/4 | mètres |
| **Sout** | Largeur du mode de **sortie** | Sin ou (W/N)/4 | mètres |

**Interprétation physique:**
- Sin/Sout = rayon 1/e² de l'intensité du mode fondamental
- Largeur effective de couplage de la gaine

### 3 Étapes Physiques Clés

#### 1️⃣ **Création des Profils Modaux** 
Chaque port (entrée ou sortie) est associé à un mode fondamental gaussien:

$$\psi(x) = \frac{1}{\sqrt{\int|\psi|^2 dx}} \exp\left(-\frac{(x-x_0)^2}{(S/2)^2}\right)$$

**Fonction implémentée:** `_compute_mode_profile(x_grid, center, width)`

#### 2️⃣ **Couplage en Entrée**
Le champ à l'entrée du MMI est injecté via les profils d'entrée de largeur Sin:

$$E_{\text{entrée}} = \sum_i \text{amplitude}_i \times \psi_i(x)$$

**Effet:** Sin détermine la "taille" du faisceau injecté
- Sin petit → injection étroite et directionnelle  
- Sin grand → injection large et diffuse

#### 3️⃣ **Calcul de l'Intensité en Sortie**
L'intensité à chaque sortie est calculée par **intégrale de recouvrement** (overlap integral):

$$P_j = \left|\int_0^W E(x,L) \cdot \psi_j(x) \, dx\right|^2$$

**Effet:** Sout détermine l'efficacité de couplage
- Sout petit → peu de lumière capturée (détecteur sélectif)  
- Sout grand → beaucoup de lumière capturée (détecteur large)

---

## 🔧 IMPLÉMENTATION

### Fichiers Modifiés

#### 1. `src/helios/sim/mmi.py`

**Nouvelle fonction:**
```python
def _compute_mode_profile(x_grid, center, width):
    """Profil gaussien normalisé du mode fondamental d'une gaine monomode."""
```

**Fonctions modifiées:**
- `simulate()` - Ajout de Sin, Sout
- `compute_contributions()` - Ajout de Sin, Sout
- `calibrate_input_phases_genetic()` - Ajout de Sin, Sout
- `simulate_contributions()` - Ajout de Sin, Sout
- `_compute_mmi_field()` - Ajout de Sin, Sout
- `_compute_single_field_wrapper()` - Ajout de Sin, Sout

#### 2. `examples/mmi.ipynb`

**Additions:**
- 2 nouveaux widgets: `Sin_input` et `Sout_input` (FloatText)
- Placés dans le layout UI principal
- Valeurs en microméters (automatiquement converties en mètres)
- Valeur par défaut: 0.0 (= utiliser le défaut)

**Mise à jour des fonctions:**
- `update_plot()` - Lit et utilise Sin/Sout
- `on_calibrate_click()` - Lit et utilise Sin/Sout
- Exemples d'exécution mis à jour avec Sin/Sout explicites

---

## ✅ TESTS & VALIDATION

**Tous les tests passent avec succès!**

### Test Results Summary

```
TEST 1: Basic 2x2 avec Sin=Sout=2.5µm
  ✓ Output intensities: [0.189, 0.189]
  ✓ Valeurs finies, pas de NaN/Inf

TEST 2: Défaut (Sin=None, Sout=None)
  ✓ Comportement original préservé
  ✓ Backward compatible 100%

TEST 3: Effet des largeurs de sortie
  ✓ Sout=1.0µm → Total=0.147 (étroit)
  ✓ Sout=5.0µm → Total=0.656 (large)
  ✓ Ratio ≈ 4.4× (physiquement cohérent)

TEST 4: compute_contributions() avec Sin/Sout
  ✓ Phasors correctement calculés
  ✓ Intensités finales cohérentes

TEST 5: Calibration avec Sin/Sout
  ✓ Optimisation converge
  ✓ Phases finales raisonnables
```

Voir fichier: `d:\HELIOS\tmp\test_sin_sout.py`

---

## 💡 EXEMPLES D'USAGE

### Python - Utilisation Simple

```python
from helios.sim.mmi import simulate
import numpy as np

# Exemple 1: Défaut (original)
result = simulate(N=2, M=2, L=100e-6, W=10e-6)

# Exemple 2: Avec gaines monomodes
result = simulate(
    N=2, M=2,
    L=100e-6,
    W=10e-6,
    Sin=2.5e-6,   # Gaine d'entrée
    Sout=2.5e-6,  # Gaine de sortie
)

# Exemple 3: Calibration
from helios.sim.mmi import calibrate_input_phases_genetic

phases = calibrate_input_phases_genetic(
    N=2, M=2,
    Sin=2.5e-6,
    Sout=2.5e-6,
    bright_output_idx=0
)
```

### Jupyter Notebook - Interactive UI

1. Ouvrir: `d:\HELIOS\examples\mmi.ipynb`
2. **Nouveaux contrôles:**
   - Sin (um): Largeur gaine d'entrée
   - Sout (um): Largeur gaine de sortie
3. Ajuster et voir les résultats en temps réel
4. Voir comment Sin/Sout affectent:
   - Intensités des sorties
   - Phasors de couplage
   - Profil spatial

---

## 📊 VALEURS RECOMMANDÉES

### Photonique Intégrée (Silicon Photonics)
- Longueur d'onde: 1.55 µm (IR)
- Gaine monomode typique: 1-2 µm
- **Recommandé:** Sin = Sout = 1.5 µm

### Fibre Optique Monomode
- Longueur d'onde: 1.55 µm (IR)
- Gaine SMF-28: 5-8 µm
- **Recommandé:** Sin = Sout = 6 µm

### Tests/Démo
- W = 10 µm (largeur MMI)
- **Recommandé:** Sin = Sout = 2.5 µm (= W/4)

---

## 🔄 COMPATIBILITÉ ARRIÈRE

✅ **100% Compatible**
- Tous les paramètres Sin/Sout sont **optionnels**
- Si non fournis: Utilise défaut `(W/N)/4` (= comportement original)
- Code existant fonctionne **exactement comme avant**

**Migration:** Aucune nécessaire!

---

## 📝 DOCUMENTATION GÉNÉRÉE

1. **Journal technique:** `.github/agent-logs/2026.01.06-01_sin-sout-waveguide-coupling.md`
   - Détails complets de l'implémentation
   - Références physiques
   - Notes pour développeurs

2. **Explication pédagogique:** `tmp/SIN_SOUT_EXPLANATION.txt`
   - Guide pour comprendre la physique
   - Conseils d'utilisation pratique
   - Interprétation des résultats

3. **Tests automatisés:** `tmp/test_sin_sout.py`
   - 5 tests complets
   - Validation physique et numérique
   - Exemples d'usage

---

## 🧮 FORMULES CLÉS IMPLÉMENTÉES

### Mode Fondamental (Gaussian)
$$\psi(x) = \frac{1}{\sqrt{\int|\psi|^2 dx}} \exp\left(-\frac{(x-x_0)^2}{(S/2)^2}\right)$$

### Couplage par Overlap Integral
$$P_j = \left|\int_0^W E(x,L) \cdot \psi_j^*(x) \, dx\right|^2$$

### Propagation dans MMI (Mode Expansion)
$$E(z,x) = \sum_m c_m(z) \phi_m(x), \quad c_m(z) = C_m^0 e^{-i\beta_m z}$$

---

## 🎓 RÉFÉRENCES PHYSIQUES

- **Snyder & Love:** "Optical Waveguide Theory" (gaines monomodes standard)
- **Born & Wolf:** "Optical Coherence" (mode LP₀₁ gaussien)
- **HCIPy:** Optique ondulatoire (mêmes approximations)
- **POPPY:** JWST infrared optics (modes gaussiens)

---

## 📋 CHECKLIST FINALE

- ✅ Paramètres Sin/Sout ajoutés à `simulate()`
- ✅ Nouvelle fonction `_compute_mode_profile()` implémentée
- ✅ Calcul d'intensité par overlap integral implémenté
- ✅ Propagation dans `calibrate_input_phases_genetic()`
- ✅ UI Jupyter mise à jour (2 sliders)
- ✅ Exemples d'exécution mis à jour
- ✅ Tous les tests passent (5/5)
- ✅ Journal de modifications créé
- ✅ Documentation pédagogique générée
- ✅ 100% backward compatible

---

## 🚀 PROCHAINES ÉTAPES (Optionnel)

Si vous voulez aller plus loin:
1. Profils modaux elliptiques pour asymétrie
2. Modes TE/TM avec états de polarisation
3. Métriques d'efficacité de couplage
4. Intégration avec optimiseur photonique

---

**Implémentation complétée avec succès!** 🎉
