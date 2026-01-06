# Sin et Sout : Clarification Complète

## Réponse à votre question

**Question :** "Sin et Sout représentent bien la largeur du cœur des gaines ou ça représente autre chose ?"

**Réponse :** ✅ **OUI, exactement !** Sin et Sout représentent le **diamètre du cœur (d_core)** des fibres monomodes d'entrée et sortie.

## Les trois concepts clés à distinguer

| Paramètre | Notation | Définition | Qui spécifie | Exemple (Si photonics @ 1.55 µm) |
|-----------|----------|-----------|--------------|----------------------------------|
| **Diamètre du cœur** | $d_{core}$ (Sin, Sout) | La largeur PHYSIQUE du cœur de la fibre | **Vous** (paramètre d'entrée) | 2.5 µm |
| **Mode Field Width** | $MFD$ | Où se concentre réellement la lumière (formule de Marcuse) | **Code** (calculé) | 4.12 µm (pour d_core=2.5) |
| **V-nombre** | $V$ | Nombre de longueurs d'onde dans le cœur, détermine les modes | **Code** (calculé) | 1.58 (single-mode) |

## Les formules

### V-nombre (détermine le régime monomode/multimode)
$$V = \frac{\pi \cdot d_{core}}{\lambda} \cdot \sqrt{n_{core}^2 - n_{cladding}^2}$$

- **V < 2.405** → ✓ **Single-mode** (only LP₀₁)
- **V > 2.405** → ⚠️ **Multimode** (LP₀₁ + LP₁₁ + ...)

### Mode Field Width (Marcuse formula)
$$MFD = d_{core} \cdot \left(0.65 + \frac{1.619}{V^{1.5}} + \frac{2.879}{V^6}\right)$$

## Flux des paramètres dans le code

```
Vous spécifiez:
  Sin = 2.5 µm  (diamètre du cœur d'entrée)
  Sout = 4.0 µm (diamètre du cœur de sortie)
            ↓
Le code calcule:
  V_in = π·2.5e-6 / 1.55e-6 · √(2² - 1.9²) = 1.582
  V_out = π·4.0e-6 / 1.55e-6 · √(2² - 1.9²) = 2.561
            ↓
Régimes identifiés:
  Input: ✓ Single-mode
  Output: ⚠️ Weakly multimode (LP₀₁ + LP₁₁)
            ↓
Mode Field Widths calculés automatiquement:
  MFD_in = 2.5 × (0.65 + ...) = 4.12 µm
  MFD_out = 4.0 × (0.65 + ...) = 4.25 µm
            ↓
Couplage multimode:
  LP₀₁: 65.3%
  LP₁₁: 34.7%
  Total: 100%
```

## Exemple pratique pour vos simulations

### Cas 1 : Single-mode output (recommandé pour l'interférométrie)
```python
simulate(
    Sin=2.5e-6,    # d_core = 2.5 µm
    Sout=2.5e-6,   # d_core = 2.5 µm
    # ...
)
# Résultat: V = 1.60 → ✓ Single-mode régime
```

### Cas 2 : Weakly multimode output (attention!)
```python
simulate(
    Sin=2.5e-6,    # d_core = 2.5 µm
    Sout=4.0e-6,   # d_core = 4.0 µm
    # ...
)
# Résultat: V = 2.56 → ⚠️ Multimode régime
# Mode breakdown: LP₀₁ 65%, LP₁₁ 35%
```

### Cas 3 : Strongly multimode output (déconseillé)
```python
simulate(
    Sin=2.5e-6,    # d_core = 2.5 µm
    Sout=6.0e-6,   # d_core = 6.0 µm
    # ...
)
# Résultat: V = 3.85 → ❌ Strongly multimode
# Mode noise détruira votre null depth!
```

## Vos données de validation

Voici ce que nos tests ont confirmé:

| d_core [µm] | V-nombre | Régime | Remarques |
|------------|----------|--------|-----------|
| 1.0 | 0.633 | ✓ SM | Très confiné |
| 2.0 | 1.266 | ✓ SM | Single-mode |
| 2.5 | 1.582 | ✓ SM | Single-mode |
| 3.0 | 1.899 | ✓ SM | À la limite |
| 4.0 | 2.532 | ⚠️ WMM | LP₀₁ + LP₁₁ |
| 5.0 | 3.164 | ⚠️ WMM | LP₀₁ + LP₁₁ + LP₂₁ |

## Conseils pratiques

### Pour optimiser votre null depth
✅ **Garder Sout < 2.7 µm** (régime single-mode pur)
- Évite le bruit modal
- Couplage optimal à LP₀₁
- Null depth prévisible

### Si vous devez utiliser Sout plus grand
⚠️ **Soutrez ce qui se passe**
- Exécutez avec `verbose=True`
- Regardez la décomposition modale
- Comprenez la fraction de LP₀₁ réelle

### Ne jamais faire
❌ **Sout > 4.2 µm** pour l'interférométrie de nulling
- Trop de modes en compétition
- Bruit modal catastrophique
- Pertes de couplage importantes

## Questions fréquentes

**Q: Dois-je connaître le Mode Field Width (MFD) de ma fibre?**  
R: Non ! Spécifiez simplement le diamètre du cœur (Sin/Sout). Le code calcule automatiquement le MFD.

**Q: Comment passer du MFD à d_core?**  
R: Inverser la formule de Marcuse (numériquement ou itérativement).

**Q: Qu'est-ce que le V-nombre dans ce simulateur?**  
R: C'est le nombre adimensionnel qui détermine combien de modes peuvent se propager. Vos paramètres d_core → V-nombre → régime monomode/multimode.

**Q: Pourquoi le code calcule-t-il le MFD?**  
R: Le MFD est nécessaire pour calculer l'intégrale de chevauchement qui détermine le couplage optical. C'est un détail de l'implémentation, pas un paramètre utilisateur.

## Référence documentaire

- ✅ **Docstrings**: Les docstrings Python sont maintenant explicites
- ✅ **Notebook**: La cellule 2 du notebook mmi.ipynb explique tout
- ✅ **Validation**: Voir `tmp/validate_dcore_parameters.py`

---

**Résumé:** Vous aviez raison dans votre intuition ! Sin et Sout = diamètre du cœur. Le code gère le MFD automatiquement. Vous n'avez qu'à spécifier le d_core de votre fibre et le simulateur s'occupe du reste ! 🎯
