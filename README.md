# Diabetes 130-US Hospitals : segmentation (k = 3) et classification du risque de réadmission

Projet mené selon la démarche CRISP-DM sur le jeu UCI *Diabetes 130-US Hospitals* (101 766 séjours, 1999-2008).

## Objectifs

| Objectif métier | Objectif data science |
| --- | --- |
| **BO 1** : décrire les profils de patients pour adapter le suivi | **DSO 1** : segmentation K-Means, validée par le taux de réadmission |
| **BO 2** : repérer, à la sortie, les patients à suivre en priorité | **DSO 2** : classification supervisée ; succès si au moins 18 % de réadmis parmi les 10 % de scores les plus élevés (taux de base 9,0 %) |

## Avancement

- [x] **I. Compréhension du métier** : contexte, état de l'art, BO / DSO, mesures
- [x] **II. Compréhension des données** : cible, séjours par patient, valeurs manquantes, âge, HbA1c, insuline, médicaments, corrélations, distributions
- [x] **III. Préparation des données** : variables dérivées, exclusions et déduplication, valeurs manquantes, regroupements, feature engineering (101 766 séjours → 69 977 patients)
- [ ] IV. Modélisation, DSO 1 : segmentation des patients
- [ ] V. Modélisation, DSO 2 : classification du risque de réadmission
- [ ] VI. Évaluation
- [ ] VII. Synthèse

## Notebook

`analysis/notebook_diabetes_dynamika.ipynb` : déjà exécuté, il se lit sans rien installer.

## Pour les relancer

Disposition attendue (le notebook cherche le dossier `data` en remontant depuis son emplacement) :

```text
data/diabetic_data.csv                  obligatoire (jeu UCI 296, non fourni ici)
analysis/notebook_diabetes_dynamika.ipynb
```

Paquets : `pip install -r requirements.txt` (Python 3.12).

Exécution : `jupyter nbconvert --to notebook --execute --inplace analysis/notebook_diabetes_dynamika.ipynb`
