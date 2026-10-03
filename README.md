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
- [x] **IV. Modélisation, DSO 1 : segmentation des patients** : 11 variables sur trois axes, choix de k (coude, silhouette, stabilité ARI), K-Means k = 3, centres standardisés, ACP, description des profils A / B / C (réadmission 7,2 / 10,5 / 12,0 %)
- [x] **V. Modélisation, DSO 2 : classification du risque de réadmission** : découpage 80 / 20 stratifié, régression logistique et deux XGBoost en validation croisée à 5 plis (top 10 % de 19,3 à 20,5 %, au-dessus du critère de 18 %), XGBoost réglé retenu, choix du seuil (liste top 10 %), courbe d'apprentissage
- [ ] VI. Évaluation
- [ ] VII. Synthèse

## Notebook

`analysis/notebook_diabetes_dynamika.ipynb` : déjà exécuté, il se lit sans rien installer.

## Pour les relancer

Disposition attendue (le notebook cherche le dossier `data` en remontant depuis son emplacement) :

```text
data/diabetic_data.csv                  obligatoire (jeu UCI 296, non fourni ici)
analysis/notebook_diabetes_dynamika.ipynb
analysis/figures_modelisation/fig25_acp_readmission.png   image chargée par la section 4.3 (fournie)
```

Le notebook écrit ses propres figures dans `analysis/figures_modelisation/` (préfixe `nbk3_`, non versionnées).

Paquets : `pip install -r requirements.txt` (Python 3.12).

Exécution : `jupyter nbconvert --to notebook --execute --inplace analysis/notebook_diabetes_dynamika.ipynb`
