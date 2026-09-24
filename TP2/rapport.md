# TP2 — Régularisation, optimisation et métriques

**Auteur :** Wassim Triki
**Identifiant TSP :** wtriki

---

## 1. Création d'un dataset personnalisé

Dataset utilisé : Cardiovascular Disease dataset (Kaggle), chargé via une classe `CardioDataset` héritant de `torch.utils.data.Dataset`.

**Résultat du test du dataset :**


**Pourquoi `StandardScaler()` sur l'ensemble du dataset avant le split est une mauvaise pratique (data leakage) ?**

Le `StandardScaler` calcule la moyenne et l'écart-type de chaque colonne à partir des données qu'on lui fournit. En l'appliquant sur l'ensemble complet (train + val + test réunis) avant de séparer les sous-ensembles, les statistiques utilisées pour normaliser le train set intègrent indirectement des informations provenant du val et du test set — c'est une fuite d'information (**data leakage**). Le modèle profite alors, même subtilement, de connaissances qui ne devraient être disponibles qu'au moment de l'évaluation finale, ce qui rend la performance mesurée artificiellement optimiste par rapport à ce qu'on observerait sur de vraies nouvelles données. La bonne pratique consiste à calculer le `fit` du scaler uniquement sur le train set, puis à appliquer cette même transformation (`transform` seul, sans refaire `fit`) au val et au test set.

**Quelle classe PyTorch utiliser si le dataset est trop volumineux pour la RAM (ex: 500 Go) ?**

`torch.utils.data.IterableDataset`. Contrairement à `Dataset`, qui charge l'intégralité des données en mémoire et permet un accès direct par index via `__getitem__`, `IterableDataset` définit une méthode `__iter__` qui génère les exemples un par un (ou par morceaux) à la demande, typiquement en lisant le fichier progressivement depuis le disque (streaming), sans jamais charger l'ensemble des données en RAM.
