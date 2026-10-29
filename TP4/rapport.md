# CSC 8607 – TP4 : Réseaux récurrents et génération de partitions ABC

**Fichiers du dossier `TP4/`**

| Fichier | Rôle |
|---|---|
| `train_music.py` | Chargement, vocabulaire, `MusicDataset`, `MusicModel` (RNN/LSTM), `evaluate_model`, `train_model` (norme des gradients, clipping, Early Stopping), `generate_music`. Modes : `--mode explore`, `quick`, `generate`, `full` |
| `run_music.sh` | Job Slurm (`sbatch run_music.sh [mode]`) |
| `figures/` | Courbes TensorBoard insérées dans ce rapport |
| `captures/` | Captures d'écran du terminal |
| `generations/` | Partitions ABC générées |
| `logs/` | Sorties Slurm des jobs |

> Les passages marqués **⚠️ À COMPLÉTER** dépendent des résultats de mes jobs.

---

## Exercice 1 – Exploration et prétraitement des partitions ABC

### Étape 1 – Chargement

```python
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
```

#### 1.a.1 Nombre de partitions

**⚠️ À COMPLÉTER**

- `train.json` : **…** partitions
- `validation.json` : **…** partitions

#### 1.a.2 Première partition et rôle de l'en-tête

**⚠️ À COMPLÉTER** (texte de `train_data[0]["abc notation"]`)

```
…
```

En-tête ABC (lignes `Lettre:valeur` avant les notes) :

| Champ | Rôle |
|---|---|
| `X:` | Numéro de référence de la partition dans le recueil. C'est obligatoirement le premier champ : il marque le début d'un nouvel air. |
| `T:` | Titre du morceau (s'il est présent). |
| `M:` | Métrique (mesure) : nombre et valeur des temps par mesure, par exemple `6/8` (2 temps de 3 croches, typique d'une *jig*) ou `4/4` (*reel*). Le lecteur s'en sert pour placer les barres et l'accentuation. |
| `L:` | Durée de note par défaut : `L:1/8` veut dire qu'une lettre seule (`A`) dure une croche ; `A2` dure deux fois plus, `A/2` deux fois moins. |
| `K:` | Tonalité (armure), par exemple `G` (sol majeur, un fa ♯) ou `Dmaj`. C'est le dernier champ de l'en-tête : tout ce qui suit est la mélodie. Elle fixe les altérations implicites des notes. |
| `R:` | (si présent) Type de danse : *reel*, *jig*, *hornpipe*… |

Pour le modèle, l'en-tête est un **contexte global** : la métrique et la tonalité conditionnent toute la suite (nombre de notes par mesure, notes « autorisées »). Un modèle auto-régressif qui a lu `M:6/8` et `K:G` en début de séquence doit en garder la trace sur des centaines de caractères, ce qui exige une mémoire à long terme.

### Étape 2 – Vocabulaire et padding

```python
unique_chars = sorted(list(set(all_text)))
char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
index_to_char = {idx: char for idx, char in enumerate(unique_chars)}

def vectorize_string(s):
    return [char_to_index[char] for char in s]

def pad_or_truncate(sequence, max_length=SEQ_LENGTH):
    if len(sequence) < max_length:
        return sequence + " " * (max_length - len(sequence))
    else:
        return sequence[:max_length]
```

#### 1.b.1 Statistiques

**⚠️ À COMPLÉTER**

- Taille du vocabulaire : **…** caractères uniques
- Longueur moyenne : **…** caractères ; longueur maximale : **…** caractères
- Partitions plus longues que 500 caractères (donc tronquées) : **…**

#### 1.b.2 Pourquoi `sorted(...)` ?

Un `set` Python n'a **pas d'ordre garanti**. Pour les chaînes de caractères, l'ordre d'itération dépend de leur valeur de hachage, et Python **randomise le hachage des chaînes à chaque lancement de l'interpréteur** (`PYTHONHASHSEED`, activé par défaut pour des raisons de sécurité). Sans `sorted()`, deux exécutions du même script peuvent donc construire des `char_to_index` différents : `'A'` peut valoir 12 aujourd'hui et 47 demain.

Si l'on sauvegardait les poids aujourd'hui et qu'on relançait le script demain pour générer :
- la ligne 12 de la matrice d'`Embedding` (apprise pour `'A'`) serait utilisée pour un autre caractère ;
- la sortie n° 47 de la couche `fc` (apprise pour prédire `'A'`) serait décodée en un autre caractère par `index_to_char`.

Le modèle se chargerait **sans aucune erreur** (les dimensions sont identiques), mais il recevrait et produirait des caractères mélangés : la musique générée serait du charabia. `sorted()` rend le vocabulaire **déterministe** : il ne dépend que de l'ensemble des caractères, pas de l'exécution. (Une alternative robuste consiste à sauvegarder `char_to_index` avec les poids.)

#### 1.b.3 Pourquoi tronquer à 500 plutôt que padder jusqu'à `max_len` ?

`max_len` (**…** caractères) est bien plus grand que la longueur typique (**…** en moyenne). Padder tout le monde jusqu'à `max_len` aurait trois coûts :

- **Temps de calcul** : un RNN traite les pas de temps **séquentiellement** (h_t dépend de h_{t-1}), on ne peut pas paralléliser sur la longueur. Le coût d'un batch est proportionnel à `seq_len` : passer de 500 à `max_len` multiplierait le temps d'une époque par environ `max_len / 500` (**…**), alors que l'immense majorité des pas de temps ajoutés ne contiendrait que des espaces.
- **Mémoire GPU** : la BPTT garde en mémoire les activations de **tous** les pas de temps pour le backward. La mémoire des activations est proportionnelle à `batch_size × seq_len × hidden_size` ; elle exploserait d'autant, au point de forcer à réduire le batch.
- **Apprentissage** : la plupart des cibles deviendraient du padding. La loss serait dominée par la tâche triviale « après un espace de padding vient un espace », ce qui dilue le signal utile et biaise le modèle vers la prédiction d'espaces. Les séquences très longues aggravent aussi le *vanishing/exploding gradient* (le gradient traverse plus de pas).

Tronquer à 500 (≈ 2 × la moyenne) garde **la quasi-totalité** des partitions entières (**…** % seulement sont coupées) et ne perd que la fin des morceaux les plus longs. Le padding reste modéré, et le coût d'un batch est maîtrisé.

---

## Exercice 2 – Dataset auto-régressif et modélisation

### Étape 1 – `MusicDataset`

```python
def __getitem__(self, idx):
    seq = self.data[idx]
    x = torch.tensor(seq[:-1], dtype=torch.long)  # tous les caractères sauf le dernier
    y = torch.tensor(seq[1:], dtype=torch.long)   # tous sauf le premier (décalage de +1)
    return x, y
```

#### 2.a Shapes et type

Avec `SEQ_LENGTH = 500` et `BATCH_SIZE = 128`, chaque séquence de 500 indices donne 499 entrées et 499 cibles :

- `x` : **`(128, 499)`**, `torch.int64`
- `y` : **`(128, 499)`**, `torch.int64`

(sauf le dernier batch de l'époque, qui contient le reste, par exemple `(16, 499)` pour 10 000 partitions : 10 000 = 78 × 128 + 16.)

Vérifié par `--mode explore` : **⚠️ À COMPLÉTER** (`Shapes d'un batch : x = (128, 499) …`). Le contrôle `x[0, 1:6] == y[0, 0:5]` confirme le décalage d'un pas de temps : la cible au temps t est l'entrée au temps t + 1.

**Pourquoi `torch.long` ?** Ces tenseurs contiennent des **indices de catégories**, pas des grandeurs continues :
- `nn.Embedding` attend des indices entiers (`LongTensor` ou `IntTensor`) pour aller chercher une ligne de sa table ; un `float` n'a pas de sens comme numéro de ligne.
- `nn.CrossEntropyLoss` attend des cibles de classe entières de type `long` (int64) pour sélectionner le logit de la bonne classe.
- Un `float32` introduirait de plus des approximations (un index n'est pas une valeur qu'on additionne ou qu'on dérive) et laisserait croire à une relation d'ordre entre caractères.

### Étape 2 – `MusicModel`

```python
self.embedding = nn.Embedding(vocab_size, embedding_dim)
self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)   # ou nn.RNN(...)
self.fc = nn.Linear(hidden_size, vocab_size)
```

#### 2.b.1 Calcul théorique des paramètres (V = 95, E = 256, H = 512)

**Embedding** : une table de V vecteurs de dimension E : 95 × 256 = **24 320**.

**RNN (Elman)** : h_t = tanh(W_ih x_t + b_ih + W_hh h_{t−1} + b_hh)

| Terme | Dimension | Paramètres |
|---|---|---|
| W_ih | H × E = 512 × 256 | 131 072 |
| W_hh | H × H = 512 × 512 | 262 144 |
| b_ih | H | 512 |
| b_hh | H | 512 |
| **Total RNN** | | **394 240** |

**LSTM** : 4 portes (entrée i, oubli f, candidat g, sortie o), chacune avec ses propres W_ih, W_hh, b_ih, b_hh de mêmes dimensions que le RNN. PyTorch les empile en matrices de 4H lignes :

| Terme | Dimension | Paramètres |
|---|---|---|
| W_ih | 4H × E = 2048 × 256 | 524 288 |
| W_hh | 4H × H = 2048 × 512 | 1 048 576 |
| b_ih | 4H | 2 048 |
| b_hh | 4H | 2 048 |
| **Total LSTM** | | **1 576 960** |

**Couche `fc`** : H × V + V = 512 × 95 + 95 = **48 735**.

| Modèle | Embedding | Bloc récurrent | fc | **Total** |
|---|---|---|---|---|
| RNN | 24 320 | 394 240 | 48 735 | **467 295** |
| LSTM | 24 320 | 1 576 960 | 48 735 | **1 650 015** |

Vérification par `count_parameters` : avec V = 95, j'obtiens bien **467 295** (RNN) et **1 650 015** (LSTM). Avec le vrai vocabulaire (V = **…**), `--mode explore` donne : **⚠️ À COMPLÉTER**.

**Le bloc LSTM a-t-il exactement 4 fois plus de paramètres que le bloc RNN ?** **Oui** : 4 × 394 240 = 1 576 960. Un RNN calcule une seule transformation affine de (x_t, h_{t−1}) ; un LSTM en calcule quatre, de mêmes dimensions (une par porte i, f, g, o). En revanche, le **modèle complet** n'a pas 4 fois plus de paramètres (1 650 015 / 467 295 ≈ 3,53), car l'Embedding et la couche `fc` sont identiques dans les deux cas.

#### 2.b.2 Pourquoi `nn.Embedding` plutôt qu'une couche linéaire 1 → E sur l'indice ?

Passer l'indice scalaire i dans `nn.Linear(1, E)` donnerait le vecteur i·w + b :
- **Ordre et distances artificiels** : l'indice est arbitraire (il vient de l'ordre alphabétique de `sorted`). Le caractère n° 10 serait traité comme « presque » le n° 11 et « très loin » du n° 80, sans aucune raison musicale.
- **Expressivité quasi nulle** : tous les caractères seraient sur **une même droite** de l'espace R^E (des multiples d'un seul vecteur w, décalés de b). Le réseau ne pourrait pas donner à chaque caractère une représentation propre.

`nn.Embedding` est une **table de correspondance** : chaque caractère a son propre vecteur de dimension E, appris librement. C'est mathématiquement équivalent à multiplier un vecteur *one-hot* de taille V par une matrice V × E, sans hypothèse d'ordre entre catégories, mais en bien plus efficace (on lit une ligne au lieu de faire un produit matriciel avec un vecteur presque vide). Le réseau peut ainsi rapprocher les caractères qui jouent des rôles voisins (les notes `A`–`G` entre elles, les chiffres de durée entre eux, les barres `|` et `:`…).

#### 2.b.3 Différence entre les `hidden_state` du RNN et du LSTM

- **`nn.RNN`** renvoie **un seul tenseur** h_n, de forme `(num_layers, batch, hidden_size)` = `(1, B, 512)` : le dernier état caché.
- **`nn.LSTM`** renvoie **un tuple `(h_n, c_n)`**, deux tenseurs de même forme `(1, B, 512)` :
  - **h_t (hidden state)** : la « mémoire de travail » ou sortie du LSTM au temps t, h_t = o_t ⊙ tanh(c_t). C'est elle qui est passée à la couche `fc` pour prédire le caractère suivant, et qui est réinjectée dans les portes au pas suivant.
  - **c_t (cell state)** : la **mémoire à long terme**, mise à jour de façon additive (c_t = f_t ⊙ c_{t−1} + i_t ⊙ g_t). Elle transporte l'information sur de longues distances (la métrique, la tonalité, le motif de la mélodie) sans être écrasée à chaque pas ; la porte de sortie o_t décide ce qu'on en expose dans h_t.

Il faut donc conserver **les deux** pour la génération : c'est ce que fait `generate_music` en repassant le tuple `hidden_state` à chaque pas.

---

## Exercice 3 – Entraînement, dynamique des gradients, RNN vs LSTM

Trous complétés (`train_music.py`) :

```python
loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
...
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
...
optimizer.step()
```

Ajouts par rapport à l'énoncé : le nom du run contient `hidden_size` et le fichier de sauvegarde s'appelle `best_{rnn_type}_h{hidden_size}.pth`, pour que l'entraînement long (h = 1024) n'écrase pas le meilleur LSTM du Run 3 (h = 512). J'affiche aussi la norme **maximale** des gradients de chaque époque et la durée d'une époque, et un paramètre `max_minutes` arrête proprement l'entraînement long avant la limite de 2 h de Slurm, pour que la génération finale ait le temps de s'exécuter.

Lancement : `sbatch run_music.sh quick` (les 3 runs à la suite, 15 époques, `subset_size = 10000`, `lr = 5e-3`).

#### 3.1 Pourquoi `logits.reshape(-1, vocab_size)` et `y.reshape(-1)` ?

`nn.CrossEntropyLoss` attend des logits de forme **`(N, C)`** (N exemples, C classes) et des cibles de forme **`(N,)`**. Or le modèle sort des logits de forme `(batch, seq_len, vocab_size)` et les cibles sont `(batch, seq_len)`. En modélisation caractère par caractère, **chaque pas de temps de chaque séquence est un problème de classification indépendant** à `vocab_size` classes. On aplatit donc les deux premières dimensions :

- `logits.reshape(-1, vocab_size)` → `(batch × seq_len, vocab_size)`
- `y.reshape(-1)` → `(batch × seq_len,)`

Le `-1` laisse PyTorch calculer la dimension : c'est le **nombre total de caractères à prédire dans le batch**, soit 128 × 499 = **63 872** pour un batch complet. La loss obtenue est la moyenne de l'entropie croisée sur ces 63 872 prédictions.

(Sans le reshape, PyTorch interpréterait `(B, T, V)` comme `(N, C, d)`, c'est-à-dire avec les classes en 2e dimension : il prendrait les 499 pas de temps pour des classes, et lèverait une erreur ou calculerait une loss fausse.)

#### 3.2 Pourquoi `clip_grad_norm_` après `loss.backward()` et avant `optimizer.step()` ?

- **Après `backward()`** : ce sont les gradients `p.grad` que l'on mesure et que l'on réduit. Ils n'existent (pour ce batch) qu'une fois la rétropropagation terminée ; avant, ils valent `None` (on vient de faire `zero_grad`).
- **Avant `step()`** : `optimizer.step()` utilise `p.grad` pour mettre à jour les poids. Il faut donc avoir rescalé les gradients **avant** la mise à jour. Le clipping appelé après `step()` ne servirait à rien : le pas trop grand aurait déjà été appliqué aux poids, et les gradients seraient remis à zéro au batch suivant.

`clip_grad_norm_` calcule la norme L2 globale ‖g‖ de tous les gradients concaténés ; si ‖g‖ > `max_norm`, il les multiplie tous par `max_norm / ‖g‖` (même direction, longueur limitée). Il renvoie la norme **avant** clipping, ce qui permet de la tracer dans TensorBoard dans les deux cas (`max_norm = inf` ne modifie rien).

#### 3.3 Analyse expérimentale

**⚠️ À COMPLÉTER** avec les courbes et le tableau des 3 runs.

![Loss de validation](figures/ex3_loss_val.png)

![Accuracy de validation](figures/ex3_acc_val.png)

![Norme des gradients](figures/ex3_gradnorm.png)

| Run | Modèle | Clipping | Meilleure Val Loss | Val Acc finale | Norme moyenne des gradients | Norme max | Temps / époque |
|---|---|---|---|---|---|---|---|
| 1 | RNN | aucun | … | … | … | … | … |
| 2 | RNN | 1,0 | … | … | … | … | … |
| 3 | LSTM | 1,0 | … | … | … | … | … |

**Run 1 vs Run 2 (effet du clipping).** **⚠️ À COMPLÉTER** (observation).

Le problème corrigé est l'**explosion du gradient** (*exploding gradient*). En BPTT, le gradient de la loss au temps t par rapport à l'état h_k (k < t) est un produit de t − k jacobiennes :

∂h_t / ∂h_k = ∏_{j=k+1}^{t} diag(tanh′(a_j)) · W_hh

Sa norme se comporte comme ‖W_hh‖^{t−k} (pondéré par les dérivées de tanh ≤ 1). Si la plus grande valeur singulière de W_hh dépasse 1 (ce qui arrive vite avec `lr = 5e-3`), ce produit **croît exponentiellement** avec la distance : sur 499 pas, quelques batches produisent des gradients énormes. La mise à jour correspondante envoie les poids dans une zone de la surface de perte où la loss remonte brutalement (les « falaises » des RNN), d'où des pics de loss et une instabilité. Le clipping **borne la longueur du pas** sans changer sa direction, et supprime ces sauts. Il ne résout pas en revanche le **vanishing gradient** (quand le produit tend vers 0), qui empêche le RNN d'apprendre les dépendances lointaines.

**Run 2 (RNN) vs Run 3 (LSTM).** **⚠️ À COMPLÉTER** (vitesse de convergence, loss et accuracy finales).

Explication mathématique. Le LSTM calcule, à partir de x_t et h_{t−1} :

- f_t = σ(W_f [h_{t−1}, x_t] + b_f) (oubli), i_t = σ(W_i […] + b_i) (entrée), o_t = σ(W_o […] + b_o) (sortie)
- g_t = tanh(W_g [h_{t−1}, x_t] + b_g) (candidat)
- **c_t = f_t ⊙ c_{t−1} + i_t ⊙ g_t**, puis h_t = o_t ⊙ tanh(c_t)

La mise à jour de la *cell state* est **additive**. En suivant le chemin direct c_{t−1} → c_t :

∂c_t / ∂c_{t−1} = diag(f_t)   donc   ∂c_t / ∂c_k = ∏_{j=k+1}^{t} diag(f_j)

Contrairement au RNN, ce produit **ne contient ni W_hh ni tanh′** : il ne dépend que des portes d'oubli, que le réseau **apprend** à maintenir proches de 1 quand une information doit être conservée. Le gradient peut alors traverser des centaines de pas sans s'annuler ni exploser (le « carrousel d'erreur constante »). La porte d'entrée i_t décide quand écrire une nouvelle information, et la porte de sortie o_t quand la lire.

Dans une partition, c'est exactement ce qu'il faut : se souvenir de la métrique `M:` et de la tonalité `K:` lues dans l'en-tête, compter les temps pour placer les barres `|` au bon endroit, et reprendre un motif ou une reprise `:|` des dizaines de caractères plus loin. Le RNN simple, dont le gradient s'évanouit au-delà de quelques dizaines de pas, n'apprend surtout que des statistiques locales (quelle note suit quelle note).

#### 3.4 Pourquoi l'`Accuracy/Validation` est-elle artificiellement optimiste ?

Une partition de 250 caractères paddée à 500 contient **250 espaces de padding**, soit la moitié des cibles. Or prédire le padding est **trivial** : une fois la partition terminée, la cible est toujours un espace, et après un espace de padding vient toujours un espace. Le modèle apprend très vite ce comportement et obtient près de 100 % de réussite sur ces positions.

L'accuracy est calculée sur tous les tokens, padding compris : pour cette partition, même un modèle qui ne prédirait correctement **aucune** note atteindrait ~50 % d'accuracy. Le chiffre affiché mesure donc en grande partie la capacité à « continuer les espaces », pas à prédire la musique. La loss est biaisée de la même façon, vers le bas.

**Correction simple** : demander à la loss d'ignorer le padding avec `ignore_index` :

```python
PAD_IDX = char_to_index[" "]
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
```

Les positions dont la cible vaut `PAD_IDX` ne contribuent alors ni à la loss ni au gradient. Comme l'espace est aussi un caractère légitime en ABC (séparation de groupes de notes), le mieux est d'utiliser un **jeton de padding dédié**, absent des partitions (par exemple un caractère spécial ajouté au vocabulaire, `<PAD>`), et `ignore_index=PAD_IDX` sur ce jeton. Il faut aussi masquer l'accuracy de la même façon : `mask = y != PAD_IDX`, puis `correct = (preds[mask] == y[mask]).sum()` et `total = mask.sum()`.

---

## Exercice 4 – Génération musicale, température et job Slurm

### 4.a Génération avec propagation de l'état caché

```python
last_logits = logits[:, -1, :] / temperature
probs = F.softmax(last_logits, dim=-1).squeeze(0)
...
logits, hidden_state = model(input_tensor, hidden_state=hidden_state)
```

#### 4.a.1 Efficacité algorithmique

- **Avec la méthode du TP** : l'amorce de longueur P est traitée une fois, puis chaque nouveau caractère coûte **un seul pas** de récurrence (entrée `(1, 1)` + `hidden_state`, qui résume tout le passé). Générer N caractères coûte **O(P + N)**, soit **O(N)**.
- **En repassant toute la séquence à chaque itération** avec `model(full_seq, None)` : à l'itération k, le réseau doit retraiter P + k caractères depuis un état vide. Le coût total vaut ∑_{k=1}^{N} (P + k) = N·P + N(N + 1)/2, soit **O(N²)**. Pour N = 300 et P ≈ 35, cela fait environ 55 650 pas de récurrence au lieu de 335 (≈ 166 fois plus), et l'écart grandit avec N.

C'est le principal avantage pratique d'un réseau récurrent pour la génération : l'état caché est une **mémoire de taille fixe**, mise à jour en temps constant à chaque pas.

#### 4.a.2 Rôle mathématique de la température

P(y_t = c) = exp(z_c / T) / ∑_{c′} exp(z_{c′} / T)

- **T → 0⁺** : en divisant par T, on multiplie les écarts entre logits par 1/T → +∞. Le terme du logit maximal z_max domine exponentiellement tous les autres : P(c) → 1 si c = argmax z, 0 sinon. La distribution devient un **Dirac (one-hot) sur le caractère le plus probable** : c'est le **décodage glouton** (*greedy decoding*, `argmax`). (En cas d'égalité, la masse se répartit entre les maxima.)
- **T = 1** : on échantillonne selon la distribution apprise par le modèle.
- **T → +∞** : z_c / T → 0 pour tous les c, donc exp(z_c / T) → 1 : la distribution tend vers la **loi uniforme** P(c) = 1/V. Le modèle n'a plus aucune influence : on tire des caractères au hasard.

La température règle donc le compromis **cohérence (T petit) / diversité (T grand)**.

#### 4.a.3 Écoute et analyse qualitative

Amorce : `"X:1\nT:Irish Reel\nM:4/4\nL:1/8\nK:D\n"`, 300 caractères générés, modèle **⚠️ À COMPLÉTER** (LSTM h = 1024 de l'entraînement long, ou LSTM h = 512 du Run 3).

**T = 0,2**

```
⚠️ À COMPLÉTER
```

**T = 0,8**

```
⚠️ À COMPLÉTER
```

**T = 2,0**

```
⚠️ À COMPLÉTER
```

**⚠️ À COMPLÉTER** : analyse après écoute sur ABC Player (validité syntaxique, richesse musicale).

### 4.b Entraînement à plus grande échelle via `sbatch`

`run_music.sh` reprend le script de l'énoncé, avec en plus `--account=csc_8607` et `--qos=normal` (désormais exigés par le cluster) et un argument optionnel de mode. Sans argument, il lance `--mode full` :

```python
train_model(rnn_type="lstm", hidden_size=1024, subset_size=50000,
            max_grad_norm=1.0, num_epochs=30, max_minutes=100)
# puis chargement de best_lstm_h1024.pth et génération à T = 0,2 / 0,8 / 2,0
```

J'ai choisi `subset_size = 50000` plutôt que `None` pour tenir dans les 2 h du job. `max_minutes = 100` garantit qu'il reste du temps pour la génération finale si les 30 époques ne tiennent pas dans le budget.

**⚠️ À COMPLÉTER** : numéro du job, durée par époque, tableau des époques, meilleure val loss, comparaison avec le Run 3.

### 4.c (Bonus) Augmentation de données par transposition

Transposer un air de k demi-tons conserve sa structure (intervalles, rythme, métrique) : c'est **le même morceau dans une autre tonalité**. Sur le format ABC, on peut l'implémenter ainsi :

1. **Séparer l'en-tête du corps** : ne transposer que les lignes de notes, et modifier le champ `K:` (par exemple `K:D` + 2 demi-tons → `K:E`, `K:Ador` → `K:Bdor`).
2. **Parser chaque note** du corps : altération éventuelle (`^` dièse, `_` bémol, `=` bécarre), lettre (`C`–`B` grave, `c`–`b` aigu), marques d'octave (`,` vers le bas, `'` vers le haut), puis la durée, inchangée. On la convertit en hauteur MIDI en tenant compte de l'armure de la tonalité d'origine.
3. **Ajouter k** à la hauteur, puis la **réécrire** en ABC selon l'armure de la nouvelle tonalité (choisir la lettre et l'altération, ajuster les marques d'octave). Les accords entre guillemets (`"G"`, `"Em"`) se transposent de la même façon.
4. **Appliquer la transposition à la volée** dans `MusicDataset.__getitem__`, avec un k tiré au hasard (par exemple entre −5 et +6) à chaque fois qu'une partition est lue, ou bien précalculer quelques versions transposées.

En pratique, on peut s'appuyer sur la bibliothèque `music21`, qui lit l'ABC et sait transposer une partition (`score.transpose(k)`), puis réexporter en ABC (ou utiliser `abc2midi` / `midi2abc`).

**Pourquoi cela réduit le sur-apprentissage** : chaque époque présente au modèle des séquences **différentes** (les mêmes airs dans d'autres tonalités). Le nombre effectif d'exemples distincts est multiplié par jusqu'à 12. Le modèle ne peut plus mémoriser une suite précise de caractères ; il est poussé à apprendre ce qui est invariant par transposition : les **intervalles**, le rythme, la structure en mesures et en phrases, et le lien entre la tonalité `K:` et les notes employées. C'est l'équivalent musical du `RandomResizedCrop` ou du flip horizontal pour les images au TP3, avec une transformation qui, comme il se doit, ne change pas la « classe » (le style musical) des données.

---

## Conclusion

**⚠️ À COMPLÉTER** après les résultats.
