# Cours de programmation des architectures multi-cœurs – TD Introduction au support d'exécution à base de tâches StarPU

## Introduction

Le support d'exécution [StarPU](https://starpu.gitlabpages.inria.fr) est développé par l'équipe de recherche [STORM](https://team.inria.fr/storm/). Il permet de gérer l'exécution parallèle des tâches d'une application sur une machine multiprocesseurs, multicœurs et/ou avec un ou plusieurs accélérateurs.

Le TD est à réaliser sur la machine PlaFRIM.

## Exemple de calcul CPU + GPU

1. Récupérez l'archive `cholesky.tar` contenant le programme `cholesky.c`, et envoyez-le sur votre compte de la machine PlaFRIM

2. Décompactez le fichier `.tar`

3. Dans le répertoire `cholesky`, vous devez avoir 3 fichiers :

    ```text
    Makefile  load-modules  cholesky.c
    ```

4. Sourcez le fichier `load-modules` pour charger les modules appropriés :

    ```bash
    source load-modules
    ```

5. Compilez le programme avec la commande `make`.

6. Lancez le programme.

    ```bash
    ./cholesky 3000 480 1 1
    ```

    Avec ces arguments, le programme génère une matrice de taille 3000 (argument 1), et effectue une factorisation de Cholesky en utilisant des tuiles de taille 480 (argument 2). L'argument 3 est un booléen (1 ou 0) qui permet d'activer ou non un calcul de vérification. L'argument 4 est un booléen qui permet d'activer ou non l'affichage d'une partie de la matrice pour vérification.

## Expérimentations

Les nœuds `sirocco` de la machine PlaFRIM sont équipés de GPUs NVIDIA. Le but est d'utiliser StarPU pour effectuer une partie de l'exécution de la factorisation de Cholesky sur ces GPUs.

1. Pour lancer le calcul sur un nœud de calcul `sirocco`, exécutez la commande suivante :

    ```bash
    srun -v -C sirocco --exclusive -t 5 ./cholesky 12000 480 0 0
    ```

    Pour demander une machine `sirocco` spécifique, par exemple `sirocco03`, remplacez `-C sirocco` par `-w sirocco03`.

    La variable d'environnement `STARPU_WORKER_STATS=1` permet d'avoir des informations sur la répartition des tâches entre les différentes unités de calcul. Exemple:

    ```bash
    STARPU_WORKER_STATS=1 srun -v -C sirocco --exclusive -t 5 ./cholesky 12000 480 0 0
    ```

2. Expérimentez avec l'ordonnanceur `dm` avec le programme `cholesky`.

    La variable d'environnement `STARPU_SCHED` permet de sélectionner un algorithme d'ordonnancement de StarPU. Par défaut, l'algorithme d'ordonnancement sélectionné est `lws` pour _locality work-stealing_. Il s'agit d'un algorithme qui fait de l'équilibrage de charge réactif en tenant compte de la localité des tâches: un `worker` effectue un vol de travail de préférence sur un worker _proche_ en terme de topologie de la machine.

    L'algorithme d'ordonnancement `dm` permet d'utiliser des modèles de performances pour déterminer si les tâches doivent plutôt être ordonnancées sur les CPUs ou sur les GPUs.

    ```bash
    STARPU_SCHED='dm' STARPU_WORKER_STATS=1 srun -v -C sirocco --exclusive -t 5 ./cholesky 12000 480 0 0
    ```

    - Qu'observez-vous au niveau de la répartition des tâches entre les différentes unités de calcul ?
    - Quelle est votre interprétation des messages de StarPU du type "model XXXX is not calibrated enough" ?

3. Comparez le temps d'exécution moyen des 4 noyaux de calcul utilisés dans la factorisation de Cholesky sur CPU et sur GPU.

    La commande `starpu_perfmodel_display -l` permet de lister les noms des modèles de performances que StarPU a enregistré pour chacun des noyaux de calcul.

    La commande `starpu_perfmodel_display -s <NOM>` permet d'afficher les données enregistrées pour un noyau sur une machine.

    - Comparez les performances CPU et GPU pour les noyaux GEMM, TRSM et SYRK.
    - Quelle observation peut-on faire sur le noyau POTRF ?

4. Expérimentez avec l'ordonnanceur `dmda` avec le programme `cholesky`.

    ```bash
    STARPU_SCHED='dmda' STARPU_WORKER_STATS=1 srun -v -C sirocco --exclusive -t 5 ./cholesky 12000 480 0 0
    ```

    - Qu'observez-vous sur le temps d'exécution par rapport à l'algorithme d'ordonnancement 'dm' ?

    La variable d'environnement `STARPU_BUS_STATS=1` permet d'afficher des statistiques de transferts de données entre les unités de calcul.

    - Comparez les volumes de transfert obtenus avec l'algorithme `dm` et `dmda`. Quel lien peut-on faire avec les performances relatives de ces deux algorithmes ?

## Produit Matrice x Vecteur

1. À partir des exemples précédents, créez un exemple `mat_vec_mult.c` qui effectue le produit d'une matrice (dense) par un vecteur, en exploitant les CPUs et les GPUs s'ils sont disponibles.

---
