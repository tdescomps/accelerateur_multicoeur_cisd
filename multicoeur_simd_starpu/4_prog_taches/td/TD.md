# Cours de programmation des architectures multi-cœurs – TD Introduction au support d'exécution à base de tâches StarPU

## Introduction

Le support d'exécution [StarPU](https://starpu.gitlabpages.inria.fr) est développé par l'équipe de recherche [STORM](https://team.inria.fr/storm/). Il permet de gérer l'exécution parallèle des tâches d'une application sur une machine multiprocesseurs, multicœurs et/ou avec un ou plusieurs accélérateurs.

Le TD est à réaliser sur la machine PlaFRIM.

## Vérification des modules

1. Loguez vous sur la machine PlaFRIM

2. Exécutez la commande suivante:

    ```bash
    module load compiler/gcc/11.2.0 hardware/hwloc/2.5.0 runtime/starpu/1.3.8/1.3.8/mpi
    ```

3. Vérifiez que le module de StarPU est bien chargé et fonctionne correctement, avec la commande suivante:

    ```bash
    starpu_machine_display
    ```

## Premier exemple, pour prendre en main StarPU: produit d'un vecteur par un scalaire

1. Récupérez l'archive `.tar` contenant le code source séquentiel du programme `vector_scale`, et envoyez-le sur votre compte de la machine PlaFRIM

2. Décompactez le fichier `.tar`

3. Dans le répertoire `vector_scale`, vous devez avoir 3 fichiers:

    ```text
    Makefile  load-modules  vector_scale.c
    ```

4. Sourcez le fichier `load-modules` pour charger les modules appropriés :

    ```bash
    source load-modules
    ```

5. Compilez le programme avec la commande `make`.

6. Lancez le programme.

    Le programme génère un vecteur aléatoire de longeur 100 (par défaut), puis calcule le produit de ce vecteur par un facteur également aléatoire.

7. Le programme prend optionnellement un entier `>= 1` en argument, pour spécifier une longueur de vecteur autre que la valeur par défaut.

    Essayez le programme en précisant une longueur, pour vérifier que tout fonctionne.

8. En vous inspirant de la présentation d'introduction de StarPU, modifiez le code source pour que le calcul du produit _vecteur_ * _scalaire_ soit effectué dans une tâche StarPU.

    Les étapes à suivre sont les suivantes:
    - Inclure le fichier `starpu.h` en début de fichier;
    - Initialiser StarPU;
    - Initialiser un _Data Handle_ de StarPU pour le vecteur `x`;
    - Déclarer une structure codelet pour la tâche qui va faire le calcul;
    - Écrire la fonction de la tâche StarPU;
    - Soumettre la tâche à StarPU;
    - Ajouter l'attente de la fin d'exécution de la tâche StarPU;
    - Dé-initialiser le _Data Handle_ de `x`;
    - Afficher le résultat;
    - Dé-initialiser StarPU.

    Si besoin, la documentation de StarPU est disponible en ligne ici: [Documentation de StarPU](https://starpu.gitlabpages.inria.fr/#doc)

9. Vérifiez que le programme fonctionne correctement.

## Utilisation du partitionnement

Dans le premier exemple, une seule tâche est créée. Une manière de permettre la création de plusieurs tâches est de __partitionner__ le _Data Handle_ du vecteur `x` en plusieurs parties, puis de soumettre une tâche pour chaque sous-partie.

1. Faites une version `vector_scale_part.c` du programme `vector_scale.c` dans laquelle vous partitionnez le _Data Handle_ du vecteur `x` et soumettez une tâche pour chaque élément de la partition.

2. Vérifiez que le programme fonctionne correctement.

## Données en lecture-écriture vs données en lecture

1. À partir de l'exemple `vector_scale.c`, créez un nouveau programme `axpy.c` qui calcule `y == alpha * x + y`, où `x` et `y` sont des vecteurs et `alpha` un facteur scalaire. Le vecteur `x` sera passé en lecture seule lors de la création de la tâche.

De la même manière que l'exemple précédent, il est préférable d'écrire d'abord la version sans partitionnement, puis de rajouter le partitionnement dans un second temps, après avoir vérifié le bon fonctionnement du code.

## Manipulation de données plus complexes

1. À partir des exemples précédents, créez un exemple `mat_vec_mult.c` qui effectue le produit d'une matrice (dense) par un vecteur.

2. La manipulation de données de type matrices denses fait intervenir la notion de _leading dimension_. Quel est le rôle de cette notion ?

---
