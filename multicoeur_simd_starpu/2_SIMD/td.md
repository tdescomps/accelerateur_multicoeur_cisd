# Cours de programmation des architectures multi-cœurs – TD Introduction à l'utilisation des routines _intrinsics_ du jeu d'instructions AVX2

## Introduction

Les processeurs généralistes contemporains sont désormais tous munis de jeux d'instructions dits SIMD (_Single Instruction Multiple Data_). Une manière d'utiliser ce type de jeux d'instructions dans un programme est de faire appel à des pseudo-routines dites _intrinsics_.

Le but de ce sujet de TD est d'explorer la programmation SIMD
à l'aide d'_intrinsics_ en s'appuyant sur le jeux d'instructions AVX2 des processeurs Intel et AMD.

## PlaFRIM – nœud frontal

1. Loguez vous sur le nœud frontal de la machine PlaFRIM de **formation**.

2. Exécutez la commande `hostname` pour vérifier que vous êtes sur la bonne machine :

    ```bash
    hostname
    ```

    La commande doit afficher :

    ```text
    miriel045.formation.cluster
    ```

3. Affichez l'information brute sur les processeurs disponibles sur cette machine:

    ```bash
    cat /proc/cpuinfo
    ```

4. Vérifiez que le champ `flags` contient bien les mots-clés `avx2` et `fma`.

## Jeu d'instructions SIMD Intel AVX2

Le jeu d'instructions AVX2 a été introduit par la société Intel en 2013 avec la microarchitecture [Haswell](https://ark.intel.com/content/www/us/en/ark/products/codename/42174/products-formerly-haswell.html). Il propose un ensemble de registres SIMD de 256 bits, et supporte des éléments de registres de type entiers (8/16/32/64 bits) ainsi que des éléments de type flottants simple précision (32 bits) ou double précision (64 bits).

La liste des pseudo-routines _intrinsics_ permettant de programmer en AVX2 est disponible sur le site [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html).

## Observations

Dans cette première partie, il s'agit de tester quelques instructions courantes AVX2 à l'aide d'un petit programme de démo. N'hésitez pas à regarder le code source pour voir comment les pseudo-routines _intrinsics_ sont utilisées.

Vous pouvez récupérer le fichier `demo_avx2.tar` sur Moodle, sur le module du cours IT390.

1. Envoyez le fichier `.tar` sur votre compte de la machine PlaFRIM.
2. Décompactez le fichier `.tar`.
3. Dans le répertoire `demo_avx2/` vous devez avoir 4 fichiers:

    ```text
    demo_float.c  demo_int32.c  load-modules  Makefile
    ```

4. Sourcez le fichier `load-modules` pour charger les modules appropriés :

    ```bash
    source load-modules
    ```

5. Compilez le programme avec la commande `make`.

    La commande `make` doit générer deux exécutables: `demo_int32` et `demo_float`. Le premier utilise des éléments de registres SIMD de type `int32_t` (entiers 32 bits), le second utilise des éléments de type `float`.

6. Expérimentez les différents _intrinsics_ illustrés par les deux exécutables et assurez-vous que vous comprenez leur action.

## Programmation

### X + Y

Le programme `xpy` est un programme destiné à calculer la somme de deux vecteurs dont les éléments sont de type flottants simple précision. La fonction chargée de faire le calcul est vide. Il faut la compléter avec les instructions nécessaires pour faire le calcul de la somme en utilisant les _intrinsics_ AVX2. Vous supposerez que la longueur des vecteurs est un multiple du nombre d'éléments des registres SIMD.

Vous pouvez récupérer le fichier `XPY.tar` sur Moodle, sur le module du cours IT390.

1. Décompactez le fichier `XPY.tar` sur PlaFRIM.

2. Ouvrez le fichier `xpy.c` avec votre éditeur de code et identifiez la fonction à compléter.

3. Implémentez la fonction calculant la somme des deux vecteurs.

4. Compilez et testez votre programme complété.

5. Expérimentez avec diverses valeurs de taille de vecteurs et boucles de répétition. Quelles observations pouvez-vous faire ? Quels sont les effets respectifs des deux paramètres sur le comportement observé du programme ?

### A . X + Y

Le programme `axpy` est une variante du programme `xpy` qui calcule `alpha * X + Y`, où `alpha` est un facteur scalaire et `X` et `Y` sont des vecteurs.

1. Récupérez le fichier `AXPY.tar` et complétez la fonction chargée de calculer `alpha * X + Y`.

2. Compilez et testez votre programme. Comment se comparent les performances de `xpy` et `axpy` pour des paramètres d'entrée identiques ?

### Min / Max

À partir du programme `demo_avx2`, implémentez une fonction minmax qui compare deux registres AVX2 élément par élément et retourne un registre contenant la valeur la plus petite pour chaque paire d'éléments comparés, ainsi qu'un second registre contenant la valeur la plus grande pour chaque paire d'éléments comparés. On prendra des éléments de type `int32_t`.

Note: pour l'intérêt de l'exercice, n'utilisez pas ici `_mm256_max_epi32()` et `_mm256_min_epi32()` !

Exemple, en entrée:

```c
reg_X = [ 5, 8, 5, 1, 5, 0, 2, 8 ]
reg_Y = [ 8, 0, 5, 4, 1, 9, 5, 3 ]
```

En sortie:

```c
reg_Min = [ 5, 0, 5, 1, 1, 0, 2, 3 ]
reg_Max = [ 8, 8, 5, 4, 5, 9, 5, 8 ]
```

### Réduction

À partir du programme `demo_avx2`, implémentez une fonction de _réduction_ qui calcule la somme des 8 éléments de type `float` d'un registre AVX2 en combinant les opérations de permutation et d'addition adéquates. Le résultat sera donné dans un registre AVX2 dont tous les éléments contiennent la somme des éléments du registre d'entrée.

Exemple, en entrée:

```c
reg_X = [ 8.0, 9.0, 5.0, 7.0, 1.0, 2.0, 9.0, 7.0 ]
```

En sortie:

```c
/* Note: 8 + 9 + 5 + 7 + 1 + 2 + 9 + 7 == 48 */
reg_Result = [ 48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 48.0 ]
```

### Produit scalaire de deux vecteurs

En vous inspirant des programmes `xpy` / `axpy`, écrivez un programme réalisant le produit scalaire de deux vecteurs. Vous supposerez que la longueur des vecteurs est un multiple du nombre d'éléments des registres SIMD.

### Nombres complexes

On suppose maintenant que les vecteurs X et Y contiennent des nombres complexes (de la forme `a+ib`, où `a` est la partie réelle et `b` la partie imaginaire). On suppose que les éléments d'indices pairs des vecteurs correspondent à la partie réelle et les indices impairs correspondent à la partie imaginaire.

1. Implémentez une fonction qui calcule la somme élément par élément de deux vecteurs de nombres complexes. Rappel:

   ```c
   (a+ib) + (c+id) == (a+c) + i(b+d)
   ```

2. Implémentez une fonction qui calcule le produit élément par élément de deux vecteurs de nombres complexes. Rappel:

   ```c
   (a+ib) * (c+id) == (a*c - b*d) + i(a*d + b*c)
   ```

### _Question subsidiaire_

Si vous avez terminé les exercices précédents en avance, essayez d'implémenter une version AVX2 de l'algorithme de Cooley & Tukey de calcul rapide de la transformée de Fourier discrète (DFT): [Cooley-Tukey Pseudocode](https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm#Pseudocode)

_Attention, la pseudo routine [`_mm256_exp_ps()`](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SVML&text=_mm256_exp_ps&ig_expand=2816,2816) n'est disponible qu'en utilisant le compilateur Intel._

---
