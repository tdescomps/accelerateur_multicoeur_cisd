# Cours de programmation des architectures multi-cœurs – TD Introduction à l'utilisation des routines _intrinsics_ du jeu d'instructions AVX2, 2ème partie

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

### Jeu d'instructions SIMD Intel AVX2

Le jeu d'instructions AVX2 a été introduit par la société Intel en 2013 avec la microarchitecture [Haswell](https://ark.intel.com/content/www/us/en/ark/products/codename/42174/products-formerly-haswell.html). Il propose un ensemble de registres SIMD de 256 bits, et supporte des éléments de registres de type entiers (8/16/32/64 bits) ainsi que des éléments de type flottants simple précision (32 bits) ou double précision (64 bits).

La liste des pseudo-routines _intrinsics_ permettant de programmer en AVX2 est disponible sur le site [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html).

## Programmation

### Symétries

Le programme `symmetry` est un programme qui manipule des tableaux de points exprimés en coordonnées 2D rectangulaires. Les éléments de ces tableaux sont déclarés comme des _structures_ C contenant un champ `x` et un champ `y`, tous deux de type flottant 32 bits (type `float`);

Les fonctions `symmetry_x` et `symmetry_y` sont à compléter dans le cadre de l'exercice. La fonction `symmetry_x` doit remplacer la coordonnée `x` de chacun des points du tableau par `-x`. De même, la fonction `symmetry_y` doit remplacer la coordonnée `y` de chaque point par `-y`.

Vous pouvez récupérer le fichier `symmetry.tar` sur Moodle, sur le module du cours IT390.

1. Décompactez le fichier `symmetry.tar` sur PlaFRIM.

2. Ouvrez le fichier `symmetry.c` avec votre éditeur de code et identifiez la fonction à compléter.

3. Implémentez la fonction `symmetry_x` en utilisant une première méthode basée sur `_mm256_mul_ps()`.

4. Compilez et testez votre programme pour la fonction `symmetry_x`.

5. Implémentez la fonction `symmetry_y` en utilisant une seconde méthode basée sur `_mm256_xor_ps()`.

6. Compilez et testez votre programme pour la fonction `symmetry_y`.

Quels sont les avantages respectifs des deux méthodes ?

### _Chroma Keying_

Le programme `chromakey` est un programme qui applique une version simplifiée d'un effet cinématographique appelé [_Chroma Keying_](https://en.wikipedia.org/wiki/Chroma_key) qui consiste à remplacer les pixels d'une image ayant une couleur spécifique (souvent le vert ou le bleu) par les pixels d'une autre image. Ce programme simplifié charge une image de premier plan (_foreground_) et une image d'arrière plan (_background_). Il remplace ensuite les pixels de couleur bleue de l'image _foreground_ par les pixels de l'image _background_. L'image résultante est ensuite écrite dans un fichier.

Les fichiers images sont au [format `PPM`](https://en.wikipedia.org/wiki/Netpbm), un format d'image rudimentaire. Les pixels sont encodés en trois composantes `{rouge, vert, bleu}`. Chaque composante est un entier compris entre `0` et `255` (donc encodable sur 8 bits).

La couleur "_chroma key_" est le bleu intense correspondant aux valeurs `{0, 0, 255}`.

La fonction `apply_chromakey()` est à compléter dans le cadre de l'exercice.

Vous pouvez récupérer le fichier `chromakey.tar` sur Moodle, sur le module du cours IT390.

1. Décompactez le fichier `chromakey.tar` sur PlaFRIM.

2. Ouvrez le fichier `chromakey.c` avec votre éditeur de code et identifiez la fonction à compléter.

3. Deux fichiers images au format `.ppm` sont fournies dans le `.tar`. Compilez et testez votre programme avant modification.

   ```bash
   make
   ./chromakey --fg foreground.ppm --bg background.ppm --output output.ppm
   ```

   Le fichier `output.ppm` doit être identique au fichier `foreground.ppm` à l'exception de la ligne de commentaire. Vous pouvez comparer les fichiers avec la commande `diff`:

   ```bash
   diff foreground.ppm output.ppm
   ```

4. Implémentez la fonction `apply_chromakey()` en utilisant les routines _intrinsics_ AVX2. Pensez à vérifier que toutes les routines dont vous avez besoin existent avant d'écrire la fonction !

5. Compilez et testez votre programme après modification.

   ```bash
   make
   ./chromakey --fg foreground.ppm --bg background.ppm --output output.ppm
   ```

6. Vous pouvez ensuite télécharger `output.ppm` vers votre machine locale pour l'ouvrir avec un programme de visualisation d'images comme `gimp`, par exemple. L'image obtenue doit être une image composite de `foreground.ppm` et `background.ppm` où les pixels bleus de `foreground.ppm` ont été remplacés par les pixels correspondants de `background.ppm`.

### Mandelbrot

Le programme `mandelbrot` est un programme qui génère une image de [l'ensemble de Mandelbrot](https://fr.wikipedia.org/wiki/Ensemble_de_Mandelbrot) en utilisant la bibliothèque graphique `Xlib.h` de l'environnement X11. La fonction `update_mand_array()` est chargée de calculer l'image de l'ensemble. Pour chaque point `(x, y)` de l'image, elle considère le point de coordonnées `(ca, cb)` correspondant dans le repère de l'ensemble de Mandelbrot, et le nombre complexe `c == ca + 𝕚.cb`, et elle détermine au bout de combien d'itérations la suite complexe `{ z_0 = 0; z_n+1 = (z_n)^2 + c}` diverge, en pratique, au bout de combien d'itérations elle excède un certain seuil `module_threshold`. Si ce nombre d'itérations est inférieur à `loop_threshold`, il détermine la couleur associée au point dans le tableau `colormap[]`; dans le cas contraire, le point est arbitrairement colorié avec la couleur d'indice 0 dans le tableau `colormap[]`.

La fonction `update_mand_array()` donnée initialement est une version séquentielle. Cette fonction est à vectoriser avec les routines _intrinsivc_ AVX2/ dans le cadre de l'exercice.

Vous pouvez récupérer le fichier `mandelbrot.tar` sur Moodle, sur le module du cours IT390.

1. Décompactez le fichier `mandelbrot.tar` sur PlaFRIM.

2. Compilez et testez le programme initial pour vérifier que l'affichage de l'ensemble de Mandelbrot fonctionne correctement avec la version séquentielle de la fonction `update_mand_array()`.

   ```bash
   make
   ./mandelbrot_x11
   ```

3. Vectorisez la fonction `update_mand_array()` en utilisant les routines _intrinsics_ AVX2. L'enjeu est notamment de gérer correctement la détection de la divergence des éléments de la suite pour chacun des points traités simultanément par les instrutions vectorielles.

4. Compilez et testez votre programme après modification. L'affichage de la version vectorisée doit être identique à l'affichage de la version séquentielle.

---
