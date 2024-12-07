# Cours de programmation des architectures multi-cœurs – TD Découverte et gestion de la topologie

## Introduction

Les architectures des machines de calcul contemporaines combinent de multiples formes de parallélisme matériel. En conséquence, la prise en compte de la topologie de ces architectures revêt un caractère important dans la recherche d'optimisation de codes.

Le but de ce sujet de TD est d'explorer la topologie des nœuds de la machine PlaFRIM et de manipuler les concepts de placement et d'affinité.

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

    Note: Le numéro exact du nœud miriel que vous obtiendrez peut changer.

3. Affichez l'information brute sur les processeurs disponibles sur cette machine:

    ```bash
    cat /proc/cpuinfo
    ```

## Bibliothèque `libhwloc`

La bibliothèque `libhwloc` est une bibliothèque logicielle fournissant des services pour découvrir la topologie des machines et contrôler le placement des processus, des threads et des données sur ces machines. La bibliothèque `libhwloc` ([link](https://www.open-mpi.org/projects/hwloc/)) peut être utilisée soit par l'intermédiaire d'outils de ligne de commande, soit par directement à l'intérieur des programmes par une interface de programmation.

1. Chargez le module `hwloc` :

    ```bash
    module load hardware/hwloc/2.5.0
    ```

2. Vérifiez que le bon module est chargé :

    ```bash
    hwloc-info --version
    ```

    La commande doit afficher :

    ```text
    hwloc-info 2.5.0
    ```

### Commande `lstopo`

La commande `lstopo` est l'un des outils en ligne de commande de `libhwloc`. Elle permet d'afficher des informations sur la topologie de la machine, sous forme textuelle ou graphique.

1. Affichez la topologie du nœud frontal sous forme textuelle avec la commande suivante :

    ```bash
    lstopo --of console
    ```

    La commande retourne un résumé des informations de topologie de la machine sous forme d'arbre.

2. Déterminez à quoi correspondent les éléments suivants de la topologie :
*utilisation de hwloc ("hardware locality")*
    - Machine
    - Package
    - L3
    - NUMANode
    - L2
    - L1d & L1i
    - Core
    - PU

3. Combien comptez vous :
    - de processeurs ?
    - de cœurs ?
    - de threads matériel (activés) ?

4. Affichez la topologie du nœud frontal sous forme graphique avec la commande suivante :

    ```bash
    lstopo
    ```

5. Affichez la matrice de distances de la topologie avec la commande suivante :

    ```bash
    lstopo --distances
    ```

## PlaFRIM – nœud de calcul

1. Demandez un nœud de calcul à Slurm:

    ```bash
    salloc -N 1 -C miriel --exclusive
    ```

    - L'option `-N 1` demande 1 nœud.
    - L'option `-C miriel` ajoute la contrainte que le nœud doit être un nœud `miriel` (il y a aussi des nœuds `sirocco` avec des caractéristiques différentes).
    - L'option `--exclusive` vous alloue le nœud exclusivement à vous. Sans cette option, le nœud peut être alloué à plusieurs utilisateurs de manière concurrente, ce qui peut altérer des mesures de performances.

    Après quelques instants, Slurm doit répondre le message suivant

    ```text
    [...]
    salloc: Nodes miriel<NNN> are ready for job
    ```

2. Loguez vous sur le nœud de calcul obtenu, depuis la machine frontale

    ```bash
    ssh -X miriel<NNN>
    ```

3. Exécutez la commande `hostname` pour vérifier que vous êtes sur la bonne machine :

    ```bash
    hostname
    ```

    La commande doit afficher :

    ```text
    miriel<NNN>.formation.cluster
    ```

4. Vérifiez la version disponible d'hwloc :

    ```bash
    hwloc-info --version
    ```

    Comparez ce numéro de version avec ce que vous avez obtenu sur le nœud frontal.
    - Quelle est votre interprétation du résultat affiché ici ?
    - Quelle action vous semble pertinente ici ?

5. Affichez la topologie du nœud de calcul sous forme graphique avec la commande suivante :

    ```bash
    lstopo
    ```

### Commande `hwloc-bind`

La commande `hwloc-bind` permet d'*attacher* des processus à un (sous-)ensemble des unités de la machine. Les threads d'un processus ainsi attaché ne peuvent pas être ordonnancé sur des unités situées en dehors du sous-ensemble sélectionné. Celà permet par exemple de restreindre un processus à un banc NUMA, ou à un processeur physique, ou bien à un seul cœur.

Elle s'utilise de la manière suivante :

```bash
hwloc-bind <location> -- command ...
```

Dans l'exemple, `<location>` indique les unités sur lesquelles le processus `command` est attaché.

Pour illustrer l'effet de la commande `hwloc-bind`, on peut l'utiliser conjointement avec la commande `lstopo --ps`. L'option `--ps` de `lstopo` affiche les processus qui sont *bindés* (attachés) et sur quelles unités il le sont.

1. Appliquez `hwloc-bind` à `lstopo --ps` pour attacher la commande `lstopo` sur le `package` numéro `0` :

    ```bash
    hwloc-bind package:0 lstopo --ps
    ```

2. Vérifiez sur l'affichage produit que la commande `lstopo` est bien attaché sur les unités demandées,

3. Réitérez le processus sur le `package` numéro `1` et vérifiez que `lstopo` est bien attaché sur les unités demandées.

4. Expérimentez également avec les niveaux: `numanode` et `core`.

5. Expérimentez avec plusieurs éléments de même niveau, par exemple :

    ```bash
    hwloc-bind numanode:0 numanode:2 lstopo --ps
    ```

    ou :

    ```bash
    hwloc-bind numanode:0 numanode:1 lstopo --ps
    ```

```
[cisd-descomp@miriel035 ~]$ hwloc-bind numanode:0 core:12 -- hwloc-bind --get
0x00000557
[cisd-descomp@miriel035 ~]$ hwloc-bind numanode:0 -- hwloc-bind --get
0x00000555
[cisd-descomp@miriel035 ~]$ hwloc-bind core:12 -- hwloc-bind --get
0x00000002
```

## Exemple avec un petit programme effectuant un calcul

Dans la suite, nous allons utiliser un petit programme utilisant la bibliothèque Intel MKL pour effectuer un calcul (en l'occurrence, une factorisation de matrice par la méthode de Cholesky).

Vous pouvez récupérer le fichier `.tar` des sources sur Moodle / cours IT390 :
`cholesky_mkl.tar` sur votre machine à l'ENSEIRB-MATMECA

1. Envoyez le fichier `.tar` sur votre compte de la machine PlaFRIM.
2. Décompactez le fichier `.tar`.
3. Dans le répertoire `cholesky_mkl/` vous devez avoir 3 fichiers :

    ```text
    Makefile  cholesky.c  load-modules
    ```

4. Sourcez le fichier `load-modules` pour charger les modules appropriés :

    ```bash
    source load-modules
    ```

5. Compilez le programme avec la commande `make`.

    Le programme `cholesky` prend un seul argument de ligne de commande : la taille de la matrice. La commande `cholesky 30000` génère une matrice carrée `30000x30000` et effectue un appel de la routine MKL de factorisation de Cholesky sur cette matrice.

6. Ouvrez une seconde fenêtre de terminal (en gardant la précédente ouverte), loguez vous sur la machine PlaFRIM puis sur votre nœud de calcul.

7. Dans le terminal 2, lancez la commande `htop`, puis lancez en parallèle dans le terminal 1 la commande `./cholesky 30000`.
    - Qu'observez-vous ?
    - Qu'en déduisez-vous sur la nature des routines de la MKL ?
    - Quel est le rôle de la commande `htop` ?
    - Quelles sont les différentes phases du programme `cholesky` ?

8. Utilisez la commande `hwloc-bind` pour confiner le programme `cholesky` sur le `package` `0` et vérifiez l'effet d'`hwloc-bind` avec `htop`.

9. Expérimentez avec différents paramètres de `hwloc-bind` pour voir l'effet sur les cœurs activés ou non.

    La bibliothèque Intel MKL peut être contrôlée indirectement avec la variable d'environnement `KMP_AFFINITY`. Cette variable permet de consulter le *binding* des routines de calcul MKL, et également de modifier ce *binding*.

10. Affichez le *binding* par défaut des routines MKL en lançant la commande suivante :

    ```bash
    KMP_AFFINITY='verbose' ./cholesky 10000
    ```

11. Observez l'effet de `hwloc-bind` avec la commande suivante :

    ```bash
    KMP_AFFINITY='verbose' hwloc-bind <LOCATION> ./cholesky 10000    
    ```

    Testez avec différentes options pour `<LOCATION>`.

    Il est possible de modifier le *binding* des routines MKL avec cette variable `KMP_AFFINITY`. Parmi les options possibles, l'option `compact` permet de favoriser l'affinité des threads des routines MKL. L'option `scatter` permet au contraire de favoriser la dissémination des threads MKL.

12. Comparez les trois commandes suivantes :

    ```bash
    ./cholesky 30000
    ```

    puis :

    ```bash
    KMP_AFFINITY='compact' ./cholesky 30000
    ```

    et enfin :

    ```bash
    KMP_AFFINITY='scatter' ./cholesky 30000
    ```

    - Qu'observez-vous ?
    - Quel est l'effet de l'option `compact` ?
    - Quel est l'effet de l'option `scatter` ?

    Note: vous pouvez vous aider des commandes `htop` et `top` pour investiguer, ainsi que de l'option `verbose` de la variable `KMP_AFFINITY`.

    Note: l'option `-d` de `htop` permet de controller la fréquence de mise à jour de l'affichage; l'option `-u <USERNAME>` permet de n'afficher que les processus et threads de l'utilisateur `USERNAME`; et il est possible de changer les colonnes affichées par la commande `htop` en utilisant la touche `F2` pour modifier la configuration.

---
