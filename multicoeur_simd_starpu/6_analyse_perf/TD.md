# Cours de programmation des architectures multi-cœurs – TD Analyse de Performances

## Introduction

Divers outils sont disponibles sur les supercalculateurs pour permettre d'analyser les performances des applications et donner des pistes d'amélioration

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

## Commande Linux `perf` et compteurs matériels

Les processeurs contemporains proposent l'accès à des compteurs
matériels indiquant le nombre d'occurrences d'événements d'intérêt,
tels que le nombre d'instructions traitées, le nombre de branches
conditionnelles, le nombre de défauts de cache, etc. Le nombre
et la nature de ces compteurs dépend directement du modèle du
processeur considéré.

La commande `perf` disponible sur les systèmes d'exploitation Linux
permet d'accéder à ces compteurs matériels et notamment de voir
comment ces compteurs évoluent lors de l'exécution d'une application.

La liste des compteurs disponible sur la machine peut être obtenue par
la commande `perf list`:

```bash
$ perf list

  branch-instructions OR branches                    [Hardware event]
  branch-misses                                      [Hardware event]
  bus-cycles                                         [Hardware event]
  cache-misses                                       [Hardware event]
  cache-references                                   [Hardware event]
  cpu-cycles OR cycles                               [Hardware event]
  instructions                                       [Hardware event]
  ref-cycles                                         [Hardware event]

  alignment-faults                                   [Software event]
  bpf-output                                         [Software event]
  context-switches OR cs                             [Software event]
  cpu-clock                                          [Software event]
  cpu-migrations OR migrations                       [Software event]

[. . .]
```

La mesure de performances est réalisée avec la commande `perf stat <application>`.
Par défaut, une sélection prédéfinies de compteurs matériels est utilisée. L'exemple
ci-dessous applique la commande `perf stat` au programme `/bin/true`:

```bash
$ perf stat -B /bin/true

 Performance counter stats for '/bin/true':

          0,364880      task-clock (msec)         #    0,404 CPUs utilized          
                 0      context-switches          #    0,000 K/sec                  
                 0      cpu-migrations            #    0,000 K/sec                  
               120      page-faults               #    0,329 M/sec                  
           765 978      cycles                    #    2,099 GHz                    
           482 338      instructions              #    0,63  insn per cycle         
            90 669      branches                  #  248,490 M/sec                  
             3 996      branch-misses             #    4,41% of all branches        

       0,000903486 seconds time elapsed
```

1. Récupérez l'archive `matrix_vector.tar` sur Moodle et envoyez le fichier sur votre compte PlaFRIM.
2. Décompactez le fichier `.tar`.
3. Chargez le module `compiler/gcc/11.2.0`
4. Compilez le programme avec la commande `make`. La compilation doit générer deux programmes: `matrix_vector` et `matrix_t_vector`. Le second programme multiplie la transposée de la matrice avec le vecteur, ce qui revient à échanger les deux dimensions de la matrice lors de son traitement dans la fonction de multiplication.
5. Expérimentez avec la commande `perf stat` sur les deux programmes `matrix_vector` et `matrix_t_vector` pour déterminer l'effet de l'échange des deux dimensions de la matrice sur les performances du programme.

6. Effacez les deux programmes compilés par le compilateur GNU `gcc` avec `make clean`.
7. Chargez le module `compiler/intel/2020_update4`.
8. Compilez le programme avec le compilateur Intel `icc`.
9. Utilisez `perf stat` pour comparer les résultats obtenus avec le compilateur Intel `icc`, et les résultats obtenus précédemment avec le compilateur GNU `gcc`.

## Rapports d'optimisation des compilateurs

Les compilateurs tels que `gcc` ou `icc` permettent d'obtenir des informations sur les optimisations qu'ils réalisent, et parfois sur les optimisations qu'ils ne parviennent pas à réaliser.

Avec le compilateur GNU `gcc`, l'option `-fopt-info` permet d'obtenir le rapport d'optimisation.

0. Faites un `make clean` et chargez le module `compiler/gcc/11.2.0`.

1. Sur l'exemple du produit Matrix x Vector, comparez le rapport d'optimisation de `gcc` pour le programme `matrix_vector` et le programme `matrix_t_vector`. Quelle(s) différence(s) observez-vous ? Quel est le lien entre ces différences et les différences de performances des deux programmes ?

2. En utilisant la commande `perf stat -e avx_insts.all <COMMANDE>`, comparez le nombre d'instructions AVX utilisées par les deux versions du programme. Est-ce cohérent avec les observations sur les rapports d'optimisation du compilateur `gcc` ?

Avec le compilateur Intel `icc`, l'option `-qopt-report` permet de générer le rapport d'optimisation dans un fichier dont le nom termine par `.optrpt`.

0. Faites un `make clean` et chargez le module `compiler/intel/2020_update4`.

1. Générez les rapports d'optimisation pour les deux variantes du programme et comparez-les. Y'a-t-il des différences entre les optimisations rapportées par `icc` et par `gcc` précédemment ?

2. En utilisant la commande `perf stat -e avx_insts.all <COMMANDE>`, comparez le nombre d'instructions AVX utilisées par les deux versions du programme. Est-ce cohérent avec les observations sur les rapports d'optimisation du compilateur `icc` ?

## Exemple d'analyse de performances avec Intel VTune

L'outil Intel VTune est un outil d'analyse de performances d'usage général. Il permet d'obtenir un aperçu du comportement d'une application et des suggestions de pistes à explorer pour essayer d'améliorer les performances.

1. Récupérez l'archive `box_muller.tar` sur Moodle et envoyez le fichier sur votre compte PlaFRIM.
2. Décompactez le fichier `.tar`.
3. Chargez les modules nécessaires avec les deux commandes suivantes:

    ```bash
    source load-modules-intel
    source /cm/shared/modules/intel/haswell/parallel_studio/2020_update4/vtune_profiler/vtune-vars.sh
    ```

4. Compilez le programme et vérifiez qu'il fonctionne:

    ```bash
    ./box_muller 100000000
    ```

5. Lancez le programme Intel VTune avec la commande `vtune-gui`. La gestion des fenêtres à distance peut être un peu lente. Il est nécessaire d'être patient et d'attendre la prise en compte de chaque clic de souris individuellement !

6. Créez un projet pour le programme `box_muller`, en précisant le programme exécutable et `100000000` en argument, puis lancez l'analyse de base. Quels sont les problèmes de performances rapportés par VTune ?

7. Modifiez le `Makefile` pour activer les lignes `CXXFLAGS` en commentaires, en enlevant le `#` initial. Modifiez le programme `box_muller.cpp` pour activer le `pragma omp` à la ligne 25 dans la fonction `box_muller()`.

8. Recompilez le programme puis relancez l'analyse avec VTune pour voir si les problèmes de performances ont été corrigés ou améliorés. Reste-t-il des éléments à améliorer ?

## Exemple d'analyse de vectorisation avec Intel Advisor

L'outil Intel Advisor permet d'analyser plus spécifiquement les problèmes de vectorisation et de parallélisation, notamment en déterminant les problèmes de dépendances de données qui peuvent gêner le compilateur.

1. Récupérez l'archive `tsvc_2.tar` sur Moodle et envoyez le fichier sur votre compte PlaFRIM.
2. Décompactez le fichier `.tar`. L'archive contient une suite de _benchmark_ appelée [TSVC](https://github.com/UoB-HPC/TSVC_2) permettant de tester les capacités de vectorisation des compilateurs.
3. Chargez les modules nécessaires avec les deux commandes suivantes:

    ```bash
    source load-modules-intel
    source /cm/shared/modules/intel/haswell/parallel_studio/2020_update4/advisor/advixe-vars.sh
    ```

4. Compilez la suite de benchmarks avec la commande `make`. Un programme exécutable est généré dans le répertoire `bin/intel/`

5. Lancez le programme Intel Advisor avec la commande `advixe-gui`. À nouveau, les interactions avec l'interface graphique peuvent être un peu lentes.

6. Créez un projet pour la suite de benchmarks TSVC_2 en indiquant le programme exécutable du répertoire 'bin/intel/'.

7. Lancez l'analyse de base. Attention, l'analyse dure environ 5 minutes, le temps de passer par toutes les routines de test.

8. Explorer le rapport d'analyse pour voir les boucles qui ont été vectorisées et celles où le compilateur a échoué. Regardez notamment les indications sur les causes des échecs de vectorisation.

9. S'il reste du temps disponible, comparez en compilant la suite de benchmarks avec le compilateur GNU `gcc`, pour voir si les capacités de vectorisation sont similaires ou non au compilateur Intel `icc`. Il faut remplacer la chaine "intel" par "GNU" dans le `Makefile` principal.

---
