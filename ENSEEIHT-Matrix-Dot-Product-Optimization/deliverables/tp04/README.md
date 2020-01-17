# TP04

## Grille 1 dimension, Block 1 dimension

```c
int j = threadIdx.x;
int i = blockIdx.x;
```

6 secondes

Performances mauvaises car on ne bénéficie pas de la mémoire partagée pour l'accès aux lignes de la matrice A

```c
int i = blockIdx.x;
int j = threadIdx.x;
```
0.3 secondes

```text
Temps de transfert = 99.249458 % du temps de calcul
```
### Conclusion

Fonctionne pour TM <= 1024 mais erreur pour TM > 1024 car trop de threads créés sur 1 dimension d'un block

## Grille 2 dimensions, Block 2 dimensions

TM = 2048
```c
taille grille : 64 - 64
taille bloc   : 32 - 32
Temps calcul : 2.354209 seconde

taille grille : 128 - 128
taille bloc   : 16 - 16
Temps calcul : 1.885472 seconde

taille grille : 256 - 256
taille bloc   : 8 - 8
Temps calcul : 4.599896 seconde

taille grille : 64 - 128
taille bloc   : 32 - 16
Temps calcul : 1.828602 seconde

taille grille : 32 - 256
taille bloc   : 64 - 8
Temps calcul : 1.787825 seconde

taille grille : 16 - 512
taille bloc   : 128 - 4
Temps calcul : 1.648855 seconde

taille grille : 16 - 1024
taille bloc   : 128 - 2
Temps calcul : 1.568426 seconde

taille grille : 512 - 16
taille bloc   : 4 - 128
Temps calcul : 6.850072 seconde

taille grille : 256 - 32
taille bloc   : 8 - 64
Temps calcul : 3.434813 seconde

taille grille : 128 - 64
taille bloc   : 16 - 32
Temps calcul : 1.934817 seconde
```

Meilleures performances pour la configuration (16, 1024) x (128, 2)
Performances la moins bonne pour la configuration (1024, 16) x (2, 128) (crash)

2048 * 4 * {TH_BLOCK_Y} = 16ko (pour TH_BLOCK_Y=2) pour la matrice A à stocker