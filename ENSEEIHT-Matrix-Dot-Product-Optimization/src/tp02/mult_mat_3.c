
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    register int i, j, jk, k, ki, msec;
    float cc;
    clock_t start, temps;
    float *MA, *MB, *MC;

    // TM saisie comme argument du main, sinon 1000 par defaut
    register const TM = (argc > 1 ? atoi(argv[1]) : 1000);

    // allocation memoire pour les matrices A, B, et C
    MA = (float *) malloc(TM * TM * sizeof(float));
    MB = (float *) malloc(TM * TM * sizeof(float));
    MC = (float *) malloc(TM * TM * sizeof(float));

    // initialisation des matrices avec des valeurs permettant de vérifier le resultat

    for(i = 0; i < TM; i++) {
        for(j = 0; j < TM; j++) {
            MA[i * TM + j] = 1.0;
            MB[i * TM + j] = 1.0;
            MC[i * TM + j] = 0.0;
            if (i == j) {
                MA[i * TM + j] = (float) (i + 1);
                MB[i * TM + j] = (float) (i + 1);
            }
        }
    }

    start = clock();

    // multiplication C = A * B
    const step = 4;
    for (ki = 0; ki < TM; ki += step) {
        for (i = 0; i < TM; ++i) {
            for (j = 0; j < TM; ++j) {
                const m = ki + step;
                for (k = ki; k < m; ++k)
                    MC[i * TM + j] += MA[i * TM + k] * MB[k * TM + j];
            }
        }
    }



    temps = clock() - start;
    msec = temps * 1000 / CLOCKS_PER_SEC;
    printf("Temps multiplication %d secondes %d millisecondes\n", msec / 1000, msec % 1000);


    // Verification des resultats
    // Si les boucles de multiplication ne sont pas executees correctement, les erreurs sont détectées et affichées à l'écran.
    for(i = 0; i < TM; i++){
        for(j = 0; j < TM; j++){
            if ((i == j) && (MC[i * TM + j] != (float) ((i + 1) * (i + 1) + TM - 1))) {
                printf("Erreur i: %d j: %d\n", i, j);
            }
            else if ((i != j) && (MC[i * TM + j] != (float) (i + j + TM))) {
                printf("Erreur i: %d j: %d\n", i, j);
            }
        }
    }

    // liberation de l'espace memoire
    free(MA);
    free(MB);
    free(MC);

    return(0);
}
