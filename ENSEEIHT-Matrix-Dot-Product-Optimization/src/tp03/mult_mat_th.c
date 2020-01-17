/* Multiplication de matrices avec decoupage vertical de B sur l'indice j et execution dans differents thread 
 */


#include < stdio.h > #include < stdlib.h > #include < pthread.h > #include < unistd.h >

#define PASI 8 /* Pas de découpage sur l'indice I */ # define PASJ 8 /* Pas de découpage sur l'indice J */ # define PASK 8 /* Pas de découpage sur l'indice K */

# define NB_TH PASJ /* nombre de threads */

/* pour que les matrices soient visibles par les threads */
float * MA, * MB, * MC;
int TM;

/* ------------------------------------------------------Fonction associée aux thread
-------------------------------------------------------*/
void * thread_fonc(void * p_data) {

    /*Compléter pour récupérer j1 et j2, bornes de la bande de B sur laquelle travaille le thread */
    //...
    //ajouter le code de multiplication des sous-matrices
    //...

    pthread_exit(NULL);
}

/*-------------------------------------------------------
 Program main
--------------------------------------------------------*/

int main(int argc, char * argv[]) {
    register int i, j, ret, nth;

    /* Structure stockant les id des threads */
    pthread_t thread_id[NB_TH];

    /*tableau des parametres des threads*/
    int ta[NB_TH][2];

    /* TM saisie comme argument du main, sinon 1000 par defaut */
    TM = 1000;
    if (argc > 1) {
        TM = atoi(argv[1]);
    }
    /* Verification de la faisabilite du decoupage */
    if ((TM % PASI) != 0 || (TM % PASJ) != 0 || (TM % PASK) != 0) {
        printf("TM doit etre multiple de %d, %d, et %d\n", PASI, PASJ, PASK);
        exit(1);
    }

    /* allocation memoire pour les matrices A, B, et C */
    MA = (float * ) malloc(TM * TM * sizeof(float));
    MB = (float * ) malloc(TM * TM * sizeof(float));
    MC = (float * ) malloc(TM * TM * sizeof(float));

    /* initialisation des matrices avec des valeurs permettant de verifier le resultat */
    for (i = 0; i < TM; i++) {
        for (j = 0; j < TM; j++) {
            MA[i * TM + j] = 1.0;
            MB[i * TM + j] = 1.0;
            MC[i * TM + j] = 0.0;
            if (i == j) {
                MA[i * TM + j] = (float)(i + 1);
                MB[i * TM + j] = (float)(i + 1);
            }
        }
    }

    /* Creation des threads */
    printf("Creation des threads  !\n");
    for (nth = 0; nth < NB_TH; nth++) {
        /* remplir dans ta les parametres pour thread_fonc */
        //......
        //......
        //......
        ret = pthread_create( & thread_id[nth], NULL, thread_fonc, (void * ) ta[nth]);
        if (ret) {
            perror("thread_create");
            exit(2);
        }
    }

    /* Attente de la fin des threads. */
    printf("Attente de la fin des threads  !\n");
    for (i = 0; i < NB_TH; i++) {
        pthread_join(thread_id[i], NULL);
    }

    /* Verification des resultats */
    for (i = 0; i < TM; i++) {
        for (j = 0; j < TM; j++) {
            if ((i == j) && (MC[i * TM + j] != (float)((i + 1) * (i + 1) + TM - 1))) {
                printf("Erreur i: %d j: %d\n", i, j);
                exit(1);
            } else if ((i != j) && (MC[i * TM + j] != (float)(i + j + TM))) {
                printf("Erreur i: %d j: %d\n", i, j);
                exit(1);
            }
        }
    }

    /* liberation de l'espace memoire */
    free(MA);
    free(MB);
    free(MC);

    return EXIT_SUCCESS;
}