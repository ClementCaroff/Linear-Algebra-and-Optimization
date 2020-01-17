
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

/********************************************************************
CUDA Kernel
*********************************************************************/
__global__ void matrixMul (float* C, float* A, float* B, int TA)
{
    /* calcul des coordonnees du point de C a calculer */
    int i = blockIdx.x;
    int j = threadIdx.x;

    /* calcul de C[i][j] */
    int cc = 0;
    for (int k = 0; k < TA; ++ k)
        cc += A[i * TA + k] * B[k * TA + j];

    /* stockage */
    C[i * TA + j] = cc;
}


/********************************************************************
Programme main
*********************************************************************/

int main (int argc, char** argv)
{
    int i, j, TM, GRID_SIZE_X, BLOCK_SIZE_X;
    clock_t startDelta, deltaTime;
    cudaError_t cerror;

    /* pour le calcul du temps de traitement sur GPU */
    float tc;
    cudaEvent_t depart, arret;

    cudaEventCreate(&depart);
    cudaEventCreate(&arret);

    /* valeurs par defaut */
    TM = 1024;

    /* TM peut etre lu comme arg1 de la commande */
    if (argc > 1) {
        TM = atoi(argv[1]);
    }

    GRID_SIZE_X = TM;
    BLOCK_SIZE_X = TM;

    /* definiton de la grille et des blocs */
    dim3 grid(GRID_SIZE_X);
    dim3 block(BLOCK_SIZE_X);
    printf("taille grille : %d \n", GRID_SIZE_X);
    printf("taille bloc : %d  \n", BLOCK_SIZE_X);

    /* allocation des matrices sur CPU */
    unsigned int msize_A = TM * TM * sizeof(float);
    unsigned int msize_B = TM * TM * sizeof(float);
    unsigned int msize_C = TM * TM * sizeof(float);

    float* h_A = (float*) malloc(msize_A);
    float* h_B = (float*) malloc(msize_B);
    float* h_C = (float*) malloc(msize_C);

    /* initialisation des matrices avec des valeurs permettant de verifier le resultat*/
    for (i = 0; i < TM; i++){
        for (j = 0; j < TM; j++){
            h_A[i * TM + j] = 1.0;
            h_B[i * TM + j] = 1.0;
            h_C[i * TM + j] = 0.0;

            if (i == j) {
                h_A[i * TM + j] = (float) (i + 1);
                h_B[i * TM + j] = (float) (i + 1);
            }
        }
    }

    /* allocation des matrices sur GPU */
    float *d_A; cudaMalloc((void**) &d_A, msize_A);
    float *d_B; cudaMalloc((void**) &d_B, msize_B);
    float *d_C; cudaMalloc((void**) &d_C, msize_C);

    /* mesure du temps : top depart */
    cudaEventRecord(depart, 0);

    /* copie des matrives A et B depuis le CPU vers le GPU */
    startDelta = clock();
    cudaMemcpy(d_A, h_A, msize_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, msize_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, msize_C, cudaMemcpyHostToDevice);
    deltaTime = clock() - startDelta;

    /* lancement des threads */
    matrixMul<<< grid, block >>>(d_C, d_A, d_B, TM);

    /* Recuperation valeur de retour GPU */
    cerror = cudaGetLastError();
    printf(" retour %d \n", (int) cerror);

    /* copie de la matrice C depuis le GPU */
    startDelta = clock();
    cudaMemcpy(h_C, d_C, msize_C, cudaMemcpyDeviceToHost);
    deltaTime += (clock() - startDelta);

    /* mesure du temps */
    const int deltaTimeMs = deltaTime * 1000.0 / CLOCKS_PER_SEC;
    cudaEventRecord(arret, 0);
    cudaEventSynchronize(arret);
    cudaEventElapsedTime(&tc, depart, arret);
    printf("Temps calcul : %f seconde\n", tc / 1000.0);
    printf("Temps transfert : %f secondes\n", deltaTimeMs / 1000.0);
    printf("Temps de transfert = %f pourcent du temps de calcul\n", 100 * deltaTimeMs / tc);


    /* verification du resultat */
    for (i = 0; i < TM; i++) {
        for (j = 0; j < TM; j++) {
            if ((i == j) && (h_C[i * TM + j] != (float)((i + 1) * (i + 1) + TM - 1))) {
                printf("Erreur i: %d j: %d %f\n", i, j, h_C[i * TM + j] ); exit(1);
            }
            else if ((i != j) && (h_C[i * TM + j] != (float)(i + j + TM))) {
                printf("Erreur i: %d j: %d\n", i, j);
                exit(1);
            }
        }
    }

    /* liberation de la memoire */
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(depart);
    cudaEventDestroy(arret);

}
