
#include <stdio.h>
#include <stdlib.h>
#include <math.h>



/********************************************************
 CUDA Kernel
********************************************************/
__global__ void matrixMul (float* C, float* A, float* B, int TA)
{
    const int TILE_SIZE = 16;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    for (int tileId = 0; tileId < TA; tileId += TILE_SIZE) {
        /* calcul des coordonnees du thread dans les matrices locales As/Bs */
        int i = threadIdx.y;
        int j = threadIdx.x;

        // copie un element de A et B vers la mémoire partagée
        As[i][j] = A[blockIdx.y * TA + tileId * TILE_SIZE];
        Bs[i][j] = B[tileId * TILE_SIZE + blockIdx.x];
        __syncthreads();

        /* calcul de c[i][j] */
        float cc = 0;
        for (int k = 0; k < TILE_SIZE; ++k) {
            float elementA, elementB;
            elementA = As[i][k];
            elementB = Bs[k][j];
            cc += elementA * elementB;
        }
        __syncthreads();

        // copier dans C ?
        C[blockIdx.y * TA + j] += cc;
    }

}


/********************************************************
 Programme main
********************************************************/
/////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    int i, j, GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE;
    int TM;
    cudaError_t cerror; /*valeur retour gpU*/

    /* pour le calcul du temps de traitement */
    float tc;
    cudaEvent_t depart, arret;
    cudaEventCreate(&depart);
    cudaEventCreate(&arret);

    /* valeurs par defaut */
    TM = 2048;
    BLOCK_SIZE_X = 16;
    BLOCK_SIZE_Y = BLOCK_SIZE_X;
    TILE_SIZE = BLOCK_SIZE_Y;

    if ((TM % BLOCK_SIZE_X) != 0 || (TM % BLOCK_SIZE_Y) != 0) {
        printf("Taille matrice non multiple des dim bloc %d, %d \n", BLOCK_SIZE_X, BLOCK_SIZE_Y);
        exit(1);
    }

    GRID_SIZE_X = TM / BLOCK_SIZE_X;
    GRID_SIZE_Y = TM / BLOCK_SIZE_Y;
    GRID_SIZE_Z = TM / TILE_SIZE;

    /* allocation des matrices sur CPU */
    unsigned int size_A = TM * TM;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);

    unsigned int size_B = TM * TM;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);

    unsigned int size_C = TM * TM;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*) malloc(mem_size_C);

    /* initialisation des matrices avec des valeurs permettant de verifier le resultat */
    for(i = 0; i < TM; i++){
        for(j = 0; j < TM; j++){
            h_A[i*TM+j] = 1.0;
            h_B[i*TM+j] = 1.0;
            h_C[i*TM+j] = 0.0;

            if (i==j) {
                h_A[i*TM+j]=(float) (i+1);
                h_B[i*TM+j]=(float) (i+1);
            }
        }
    }

    /* allocation des matrices sur GPU */
    float* d_A;
    float* d_B;
    cudaMalloc((void**) &d_A, mem_size_A);
    cudaMalloc((void**) &d_B, mem_size_B);
    float* d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    /* top depart pour le calcul du temps de transfert */
    cudaEventRecord(depart,0);

    /* copie des matrives A et B depuis le CPU vers le GPU */
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    cudaEventRecord(arret,0);
    cudaEventSynchronize(arret);
    cudaEventElapsedTime(&tc,depart, arret);
    printf("Temps de transfert host vers device : %f seconde\n", tc/1000.0);

    /* definiton de la grille et des blocs */
    dim3 grid(GRID_SIZE_X, GRID_SIZE_Y);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    printf("grille  %d, %d \n", GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
    printf("bloc %d, %d  \n", BLOCK_SIZE_X, BLOCK_SIZE_Y);

    cudaEventRecord(depart,0);

    /* execution du kernel */
    matrixMul<<< grid, block >>>(d_C, d_A, d_B, TM);

    cudaEventRecord(arret,0);
    cudaEventSynchronize(arret);
    cudaEventElapsedTime(&tc,depart, arret);
    printf("Temps de calcul : %f seconde\n", tc/1000.0);

    /* valeur retour GPU : 0 = OK, sinon erreur */
    cerror=cudaGetLastError();
    printf(" retour GPU = %d \n", (int) cerror);

    cudaEventRecord(depart,0);
    /* copie de la matrive C depuis le GPU */
    cudaMemcpy(h_C, d_C, mem_size_C,
               cudaMemcpyDeviceToHost);

    cudaEventRecord(arret,0);
    cudaEventSynchronize(arret);
    cudaEventElapsedTime(&tc,depart, arret);
    printf("Temps transfert device vers host : %f seconde\n", tc/1000.0);

    /* verification du resultat */
    for (i = 0; i < TM; i++){
        for (j = 0; j < TM; j++){
            if ((i==j) && (h_C[i*TM+j] != (float)((i+1)*(i+1)+TM-1))) {
                printf("Erreur i: %d j: %d %f\n", i, j, h_C[i*TM+j] );
                exit(1);
            } else if ((i!=j) && (h_C[i*TM+j] != (float) (i + j + TM))) {
                printf("Erreur i: %d j: %d\n", i, j);
                exit(1);
            }
        }
    }

    cudaEventDestroy(depart);
    cudaEventDestroy(arret);

    /* liberation de la memoire */
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}