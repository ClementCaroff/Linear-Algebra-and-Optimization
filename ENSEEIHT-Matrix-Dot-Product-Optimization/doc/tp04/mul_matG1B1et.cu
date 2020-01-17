 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h> 

/********************************************************************
CUDA Kernel
*********************************************************************/
__global__ void matrixMul( float* C, float* A, float* B, int TA)
{
   /* calcul des coordonnees du point de C a calculer */
   int i = ... ;
   int j = ... ;
 
   /* calcul de C[i][j] */
   // ...
}
 
 
/********************************************************************
Programme main
*********************************************************************/
 
int main(int argc, char** argv)
{
	int i, j, TM, GRID_SIZE_X, BLOCK_SIZE_X;
	cudaError_t cerror;
	
	/* pour le calcul du temps de traitement sur GPU */
	float tc;
    cudaEvent_t depart, arret;
	
    cudaEventCreate(&depart);
    cudaEventCreate(&arret);
	
	/* valeurs par defaut */
	TM=1024;
	
	/* TM peut etre lu comme arg1 de la commande */
	if (argc>1) {
		 TM=atoi(argv[1]);
    }

	GRID_SIZE_X = TM;
	BLOCK_SIZE_X = TM;
	
	/* definiton de la grille et des blocs */
    dim3 block(BLOCK_SIZE_X);
    dim3 grid(GRID_SIZE_X);
 	printf("taille bloc : %d  \n", BLOCK_SIZE_X);
 	printf("taille grille : %d \n", GRID_SIZE_X);

   /* allocation des matrices sur CPU */
   unsigned int msize_A = TM * TM * sizeof(float);
   
   float* h_A = (float*) malloc(msize_A);
 
   //....
   
   /* initialisation des matrices avec des valeurs permettant de verifier le resultat*/
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
   float *d_A;
   cudaMalloc((void**) &d_A, msize_A);
   //...

   /* mesure du temps : top depart */
   cudaEventRecord(depart,0);
   
   /* copie des matrives A et B depuis le CPU vers le GPU */
   cudaMemcpy(d_A, h_A, msize_A, cudaMemcpyHostToDevice);
   //...
   
   /* lancement des threads */
   matrixMul<<< grid, block >>>(d_C, d_A, d_B, TM);

   /* Recuperation valeur de retour GPU */
   cerror=cudaGetLastError();
   printf(" retour %d \n", (int) cerror);

   /* copie de la matrive C depuis le GPU */
   // ...

   /* mesure du temps */
   cudaEventRecord(arret,0);
   cudaEventSynchronize(arret);
   cudaEventElapsedTime(&tc,depart, arret);
   printf("Temps calcul : %f seconde\n", tc/1000.0);

   /* verification du resultat */
  for(i = 0; i < TM; i++){
    for(j = 0; j < TM; j++){
	if ((i==j) && (h_C[i*TM+j] != (float)((i+1)*(i+1)+TM-1))) 	   	{
		printf("Erreur i: %d j: %d %f\n", i, j, h_C[i*TM+j] ); exit(1);
		}
		else if ((i!=j) && (h_C[i*TM+j] != (float)(i+j+TM))) 			{
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
