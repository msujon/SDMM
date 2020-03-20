#include <stdio.h>
#include <random>
#include <cassert>

#include "CSC.h"
#include "CSR.h"
#include "commonutility.h"
#include "utility.h"

#define DREAL 1 // must defined it before including simd.h... not effective otherwise 
#include "simd.h"

#define INDEXTYPE int
#define VALUETYPE double 


void Usage()
{
   printf("\n");
   printf("Usage for CompAlgo:\n");
   printf("-input <string>, full path of input file (required).\n");

}

void GetFlags(int narg, char **argv, string &inputfile, int &option, 
      INDEXTYPE &D, INDEXTYPE &M)
{
   option = 1; 
   inputfile = "";
   //D = 256; 
   D = 128; 
   M = 0; 

   for(int p = 0; p < narg; p++)
   {
      if(strcmp(argv[p], "-input") == 0)
      {
	 inputfile = argv[p+1];
      }
      else if(strcmp(argv[p], "-option") == 0)
      {
	 option = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-D") == 0)
      {
	 D = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-M") == 0)
      {
	 M = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-h") == 0)
      {
         Usage();
         exit(1);
      }
   }
   if (inputfile == "")
   {
      cout << "Need input file ??? " << endl;
      exit(1);
   }
}

/*
 * base implementation of Sparse-Dense Matrix Multiplication: CSR-IKJ 
 *    B, C matrix: row-major matrix 
 *    A: sparse matrix, CSR format 
 *    C = A*B + C
         // AXPY ... can aply optimization for AXPY
         
          * make N (D) as compile time paramter... 
          * unroll and vectorize j-loop ... 
          *    for example, N=128, use 16 AVX512 register and use register blocking
          
 */
template <typename IT, typename NT>
void Trusted_SDMM_CSR_IKJ(IT M, IT N, IT K, const CSR<IT, NT>& A, NT *B, IT ldb,
      NT *C, IT ldc)
{
   for (IT i=0; i < M; i++)
   {
      IT ia0 = A.rowptr[i];
      IT ia1 = A.rowptr[i+1]; 
      for (IT k=ia0; k < ia1; k++)
      {
         NT a0 = A.values[k];
         IT ja0 = A.colids[k];
         for (IT j=0; j < N; j++)
            C[i*ldc+j] += a0 * B[ja0*ldb + j];  // row-major C  
      }
   }
}

void PrintVector(char*name, VTYPE v)
{
   int i;
#ifdef DREAL 
   double *fptr = (double*) &v;
#else
   float *fptr = (float*) &v;
#endif
   fprintf(stdout, "vector-%s: < ", name);
   for (i=0; i < VLEN; i++)
      fprintf(stdout, "%lf, ", fptr[i]);
   fprintf(stdout, ">\n");
}


template <typename IT, typename NT>
void Test_SDMM_CSR_IKJ_D128(IT M, IT N, IT K, const CSR<IT, NT>& A, NT *B, IT ldb,
      NT *C, IT ldc)
{
   for (IT i=0; i < M; i++)
   {
      IT ia0 = A.rowptr[i];
      IT ia1 = A.rowptr[i+1]; 
/*
 *    register block both B and C  
 */
      VTYPE VC0, VC1, VC2, VC3, VC4, VC5, VC6, VC7, 
            VC8, VC9, VC10, VC11, VC12, VC13, VC14, VC15; 
/*
 *    D=128:  AVXZ: #8 double
 *       128/8 = 16 
 */
   #if 0
         fprintf(stdout, "--- before c[%d,0]= %f \n", i, C[i*ldc]);
   #endif
      BCL_vldu(VC0, C+i*ldc+VLEN*0); 
      BCL_vldu(VC1, C+i*ldc+VLEN*1); 
      BCL_vldu(VC2, C+i*ldc+VLEN*2); 
      BCL_vldu(VC3, C+i*ldc+VLEN*3); 
      BCL_vldu(VC4, C+i*ldc+VLEN*4); 
      BCL_vldu(VC5, C+i*ldc+VLEN*5); 
      BCL_vldu(VC6, C+i*ldc+VLEN*6); 
      BCL_vldu(VC7, C+i*ldc+VLEN*7); 
      BCL_vldu(VC8, C+i*ldc+VLEN*8); 
      BCL_vldu(VC9, C+i*ldc+VLEN*9); 
      BCL_vldu(VC10, C+i*ldc+VLEN*10); 
      BCL_vldu(VC11, C+i*ldc+VLEN*11); 
      BCL_vldu(VC12, C+i*ldc+VLEN*12); 
      BCL_vldu(VC13, C+i*ldc+VLEN*13); 
      BCL_vldu(VC14, C+i*ldc+VLEN*14); 
      BCL_vldu(VC15, C+i*ldc+VLEN*15); 

      for (IT k=ia0; k < ia1; k++)
      {
         VTYPE VB0, VB1, VB2, VB3, VB4, VB5, VB6, VB7, 
            VB8, VB9, VB10, VB11, VB12, VB13, VB14, VB15; 
         VTYPE VA0; 
         NT a0 = A.values[k];
         IT ja0 = A.colids[k];
      #if 0
         for (IT j=0; j < N; j++)
            C[i*ldc+j] += a0 * B[ja0*ldb + j];  // row-major C  
      #else
         
         BCL_vset1(VA0, a0);

         BCL_vldu(VB0, B+ja0*ldb+VLEN*0); 
         BCL_vldu(VB1, B+ja0*ldb+VLEN*1); 
         BCL_vldu(VB2, B+ja0*ldb+VLEN*2); 
         BCL_vldu(VB3, B+ja0*ldb+VLEN*3); 
         BCL_vldu(VB4, B+ja0*ldb+VLEN*4); 
         BCL_vldu(VB5, B+ja0*ldb+VLEN*5); 
         BCL_vldu(VB6, B+ja0*ldb+VLEN*6); 
         BCL_vldu(VB7, B+ja0*ldb+VLEN*7); 
         BCL_vldu(VB8, B+ja0*ldb+VLEN*8); 
         BCL_vldu(VB9, B+ja0*ldb+VLEN*9); 
         BCL_vldu(VB10, B+ja0*ldb+VLEN*10); 
         BCL_vldu(VB11, B+ja0*ldb+VLEN*11); 
         BCL_vldu(VB12, B+ja0*ldb+VLEN*12); 
         BCL_vldu(VB13, B+ja0*ldb+VLEN*13); 
         BCL_vldu(VB14, B+ja0*ldb+VLEN*14); 
         BCL_vldu(VB15, B+ja0*ldb+VLEN*15); 
        
         BCL_vmac(VC0, VA0, VB0); 
         BCL_vmac(VC1, VA0, VB1); 
         BCL_vmac(VC2, VA0, VB2); 
         BCL_vmac(VC3, VA0, VB3); 
         BCL_vmac(VC4, VA0, VB4); 
         BCL_vmac(VC5, VA0, VB5); 
         BCL_vmac(VC6, VA0, VB6); 
         BCL_vmac(VC7, VA0, VB7); 
         BCL_vmac(VC8, VA0, VB8); 
         BCL_vmac(VC9, VA0, VB9); 
         BCL_vmac(VC10, VA0, VB10); 
         BCL_vmac(VC11, VA0, VB11); 
         BCL_vmac(VC12, VA0, VB12); 
         BCL_vmac(VC13, VA0, VB13); 
         BCL_vmac(VC14, VA0, VB14);
         BCL_vmac(VC15, VA0, VB15); 
      #endif
   #if 0
         fprintf(stdout, "---a0= %f ", a0);
         fprintf(stdout, "b0= %f ", B[ja0*ldb]);
         PrintVector("VA0", VA0);
         PrintVector("VB0", VB0);
         PrintVector("VC0", VC0);
   #endif
      }
      BCL_vstu(C+i*ldc+VLEN*0, VC0); 
      BCL_vstu(C+i*ldc+VLEN*1, VC1); 
      BCL_vstu(C+i*ldc+VLEN*2, VC2); 
      BCL_vstu(C+i*ldc+VLEN*3, VC3); 
      BCL_vstu(C+i*ldc+VLEN*4, VC4); 
      BCL_vstu(C+i*ldc+VLEN*5, VC5); 
      BCL_vstu(C+i*ldc+VLEN*6, VC6); 
      BCL_vstu(C+i*ldc+VLEN*7, VC7); 
      BCL_vstu(C+i*ldc+VLEN*8, VC8); 
      BCL_vstu(C+i*ldc+VLEN*9, VC9); 
      BCL_vstu(C+i*ldc+VLEN*10, VC10); 
      BCL_vstu(C+i*ldc+VLEN*11, VC11); 
      BCL_vstu(C+i*ldc+VLEN*12, VC12); 
      BCL_vstu(C+i*ldc+VLEN*13, VC13); 
      BCL_vstu(C+i*ldc+VLEN*14, VC14); 
      BCL_vstu(C+i*ldc+VLEN*15, VC15); 
   #if 0
         fprintf(stdout, "c0= %f \n", C[i*ldc]);
   #endif
   }

}
/*
 * CSC_KIJ 
 */
   
template <typename IT, typename NT>
void SDMM_CSC_KIJ(IT M, IT N, IT K, const CSC<IT, NT>& A, NT *B, IT ldb,
      NT *C, IT ldc)
{
   for (IT k=0; k < K; k++)
   {
      IT ja0 = A.colptr[k];
      IT ja1 = A.colptr[k+1]; 
      for (IT i=ja0; i < ja1; i++)
      {
         NT a0 = A.values[i];
         IT ia0 = A.rowids[i];
         for (IT j=0; j < N; j++)
            C[ia0*ldc+j] += a0 * B[k*ldb + j];  // row-major C  
      }
   }
}

/*
 * CSC_KIJ:  Register blocking  
 */
   
template <typename IT, typename NT>
void SDMM_CSC_KIJ_D128(IT M, IT N, IT K, const CSC<IT, NT>& A, NT *B, IT ldb,
      NT *C, IT ldc)
{
   for (IT k=0; k < K; k++)
   {
      IT ja0 = A.colptr[k];
      IT ja1 = A.colptr[k+1]; 
      for (IT i=ja0; i < ja1; i++)
      {
         NT a0 = A.values[i];
         IT ia0 = A.rowids[i];
         for (IT j=0; j < N; j++)
            C[ia0*ldc+j] += a0 * B[k*ldb + j];  // row-major C  
      }
   }
}
// from ATLAS;s ATL_epsilon.c 
template <typename NT> 
NT Epsilon(void)
{
   static NT eps; 
   const NT half=0.5; 
   volatile NT maxval, f1=0.5; 

   do
   {
      eps = f1;
      f1 *= half;
      maxval = 1.0 + f1;
   }
   while(maxval != 1.0);
   return(eps);
}

template <typename IT, typename NT>
int doTesting(IT NNZA, IT M, IT N, NT *C, NT *D, IT ldc)
{
   IT i, j, k;
   NT diff, EPS; 
   double ErrBound; 

   int nerr = 0;
/*
 * Error bound : computation M*NNZ FMAC
 *
 */
   EPS = Epsilon<VALUETYPE>();
   cout << "--- EPS = " << EPS << endl; 
   // the idea is how many flop one element needs 
   ErrBound = 2 * (NNZA/N) * EPS; 
   //cout << "--- ErrBound = " << ErrBound << " NNZ(A) = " << NNZA 
   //     << " N = " << N  <<endl; 
   // row major! 
   for (i=0; i < M; i++)
   {
      for (j=0; j < N; j++)
      {
         k = i*ldc + j;
         diff = C[k] - D[k];
         if (diff < 0.0) diff = -diff; 
         if (diff > ErrBound)
         {
      #if 0
            fprintf(stderr, "C(%d,%d) : expected=%e, got=%e, diff=%e\n",
                    i, j, C[k], D[k], diff);
      #else // print single value... 
            if (!i && !j)
               fprintf(stderr, "C(%d,%d) : expected=%e, got=%e, diff=%e\n",
                       i, j, C[k], D[k], diff);
      #endif
            nerr++;
         }
         else if (D[k] != D[k]) /* test for NaNs */
         {
            fprintf(stderr, "C(%d,%d) : expected=%e, got=%e\n",
                    i, j, C[k], D[k]);
            nerr++;

         }
      }
   }
   return(nerr);
}

void GetSpeedup(string inputfile, int option, INDEXTYPE D, INDEXTYPE M)
{
/*
 * get the input matrix as CSR format 
 */
   int nerr;
   INDEXTYPE N; /* A->MxN, B-> NxD, C-> MxD */ 
   double start, end, t0, t1, t2; 
   CSR<INDEXTYPE, VALUETYPE> A_csr; 
   CSC<INDEXTYPE, VALUETYPE> A_csc; 

   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(0.0,1.0);

   SetInputMatricesAsCSC(A_csc, inputfile);
   A_csc.Sorted(); 

#if 0
   printCSC(A_csc);
#endif
   // genetare CSR version of A  
   //SetInputMatricesAsCSR(A_csr, inputfile);
   A_csr.make_empty(); 
   A_csr = *(new CSR<INDEXTYPE, VALUETYPE>(A_csc));
   A_csr.Sorted();

   N = A_csr.rows; 

   if (!M || M > N)
      M = N;

#if 0
   cout << "CSR Info: " << endl;
   cout << "Rows = " << A_csr.rows << " Cols = " << A_csr.cols << " nnz = " 
        << A_csr.nnz << endl;
#endif
#if 1
   cout << "(M,N,D) = " << M << " , "<< N << " , " << D <<endl; 
#endif
/*
 * generate row-major matrix:
 *    B -> NxD, C->NxD  
 */
#if 0
   VALUETYPE *B = my_malloc<VALUETYPE>(N*D);
   VALUETYPE *C0 = my_malloc<VALUETYPE>(N*D);
   VALUETYPE *C = my_malloc<VALUETYPE>(N*D);
#else
   VALUETYPE *B = (VALUETYPE*) malloc(sizeof(VALUETYPE)*N*D);
   VALUETYPE *C = (VALUETYPE*)malloc(sizeof(VALUETYPE)*M*D);
   VALUETYPE *C0 = (VALUETYPE*)malloc(sizeof(VALUETYPE)*M*D);
   VALUETYPE *C1 = (VALUETYPE*)malloc(sizeof(VALUETYPE)*M*D);

   assert(B && C0 && C && C1);
#endif
/*
 * init dense matrix 
 */
   cout << "intializing dense matrix ..." << endl;
   for (INDEXTYPE i=0; i < N*D; i++)
   {
   #if 0
      B[i] = distribution(generator);  
   #else
      B[i] = 0.5; 
   #endif
   }
   for (INDEXTYPE i=0; i < M*D; i++)
      C[i] = C0[i] = 0.0;  // init with 0


/*
 * Need some kind of cache flushing 
 */
  
/*
 * will call robust timer later, for now just time it 
 */
   cout << "calling trusted SDMM first ... " << endl;
   start = omp_get_wtime(); 
   Trusted_SDMM_CSR_IKJ<INDEXTYPE, VALUETYPE>(M, D, N, A_csr, B, D, C0, D);
   end = omp_get_wtime(); 
   t0 = end-start; 
   fprintf(stdout, "Time SDMM (M=%d,N=%d,D=%d) = %e\n", M, N, D, t0); 

/*
 *    just rough timing.... need a timer considering cache flusing 
 */
   cout << "calling test SDMM first ... " << endl;
   start = omp_get_wtime(); 
   Test_SDMM_CSR_IKJ_D128<INDEXTYPE, VALUETYPE>(M, D, N, A_csr, B, D, C, D);
   end = omp_get_wtime();
   t1 = end-start;
   fprintf(stdout, "Time SDMM (M=%d,N=%d,D=%d) = %e\n", M, N, D, t1); 

  fprintf(stdout, "**** Speedup of register block verison of CSR_IKJ = %.2f\n", 
          t0/t1); 

/*
 * Test the result  
 */
   cout << "testing SDMM of resgister blocking... " << endl;
   nerr = doTesting(A_csr.nnz, M, D, C0, C, D); 
   if (!nerr)
      fprintf(stdout, "PASSED TEST\n");
   else
      fprintf(stdout, "FAILED TEST, %d ELEMENTS\n", nerr);

   //fprintf(stderr, "********** VLEN = %d\n", VLEN);  
/*
 * CSC_KIJ 
 */
   cout << "Calling CSC_KIJ version ..." << endl;
   start = omp_get_wtime(); 
   SDMM_CSC_KIJ<INDEXTYPE, VALUETYPE>(M, D, N, A_csc, B, D, C1, D);
   end = omp_get_wtime(); 
   t2 = end-start; 
   fprintf(stdout, "Time SDMM (M=%d,N=%d,D=%d) = %e\n", M, N, D, t2); 
   fprintf(stdout, "**** Speedup of CSC_KIJ over CSR_IKJ = %.2f\n", t0/t2); 

/*
 * Test the result  
 */
   cout << "testing SDMM of resgister blocking... " << endl;
   nerr = doTesting(A_csr.nnz, M, D, C0, C1, D); 
   if (!nerr)
      fprintf(stdout, "PASSED TEST\n");
   else
      fprintf(stdout, "FAILED TEST, %d ELEMENTS\n", nerr);

}

int main(int narg, char **argv)
{
   INDEXTYPE D, M; 
   int option;
   string inputfile; 
   GetFlags(narg, argv, inputfile, option, D, M);
   GetSpeedup(inputfile, option, D, M);

   return 0;
}

