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

#if 0
template<typename IT, typename NT>
typedef void (*csr_kernel_t)(IT M, IT N, IT K, const CSR<IT, VALUETTYPE>&A, 
      NT *B, IT rldb, NT *C, IT rldc); 

template<typename IT, typename NT>
typedef void (*csc_kernel_t)(IT M, IT N, IT K, const CSC<IT, VALUETTYPE>&A, 
      NT *B, IT rldb, NT *C, IT rldc); 
#else
typedef void (*csr_kernel_t)(INDEXTYPE M, INDEXTYPE N, INDEXTYPE K, 
      const CSR<INDEXTYPE, VALUETYPE>&A, VALUETYPE *B, INDEXTYPE rldb, 
      VALUETYPE *C, INDEXTYPE rldc); 

typedef void (*csc_kernel_t)(INDEXTYPE M, INDEXTYPE N, INDEXTYPE K, 
      const CSC<INDEXTYPE, VALUETYPE>&A, VALUETYPE *B, INDEXTYPE rldb, 
      VALUETYPE *C, INDEXTYPE rldc); 
#endif
/*
 * some misc definition: will move to another file later 
 *    from ATLAS 
 */
#define ATL_MaxMalloc 268435456UL
#define ATL_Cachelen 64
   #define ATL_MulByCachelen(N_) ( (N_) << 6 )
   #define ATL_DivByCachelen(N_) ( (N_) >> 6 )

#define ATL_AlignPtr(vp) (void*) \
        ATL_MulByCachelen(ATL_DivByCachelen((((size_t)(vp))+ATL_Cachelen-1)))

/*
 *    CSC class members: 
 *       nnz, rows, cols, totalcols(for parallel case)
 *       colptr, rowids, values 
 *
 */


#if 0
template<typename IT, typename NT>
void ConvertMNcsc2NNcsc(IT M, const CSC<IT, NT> &A, CSC<IT, NT> &Am)
{
   IT nnz=0; 
   #if 0
   fprintf(stdout, "nnz=%d, rows=%d, cols=%d M=%d\n", A.nnz, A.rows, A.cols, M);
   #endif
   
/*
 * will write a member function in CSC if this experiment becomes successful 
 */
/*
 * not optimized, not parallelized... will consider it later 
 */
   for (IT i=0, nnz=0; i < A.cols; i++)
   {
      IT cl0 = A.colptr[i];
      IT cl1 = A.colptr[i+1]; 

      Am.colptr[i] = nnz;

      for (IT j=cl0; j < cl1; j++)
      {
         IT ia0 = A.rowids[j]; 
         if (ia0 >= M) break;  // assumption: rowids are sorted 
         Am.rowids[nnz] = A.rowids[j]; 
         Am.values[nnz] = A.values[j]; 
         nnz++;
      }
      Am.colptr[i+1] = nnz;
#if 0
      fprintf(stdout, "--- i=%d, i+1=%d, cp[i]=%d, cp[i+1]=%d\n", 
              i, i+1, Am.colptr[i], Am.colptr[i+1]);
#endif
   }
}

void convertMNcsr2BMNcsr(IT M, IT K, IT BM, IT BK, const CSR<IT, NT> &A, 
      CSR<IT, NT> &Ablk)
{

}

#endif


void Usage()
{
   printf("\n");
   printf("Usage for CompAlgo:\n");
   printf("-input <string>, full path of input file (required).\n");

}

void GetFlags(int narg, char **argv, string &inputfile, int &option, 
      INDEXTYPE &D, INDEXTYPE &M, int &csKB, int &nrep)
{
   option = 1; 
   inputfile = "";
   //D = 256; 
   D = 128; 
   M = 0;
   nrep = 1;
   //csKB = 1024; // L2 in KB 
   csKB = 25344; // L3 in KB 

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
      else if(strcmp(argv[p], "-C") == 0)
      {
	 csKB = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-nrep") == 0)
      {
	 nrep = atoi(argv[p+1]);
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
//template <typename IT, typename NT>
void Trusted_SDMM_CSR_IKJ(INDEXTYPE M, INDEXTYPE N, INDEXTYPE K, const CSR<INDEXTYPE, VALUETYPE>& A, VALUETYPE *B, INDEXTYPE ldb,
      VALUETYPE *C, INDEXTYPE ldc)
{
#if 0
   cout << "Inside trusted kernel " << endl;
#endif
   for (INDEXTYPE i=0; i < M; i++)
   {
      INDEXTYPE ia0 = A.rowptr[i];
      INDEXTYPE ia1 = A.rowptr[i+1]; 
      for (INDEXTYPE k=ia0; k < ia1; k++)
      {
         VALUETYPE a0 = A.values[k];
         INDEXTYPE ja0 = A.colids[k];
         for (INDEXTYPE j=0; j < N; j++)
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


void SDMM_CSR_IKJ_D128(INDEXTYPE M, INDEXTYPE N, INDEXTYPE K, 
      const CSR<INDEXTYPE, VALUETYPE>& A, VALUETYPE *B, INDEXTYPE ldb,
      VALUETYPE *C, INDEXTYPE ldc)
{

   for (INDEXTYPE i=0; i < M; i++)
   {
      INDEXTYPE ia0 = A.rowptr[i];
      INDEXTYPE ia1 = A.rowptr[i+1]; 
/*
 *    register block both B and C  
 */
      VTYPE VC0, VC1, VC2, VC3, VC4, VC5, VC6, VC7, 
            VC8, VC9, VC10, VC11, VC12, VC13, VC14, VC15; 
/*
 *    D=128:  AVXZ: #8 VALUETYPE
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

      for (INDEXTYPE k=ia0; k < ia1; k++)
      {
         VTYPE VB0, VB1, VB2, VB3, VB4, VB5, VB6, VB7, 
            VB8, VB9, VB10, VB11, VB12, VB13, VB14, VB15; 
         VTYPE VA0; 
         VALUETYPE a0 = A.values[k];
         INDEXTYPE ja0 = A.colids[k];
      #if 0
         for (INDEXTYPE j=0; j < N; j++)
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

void SDMM_CSR_IKJ_D128_Aligned(INDEXTYPE M, INDEXTYPE N, INDEXTYPE K, 
      const CSR<INDEXTYPE, VALUETYPE>& A, VALUETYPE *B, INDEXTYPE ldb,
      VALUETYPE *C, INDEXTYPE ldc)
{

   for (INDEXTYPE i=0; i < M; i++)
   {
      INDEXTYPE ia0 = A.rowptr[i];
      INDEXTYPE ia1 = A.rowptr[i+1]; 
/*
 *    register block both B and C  
 */
      VTYPE VC0, VC1, VC2, VC3, VC4, VC5, VC6, VC7, 
            VC8, VC9, VC10, VC11, VC12, VC13, VC14, VC15; 
/*
 *    D=128:  AVXZ: #8 VALUETYPE
 *       128/8 = 16 
 */
   #if 0
         fprintf(stdout, "--- before c[%d,0]= %f \n", i, C[i*ldc]);
   #endif
      BCL_vld(VC0, C+i*ldc+VLEN*0); 
      BCL_vld(VC1, C+i*ldc+VLEN*1); 
      BCL_vld(VC2, C+i*ldc+VLEN*2); 
      BCL_vld(VC3, C+i*ldc+VLEN*3); 
      BCL_vld(VC4, C+i*ldc+VLEN*4); 
      BCL_vld(VC5, C+i*ldc+VLEN*5); 
      BCL_vld(VC6, C+i*ldc+VLEN*6); 
      BCL_vld(VC7, C+i*ldc+VLEN*7); 
      BCL_vld(VC8, C+i*ldc+VLEN*8); 
      BCL_vld(VC9, C+i*ldc+VLEN*9); 
      BCL_vld(VC10, C+i*ldc+VLEN*10); 
      BCL_vld(VC11, C+i*ldc+VLEN*11); 
      BCL_vld(VC12, C+i*ldc+VLEN*12); 
      BCL_vld(VC13, C+i*ldc+VLEN*13); 
      BCL_vld(VC14, C+i*ldc+VLEN*14); 
      BCL_vld(VC15, C+i*ldc+VLEN*15); 

      for (INDEXTYPE k=ia0; k < ia1; k++)
      {
         VTYPE VB0, VB1, VB2, VB3, VB4, VB5, VB6, VB7, 
            VB8, VB9, VB10, VB11, VB12, VB13, VB14, VB15; 
         VTYPE VA0; 
         VALUETYPE a0 = A.values[k];
         INDEXTYPE ja0 = A.colids[k];
      #if 0
         for (INDEXTYPE j=0; j < N; j++)
            C[i*ldc+j] += a0 * B[ja0*ldb + j];  // row-major C  
      #else
         
         BCL_vset1(VA0, a0);

         BCL_vld(VB0, B+ja0*ldb+VLEN*0); 
         BCL_vld(VB1, B+ja0*ldb+VLEN*1); 
         BCL_vld(VB2, B+ja0*ldb+VLEN*2); 
         BCL_vld(VB3, B+ja0*ldb+VLEN*3); 
         BCL_vld(VB4, B+ja0*ldb+VLEN*4); 
         BCL_vld(VB5, B+ja0*ldb+VLEN*5); 
         BCL_vld(VB6, B+ja0*ldb+VLEN*6); 
         BCL_vld(VB7, B+ja0*ldb+VLEN*7); 
         BCL_vld(VB8, B+ja0*ldb+VLEN*8); 
         BCL_vld(VB9, B+ja0*ldb+VLEN*9); 
         BCL_vld(VB10, B+ja0*ldb+VLEN*10); 
         BCL_vld(VB11, B+ja0*ldb+VLEN*11); 
         BCL_vld(VB12, B+ja0*ldb+VLEN*12); 
         BCL_vld(VB13, B+ja0*ldb+VLEN*13); 
         BCL_vld(VB14, B+ja0*ldb+VLEN*14); 
         BCL_vld(VB15, B+ja0*ldb+VLEN*15); 
       
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
      }
      BCL_vst(C+i*ldc+VLEN*0, VC0); 
      BCL_vst(C+i*ldc+VLEN*1, VC1); 
      BCL_vst(C+i*ldc+VLEN*2, VC2); 
      BCL_vst(C+i*ldc+VLEN*3, VC3); 
      BCL_vst(C+i*ldc+VLEN*4, VC4); 
      BCL_vst(C+i*ldc+VLEN*5, VC5); 
      BCL_vst(C+i*ldc+VLEN*6, VC6); 
      BCL_vst(C+i*ldc+VLEN*7, VC7); 
      BCL_vst(C+i*ldc+VLEN*8, VC8); 
      BCL_vst(C+i*ldc+VLEN*9, VC9); 
      BCL_vst(C+i*ldc+VLEN*10, VC10); 
      BCL_vst(C+i*ldc+VLEN*11, VC11); 
      BCL_vst(C+i*ldc+VLEN*12, VC12); 
      BCL_vst(C+i*ldc+VLEN*13, VC13); 
      BCL_vst(C+i*ldc+VLEN*14, VC14); 
      BCL_vst(C+i*ldc+VLEN*15, VC15); 
   }
}
#if 0
/*
 *    NOTE: need to consider dynamic blocking later 
 */
template <typename IT, typename NT>
void Test_SDMM_CSR_IKJ_BK_BM
(
 IT M, 
 IT N, 
 IT K, 
 IT BM,  /* blocking factor along M direction */
 IT BK,  /* blocking factor along K dimension */
 const CSR<IT, NT>& A, 
 NT *B, 
 IT ldb,
 NT *C, 
 IT ldc)
{
/*
 * Main Idea: unroll in M dimension and sort access of B... 
 *    a. it will regularize B access with the cost of irregular C write 
 */
   // just started working with it 
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
#endif

/*============================================================================
 * CSC_KIJ 
 *============================================================================*/
   
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
   #if 0
         if (ia0 >= M)
            break;
   #endif
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
      VTYPE VB0, VB1, VB2, VB3, VB4, VB5, VB6, VB7, 
            VB8, VB9, VB10, VB11, VB12, VB13, VB14, VB15; 
      IT ja0 = A.colptr[k];
      IT ja1 = A.colptr[k+1]; 

         BCL_vldu(VB0, B+k*ldb+VLEN*0); 
         BCL_vldu(VB1, B+k*ldb+VLEN*1); 
         BCL_vldu(VB2, B+k*ldb+VLEN*2); 
         BCL_vldu(VB3, B+k*ldb+VLEN*3); 
         BCL_vldu(VB4, B+k*ldb+VLEN*4); 
         BCL_vldu(VB5, B+k*ldb+VLEN*5); 
         BCL_vldu(VB6, B+k*ldb+VLEN*6); 
         BCL_vldu(VB7, B+k*ldb+VLEN*7); 
         BCL_vldu(VB8, B+k*ldb+VLEN*8); 
         BCL_vldu(VB9, B+k*ldb+VLEN*9); 
         BCL_vldu(VB10, B+k*ldb+VLEN*10); 
         BCL_vldu(VB11, B+k*ldb+VLEN*11); 
         BCL_vldu(VB12, B+k*ldb+VLEN*12); 
         BCL_vldu(VB13, B+k*ldb+VLEN*13); 
         BCL_vldu(VB14, B+k*ldb+VLEN*14); 
         BCL_vldu(VB15, B+k*ldb+VLEN*15); 

      for (IT i=ja0; i < ja1; i++)
      {
         VTYPE VC0, VC1, VC2, VC3, VC4, VC5, VC6, VC7, 
               VC8, VC9, VC10, VC11, VC12, VC13, VC14, VC15; 
         VTYPE VA0; 
         
         NT a0 = A.values[i];
         IT ia0 = A.rowids[i];
      #if 0 
         if (ia0 >= M) break; 
      #endif
         BCL_vset1(VA0, a0);
      #if 0
         for (IT j=0; j < N; j++)
            C[ia0*ldc+j] += a0 * B[k*ldb + j];  // row-major C  
      #endif
         BCL_vldu(VC0, C+ia0*ldc+VLEN*0); 
         BCL_vldu(VC1, C+ia0*ldc+VLEN*1); 
         BCL_vldu(VC2, C+ia0*ldc+VLEN*2); 
         BCL_vldu(VC3, C+ia0*ldc+VLEN*3); 
         BCL_vldu(VC4, C+ia0*ldc+VLEN*4); 
         BCL_vldu(VC5, C+ia0*ldc+VLEN*5); 
         BCL_vldu(VC6, C+ia0*ldc+VLEN*6); 
         BCL_vldu(VC7, C+ia0*ldc+VLEN*7); 
         BCL_vldu(VC8, C+ia0*ldc+VLEN*8); 
         BCL_vldu(VC9, C+ia0*ldc+VLEN*9); 
         BCL_vldu(VC10, C+ia0*ldc+VLEN*10); 
         BCL_vldu(VC11, C+ia0*ldc+VLEN*11); 
         BCL_vldu(VC12, C+ia0*ldc+VLEN*12); 
         BCL_vldu(VC13, C+ia0*ldc+VLEN*13); 
         BCL_vldu(VC14, C+ia0*ldc+VLEN*14); 
         BCL_vldu(VC15, C+ia0*ldc+VLEN*15); 
         
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
      
         BCL_vstu(C+ia0*ldc+VLEN*0, VC0); 
         BCL_vstu(C+ia0*ldc+VLEN*1, VC1); 
         BCL_vstu(C+ia0*ldc+VLEN*2, VC2); 
         BCL_vstu(C+ia0*ldc+VLEN*3, VC3); 
         BCL_vstu(C+ia0*ldc+VLEN*4, VC4); 
         BCL_vstu(C+ia0*ldc+VLEN*5, VC5); 
         BCL_vstu(C+ia0*ldc+VLEN*6, VC6); 
         BCL_vstu(C+ia0*ldc+VLEN*7, VC7); 
         BCL_vstu(C+ia0*ldc+VLEN*8, VC8); 
         BCL_vstu(C+ia0*ldc+VLEN*9, VC9); 
         BCL_vstu(C+ia0*ldc+VLEN*10, VC10); 
         BCL_vstu(C+ia0*ldc+VLEN*11, VC11); 
         BCL_vstu(C+ia0*ldc+VLEN*12, VC12); 
         BCL_vstu(C+ia0*ldc+VLEN*13, VC13); 
         BCL_vstu(C+ia0*ldc+VLEN*14, VC14); 
         BCL_vstu(C+ia0*ldc+VLEN*15, VC15); 
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
/*
 * NOTE: 
 * FIXME: When we want to run timer multiple times to average it out, we need to
 * be careful. When A/B/C is small enough to fit in cache, running multiple times
 * keep it (A/B/C) in cache and provide irrealistic results... We need to 
 * allocate large work-space for A/B/C to make it out of cache and shift the 
 * worksace each time
 */
template<csr_kernel_t CSR_KERNEL>
double doTiming_Acsr_CacheFlushing
(
 CSR<INDEXTYPE, VALUETYPE> A_csr, 
 INDEXTYPE M, 
 INDEXTYPE N, 
 INDEXTYPE D, 
 int csKB, 
 int nrep     /* if nrep == 0, nrep = number of wset fit in cache */
 )
{
   int i, j;
   size_t sz, szB, szC;
   size_t cs, setsz, nset, Nt; 
   VALUETYPE *wp, *b, *c, *stc, *stb; 
   double start, end;
   
   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(0.0,1.0);
/*
 * flushing strategy: 
 *    A: Assuming A (NNZ) is large enough that it won't fit in cache
 *       So we can use same A in multiple iteration 
 *    B & C: if flush = 1, we time for out of cache...otherwise in cache 
 */
   cs = csKB*(1024/sizeof(VALUETYPE)); // number elem fit in cache

   szB = ((N*D+VLEN-1)/VLEN)*VLEN;  // szB in element
   szC = ((M*D+VLEN-1)/VLEN)*VLEN;  // szC in element 
   
   setsz = szB + szC; // working set in element, multiple of VLEN 
   nset = (cs + setsz - 1)/setsz; // number of working set fit in cache  
   if (nset < 1) nset = 1;

   Nt = nset * setsz + 2*ATL_Cachelen/sizeof(VALUETYPE); // keep extra to align  
   wp = (VALUETYPE*)malloc(Nt*sizeof(VALUETYPE));
   assert(wp);

   c = stc = (VALUETYPE*) ATL_AlignPtr(wp);
   b = c + szC;
   b = stb = (VALUETYPE*) ATL_AlignPtr(b);
  

   // it's not tester, so just init all with random value 
   for (i=0; i < Nt; i++)
      wp[i] = distribution(generator);  

   /* start timer, will use ATLAS's timer later */

   if (nrep < 1) nrep = 1; // user repeatation 

   fprintf(stderr, "nrep = %d, nset = %d\n", nrep, nset);
   
   start = omp_get_wtime();
   for (i=0, j=nset; i < nrep; i++)
   {
         //CSR_KERNEL<INDEXTYPE,VALUETYPE>(M, D, N, A_csr, b, D, c, D); 
         CSR_KERNEL(M, D, N, A_csr, b, D, c, D); 
         b += setsz; 
         c += setsz;
         j--; 
         if (!j) 
         {
            b = stb; c = stc;
            j = nset;
         }
   }
   end = omp_get_wtime();

   free(wp);
   return((end-start)/((double)nrep));
}
/*
 * Assuming large working set, sizeof B+D > L3 cache 
 */
template<csr_kernel_t CSR_KERNEL>
double doTiming_Acsr
(
 CSR<INDEXTYPE, VALUETYPE> A_csr, 
 INDEXTYPE M, 
 INDEXTYPE N, 
 INDEXTYPE D, 
 int csKB, 
 int nrep     /* if nrep == 0, nrep = number of wset fit in cache */
 )
{
   int i, j;
   double start, end;
   size_t szB, szC; 
   VALUETYPE *pb, *b, *pc, *c;

   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(0.0,1.0);

   szB = ((N*D+VLEN-1)/VLEN)*VLEN;  // szB in element
   szC = ((M*D+VLEN-1)/VLEN)*VLEN;  // szC in element 

   pb = (VALUETYPE*)malloc(szB*sizeof(VALUETYPE)+ATL_Cachelen);
   assert(pb);
   b = (VALUETYPE*) ATL_AlignPtr(pb);
   
   pc = (VALUETYPE*)malloc(szC*sizeof(VALUETYPE)+ATL_Cachelen);
   assert(pc);
   c = (VALUETYPE*) ATL_AlignPtr(pc); 
   
   for (i=0; i < szB; i++)
      b[i] = distribution(generator);  
   for (i=0; i < szC; i++)
      c[i] = distribution(generator);  

/*
 * NOTE: with small working set, we should not skip the first iteration 
 * (warm cache), because we want to time out of cache... 
 * We run this timer either for in-cache data or large working set
 * So we can safely skip 1st iteration... C will be in cache then
 */

   CSR_KERNEL(M, D, N, A_csr, b, D, c, D);  // skip it's timing 
   
   start = omp_get_wtime();
   for (i=0; i < nrep; i++)
      CSR_KERNEL(M, D, N, A_csr, b, D, c, D); 
   end = omp_get_wtime();
   
   free(pb);
   free(pc);
   
   return((end-start)/((double)nrep));
}



#if 0
void GetSpeedup(string inputfile, int option, INDEXTYPE D, INDEXTYPE M, int csKB)
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

   N = A_csc.cols; 
   if (!M || M > N)
      M = N;
/*
 * convert full A_csc to A_Mcsc 
 */
   CSC<INDEXTYPE, VALUETYPE> A_Mcsc(A_csc.nnz, A_csc.rows, A_csc.cols, A_csc.nnz); // copying all, not efficient
   ConvertMNcsc2NNcsc(M, A_csc, A_Mcsc);
#if 0
   printCSC(A_csc);
#endif
   // genetare CSR version of A  
   //SetInputMatricesAsCSR(A_csr, inputfile);
   A_csr.make_empty(); 
   A_csr = *(new CSR<INDEXTYPE, VALUETYPE>(A_csc));
   A_csr.Sorted();

   //N = A_csr.rows; 


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
   #if 1
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
   if (D == 128) // intrinsic code only for D=128 for testing now 
   {
      cout << "calling test SDMM first ... " << endl;
      start = omp_get_wtime(); 
      Test_SDMM_CSR_IKJ_D128<INDEXTYPE, VALUETYPE>(M, D, N, A_csr, B, D, C, D);
      end = omp_get_wtime();
      t1 = end-start;
      fprintf(stdout, "Time SDMM (M=%d,N=%d,D=%d) = %e\n", M, N, D, t1); 

      fprintf(stdout, "**** Speedup of register block verison of CSR_IKJ = %.2f\n", 
          t0/t1); 

/*
 *    Test the result  
 */
      cout << "testing SDMM of resgister blocking... " << endl;
      nerr = doTesting(A_csr.nnz, M, D, C0, C, D); 
      if (!nerr)
         fprintf(stdout, "PASSED TEST\n");
      else
         fprintf(stdout, "FAILED TEST, %d ELEMENTS\n", nerr);
   }

   //fprintf(stderr, "********** VLEN = %d\n", VLEN);  
/*
 * CSC_KIJ 
 */
   cout << "Calling CSC_KIJ version ..." << endl;
#if 0
   //SDMM_CSC_KIJ<INDEXTYPE, VALUETYPE>(M, D, N, A_csc, B, D, C1, D);
   SDMM_CSC_KIJ_D128<INDEXTYPE, VALUETYPE>(M, D, N, A_csc, B, D, C1, D);
#else
   //SDMM_CSC_KIJ<INDEXTYPE, VALUETYPE>(M, D, N, A_Mcsc, B, D, C1, D);
   if (D == 128)
   {
      start = omp_get_wtime(); 
      SDMM_CSC_KIJ_D128<INDEXTYPE, VALUETYPE>(M, D, N, A_Mcsc, B, D, C1, D); 
      end = omp_get_wtime(); 
   }
   else
   {
      fprintf(stdout, "D != 128, calling rolled kernel!\n");
      start = omp_get_wtime(); 
      SDMM_CSC_KIJ<INDEXTYPE, VALUETYPE>(M, D, N, A_Mcsc, B, D, C1, D);
      end = omp_get_wtime(); 
   }

#endif
   t2 = end-start; 
   fprintf(stdout, "Time SDMM (M=%d,N=%d,D=%d) = %e\n", M, N, D, t2); 
   fprintf(stdout, "**** Speedup of CSC_KIJ over CSR_IKJ = %.4f\n", t0/t2); 

/*
 * Test the result  
 */
   cout << "testing SDMM of CSC_KIJ... " << endl;
   nerr = doTesting(A_csr.nnz, M, D, C0, C1, D); 
   if (!nerr)
      fprintf(stdout, "PASSED TEST\n");
   else
      fprintf(stdout, "FAILED TEST, %d ELEMENTS\n", nerr);

}
#else

void GetSpeedup(string inputfile, int option, INDEXTYPE D, INDEXTYPE M, 
      int csKB, int nrep)
{
   int nerr;
   INDEXTYPE N; /* A->MxN, B-> NxD, C-> MxD */ 
   double t0, t1, t2; 
   CSR<INDEXTYPE, VALUETYPE> A_csr0; 
   CSR<INDEXTYPE, VALUETYPE> A_csr1; 
   CSC<INDEXTYPE, VALUETYPE> A_csc;
   
   //csr_kernel_t TRUSTED = Trusted_SDMM_CSR_IKJ; 
   //csr_kernel_t TEST = SDMM_CSR_IKJ_D128; 

   SetInputMatricesAsCSC(A_csc, inputfile);
   A_csc.Sorted(); 

   N = A_csc.cols; 
   
   if (!M || M > N)
      M = N;
   
   // genetare CSR version of A  
   A_csr0.make_empty(); 
   A_csr0 = *(new CSR<INDEXTYPE, VALUETYPE>(A_csc));
   A_csr0.Sorted();
   
   // copy constructor
   A_csr1 = A_csr0;

#if 0
   t1 = doTiming_Acsr_CacheFlushing<SDMM_CSR_IKJ_D128>(A_csr1, M, N, D, csKB, 
         nrep);
   fprintf(stdout, "test time = %e\n", t1); 

   t0 = doTiming_Acsr_CacheFlushing<Trusted_SDMM_CSR_IKJ>(A_csr0, M, N, D, csKB,
         nrep);
   fprintf(stdout, "Trusted time = %e\n", t0); 
#else
   t0 = doTiming_Acsr<Trusted_SDMM_CSR_IKJ>(A_csr0, M, N, D, csKB, nrep);
   //fprintf(stdout, "trusted time = %e\n", t0); 
   
   //t1 = doTiming_Acsr<SDMM_CSR_IKJ_D128>(A_csr1, M, N, D, csKB, nrep);
   t1 = doTiming_Acsr<SDMM_CSR_IKJ_D128_Aligned>(A_csr1, M, N, D, csKB, nrep);
   //fprintf(stdout, "test time = %e\n", t1); 

#endif

   //fprintf(stdout, "Speedup = %.2f\n", t0/t1); 
   
   //fprintf(stdout, "Filename  \tNNZ \tM \tN \tD \ttrusted time \ttesttime  \tspeedup \n" );
   //fprintf(stdout, "%s, \t%ld, \t%ld, \t%ld, \t%d, %e, %e, %.2f\n", 
   //      inputfile, A_csr0.nnz, M, N, D, t0, t1, t0/t1);
   cout << inputfile << ",\t" << A_csr0.nnz << ",\t" << M << ",\t" << N 
        << ",\t" << D << ",\t" << t0 << ",\t" << t1 << "   ,\t" << t0/t1 << endl;


   return;
}
#endif

int main(int narg, char **argv)
{
   INDEXTYPE D, M; 
   int option, csKB, nrep;
   string inputfile; 
   GetFlags(narg, argv, inputfile, option, D, M, csKB, nrep);
   GetSpeedup(inputfile, option, D, M, csKB, nrep);
   return 0;
}

