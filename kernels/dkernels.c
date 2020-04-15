
#ifdef __cplusplus
   extern "C"
   {
#endif

#include <stdio.h>

#define DREAL 1          /* needed in simd.h */ 
#include "simd.h"
//#include "dkernels.h"
#include "dkernels_D128.h"

#define DEBUG 0 


void dcsrmm_IKJ_a1b1 (const char transa, const BCL_INT m, const BCL_INT n, 
      const BCL_INT k,const double alpha, const char *matdescra, 
      const BCL_INT nnz, const BCL_INT rows, const BCL_INT cols, 
      const double *val, const BCL_INT *indx, const BCL_INT *pntrb, 
      const BCL_INT *pntre, const double *B, const BCL_INT ldb, 
      const double beta, double *C, const BCL_INT ldc);

void dcsrmm_IKJ_aXbX (const char transa, const BCL_INT m, const BCL_INT n, 
      const BCL_INT k,const double alpha, const char *matdescra, 
      const BCL_INT nnz, const BCL_INT rows, const BCL_INT cols, 
      const double *val, const BCL_INT *indx, const BCL_INT *pntrb, 
      const BCL_INT *pntre, const double *B, const BCL_INT ldb, 
      const double beta, double *C, const BCL_INT ldc);

void dcsrmm_KIJ_a1b1 (const char transa, const BCL_INT m, const BCL_INT n, 
      const BCL_INT k,const double alpha, const char *matdescra, 
      const BCL_INT nnz, const BCL_INT rows, const BCL_INT cols, 
      const double *val, const BCL_INT *indx, const BCL_INT *pntrb, 
      const BCL_INT *pntre, const double *B, const BCL_INT ldb, 
      const double beta, double *C, const BCL_INT ldc);

/*
 * for Debug 
 */
#ifdef DEBUG
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
#endif
/*=============================================================================
 *                            CSR_IKJ
 *============================================================================*/

void dcsrmm_IKJ_a1b1   
(
   const char transa,     // 'N', 'T' , 'C' 
   const BCL_INT m,     // number of rows of A 
   const BCL_INT n,     // number of cols of C
   const BCL_INT k,     // number of cols of A
   const double alpha, // double scalar ?? why ptr 
   const char *matdescra,  // 6 characr array descriptor for A:
                           // [G/S/H/T/A/D],[L/U],[N/U],[F/C] -> [G,X,X,C] 
   const BCL_INT nnz,   // nonzeros: need to recreate csr with mkl 
   const BCL_INT rows,  // number of rows
   const BCL_INT cols,  // number of columns 
   const double *val,   // NNZ value  
   const BCL_INT *indx,  // colids -> column indices 
   const BCL_INT *pntrb, // starting index for rowptr
   const BCL_INT *pntre, // ending index for rowptr
   const double *B,     // Dense B matrix
   const BCL_INT ldb,   // 2nd dimension of b for zero-based indexing  
   const double beta,  // double scalar beta[0] 
   double *C,           // Dense matrix c
   const BCL_INT ldc    // 2nd dimension size of b 
)
{
#if 0
   fprintf(stdout, "***** KERNEL: m=%d, n=%d, k=%d, ldb=%d, ldc=%d\n",
           m, n, k, ldb, ldc);
   fprintf(stdout, "          C = %p , B = %p , pntrb = %p , pntre = %p\n", 
           C, B, pntrb, pntre);
#endif
   for (BCL_INT i=0; i < m; i++)
   {
      BCL_INT ia0 = pntrb[i];
      BCL_INT ia1 = pntre[i]; 
   
      //fprintf(stdout, "--- ia0=%d, ia1=%d, ia1-ia0 = %d\n", ia0, ia1, ia1-ia0);
      for (BCL_INT kk=ia0; kk < ia1; kk++)
      {
         double a0 = val[kk];
         BCL_INT ja0 = indx[kk];

         for (BCL_INT j=0; j < n; j++)
            C[i*ldc+j] += a0 * B[ja0*ldb + j];  // row-major C  
      }
   }
}
/*
 * alpha=X, beta=X, not optimized at all  
 */
void dcsrmm_IKJ_aXbX   
(
   const char transa,     // 'N', 'T' , 'C' 
   const BCL_INT m,     // number of rows of A 
   const BCL_INT n,     // number of cols of C
   const BCL_INT k,     // number of cols of A
   const double alpha, // double scalar ?? why ptr 
   const char *matdescra,  // 6 characr array descriptor for A:
                           // [G/S/H/T/A/D],[L/U],[N/U],[F/C] -> [G,X,X,C] 
   const BCL_INT nnz,   // nonzeros: need to recreate csr with mkl 
   const BCL_INT rows,  // number of rows
   const BCL_INT cols,  // number of columns 
   const double *val,   // NNZ value  
   const BCL_INT *indx,  // colids -> column indices 
   const BCL_INT *pntrb, // starting index for rowptr
   const BCL_INT *pntre, // ending index for rowptr
   const double *B,     // Dense B matrix
   const BCL_INT ldb,   // 2nd dimension of b for zero-based indexing  
   const double beta,  // double scalar beta[0] 
   double *C,           // Dense matrix c
   const BCL_INT ldc    // 2nd dimension size of b 
)
{
   for (BCL_INT i=0; i < m; i++)
   {
      BCL_INT ia0 = pntrb[i];
      BCL_INT ia1 = pntre[i]; 
   
      // update with beta first
      for (BCL_INT j=0; j < n; j++)
            C[i*ldc+j] = beta * C[i*ldc+j];

      for (BCL_INT kk=ia0; kk < ia1; kk++)
      {
         double a0 = alpha * val[kk];
         BCL_INT ja0 = indx[kk];

         for (BCL_INT j=0; j < n; j++)
            C[i*ldc+j] += a0 * B[ja0*ldb + j];  // row-major C  
      }
   }
}
/*
 * Wrapper function, will call appropriate kernels from here 
 */

void dcsrmm_IKJ
(
   const char transa,     // 'N', 'T' , 'C' 
   const BCL_INT m,     // number of rows of A 
   const BCL_INT n,     // number of cols of C
   const BCL_INT k,     // number of cols of A
   const double alpha, // double scalar ?? why ptr 
   const char *matdescra,  // 6 characr array descriptor for A:
                           // [G/S/H/T/A/D],[L/U],[N/U],[F/C] -> [G,X,X,C] 
   const BCL_INT nnz,   // nonzeros: need to recreate csr with mkl 
   const BCL_INT rows,  // number of rows
   const BCL_INT cols,  // number of columns 
   const double *val,   // NNZ value  
   const BCL_INT *indx,  // colids -> column indices 
   const BCL_INT *pntrb, // starting index for rowptr
   const BCL_INT *pntre, // ending index for rowptr
   const double *B,     // Dense B matrix
   const BCL_INT ldb,   // 2nd dimension of b for zero-based indexing  
   const double beta,  // double scalar beta[0] 
   double *C,           // Dense matrix c
   const BCL_INT ldc    // 2nd dimension size of c 
)
{
/*
 * alpha and beta value: 0.0, 1.0, X (anything else)
 */
   if (alpha == 0.0)
   {
      fprintf(stderr, "Just call SCALE blas routine, not considered here!");
      exit(1);
   }
   else if (alpha == 1.0)
   {
      if (beta == 1.0)
         dcsrmm_IKJ_a1b1(transa, m, n, k, alpha, matdescra, nnz, rows, 
               cols, val, indx, pntrb, pntre, B, ldb, beta, C, ldc);
      else /*beta==0.0 && beta == X */
      {
         fprintf(stderr, "not considered here yet!");
         exit(1);
      }
   }
   else /* alpha == X */
   {
      if (beta == 0.0 || beta == 1.0)
      {
         fprintf(stderr, "not considered here yet!");
         exit(1);
      }
      else /* beta == X*/
         dcsrmm_IKJ_aXbX(transa, m, n, k, alpha, matdescra, nnz, rows, 
               cols, val, indx, pntrb, pntre, B, ldb, beta, C, ldc);

   }
}

/*
 * Wrapper function, will call appropriate kernels from here 
 */

void dcsrmm_IKJ_D128
(
   const char transa,     // 'N', 'T' , 'C' 
   const BCL_INT m,     // number of rows of A 
   const BCL_INT n,     // number of cols of C
   const BCL_INT k,     // number of cols of A
   const double alpha, // double scalar ?? why ptr 
   const char *matdescra,  // 6 characr array descriptor for A:
                           // [G/S/H/T/A/D],[L/U],[N/U],[F/C] -> [G,X,X,C] 
   const BCL_INT nnz,   // nonzeros: need to recreate csr with mkl 
   const BCL_INT rows,  // number of rows
   const BCL_INT cols,  // number of columns 
   const double *val,   // NNZ value  
   const BCL_INT *indx,  // colids -> column indices 
   const BCL_INT *pntrb, // starting index for rowptr
   const BCL_INT *pntre, // ending index for rowptr
   const double *B,     // Dense B matrix
   const BCL_INT ldb,   // 2nd dimension of b for zero-based indexing  
   const double beta,  // double scalar beta[0] 
   double *C,           // Dense matrix c
   const BCL_INT ldc    // 2nd dimension size of c 
)
{
/*
 * alpha and beta value: 0.0, 1.0, X (anything else)
 */
   if (alpha == 0.0)
   {
      fprintf(stderr, "Just call SCALE blas routine, not considered here!");
      exit(1);
   }
   else if (alpha == 1.0)
   {
      if (beta == 1.0)
         dcsrmm_IKJ_D128_a1b1(transa, m, n, k, alpha, matdescra, nnz, rows, 
               cols, val, indx, pntrb, pntre, B, ldb, beta, C, ldc);
      else /*beta==0.0 && beta == X */
      {
         fprintf(stderr, "not considered here yet!");
         exit(1);
      }
   }
   else /* alpha == X */
   {
      if (beta == 0.0 || beta == 1.0)
      {
         fprintf(stderr, "not considered here yet!");
         exit(1);
      }
      else /* beta == X*/
         dcsrmm_IKJ_D128_aXbX(transa, m, n, k, alpha, matdescra, nnz, rows, 
               cols, val, indx, pntrb, pntre, B, ldb, beta, C, ldc);

   }
}

/*============================================================================
 *                            CSC_KIJ 
 *============================================================================*/
   
void dcscmm_KIJ_a1b1   
(
   const char transa,     // 'N', 'T' , 'C' 
   const BCL_INT m,     // number of rows of A 
   const BCL_INT n,     // number of cols of C
   const BCL_INT k,     // number of cols of A
   const double alpha, // double scalar ?? why ptr 
   const char *matdescra,  // 6 characr array descriptor for A:
                           // [G/S/H/T/A/D],[L/U],[N/U],[F/C] -> [G,X,X,C] 
   const BCL_INT nnz,   // nonzeros: need to recreate csr with mkl 
   const BCL_INT rows,  // number of rows
   const BCL_INT cols,  // number of columns 
   const double *val,   // NNZ value  
   const BCL_INT *indx,  // rowids -> row indices 
   const BCL_INT *pntrb, // starting index for colptr
   const BCL_INT *pntre, // ending index for colptr
   const double *B,     // Dense B matrix
   const BCL_INT ldb,   // 2nd dimension of b for zero-based indexing  
   const double beta,  // double scalar beta[0] 
   double *C,           // Dense matrix c
   const BCL_INT ldc    // 2nd dimension size of b 
)
{
   for (BCL_INT kk=0; kk < k; kk++)
   {
      BCL_INT ja0 = pntrb[kk];
      BCL_INT ja1 = pntre[kk];

      for (BCL_INT i=ja0; i < ja1; i++)
      {
         double a0 = val[i];
         BCL_INT ia0 = indx[i];
         for (BCL_INT j=0; j < n; j++)
            C[ia0*ldc+j] += a0 * B[k*ldb + j];  // row-major C  
      }
   }
}

#ifdef __cplusplus
   }
#endif

   
