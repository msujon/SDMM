#include "simd.h"

#define DEBUG 1 

#include "dkernels_avxz.h"



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
   const INDEXTYPE m,     // number of rows of A 
   const INDEXTYPE n,     // number of cols of C
   const INDEXTYPE k,     // number of cols of A
   const VALUETYPE alpha, // double scalar ?? why ptr 
   const char *matdescra,  // 6 characr array descriptor for A:
                           // [G/S/H/T/A/D],[L/U],[N/U],[F/C] -> [G,X,X,C] 
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
   const VALUETYPE *B,     // Dense B matrix
   const INDEXTYPE ldb,   // 2nd dimension of b for zero-based indexing  
   const VALUETYPE beta,  // double scalar beta[0] 
   VALUETYPE *C,           // Dense matrix c
   const INDEXTYPE ldc    // 2nd dimension size of b 
)
{
   for (INDEXTYPE i=0; i < m; i++)
   {
      INDEXTYPE ia0 = pntrb[i];
      INDEXTYPE ia1 = pntre[i]; 

      for (INDEXTYPE kk=ia0; kk < ia1; kk++)
      {
         VALUETYPE a0 = val[kk];
         INDEXTYPE ja0 = indx[kk];
         for (INDEXTYPE j=0; j < n; j++)
            C[i*ldc+j] += a0 * B[ja0*ldb + j];  // row-major C  
      }
   }
}

/*============================================================================
 *                            CSC_KIJ 
 *============================================================================*/
   
void dcscmm_KIJ_a1b1   
(
   const char transa,     // 'N', 'T' , 'C' 
   const INDEXTYPE m,     // number of rows of A 
   const INDEXTYPE n,     // number of cols of C
   const INDEXTYPE k,     // number of cols of A
   const double alpha, // double scalar ?? why ptr 
   const char *matdescra,  // 6 characr array descriptor for A:
                           // [G/S/H/T/A/D],[L/U],[N/U],[F/C] -> [G,X,X,C] 
   const double *val,   // NNZ value  
   const INDEXTYPE *indx,  // rowids -> row indices 
   const INDEXTYPE *pntrb, // starting index for colptr
   const INDEXTYPE *pntre, // ending index for colptr
   const VALUETYPE *B,     // Dense B matrix
   const INDEXTYPE ldb,   // 2nd dimension of b for zero-based indexing  
   const double beta,  // double scalar beta[0] 
   double *C,           // Dense matrix c
   const INDEXTYPE ldc    // 2nd dimension size of b 
)
{
   for (INDEXTYPE kk=0; kk < k; kk++)
   {
      INDEXTYPE ja0 = pntrb[kk];
      INDEXTYPE ja1 = pntre[kk];

      for (INDEXTYPE i=ja0; i < ja1; i++)
      {
         double a0 = val[i];
         INDEXTYPE ia0 = indx[i];
         for (INDEXTYPE j=0; j < n; j++)
            C[ia0*ldc+j] += a0 * B[k*ldb + j];  // row-major C  
      }
   }
}

   
