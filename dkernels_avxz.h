
#include "simd.h"

#define DEBUG 1 

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
/*
 * case: double precision, loop-order = IKJ, D=128, alpha=1.0, beta=1.0
 *       B & C row-major matrix and aligned to VLENb  
 */
void dcsrmm_IKJ_D128_a1b1   
(
   const char transa,     // 'N', 'T' , 'C' 
   const INDEXTYPE m,     // number of rows of A 
   const INDEXTYPE n,     // number of cols of C
   const INDEXTYPE k,     // number of cols of A
   const double alpha, // double scalar ?? why ptr 
   const char *matdescra,  // 6 characr array descriptor for A:
                           // [G/S/H/T/A/D],[L/U],[N/U],[F/C] -> [G,X,X,C] 
   const INDEXTYPE nnz,   // nonzeros: need to recreate csr with mkl 
   const INDEXTYPE rows,  // number of rows
   const INDEXTYPE cols,  // number of columns 
   const double *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
   const double *B,     // Dense B matrix
   const INDEXTYPE ldb,   // 2nd dimension of b for zero-based indexing  
   const double beta,  // double scalar beta[0] 
   double *C,           // Dense matrix c
   const INDEXTYPE ldc    // 2nd dimension size of c 
)
{
/*
 * we consider only A->notrans case for now
 *    *transa == 'N' matdescra="GXXC" alpha=1.0 beta=1.0  
 */
   for (INDEXTYPE i=0; i < m; i++)
   {
      INDEXTYPE ia0 = pntrb[i];
      INDEXTYPE ia1 = pntre[i]; 
/*
 *    register block C  
 */
      VTYPE VC0, VC1, VC2, VC3, VC4, VC5, VC6, VC7, 
            VC8, VC9, VC10, VC11, VC12, VC13, VC14, VC15; 
/*
 *    Assumption: C is aligned by VLENb, use vldu otherwise  
 */
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
      
      for (INDEXTYPE kk=ia0; kk < ia1; kk++)
      {
         VTYPE VB0, VB1, VB2, VB3, VB4, VB5, VB6, VB7, 
            VB8, VB9, VB10, VB11, VB12, VB13, VB14, VB15; 
         VTYPE VA0;
         double a0 = val[kk];
         INDEXTYPE ja0 = indx[kk];
         
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

/*=============================================================================
 *                         CSC_KIJ 
 *============================================================================*/

/*
 * case: double precision, loop-order = KIJ, D=128, alpha=1.0, beta=1.0
 *       B & C row-major matrix and aligned to VLENb  
 */
void dcscmm_KIJ_D128_a1b1   
(
   const char transa,     // 'N', 'T' , 'C' 
   const INDEXTYPE m,     // number of rows of A 
   const INDEXTYPE n,     // number of cols of C
   const INDEXTYPE k,     // number of cols of A
   const double alpha, // double scalar ?? why ptr 
   const char *matdescra,  // 6 characr array descriptor for A:
                           // [G/S/H/T/A/D],[L/U],[N/U],[F/C] -> [G,X,X,C] 
   const INDEXTYPE nnz,   // nonzeros: need to recreate csr with mkl 
   const INDEXTYPE rows,  // number of rows
   const INDEXTYPE cols,  // number of columns 
   const double *val,   // NNZ value  
   const INDEXTYPE *indx,  // rowids -> row indices 
   const INDEXTYPE *pntrb, // starting index for colptr
   const INDEXTYPE *pntre, // ending index for colptr
   const double *B,     // Dense B matrix
   const INDEXTYPE ldb,   // 2nd dimension of b for zero-based indexing  
   const double beta,  // double scalar beta[0] 
   double *C,           // Dense matrix c
   const INDEXTYPE ldc    // 2nd dimension size of b 
)
{
   for (INDEXTYPE kk=0; kk < k; kk++)
   {
      VTYPE VB0, VB1, VB2, VB3, VB4, VB5, VB6, VB7, 
            VB8, VB9, VB10, VB11, VB12, VB13, VB14, VB15; 
      INDEXTYPE ja0 = pntrb[kk];
      INDEXTYPE ja1 = pntre[kk]; 

         BCL_vld(VB0, B+k*ldb+VLEN*0); 
         BCL_vld(VB1, B+k*ldb+VLEN*1); 
         BCL_vld(VB2, B+k*ldb+VLEN*2); 
         BCL_vld(VB3, B+k*ldb+VLEN*3); 
         BCL_vld(VB4, B+k*ldb+VLEN*4); 
         BCL_vld(VB5, B+k*ldb+VLEN*5); 
         BCL_vld(VB6, B+k*ldb+VLEN*6); 
         BCL_vld(VB7, B+k*ldb+VLEN*7); 
         BCL_vld(VB8, B+k*ldb+VLEN*8); 
         BCL_vld(VB9, B+k*ldb+VLEN*9); 
         BCL_vld(VB10, B+k*ldb+VLEN*10); 
         BCL_vld(VB11, B+k*ldb+VLEN*11); 
         BCL_vld(VB12, B+k*ldb+VLEN*12); 
         BCL_vld(VB13, B+k*ldb+VLEN*13); 
         BCL_vld(VB14, B+k*ldb+VLEN*14); 
         BCL_vld(VB15, B+k*ldb+VLEN*15); 

      for (INDEXTYPE i=ja0; i < ja1; i++)
      {
         VTYPE VC0, VC1, VC2, VC3, VC4, VC5, VC6, VC7, 
               VC8, VC9, VC10, VC11, VC12, VC13, VC14, VC15; 
         VTYPE VA0; 
         
         double a0 = val[i];
         INDEXTYPE ia0 = indx[i];
         
         BCL_vset1(VA0, a0);
         
         BCL_vld(VC0, C+ia0*ldc+VLEN*0); 
         BCL_vld(VC1, C+ia0*ldc+VLEN*1); 
         BCL_vld(VC2, C+ia0*ldc+VLEN*2); 
         BCL_vld(VC3, C+ia0*ldc+VLEN*3); 
         BCL_vld(VC4, C+ia0*ldc+VLEN*4); 
         BCL_vld(VC5, C+ia0*ldc+VLEN*5); 
         BCL_vld(VC6, C+ia0*ldc+VLEN*6); 
         BCL_vld(VC7, C+ia0*ldc+VLEN*7); 
         BCL_vld(VC8, C+ia0*ldc+VLEN*8); 
         BCL_vld(VC9, C+ia0*ldc+VLEN*9); 
         BCL_vld(VC10, C+ia0*ldc+VLEN*10); 
         BCL_vld(VC11, C+ia0*ldc+VLEN*11); 
         BCL_vld(VC12, C+ia0*ldc+VLEN*12); 
         BCL_vld(VC13, C+ia0*ldc+VLEN*13); 
         BCL_vld(VC14, C+ia0*ldc+VLEN*14); 
         BCL_vld(VC15, C+ia0*ldc+VLEN*15); 
         
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
      
         BCL_vst(C+ia0*ldc+VLEN*0, VC0); 
         BCL_vst(C+ia0*ldc+VLEN*1, VC1); 
         BCL_vst(C+ia0*ldc+VLEN*2, VC2); 
         BCL_vst(C+ia0*ldc+VLEN*3, VC3); 
         BCL_vst(C+ia0*ldc+VLEN*4, VC4); 
         BCL_vst(C+ia0*ldc+VLEN*5, VC5); 
         BCL_vst(C+ia0*ldc+VLEN*6, VC6); 
         BCL_vst(C+ia0*ldc+VLEN*7, VC7); 
         BCL_vst(C+ia0*ldc+VLEN*8, VC8); 
         BCL_vst(C+ia0*ldc+VLEN*9, VC9); 
         BCL_vst(C+ia0*ldc+VLEN*10, VC10); 
         BCL_vst(C+ia0*ldc+VLEN*11, VC11); 
         BCL_vst(C+ia0*ldc+VLEN*12, VC12); 
         BCL_vst(C+ia0*ldc+VLEN*13, VC13); 
         BCL_vst(C+ia0*ldc+VLEN*14, VC14); 
         BCL_vst(C+ia0*ldc+VLEN*15, VC15); 
      }
   }
}
