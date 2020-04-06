
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
#ifdef PTTIME
   #pragma omp parallel for schedule(static)
#endif
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
/*
 * case: double precision, loop-order = IKJ, D=128, alpha=X, beta=X
 *       B & C row-major matrix and aligned to VLENb  
 */
void dcsrmm_IKJ_D128_aXbX   
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
#ifdef PTTIME
   #pragma omp parallel for schedule(static)
#endif
   for (INDEXTYPE i=0; i < m; i++)
   {
      INDEXTYPE ia0 = pntrb[i];
      INDEXTYPE ia1 = pntre[i]; 
/*
 *    register block C  
 */
      VTYPE Valpha, Vbeta;
      VTYPE VC0, VC1, VC2, VC3, VC4, VC5, VC6, VC7, 
            VC8, VC9, VC10, VC11, VC12, VC13, VC14, VC15; 
      
      VTYPE tVC0, tVC1, tVC2, tVC3, tVC4, tVC5, tVC6, tVC7, 
            tVC8, tVC9, tVC10, tVC11, tVC12, tVC13, tVC14, tVC15; 
      
      BCL_vzero(VC0);
      BCL_vzero(VC1);
      BCL_vzero(VC2);
      BCL_vzero(VC3);
      BCL_vzero(VC4);
      BCL_vzero(VC5);
      BCL_vzero(VC6);
      BCL_vzero(VC7);
      BCL_vzero(VC8);
      BCL_vzero(VC9);
      BCL_vzero(VC10);
      BCL_vzero(VC11);
      BCL_vzero(VC12);
      BCL_vzero(VC13);
      BCL_vzero(VC14);
      BCL_vzero(VC15);

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

      BCL_vset1(Valpha, alpha);
      BCL_vset1(Vbeta, beta);
      
/*
 *    Assumption: C is aligned by VLENb, use vldu otherwise  
 */
      // t = [C]
      BCL_vld(tVC0, C+i*ldc+VLEN*0); 
      BCL_vld(tVC1, C+i*ldc+VLEN*1); 
      BCL_vld(tVC2, C+i*ldc+VLEN*2); 
      BCL_vld(tVC3, C+i*ldc+VLEN*3); 
      BCL_vld(tVC4, C+i*ldc+VLEN*4); 
      BCL_vld(tVC5, C+i*ldc+VLEN*5); 
      BCL_vld(tVC6, C+i*ldc+VLEN*6); 
      BCL_vld(tVC7, C+i*ldc+VLEN*7); 
      BCL_vld(tVC8, C+i*ldc+VLEN*8); 
      BCL_vld(tVC9, C+i*ldc+VLEN*9); 
      BCL_vld(tVC10, C+i*ldc+VLEN*10); 
      BCL_vld(tVC11, C+i*ldc+VLEN*11); 
      BCL_vld(tVC12, C+i*ldc+VLEN*12); 
      BCL_vld(tVC13, C+i*ldc+VLEN*13); 
      BCL_vld(tVC14, C+i*ldc+VLEN*14); 
      BCL_vld(tVC15, C+i*ldc+VLEN*15); 

      // t = beta * t
      BCL_vmul(tVC0, Vbeta, tVC0); 
      BCL_vmul(tVC1, Vbeta, tVC1); 
      BCL_vmul(tVC2, Vbeta, tVC2); 
      BCL_vmul(tVC3, Vbeta, tVC3); 
      BCL_vmul(tVC4, Vbeta, tVC4); 
      BCL_vmul(tVC5, Vbeta, tVC5); 
      BCL_vmul(tVC6, Vbeta, tVC6); 
      BCL_vmul(tVC7, Vbeta, tVC7); 
      BCL_vmul(tVC8, Vbeta, tVC8); 
      BCL_vmul(tVC9, Vbeta, tVC9); 
      BCL_vmul(tVC10, Vbeta, tVC10); 
      BCL_vmul(tVC11, Vbeta, tVC11); 
      BCL_vmul(tVC12, Vbeta, tVC12); 
      BCL_vmul(tVC13, Vbeta, tVC13); 
      BCL_vmul(tVC14, Vbeta, tVC14); 
      BCL_vmul(tVC15, Vbeta, tVC15); 


      // t += alpha * c
      BCL_vmac(tVC0, Valpha, VC0); 
      BCL_vmac(tVC1, Valpha, VC1); 
      BCL_vmac(tVC2, Valpha, VC2); 
      BCL_vmac(tVC3, Valpha, VC3); 
      BCL_vmac(tVC4, Valpha, VC4); 
      BCL_vmac(tVC5, Valpha, VC5); 
      BCL_vmac(tVC6, Valpha, VC6); 
      BCL_vmac(tVC7, Valpha, VC7); 
      BCL_vmac(tVC8, Valpha, VC8); 
      BCL_vmac(tVC9, Valpha, VC9); 
      BCL_vmac(tVC10, Valpha, VC10); 
      BCL_vmac(tVC11, Valpha, VC11); 
      BCL_vmac(tVC12, Valpha, VC12); 
      BCL_vmac(tVC13, Valpha, VC13); 
      BCL_vmac(tVC14, Valpha, VC14); 
      BCL_vmac(tVC15, Valpha, VC15); 

      // [C] = t
      BCL_vst(C+i*ldc+VLEN*0, tVC0); 
      BCL_vst(C+i*ldc+VLEN*1, tVC1); 
      BCL_vst(C+i*ldc+VLEN*2, tVC2); 
      BCL_vst(C+i*ldc+VLEN*3, tVC3); 
      BCL_vst(C+i*ldc+VLEN*4, tVC4); 
      BCL_vst(C+i*ldc+VLEN*5, tVC5); 
      BCL_vst(C+i*ldc+VLEN*6, tVC6); 
      BCL_vst(C+i*ldc+VLEN*7, tVC7); 
      BCL_vst(C+i*ldc+VLEN*8, tVC8); 
      BCL_vst(C+i*ldc+VLEN*9, tVC9); 
      BCL_vst(C+i*ldc+VLEN*10, tVC10); 
      BCL_vst(C+i*ldc+VLEN*11, tVC11); 
      BCL_vst(C+i*ldc+VLEN*12, tVC12); 
      BCL_vst(C+i*ldc+VLEN*13, tVC13); 
      BCL_vst(C+i*ldc+VLEN*14, tVC14); 
      BCL_vst(C+i*ldc+VLEN*15, tVC15); 
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
#ifdef PTTIME
   // FIXME: Not implemented yet  
#endif
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
