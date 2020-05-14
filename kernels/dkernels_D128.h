#ifndef DKERNEL_D128_H
#define DKERNEL_D128_H

#include "kernels.h"
#include <assert.h> 

// consider aligned, make sure to call with aligned memory 
//#ifdef ALIGNED
#if 1
   #define BCL_VLD BCL_vld 
   #define BCL_VST BCL_vst 
#else
   #define BCL_VLD BCL_vldu 
   #define BCL_VST BCL_vstu 
#endif

void dcsrmm_IKJ_D128_a1b1 (const char transa, const BCL_INT m, const BCL_INT n, 
      const BCL_INT k,const double alpha, const char *matdescra, 
      const BCL_INT nnz, const BCL_INT rows, const BCL_INT cols, 
      const double *val, const BCL_INT *indx, const BCL_INT *pntrb, 
      const BCL_INT *pntre, const double *B, const BCL_INT ldb, 
      const double beta, double *C, const BCL_INT ldc);

void dcsrmm_IKJ_D128_LDB_a1b1 (const char transa, const BCL_INT m, const BCL_INT n, 
      const BCL_INT k,const double alpha, const char *matdescra, 
      const BCL_INT nnz, const BCL_INT rows, const BCL_INT cols, 
      const double *val, const BCL_INT *indx, const BCL_INT *pntrb, 
      const BCL_INT *pntre, const double *B, const BCL_INT ldb, 
      const double beta, double *C, const BCL_INT ldc);

void dcsrmm_IKJ_D128_aXbX (const char transa, const BCL_INT m, const BCL_INT n, 
      const BCL_INT k,const double alpha, const char *matdescra, 
      const BCL_INT nnz, const BCL_INT rows, const BCL_INT cols, 
      const double *val, const BCL_INT *indx, const BCL_INT *pntrb, 
      const BCL_INT *pntre, const double *B, const BCL_INT ldb, 
      const double beta, double *C, const BCL_INT ldc);

void dcsrmm_IKJ_D128_LDB_aXbX (const char transa, const BCL_INT m, const BCL_INT n, 
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

/*
 * case: double precision, loop-order = IKJ, D=128, alpha=1.0, beta=1.0
 *       B & C row-major matrix and aligned to VLENb  
 */
void dcsrmm_IKJ_D128_a1b1   
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
 * we consider only A->notrans case for now
 *    *transa == 'N' matdescra="GXXC" alpha=1.0 beta=1.0  
 */
#ifdef PTTIME
   #pragma omp parallel for schedule(static)
#endif
   for (BCL_INT i=0; i < m; i++)
   {
      BCL_INT ia0 = pntrb[i];
      BCL_INT ia1 = pntre[i]; 
/*
 *    register block C  
 */
      VTYPE VC0, VC1, VC2, VC3, VC4, VC5, VC6, VC7, 
            VC8, VC9, VC10, VC11, VC12, VC13, VC14, VC15; 
/*
 *    Assumption: C is aligned by VLENb, use vldu otherwise  
 */
      BCL_VLD(VC0, C+i*ldc+VLEN*0); 
      BCL_VLD(VC1, C+i*ldc+VLEN*1); 
      BCL_VLD(VC2, C+i*ldc+VLEN*2); 
      BCL_VLD(VC3, C+i*ldc+VLEN*3); 
      BCL_VLD(VC4, C+i*ldc+VLEN*4); 
      BCL_VLD(VC5, C+i*ldc+VLEN*5); 
      BCL_VLD(VC6, C+i*ldc+VLEN*6); 
      BCL_VLD(VC7, C+i*ldc+VLEN*7); 
      BCL_VLD(VC8, C+i*ldc+VLEN*8); 
      BCL_VLD(VC9, C+i*ldc+VLEN*9); 
      BCL_VLD(VC10, C+i*ldc+VLEN*10); 
      BCL_VLD(VC11, C+i*ldc+VLEN*11); 
      BCL_VLD(VC12, C+i*ldc+VLEN*12); 
      BCL_VLD(VC13, C+i*ldc+VLEN*13); 
      BCL_VLD(VC14, C+i*ldc+VLEN*14); 
      BCL_VLD(VC15, C+i*ldc+VLEN*15); 
      
      for (BCL_INT kk=ia0; kk < ia1; kk++)
      {
         VTYPE VB0, VB1, VB2, VB3, VB4, VB5, VB6, VB7, 
            VB8, VB9, VB10, VB11, VB12, VB13, VB14, VB15; 
         VTYPE VA0;
         double a0 = val[kk];
         BCL_INT ja0 = indx[kk];
         
         BCL_vset1(VA0, a0);
         BCL_VLD(VB0, B+ja0*ldb+VLEN*0); 
         BCL_VLD(VB1, B+ja0*ldb+VLEN*1); 
         BCL_VLD(VB2, B+ja0*ldb+VLEN*2); 
         BCL_VLD(VB3, B+ja0*ldb+VLEN*3); 
         BCL_VLD(VB4, B+ja0*ldb+VLEN*4); 
         BCL_VLD(VB5, B+ja0*ldb+VLEN*5); 
         BCL_VLD(VB6, B+ja0*ldb+VLEN*6); 
         BCL_VLD(VB7, B+ja0*ldb+VLEN*7); 
         BCL_VLD(VB8, B+ja0*ldb+VLEN*8); 
         BCL_VLD(VB9, B+ja0*ldb+VLEN*9); 
         BCL_VLD(VB10, B+ja0*ldb+VLEN*10); 
         BCL_VLD(VB11, B+ja0*ldb+VLEN*11); 
         BCL_VLD(VB12, B+ja0*ldb+VLEN*12); 
         BCL_VLD(VB13, B+ja0*ldb+VLEN*13); 
         BCL_VLD(VB14, B+ja0*ldb+VLEN*14); 
         BCL_VLD(VB15, B+ja0*ldb+VLEN*15); 
       
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
      BCL_VST(C+i*ldc+VLEN*0, VC0); 
      BCL_VST(C+i*ldc+VLEN*1, VC1); 
      BCL_VST(C+i*ldc+VLEN*2, VC2); 
      BCL_VST(C+i*ldc+VLEN*3, VC3); 
      BCL_VST(C+i*ldc+VLEN*4, VC4); 
      BCL_VST(C+i*ldc+VLEN*5, VC5); 
      BCL_VST(C+i*ldc+VLEN*6, VC6); 
      BCL_VST(C+i*ldc+VLEN*7, VC7); 
      BCL_VST(C+i*ldc+VLEN*8, VC8); 
      BCL_VST(C+i*ldc+VLEN*9, VC9); 
      BCL_VST(C+i*ldc+VLEN*10, VC10); 
      BCL_VST(C+i*ldc+VLEN*11, VC11); 
      BCL_VST(C+i*ldc+VLEN*12, VC12); 
      BCL_VST(C+i*ldc+VLEN*13, VC13); 
      BCL_VST(C+i*ldc+VLEN*14, VC14); 
      BCL_VST(C+i*ldc+VLEN*15, VC15); 
   }
}
/*
 * case: double precision, loop-order = IKJ, D=128, alpha=X, beta=X
 *       B & C row-major matrix and aligned to VLENb  
 */
void dcsrmm_IKJ_D128_aXbX   
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
 * we consider only A->notrans case for now
 *    *transa == 'N' matdescra="GXXC" alpha=1.0 beta=1.0  
 */
#ifdef PTTIME
   #pragma omp parallel for schedule(static)
#endif
   for (BCL_INT i=0; i < m; i++)
   {
      BCL_INT ia0 = pntrb[i];
      BCL_INT ia1 = pntre[i]; 
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

      for (BCL_INT kk=ia0; kk < ia1; kk++)
      {
         VTYPE VB0, VB1, VB2, VB3, VB4, VB5, VB6, VB7, 
            VB8, VB9, VB10, VB11, VB12, VB13, VB14, VB15; 
         VTYPE VA0;
         double a0 = val[kk];
         BCL_INT ja0 = indx[kk];
         
         BCL_vset1(VA0, a0);
         BCL_VLD(VB0, B+ja0*ldb+VLEN*0); 
         BCL_VLD(VB1, B+ja0*ldb+VLEN*1); 
         BCL_VLD(VB2, B+ja0*ldb+VLEN*2); 
         BCL_VLD(VB3, B+ja0*ldb+VLEN*3); 
         BCL_VLD(VB4, B+ja0*ldb+VLEN*4); 
         BCL_VLD(VB5, B+ja0*ldb+VLEN*5); 
         BCL_VLD(VB6, B+ja0*ldb+VLEN*6); 
         BCL_VLD(VB7, B+ja0*ldb+VLEN*7); 
         BCL_VLD(VB8, B+ja0*ldb+VLEN*8); 
         BCL_VLD(VB9, B+ja0*ldb+VLEN*9); 
         BCL_VLD(VB10, B+ja0*ldb+VLEN*10); 
         BCL_VLD(VB11, B+ja0*ldb+VLEN*11); 
         BCL_VLD(VB12, B+ja0*ldb+VLEN*12); 
         BCL_VLD(VB13, B+ja0*ldb+VLEN*13); 
         BCL_VLD(VB14, B+ja0*ldb+VLEN*14); 
         BCL_VLD(VB15, B+ja0*ldb+VLEN*15); 
       
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
 *    Assumption: C is aligned by VLENb, use VLDu otherwise  
 */
      // t = [C]
      BCL_VLD(tVC0, C+i*ldc+VLEN*0); 
      BCL_VLD(tVC1, C+i*ldc+VLEN*1); 
      BCL_VLD(tVC2, C+i*ldc+VLEN*2); 
      BCL_VLD(tVC3, C+i*ldc+VLEN*3); 
      BCL_VLD(tVC4, C+i*ldc+VLEN*4); 
      BCL_VLD(tVC5, C+i*ldc+VLEN*5); 
      BCL_VLD(tVC6, C+i*ldc+VLEN*6); 
      BCL_VLD(tVC7, C+i*ldc+VLEN*7); 
      BCL_VLD(tVC8, C+i*ldc+VLEN*8); 
      BCL_VLD(tVC9, C+i*ldc+VLEN*9); 
      BCL_VLD(tVC10, C+i*ldc+VLEN*10); 
      BCL_VLD(tVC11, C+i*ldc+VLEN*11); 
      BCL_VLD(tVC12, C+i*ldc+VLEN*12); 
      BCL_VLD(tVC13, C+i*ldc+VLEN*13); 
      BCL_VLD(tVC14, C+i*ldc+VLEN*14); 
      BCL_VLD(tVC15, C+i*ldc+VLEN*15); 

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
      BCL_VST(C+i*ldc+VLEN*0, tVC0); 
      BCL_VST(C+i*ldc+VLEN*1, tVC1); 
      BCL_VST(C+i*ldc+VLEN*2, tVC2); 
      BCL_VST(C+i*ldc+VLEN*3, tVC3); 
      BCL_VST(C+i*ldc+VLEN*4, tVC4); 
      BCL_VST(C+i*ldc+VLEN*5, tVC5); 
      BCL_VST(C+i*ldc+VLEN*6, tVC6); 
      BCL_VST(C+i*ldc+VLEN*7, tVC7); 
      BCL_VST(C+i*ldc+VLEN*8, tVC8); 
      BCL_VST(C+i*ldc+VLEN*9, tVC9); 
      BCL_VST(C+i*ldc+VLEN*10, tVC10); 
      BCL_VST(C+i*ldc+VLEN*11, tVC11); 
      BCL_VST(C+i*ldc+VLEN*12, tVC12); 
      BCL_VST(C+i*ldc+VLEN*13, tVC13); 
      BCL_VST(C+i*ldc+VLEN*14, tVC14); 
      BCL_VST(C+i*ldc+VLEN*15, tVC15); 
   }
}

#ifdef LOAD_BALANCE

   #ifndef NTHREADS
      #define NTHREADS 18 
   #endif
/*
 * load balancing with row element count 
 */
#if 0
void dcsrmm_IKJ_D128_LDB_a1b1 
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
 * we consider only A->notrans case for now
 *    *transa == 'N' matdescra="GXXC" alpha=1.0 beta=1.0  
 */
   TH_INT *ThRowId; 
   int RowPerThd; 
   BCL_INT i;
   BCL_INT Mnnz = 0; // Mnnz is the non zero value in M rows 
   for (i=0; i < m; i++)
      Mnnz += (pntre[i] - pntrb[i]); 

   omp_set_num_threads(NTHREADS);
/*
 * FIXME: what if, NTHREADS and nthreads are not same!!!
 */
/*
 *    NOTE: now the idea is to distribute rows among threads by nnz count... 
 *    We will populate ThRowId just like rowptr, but with rowid
 *       ThRowId[i] -> indicates the row id i-thd start working with
 */
#if 1
   {
      RowPerThd = Mnnz / NTHREADS; 
      ThRowId = malloc(sizeof(TH_INT)*(NTHREADS+1)); //
      assert(ThRowId);
      ThRowId[0] = 0;
      for (BCL_INT t=1, i=0; t < NTHREADS+1; t++)
      {
         if (i > m)
            ThRowId[t] = 0;
         else
         {
            while(pntre[i] < t*RowPerThd)
               i++;
            ThRowId[t] = (TH_INT)(++i);
         }
      }
/*
 *       NOTE: Why may not match: 
 *          Later rows has very few non zeros 
 */
      if (ThRowId[NTHREADS] != m)
      {
            //fprintf(stderr, "Last thread has to cover all remaining rows");
            ThRowId[NTHREADS] = (TH_INT)(m);
      }

   }
   #if 0
      fprintf(stderr, "NTHREADS = %d\n", NTHREADS);
      for (i=0; i < NTHREADS+1; i++)
         fprintf(stderr, "%d, ", ThRowId[i]);
      fprintf(stderr, "\n");
   #endif
#endif

   #pragma omp parallel
   {
      int i, j, t;
      int id = omp_get_thread_num();
      int nthreads = omp_get_num_threads(); 
#if 1 
      assert(NTHREADS == nthreads);
#endif
      for (i=ThRowId[id]; i < ThRowId[id+1]; i++)
      {
      #if 0
         fprintf(stderr, " rowid (t%d) = %d\n", id, i);
      #endif
         BCL_INT ia0 = pntrb[i];
         BCL_INT ia1 = pntre[i]; 
/*
 *       register block C  
 */
         VTYPE VC0, VC1, VC2, VC3, VC4, VC5, VC6, VC7, 
               VC8, VC9, VC10, VC11, VC12, VC13, VC14, VC15; 
/*
 *       Assumption: C is aligned by VLENb, use vldu otherwise  
 */
         BCL_VLD(VC0, C+i*ldc+VLEN*0); 
         BCL_VLD(VC1, C+i*ldc+VLEN*1); 
         BCL_VLD(VC2, C+i*ldc+VLEN*2); 
         BCL_VLD(VC3, C+i*ldc+VLEN*3); 
         BCL_VLD(VC4, C+i*ldc+VLEN*4); 
         BCL_VLD(VC5, C+i*ldc+VLEN*5); 
         BCL_VLD(VC6, C+i*ldc+VLEN*6); 
         BCL_VLD(VC7, C+i*ldc+VLEN*7); 
         BCL_VLD(VC8, C+i*ldc+VLEN*8); 
         BCL_VLD(VC9, C+i*ldc+VLEN*9); 
         BCL_VLD(VC10, C+i*ldc+VLEN*10); 
         BCL_VLD(VC11, C+i*ldc+VLEN*11); 
         BCL_VLD(VC12, C+i*ldc+VLEN*12); 
         BCL_VLD(VC13, C+i*ldc+VLEN*13); 
         BCL_VLD(VC14, C+i*ldc+VLEN*14); 
         BCL_VLD(VC15, C+i*ldc+VLEN*15); 
      
         for (BCL_INT kk=ia0; kk < ia1; kk++)
         {
            VTYPE VB0, VB1, VB2, VB3, VB4, VB5, VB6, VB7, 
               VB8, VB9, VB10, VB11, VB12, VB13, VB14, VB15; 
            VTYPE VA0;
            double a0 = val[kk];
            BCL_INT ja0 = indx[kk];
         
            BCL_vset1(VA0, a0);
            BCL_VLD(VB0, B+ja0*ldb+VLEN*0); 
            BCL_VLD(VB1, B+ja0*ldb+VLEN*1); 
            BCL_VLD(VB2, B+ja0*ldb+VLEN*2); 
            BCL_VLD(VB3, B+ja0*ldb+VLEN*3); 
            BCL_VLD(VB4, B+ja0*ldb+VLEN*4); 
            BCL_VLD(VB5, B+ja0*ldb+VLEN*5); 
            BCL_VLD(VB6, B+ja0*ldb+VLEN*6); 
            BCL_VLD(VB7, B+ja0*ldb+VLEN*7); 
            BCL_VLD(VB8, B+ja0*ldb+VLEN*8); 
            BCL_VLD(VB9, B+ja0*ldb+VLEN*9); 
            BCL_VLD(VB10, B+ja0*ldb+VLEN*10); 
            BCL_VLD(VB11, B+ja0*ldb+VLEN*11); 
            BCL_VLD(VB12, B+ja0*ldb+VLEN*12); 
            BCL_VLD(VB13, B+ja0*ldb+VLEN*13); 
            BCL_VLD(VB14, B+ja0*ldb+VLEN*14); 
            BCL_VLD(VB15, B+ja0*ldb+VLEN*15); 
       
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
         BCL_VST(C+i*ldc+VLEN*0, VC0); 
         BCL_VST(C+i*ldc+VLEN*1, VC1); 
         BCL_VST(C+i*ldc+VLEN*2, VC2); 
         BCL_VST(C+i*ldc+VLEN*3, VC3); 
         BCL_VST(C+i*ldc+VLEN*4, VC4); 
         BCL_VST(C+i*ldc+VLEN*5, VC5); 
         BCL_VST(C+i*ldc+VLEN*6, VC6); 
         BCL_VST(C+i*ldc+VLEN*7, VC7); 
         BCL_VST(C+i*ldc+VLEN*8, VC8); 
         BCL_VST(C+i*ldc+VLEN*9, VC9); 
         BCL_VST(C+i*ldc+VLEN*10, VC10); 
         BCL_VST(C+i*ldc+VLEN*11, VC11); 
         BCL_VST(C+i*ldc+VLEN*12, VC12); 
         BCL_VST(C+i*ldc+VLEN*13, VC13); 
         BCL_VST(C+i*ldc+VLEN*14, VC14); 
         BCL_VST(C+i*ldc+VLEN*15, VC15); 
      }
   }
   free(ThRowId);
}
#else
void dcsrmm_IKJ_D128_LDB_a1b1 
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
 * we consider only A->notrans case for now
 *    *transa == 'N' matdescra="GXXC" alpha=1.0 beta=1.0  
 */
   omp_set_num_threads(NTHREADS);

   #pragma omp parallel
   {
      int i, j, t;
      int RowPerThd, tt;
      BCL_INT rowb, rowe;
      BCL_INT Mnnz = 0; // Mnnz is the non zero value in M rows 
      int deg, cumRow, curRow;
      int id = omp_get_thread_num();
      int nthreads = omp_get_num_threads(); 
      
      for (i=0; i < m; i++)
         Mnnz += (pntre[i] - pntrb[i]); 
      RowPerThd = Mnnz / nthreads; 
      
      curRow = cumRow = 0; 
      tt = 1; 
      
      // set rowstart for 1st thread  
      if (id == 0) 
         rowb = 0;

      for (i=0; i < m; i++)
      {
         deg = pntre[i] - pntrb[i]; 
         cumRow += deg;
         curRow += deg;
         if (curRow > RowPerThd)
         {
            //ThRowId[tt] = i;
            if (tt == id)
               rowb = i; 
            if (tt == id+1)
               rowe = i; 
            curRow = 0;
            RowPerThd = (Mnnz - cumRow) / (nthreads - tt);
            tt += 1; 
         }
      }
      #if 0
      fprintf(stdout, "---tt = %d\n", tt);
      #endif
      if (tt == id+1)
         rowe = m; 


      for (i=rowb; i < rowe; i++)
      {
      #if 0
         fprintf(stderr, " rowid (t%d) = %d\n", id, i);
      #endif
         BCL_INT ia0 = pntrb[i];
         BCL_INT ia1 = pntre[i]; 
/*
 *       register block C  
 */
         VTYPE VC0, VC1, VC2, VC3, VC4, VC5, VC6, VC7, 
               VC8, VC9, VC10, VC11, VC12, VC13, VC14, VC15; 
/*
 *       Assumption: C is aligned by VLENb, use vldu otherwise  
 */
         BCL_VLD(VC0, C+i*ldc+VLEN*0); 
         BCL_VLD(VC1, C+i*ldc+VLEN*1); 
         BCL_VLD(VC2, C+i*ldc+VLEN*2); 
         BCL_VLD(VC3, C+i*ldc+VLEN*3); 
         BCL_VLD(VC4, C+i*ldc+VLEN*4); 
         BCL_VLD(VC5, C+i*ldc+VLEN*5); 
         BCL_VLD(VC6, C+i*ldc+VLEN*6); 
         BCL_VLD(VC7, C+i*ldc+VLEN*7); 
         BCL_VLD(VC8, C+i*ldc+VLEN*8); 
         BCL_VLD(VC9, C+i*ldc+VLEN*9); 
         BCL_VLD(VC10, C+i*ldc+VLEN*10); 
         BCL_VLD(VC11, C+i*ldc+VLEN*11); 
         BCL_VLD(VC12, C+i*ldc+VLEN*12); 
         BCL_VLD(VC13, C+i*ldc+VLEN*13); 
         BCL_VLD(VC14, C+i*ldc+VLEN*14); 
         BCL_VLD(VC15, C+i*ldc+VLEN*15); 
      
         for (BCL_INT kk=ia0; kk < ia1; kk++)
         {
            VTYPE VB0, VB1, VB2, VB3, VB4, VB5, VB6, VB7, 
               VB8, VB9, VB10, VB11, VB12, VB13, VB14, VB15; 
            VTYPE VA0;
            double a0 = val[kk];
            BCL_INT ja0 = indx[kk];
         
            BCL_vset1(VA0, a0);
            BCL_VLD(VB0, B+ja0*ldb+VLEN*0); 
            BCL_VLD(VB1, B+ja0*ldb+VLEN*1); 
            BCL_VLD(VB2, B+ja0*ldb+VLEN*2); 
            BCL_VLD(VB3, B+ja0*ldb+VLEN*3); 
            BCL_VLD(VB4, B+ja0*ldb+VLEN*4); 
            BCL_VLD(VB5, B+ja0*ldb+VLEN*5); 
            BCL_VLD(VB6, B+ja0*ldb+VLEN*6); 
            BCL_VLD(VB7, B+ja0*ldb+VLEN*7); 
            BCL_VLD(VB8, B+ja0*ldb+VLEN*8); 
            BCL_VLD(VB9, B+ja0*ldb+VLEN*9); 
            BCL_VLD(VB10, B+ja0*ldb+VLEN*10); 
            BCL_VLD(VB11, B+ja0*ldb+VLEN*11); 
            BCL_VLD(VB12, B+ja0*ldb+VLEN*12); 
            BCL_VLD(VB13, B+ja0*ldb+VLEN*13); 
            BCL_VLD(VB14, B+ja0*ldb+VLEN*14); 
            BCL_VLD(VB15, B+ja0*ldb+VLEN*15); 
       
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
         BCL_VST(C+i*ldc+VLEN*0, VC0); 
         BCL_VST(C+i*ldc+VLEN*1, VC1); 
         BCL_VST(C+i*ldc+VLEN*2, VC2); 
         BCL_VST(C+i*ldc+VLEN*3, VC3); 
         BCL_VST(C+i*ldc+VLEN*4, VC4); 
         BCL_VST(C+i*ldc+VLEN*5, VC5); 
         BCL_VST(C+i*ldc+VLEN*6, VC6); 
         BCL_VST(C+i*ldc+VLEN*7, VC7); 
         BCL_VST(C+i*ldc+VLEN*8, VC8); 
         BCL_VST(C+i*ldc+VLEN*9, VC9); 
         BCL_VST(C+i*ldc+VLEN*10, VC10); 
         BCL_VST(C+i*ldc+VLEN*11, VC11); 
         BCL_VST(C+i*ldc+VLEN*12, VC12); 
         BCL_VST(C+i*ldc+VLEN*13, VC13); 
         BCL_VST(C+i*ldc+VLEN*14, VC14); 
         BCL_VST(C+i*ldc+VLEN*15, VC15); 
      }
   }
}

#endif

#if 0
void dcsrmm_IKJ_D128_LDB_aXbX   
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
 * we consider only A->notrans case for now
 *    *transa == 'N' matdescra="GXXC" alpha=1.0 beta=1.0  
 */
   TH_INT *ThRowId; 
   int RowPerThd, tt;
   int deg, cumRow, curRow; 
   BCL_INT i;
   BCL_INT Mnnz = 0; // Mnnz is the non zero value in M rows 
   
   omp_set_num_threads(NTHREADS);
   
   
   for (i=0; i < m; i++)
      Mnnz += (pntre[i] - pntrb[i]); 

/*
 * FIXME: what if, NTHREADS and nthreads are not same!!!
 */
/*
 *    NOTE: now the idea is to distribute rows among threads by nnz count... 
 *    We will populate ThRowId just like rowptr, but with rowid
 *       ThRowId[i] -> indicates the row id i-thd start working with
 */
#if 0
   {
      RowPerThd = Mnnz / NTHREADS; 
      ThRowId = malloc(sizeof(TH_INT)*(NTHREADS+1)); //
      assert(ThRowId);
      ThRowId[0] = 0;
      for (BCL_INT t=1, i=0; t < NTHREADS+1; t++)
      {
         if (i > m)
            ThRowId[t] = 0;
         else
         {
            while(pntre[i] < t*RowPerThd)
               i++;
            ThRowId[t] = (TH_INT)(++i);
         }
      }
/*
 *       NOTE: Why may not match: 
 *          Later rows has very few non zeros 
 */
      if (ThRowId[NTHREADS] != m)
      {
            //fprintf(stderr, "Last thread has to cover all remaining rows");
            ThRowId[NTHREADS] = (TH_INT)(m);
      }

   }
   #if 0
      fprintf(stderr, "NTHREADS = %d\n", NTHREADS);
      for (i=0; i < NTHREADS+1; i++)
         fprintf(stderr, "%d, ", ThRowId[i]);
      fprintf(stderr, "\n");
   #endif
#else    /* another implementation */
   RowPerThd = Mnnz / NTHREADS; 
   ThRowId = malloc(sizeof(TH_INT)*(NTHREADS+1)); //
   assert(ThRowId);
   
   ThRowId[0] = 0; 
   curRow = cumRow = 0;
   tt = 1;
   for (i=0; i < m; i++)
   {
      deg = pntre[i] - pntrb[i]; 
      cumRow += deg;
      curRow += deg;
      if (curRow > RowPerThd)
      {
         ThRowId[tt] = i; 
         curRow = 0;
         RowPerThd = (Mnnz - cumRow) / (NTHREADS - tt);
         tt += 1; 
      }
   }
   ThRowId[tt] = m;
   assert(cumRow == Mnnz);
#endif
   
   #pragma omp parallel
   {
      TH_INT i;  // assumming M fits in short int 
      int id = omp_get_thread_num();
      int nthreads = omp_get_num_threads(); 
#if 1 
      assert(NTHREADS == nthreads);
#endif
      for (i=ThRowId[id]; i < ThRowId[id+1]; i++)
      {
         BCL_INT ia0 = pntrb[i];
         BCL_INT ia1 = pntre[i]; 
/*
 *       register block C  
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

         for (BCL_INT kk=ia0; kk < ia1; kk++)
         {
            VTYPE VB0, VB1, VB2, VB3, VB4, VB5, VB6, VB7, 
               VB8, VB9, VB10, VB11, VB12, VB13, VB14, VB15; 
            VTYPE VA0;
            double a0 = val[kk];
            BCL_INT ja0 = indx[kk];
         
            BCL_vset1(VA0, a0);
            BCL_VLD(VB0, B+ja0*ldb+VLEN*0); 
            BCL_VLD(VB1, B+ja0*ldb+VLEN*1); 
            BCL_VLD(VB2, B+ja0*ldb+VLEN*2); 
            BCL_VLD(VB3, B+ja0*ldb+VLEN*3); 
            BCL_VLD(VB4, B+ja0*ldb+VLEN*4); 
            BCL_VLD(VB5, B+ja0*ldb+VLEN*5); 
            BCL_VLD(VB6, B+ja0*ldb+VLEN*6); 
            BCL_VLD(VB7, B+ja0*ldb+VLEN*7); 
            BCL_VLD(VB8, B+ja0*ldb+VLEN*8); 
            BCL_VLD(VB9, B+ja0*ldb+VLEN*9); 
            BCL_VLD(VB10, B+ja0*ldb+VLEN*10); 
            BCL_VLD(VB11, B+ja0*ldb+VLEN*11); 
            BCL_VLD(VB12, B+ja0*ldb+VLEN*12); 
            BCL_VLD(VB13, B+ja0*ldb+VLEN*13); 
            BCL_VLD(VB14, B+ja0*ldb+VLEN*14); 
            BCL_VLD(VB15, B+ja0*ldb+VLEN*15); 
       
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
 *       Assumption: C is aligned by VLENb, use VLDu otherwise  
 */
         // t = [C]
         BCL_VLD(tVC0, C+i*ldc+VLEN*0); 
         BCL_VLD(tVC1, C+i*ldc+VLEN*1); 
         BCL_VLD(tVC2, C+i*ldc+VLEN*2); 
         BCL_VLD(tVC3, C+i*ldc+VLEN*3); 
         BCL_VLD(tVC4, C+i*ldc+VLEN*4); 
         BCL_VLD(tVC5, C+i*ldc+VLEN*5); 
         BCL_VLD(tVC6, C+i*ldc+VLEN*6); 
         BCL_VLD(tVC7, C+i*ldc+VLEN*7); 
         BCL_VLD(tVC8, C+i*ldc+VLEN*8); 
         BCL_VLD(tVC9, C+i*ldc+VLEN*9); 
         BCL_VLD(tVC10, C+i*ldc+VLEN*10); 
         BCL_VLD(tVC11, C+i*ldc+VLEN*11); 
         BCL_VLD(tVC12, C+i*ldc+VLEN*12); 
         BCL_VLD(tVC13, C+i*ldc+VLEN*13); 
         BCL_VLD(tVC14, C+i*ldc+VLEN*14); 
         BCL_VLD(tVC15, C+i*ldc+VLEN*15); 

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
         BCL_VST(C+i*ldc+VLEN*0, tVC0); 
         BCL_VST(C+i*ldc+VLEN*1, tVC1); 
         BCL_VST(C+i*ldc+VLEN*2, tVC2); 
         BCL_VST(C+i*ldc+VLEN*3, tVC3); 
         BCL_VST(C+i*ldc+VLEN*4, tVC4); 
         BCL_VST(C+i*ldc+VLEN*5, tVC5); 
         BCL_VST(C+i*ldc+VLEN*6, tVC6); 
         BCL_VST(C+i*ldc+VLEN*7, tVC7); 
         BCL_VST(C+i*ldc+VLEN*8, tVC8); 
         BCL_VST(C+i*ldc+VLEN*9, tVC9); 
         BCL_VST(C+i*ldc+VLEN*10, tVC10); 
         BCL_VST(C+i*ldc+VLEN*11, tVC11); 
         BCL_VST(C+i*ldc+VLEN*12, tVC12); 
         BCL_VST(C+i*ldc+VLEN*13, tVC13); 
         BCL_VST(C+i*ldc+VLEN*14, tVC14); 
         BCL_VST(C+i*ldc+VLEN*15, tVC15); 
      }
   }
   free(ThRowId);
}
#else /* imp of LDB without extra memory */

void dcsrmm_IKJ_D128_LDB_aXbX   
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
 * we consider only A->notrans case for now
 *    *transa == 'N' matdescra="GXXC" alpha=1.0 beta=1.0  
 */ 
   
   omp_set_num_threads(NTHREADS);
   
   #pragma omp parallel
   {
      int RowPerThd, tt;
      BCL_INT i, rowb, rowe;
      BCL_INT Mnnz = 0; // Mnnz is the non zero value in M rows 
      int deg, cumRow, curRow;
      int id = omp_get_thread_num();
      int nthreads = omp_get_num_threads(); 
      
      for (i=0; i < m; i++)
         Mnnz += (pntre[i] - pntrb[i]); 
      RowPerThd = Mnnz / nthreads; 
      
      curRow = cumRow = 0; 
      tt = 1; 
      
      // set rowstart for 1st thread  
      if (id == 0) 
         rowb = 0;

      for (i=0; i < m; i++)
      {
         deg = pntre[i] - pntrb[i]; 
         cumRow += deg;
         curRow += deg;
         if (curRow > RowPerThd)
         {
            //ThRowId[tt] = i;
            if (tt == id)
               rowb = i; 
            if (tt == id+1)
               rowe = i; 
            curRow = 0;
            RowPerThd = (Mnnz - cumRow) / (nthreads - tt);
            tt += 1; 
         }
      }
      #if 0
      fprintf(stdout, "---tt = %d\n", tt);
      #endif
      if (tt == id+1)
         rowe = m; 


      for (i=rowb; i < rowe; i++)
      {
         BCL_INT ia0 = pntrb[i];
         BCL_INT ia1 = pntre[i]; 
/*
 *       register block C  
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

         for (BCL_INT kk=ia0; kk < ia1; kk++)
         {
            VTYPE VB0, VB1, VB2, VB3, VB4, VB5, VB6, VB7, 
               VB8, VB9, VB10, VB11, VB12, VB13, VB14, VB15; 
            VTYPE VA0;
            double a0 = val[kk];
            BCL_INT ja0 = indx[kk];
         
            BCL_vset1(VA0, a0);
            BCL_VLD(VB0, B+ja0*ldb+VLEN*0); 
            BCL_VLD(VB1, B+ja0*ldb+VLEN*1); 
            BCL_VLD(VB2, B+ja0*ldb+VLEN*2); 
            BCL_VLD(VB3, B+ja0*ldb+VLEN*3); 
            BCL_VLD(VB4, B+ja0*ldb+VLEN*4); 
            BCL_VLD(VB5, B+ja0*ldb+VLEN*5); 
            BCL_VLD(VB6, B+ja0*ldb+VLEN*6); 
            BCL_VLD(VB7, B+ja0*ldb+VLEN*7); 
            BCL_VLD(VB8, B+ja0*ldb+VLEN*8); 
            BCL_VLD(VB9, B+ja0*ldb+VLEN*9); 
            BCL_VLD(VB10, B+ja0*ldb+VLEN*10); 
            BCL_VLD(VB11, B+ja0*ldb+VLEN*11); 
            BCL_VLD(VB12, B+ja0*ldb+VLEN*12); 
            BCL_VLD(VB13, B+ja0*ldb+VLEN*13); 
            BCL_VLD(VB14, B+ja0*ldb+VLEN*14); 
            BCL_VLD(VB15, B+ja0*ldb+VLEN*15); 
       
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
 *       Assumption: C is aligned by VLENb, use VLDu otherwise  
 */
         // t = [C]
         BCL_VLD(tVC0, C+i*ldc+VLEN*0); 
         BCL_VLD(tVC1, C+i*ldc+VLEN*1); 
         BCL_VLD(tVC2, C+i*ldc+VLEN*2); 
         BCL_VLD(tVC3, C+i*ldc+VLEN*3); 
         BCL_VLD(tVC4, C+i*ldc+VLEN*4); 
         BCL_VLD(tVC5, C+i*ldc+VLEN*5); 
         BCL_VLD(tVC6, C+i*ldc+VLEN*6); 
         BCL_VLD(tVC7, C+i*ldc+VLEN*7); 
         BCL_VLD(tVC8, C+i*ldc+VLEN*8); 
         BCL_VLD(tVC9, C+i*ldc+VLEN*9); 
         BCL_VLD(tVC10, C+i*ldc+VLEN*10); 
         BCL_VLD(tVC11, C+i*ldc+VLEN*11); 
         BCL_VLD(tVC12, C+i*ldc+VLEN*12); 
         BCL_VLD(tVC13, C+i*ldc+VLEN*13); 
         BCL_VLD(tVC14, C+i*ldc+VLEN*14); 
         BCL_VLD(tVC15, C+i*ldc+VLEN*15); 

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
         BCL_VST(C+i*ldc+VLEN*0, tVC0); 
         BCL_VST(C+i*ldc+VLEN*1, tVC1); 
         BCL_VST(C+i*ldc+VLEN*2, tVC2); 
         BCL_VST(C+i*ldc+VLEN*3, tVC3); 
         BCL_VST(C+i*ldc+VLEN*4, tVC4); 
         BCL_VST(C+i*ldc+VLEN*5, tVC5); 
         BCL_VST(C+i*ldc+VLEN*6, tVC6); 
         BCL_VST(C+i*ldc+VLEN*7, tVC7); 
         BCL_VST(C+i*ldc+VLEN*8, tVC8); 
         BCL_VST(C+i*ldc+VLEN*9, tVC9); 
         BCL_VST(C+i*ldc+VLEN*10, tVC10); 
         BCL_VST(C+i*ldc+VLEN*11, tVC11); 
         BCL_VST(C+i*ldc+VLEN*12, tVC12); 
         BCL_VST(C+i*ldc+VLEN*13, tVC13); 
         BCL_VST(C+i*ldc+VLEN*14, tVC14); 
         BCL_VST(C+i*ldc+VLEN*15, tVC15); 
      }
   }
}

#endif


/*
 * for testing, setup some parameter 
 */
   //#define BN 512    // 512 * 8 = 4KB, page size  
   #define BN 512 * 32    // 32 TLB entry for B  
   


/*
 *    Bn blocking...
 *       the idea is limit access of B until Bn for each iteration 
 */ 

void dcsrmm_IKJ_D128_BN_aXbX   
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
 * we consider only A->notrans case for now
 *    *transa == 'N' matdescra="GXXC" alpha=1.0 beta=1.0  
 */
   BCL_INT i, j; 
/*
 * for test
 */
   BCL_INT NN = (n+BN-1)/BN;  // ceil (N/BN)


   omp_set_num_threads(NTHREADS);



   #pragma omp parallel 
   {
      int id, nthreads; 
      int chunksize; 

      id = omp_get_thread_num(); 
      nthreads = omp_get_num_threads();
   
      chunksize = m / nthreads; 
/*
 *    How to improve static scheduling (load balancing): 
 *       We can count non-zero in each rows and divide number of rows so that
 *       each thread will have similar non-zeros to work with... 
 *       note: don't want to spli the row to different threads!!! 
 */

      for (i=id*chunksize; i < (id+1)*chunksize; i++)
      {
         BCL_INT ia0 = pntrb[i];
         BCL_INT ia1 = pntre[i]; 
/*
 *       BLOCK: B-blocking to limit TLB problem  
 */

         for (j=0; j < NN; j++)  /// how to handle .... block????
         {

/*
 *       register block C  
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

         for (BCL_INT kk=ia0; kk < ia1; kk++)
         {
            VTYPE VB0, VB1, VB2, VB3, VB4, VB5, VB6, VB7, 
               VB8, VB9, VB10, VB11, VB12, VB13, VB14, VB15; 
            VTYPE VA0;
            double a0 = val[kk];
            BCL_INT ja0 = indx[kk];
         
            BCL_vset1(VA0, a0);
            BCL_VLD(VB0, B+ja0*ldb+VLEN*0); 
            BCL_VLD(VB1, B+ja0*ldb+VLEN*1); 
            BCL_VLD(VB2, B+ja0*ldb+VLEN*2); 
            BCL_VLD(VB3, B+ja0*ldb+VLEN*3); 
            BCL_VLD(VB4, B+ja0*ldb+VLEN*4); 
            BCL_VLD(VB5, B+ja0*ldb+VLEN*5); 
            BCL_VLD(VB6, B+ja0*ldb+VLEN*6); 
            BCL_VLD(VB7, B+ja0*ldb+VLEN*7); 
            BCL_VLD(VB8, B+ja0*ldb+VLEN*8); 
            BCL_VLD(VB9, B+ja0*ldb+VLEN*9); 
            BCL_VLD(VB10, B+ja0*ldb+VLEN*10); 
            BCL_VLD(VB11, B+ja0*ldb+VLEN*11); 
            BCL_VLD(VB12, B+ja0*ldb+VLEN*12); 
            BCL_VLD(VB13, B+ja0*ldb+VLEN*13); 
            BCL_VLD(VB14, B+ja0*ldb+VLEN*14); 
            BCL_VLD(VB15, B+ja0*ldb+VLEN*15); 
       
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
 *       Assumption: C is aligned by VLENb, use VLDu otherwise  
 */
      // t = [C]
         BCL_VLD(tVC0, C+i*ldc+VLEN*0); 
         BCL_VLD(tVC1, C+i*ldc+VLEN*1); 
         BCL_VLD(tVC2, C+i*ldc+VLEN*2); 
         BCL_VLD(tVC3, C+i*ldc+VLEN*3); 
         BCL_VLD(tVC4, C+i*ldc+VLEN*4); 
         BCL_VLD(tVC5, C+i*ldc+VLEN*5); 
         BCL_VLD(tVC6, C+i*ldc+VLEN*6); 
         BCL_VLD(tVC7, C+i*ldc+VLEN*7); 
         BCL_VLD(tVC8, C+i*ldc+VLEN*8); 
         BCL_VLD(tVC9, C+i*ldc+VLEN*9); 
         BCL_VLD(tVC10, C+i*ldc+VLEN*10); 
         BCL_VLD(tVC11, C+i*ldc+VLEN*11); 
         BCL_VLD(tVC12, C+i*ldc+VLEN*12); 
         BCL_VLD(tVC13, C+i*ldc+VLEN*13); 
         BCL_VLD(tVC14, C+i*ldc+VLEN*14); 
         BCL_VLD(tVC15, C+i*ldc+VLEN*15); 

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
         BCL_VST(C+i*ldc+VLEN*0, tVC0); 
         BCL_VST(C+i*ldc+VLEN*1, tVC1); 
         BCL_VST(C+i*ldc+VLEN*2, tVC2); 
         BCL_VST(C+i*ldc+VLEN*3, tVC3); 
         BCL_VST(C+i*ldc+VLEN*4, tVC4); 
         BCL_VST(C+i*ldc+VLEN*5, tVC5); 
         BCL_VST(C+i*ldc+VLEN*6, tVC6); 
         BCL_VST(C+i*ldc+VLEN*7, tVC7); 
         BCL_VST(C+i*ldc+VLEN*8, tVC8); 
         BCL_VST(C+i*ldc+VLEN*9, tVC9); 
         BCL_VST(C+i*ldc+VLEN*10, tVC10); 
         BCL_VST(C+i*ldc+VLEN*11, tVC11); 
         BCL_VST(C+i*ldc+VLEN*12, tVC12); 
         BCL_VST(C+i*ldc+VLEN*13, tVC13); 
         BCL_VST(C+i*ldc+VLEN*14, tVC14); 
         BCL_VST(C+i*ldc+VLEN*15, tVC15); 
      }
   }
   }
}
#endif

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
#ifdef PTTIME
   // FIXME: Not implemented yet  
#endif
   for (BCL_INT kk=0; kk < k; kk++)
   {
      VTYPE VB0, VB1, VB2, VB3, VB4, VB5, VB6, VB7, 
            VB8, VB9, VB10, VB11, VB12, VB13, VB14, VB15; 
      BCL_INT ja0 = pntrb[kk];
      BCL_INT ja1 = pntre[kk]; 

         BCL_VLD(VB0, B+k*ldb+VLEN*0); 
         BCL_VLD(VB1, B+k*ldb+VLEN*1); 
         BCL_VLD(VB2, B+k*ldb+VLEN*2); 
         BCL_VLD(VB3, B+k*ldb+VLEN*3); 
         BCL_VLD(VB4, B+k*ldb+VLEN*4); 
         BCL_VLD(VB5, B+k*ldb+VLEN*5); 
         BCL_VLD(VB6, B+k*ldb+VLEN*6); 
         BCL_VLD(VB7, B+k*ldb+VLEN*7); 
         BCL_VLD(VB8, B+k*ldb+VLEN*8); 
         BCL_VLD(VB9, B+k*ldb+VLEN*9); 
         BCL_VLD(VB10, B+k*ldb+VLEN*10); 
         BCL_VLD(VB11, B+k*ldb+VLEN*11); 
         BCL_VLD(VB12, B+k*ldb+VLEN*12); 
         BCL_VLD(VB13, B+k*ldb+VLEN*13); 
         BCL_VLD(VB14, B+k*ldb+VLEN*14); 
         BCL_VLD(VB15, B+k*ldb+VLEN*15); 

      for (BCL_INT i=ja0; i < ja1; i++)
      {
         VTYPE VC0, VC1, VC2, VC3, VC4, VC5, VC6, VC7, 
               VC8, VC9, VC10, VC11, VC12, VC13, VC14, VC15; 
         VTYPE VA0; 
         
         double a0 = val[i];
         BCL_INT ia0 = indx[i];
         
         BCL_vset1(VA0, a0);
         
         BCL_VLD(VC0, C+ia0*ldc+VLEN*0); 
         BCL_VLD(VC1, C+ia0*ldc+VLEN*1); 
         BCL_VLD(VC2, C+ia0*ldc+VLEN*2); 
         BCL_VLD(VC3, C+ia0*ldc+VLEN*3); 
         BCL_VLD(VC4, C+ia0*ldc+VLEN*4); 
         BCL_VLD(VC5, C+ia0*ldc+VLEN*5); 
         BCL_VLD(VC6, C+ia0*ldc+VLEN*6); 
         BCL_VLD(VC7, C+ia0*ldc+VLEN*7); 
         BCL_VLD(VC8, C+ia0*ldc+VLEN*8); 
         BCL_VLD(VC9, C+ia0*ldc+VLEN*9); 
         BCL_VLD(VC10, C+ia0*ldc+VLEN*10); 
         BCL_VLD(VC11, C+ia0*ldc+VLEN*11); 
         BCL_VLD(VC12, C+ia0*ldc+VLEN*12); 
         BCL_VLD(VC13, C+ia0*ldc+VLEN*13); 
         BCL_VLD(VC14, C+ia0*ldc+VLEN*14); 
         BCL_VLD(VC15, C+ia0*ldc+VLEN*15); 
         
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
      
         BCL_VST(C+ia0*ldc+VLEN*0, VC0); 
         BCL_VST(C+ia0*ldc+VLEN*1, VC1); 
         BCL_VST(C+ia0*ldc+VLEN*2, VC2); 
         BCL_VST(C+ia0*ldc+VLEN*3, VC3); 
         BCL_VST(C+ia0*ldc+VLEN*4, VC4); 
         BCL_VST(C+ia0*ldc+VLEN*5, VC5); 
         BCL_VST(C+ia0*ldc+VLEN*6, VC6); 
         BCL_VST(C+ia0*ldc+VLEN*7, VC7); 
         BCL_VST(C+ia0*ldc+VLEN*8, VC8); 
         BCL_VST(C+ia0*ldc+VLEN*9, VC9); 
         BCL_VST(C+ia0*ldc+VLEN*10, VC10); 
         BCL_VST(C+ia0*ldc+VLEN*11, VC11); 
         BCL_VST(C+ia0*ldc+VLEN*12, VC12); 
         BCL_VST(C+ia0*ldc+VLEN*13, VC13); 
         BCL_VST(C+ia0*ldc+VLEN*14, VC14); 
         BCL_VST(C+ia0*ldc+VLEN*15, VC15); 
      }
   }
}

#endif 
