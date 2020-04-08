#ifndef KERNEL_H
#define KERNEL_H

#ifdef __cplusplus 
   extern "C"
   {
#endif

/* 
 * INT type can be changed here, implemenetation not dependent on int  
 */
#define BCL_INT int 
#define BCL_VLEN 8   // depends on arch, need to guard it with macro  

// double function prototypes  
void dcsrmm_IKJ (const char transa, const BCL_INT m, const BCL_INT n, 
      const BCL_INT k,const double alpha, const char *matdescra, 
      const BCL_INT nnz, const BCL_INT rows, const BCL_INT cols, 
      const double *val, const BCL_INT *indx, const BCL_INT *pntrb, 
      const BCL_INT *pntre, const double *B, const BCL_INT ldb, 
      const double beta, double *C, const BCL_INT ldc);


void dcsrmm_IKJ_D128 (const char transa, const BCL_INT m, const BCL_INT n, 
      const BCL_INT k,const double alpha, const char *matdescra, 
      const BCL_INT nnz, const BCL_INT rows, const BCL_INT cols, 
      const double *val, const BCL_INT *indx, const BCL_INT *pntrb, 
      const BCL_INT *pntre, const double *B, const BCL_INT ldb, 
      const double beta, double *C, const BCL_INT ldc);


void dcsrmm_KIJ (const char transa, const BCL_INT m, const BCL_INT n, 
      const BCL_INT k,const double alpha, const char *matdescra, 
      const BCL_INT nnz, const BCL_INT rows, const BCL_INT cols, 
      const double *val, const BCL_INT *indx, const BCL_INT *pntrb, 
      const BCL_INT *pntre, const double *B, const BCL_INT ldb, 
      const double beta, double *C, const BCL_INT ldc);


void dcsrmm_KIJ_D128 (const char transa, const BCL_INT m, const BCL_INT n, 
      const BCL_INT k,const double alpha, const char *matdescra, 
      const BCL_INT nnz, const BCL_INT rows, const BCL_INT cols, 
      const double *val, const BCL_INT *indx, const BCL_INT *pntrb, 
      const BCL_INT *pntre, const double *B, const BCL_INT ldb, 
      const double beta, double *C, const BCL_INT ldc);

#ifdef __cplusplus 
   }  // extern "C"
#endif

#endif
