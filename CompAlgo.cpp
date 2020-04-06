#include <stdio.h>
#include <random>
#include <cassert>

#include "CSC.h"
#include "CSR.h"
#include "commonutility.h"
#include "utility.h"

#define INDEXTYPE int
//#define INDEXTYPE long long


#define VALUETYPE double 
#define DREAL 1 

/*
 * NOTE: must defined type (e.g., DREAL) before including kernels.h 
 */
#include "kernels.h" 


#if 1  // should already be defined in simd.h
   #ifndef VLEN
      #ifdef DREAL 
         #define VLEN 4
      #else // SREAL 
         #define VLEN 8
      #endif
   #endif
#endif

/*==============================================================================*
 *                   API FOR ALL CSR & CSC BASED KERNELS
 *                   ------------------------------------
 * NOTE:
 * 1. It is actually based on MKL's old API  
 * 2. only supports zero based index so far, will extend it later
 * 3. m,n,k can be different than actually rows, cols of matrix 
 *============================================================================*/
typedef void (*csr_mm_t) 
(
   const char transa,     // 'N', 'T' , 'C' 
   const INDEXTYPE m,     // number of rows of A in computation  
   const INDEXTYPE n,     // number of cols of C
   const INDEXTYPE k,     // number of cols of A in computation 
   const VALUETYPE alpha, // alpha value  
   const char *matdescra,  // 6 characr array descriptor for A:
                           // [G/S/H/T/A/D],[L/U],[N/U],[F/C] -> [G,X,X,C] 
   const INDEXTYPE nnz,   // nonzeros: need to recreate csr with mkl 
   const INDEXTYPE rows,  // number of rows
   const INDEXTYPE cols,  // number of columns 
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
   const VALUETYPE *b,     // Dense B matrix
   const INDEXTYPE ldb,   // 2nd dimension of b (col size since row-major)  
   const VALUETYPE beta,  // beta value 
   VALUETYPE *c,           // Dense matrix c
   const INDEXTYPE ldc    // 2nd dimension size of c (col size since roa-major) 
);

typedef void (*csc_mm_t) 
(
   const char transa,     // 'N', 'T' , 'C' 
   const INDEXTYPE m,     // number of rows of A in computation 
   const INDEXTYPE n,     // number of cols of C
   const INDEXTYPE k,     // number of cols of A in computation 
   const VALUETYPE alpha, // alpha 
   const char *matdescra,  // 6 characr array descriptor for A:
                           // [G/S/H/T/A/D],[L/U],[N/U],[F/C] -> [G,X,X,C] 
   const INDEXTYPE nnz,   // nonzeros: need to recreate csr with mkl 
   const INDEXTYPE rows,  // number of rows
   const INDEXTYPE cols,  // number of columns 
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // rowids -> row indices 
   const INDEXTYPE *pntrb, // starting index for colptr
   const INDEXTYPE *pntre, // ending index for colptr
   const VALUETYPE *b,     // Dense B matrix
   const INDEXTYPE ldb,   // 2nd dimension of b for zero-based indexing  
   const VALUETYPE beta,  // beta  
   VALUETYPE *c,           // Dense matrix c
   const INDEXTYPE ldc    // 2nd dimension size of b 
);

/*=============================================================================*
 *                      MKL's API 
 *                      ----------
 * NOTE: I'm not gonna change my API... using a wrapper function to call 
 * MKL routine for trusted csr_mm 
 * 
 * new API: 
 * -------- 
 
 sparse_status_t mkl_sparse_d_mm 
 (
   const sparse_operation_t operation, 
                           -> SPARSE_OPERATION_NON_TRANSPOSE
                           -> SPARSE_OPERATION_TRANSPOSE
                           -> SPARSE_OPERATION_CONJUGATE_TRANSPOSE
   const double alpha, 
   const sparse_matrix_t A,
                           -> need to create first using mkl_sparse_d_create_csr 
   struct matrix_descr descr
   {
         sparse_matrix_type_t type;
         sparse_fill_mode_t mode;
         sparse_dia_type_t diag;
   }

   sparse_matrix_type_t
                           -> SPARSE_MATRIX_TYPE_GENERAL
                           -> SPARSE_MATRIX_TYPE_SYMMETRIC
                           -> SPARSE_MATRIX_TYPE_HERMITIAN
                           -> SPARSE_MATRIX_TYPE_TRIANGULAR
                           -> SPARSE_MATRIX_TYPE_DIAGONAL
                           -> SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR
                           -> SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL 
   sparse_fill_mode_t
                           -> SPARSE_FILL_MODE_LOWER
                           -> SPARSE_FILL_MODE_UPPER
   sparse_dia_type_t
                           -> SPARSE_DIAG_NON_UNIT
                           -> SPARSE_DIAG_UNIT



   const sparse_layout_t layout, // for dense matrices
                           -> SPARSE_LAYOUT_COLUMN_MAJOR 
                           -> SPARSE_LAYOUT_ROW_MAJOR  
   const double *B, 
   const MKL_INT columns, 
   const MKL_INT ldb, 
   const double beta, 
   double *C, 
   const MKL_INT ldc
   );
 *
 * Return statue:  sparse_status_t
 *    -> SPARSE_STATUS_SUCCESS
 *    -> SPARSE_STATUS_NOT_INITIALIZED
 *    -> SPARSE_STATUS_ALLOC_FAILED
 *    -> SPARSE_STATUS_EXECUTION_FAILED
 *    -> SPARSE_STATUS_INTERNAL_ERROR
 *    -> SPARSE_STATUS_NOT_SUPPORTED
 * 
 * indexing 
 *    -> SPARSE_INDEX_BASE_ZERO
 *    -> SPARSE_INDEX_BASE_ONE
 *
 * Need to apply mkl_sparse_d_create_csr routine to create A frist 
 * 
 *
 *=============================================================================*/

#define TIME_MKL 1 

#ifdef TIME_MKL

#include "mkl_spblas.h"
#include "mkl_types.h"

void MKL_csr_mm
(
   const char transa,     // 'N', 'T' , 'C' 
   const INDEXTYPE m,     // number of rows of A needed to compute 
   const INDEXTYPE n,     // number of cols of C
   const INDEXTYPE k,     // number of cols of A
   const VALUETYPE alpha, // double scalar ?? why ptr 
   const char *matdescra, // 6 characr array descriptor for A:
                          // [G/S/H/T/A/D],[L/U],[N/U],[F/C] -> [G,X,X,C] 
   const INDEXTYPE nnz,   // nonzeros: need to recreate csr with mkl 
   const INDEXTYPE rows,  // number of rows
   const INDEXTYPE cols,  // number of columns 
   const VALUETYPE *val,  // NNZ value  
   const INDEXTYPE *indx, // colids -> column indices 
   const INDEXTYPE *pntrb,// starting index for rowptr
   const INDEXTYPE *pntre,// ending index for rowptr
   const VALUETYPE *b,    // Dense B matrix
   const INDEXTYPE ldb,   // 2nd dimension of b for zero-based indexing  
   const VALUETYPE beta,  // double scalar beta[0] 
   VALUETYPE *c,          // Dense matrix c
   const INDEXTYPE ldc    // 2nd dimension size of b 
)
{
   INDEXTYPE i; 
   sparse_status_t stat; 
   sparse_matrix_t A = NULL; 
   struct matrix_descr Adsc; 

   // 1. inspection stage
/*
   sparse_status_t mkl_sparse_d_create_csr 
   (
      sparse_matrix_t *A, 
      const sparse_index_base_t indexing, 
      const MKL_INT rows, 
      const MKL_INT cols, 
      MKL_INT *rows_start,  // not const !!
      MKL_INT *rows_end,    // not const !!
      MKL_INT *col_indx,    // not const !!
      double *values        // not const
   );
   
   NOTE: NOTE: 
   -----------
   create_csr will overwrote rows_start, rows_end, col_indx and values
   So, we need to copy those here  
*/
   // copying CSR data, we will skip this copy in timing  
   MKL_INT M; 
   MKL_INT *rowptr;
   MKL_INT *col_indx;
   VALUETYPE *values; 

#if 1
   fprintf(stdout, "rows=%ld, cols=%d, nnz=%ld\nm=%ld, n=%d, k=%ld\n",
           rows, cols, nnz, m, n, k);
#endif

   // want to keep only one array for rowptr 
   //rowptr = (MKL_INT*) malloc((rows+1)*sizeof(MKL_INT));
   
   //M = rows; 
   M = m; 
   
   rowptr = (MKL_INT*) malloc((M+1)*sizeof(MKL_INT)); // just allocate upto M
   assert(rowptr);
   for (i=0; i < M; i++)
      rowptr[i] = pntrb[i];
   rowptr[i] = pntre[i-1];
   
   col_indx = (MKL_INT*) malloc(nnz*sizeof(MKL_INT));
   assert(col_indx);
   for (i=0; i < nnz; i++)
      col_indx[i] = indx[i]; 
   
   values = (VALUETYPE*) malloc(nnz*sizeof(VALUETYPE));
   assert(col_indx);
   for (i=0; i < nnz; i++)
      values[i] = val[i]; 

   cout << "--- Running inspector for MKL" << endl;

#ifdef DREAL 
   stat = mkl_sparse_d_create_csr (&A, SPARSE_INDEX_BASE_ZERO, M, cols, 
            rowptr, rowptr+1, col_indx, values);  
#else
   stat = mkl_sparse_s_create_csr (&A, SPARSE_INDEX_BASE_ZERO, M, cols, 
            rowptr, rowptr+1, col_indx, values);  
#endif

   if (stat != SPARSE_STATUS_SUCCESS)
   {
      cout << "creating csr for MKL failed!";
      exit(1);
   }
   // 2. execution stage 
/*
   sparse_status_t mkl_sparse_d_mm 
   (
      const sparse_operation_t operation, 
      const double alpha, 
      const sparse_matrix_t A, 
      const struct matrix_descr descr, 
      const sparse_layout_t layout, 
      const double *B, 
      const MKL_INT columns, 
      const MKL_INT ldb, 
      const double beta, 
      double *C, 
      const MKL_INT ldc);
*/ 
   Adsc.type = SPARSE_MATRIX_TYPE_GENERAL;
   //Adsc.fill // no need for general matrix  
   Adsc.diag =  SPARSE_DIAG_NON_UNIT;  // no need for general 
   
   cout << "--- Running executor for MKL" << endl;
#ifdef DREAL 
   stat = mkl_sparse_d_mm (SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, Adsc, 
                           SPARSE_LAYOUT_ROW_MAJOR, b, n, ldb, beta, c, ldc); 
#else
   stat = mkl_sparse_s_mm (SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, Adsc, 
                           SPARSE_LAYOUT_ROW_MAJOR, b, n, ldb, beta, c, ldc); 
#endif

   if (stat != SPARSE_STATUS_SUCCESS)
   {
      cout << "creating csr for MKL failed!";
      exit(1);
   }
   cout << "--- Done calling MKL's API" << endl;
/*
 * free all data 
 */
   free(rowptr);
   free(col_indx);
   free(values);
   mkl_sparse_destroy(A);
}

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
   printf("-M <number>, rows of M (can be less than actual rows of A).\n");
   printf("-D <number>, number of cols of B & C [range = 1 to 256] \n");
   printf("-C <number>, Cachesize in KB to flush it for small workset \n");
   printf("-nrep <number>, number of repeatation \n");
   printf("-T <0,1>, 1 means, run tester as well  \n");
   printf("-skHd<1>, 1 means, skip header of the printed results  \n");
   printf("-h, show this usage message  \n");

}

void GetFlags(int narg, char **argv, string &inputfile, int &option, 
      INDEXTYPE &D, INDEXTYPE &M, int &csKB, int &nrep, int &isTest, int &skHd)
{
   option = 1; 
   inputfile = "";
   //D = 256; 
   D = 128; 
   M = 0;
   isTest = 0; 
   nrep = 1;
   skHd = 0; // by default print header
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
      else if(strcmp(argv[p], "-T") == 0)
      {
	 isTest = atoi(argv[p+1]);
      }
      else if(strcmp(argv[p], "-skHd") == 0)
      {
	 skHd = atoi(argv[p+1]);
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
int doChecking(IT NNZA, IT M, IT N, NT *C, NT *D, IT ldc)
{
   IT i, j, k;
   NT diff, EPS; 
   double ErrBound; 

   int nerr = 0;
/*
 * Error bound : computation M*NNZ FMAC
 *
 */
   EPS = Epsilon<NT>();
   // the idea is how many flop one element needs, should be max degree
   // avg degree is not perfect  
   // ErrBound = 2 * (NNZA/N) * EPS; 
   ErrBound = 2 * (NNZA) * EPS; 
   cout << "--- EPS = " << EPS << " ErrBound = " << ErrBound << endl; 
   //cout << "--- ErrBound = " << ErrBound << " NNZ(A) = " << NNZA << " N = " << N  <<endl; 
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
      #if 1
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

template <csr_mm_t trusted, csr_mm_t test>
int doTesting_Acsr(CSR<INDEXTYPE, VALUETYPE> &A, INDEXTYPE M, INDEXTYPE N, 
      INDEXTYPE K)
{
   int nerr; 
   size_t i, szB, szC, ldc, ldb; 
   VALUETYPE *pb, *b, *pc0, *c0, *pc, *c;
   INDEXTYPE nnz=A.nnz;
   VALUETYPE *values;  // want to generate random value  

   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(0.0,1.0);

/*
 * NOTE: we are considering only row major B and C storage now
 */
   ldb = ldc = N; // both row major 

   VALUETYPE beta = 1.0;
   VALUETYPE alpha = 1.0;
      
   szB = ((K*ldb+VLEN-1)/VLEN)*VLEN;  // szB in element
   
   //szC = ((M*ldc+VLEN-1)/VLEN)*VLEN;  // szC in element 
   // changes to test mkl 
   szC = ((A.rows*ldc+VLEN-1)/VLEN)*VLEN;  // szC in element 
      
   pb = (VALUETYPE*)malloc(szB*sizeof(VALUETYPE)+2*ATL_Cachelen);
   assert(pb);
   b = (VALUETYPE*) ATL_AlignPtr(pb);

   pc0 = (VALUETYPE*)malloc(szC*sizeof(VALUETYPE)+2*ATL_Cachelen);
   assert(pc0);
   c0 = (VALUETYPE*) ATL_AlignPtr(pc0); 
      
   pc = (VALUETYPE*)malloc(szC*sizeof(VALUETYPE)+2*ATL_Cachelen);
   assert(pc);
   c = (VALUETYPE*) ATL_AlignPtr(pc); 
   
   // init   
   for (i=0; i < szB; i++)
   {
   #if 1
      b[i] = distribution(generator);  
   #else
      //b[i] = 1.0*i;  
      b[i] = 0.5;  
   #endif
   }
   for (i=0; i < szC; i++)
   {
   #if 1
      c[i] = c0[i] = distribution(generator);  
   #else
      c[i] = 0.0; c0[i] = 0.0;
   #endif
   }
/*
 * csr may consists all 1 as values... init with random values
 */
   values = (VALUETYPE*)malloc(A.nnz*sizeof(VALUETYPE));
   assert(values);
   for (i=0; i < A.nnz; i++)
      values[i] = distribution(generator);  

   
   fprintf(stderr, "Applying trusted kernel\n");
   trusted('N', M, N, K, alpha, "GXXC", A.nnz, A.rows, A.cols, values, 
           A.colids, A.rowptr, (A.rowptr)+1, b, ldb, beta, c0, ldc);   
   
   fprintf(stderr, "Applying test kernel\n");
   test('N', M, N, K, alpha, "GXXC", A.nnz, A.rows, A.cols, values, A.colids, 
            A.rowptr, (A.rowptr)+1, b, ldb, beta, c, ldc);  

   nerr = doChecking<INDEXTYPE, VALUETYPE>(nnz, M, N, c0, c, ldc);

   free(pc0);
   free(pc);
   free(pb);

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
template<csr_mm_t CSR_KERNEL>
double doTiming_Acsr_CacheFlushing
(
 CSR<INDEXTYPE, VALUETYPE> &A, 
 INDEXTYPE M, 
 INDEXTYPE N, 
 INDEXTYPE K, 
 int csKB, 
 int nrep     /* if nrep == 0, nrep = number of wset fit in cache */
 )
{
   int i, j;
   size_t sz, szB, szC, ldb, ldc;
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

   ldb = ldc = N; // considering both row-major 
   szB = ((K*ldb+VLEN-1)/VLEN)*VLEN;  // szB in element
   szC = ((M*ldc+VLEN-1)/VLEN)*VLEN;  // szC in element 
   
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

   //fprintf(stderr, "nrep = %d, nset = %d\n", nrep, nset);
   
   start = omp_get_wtime();
   for (i=0, j=nset; i < nrep; i++)
   {
         //a1b1 kernel
         CSR_KERNEL('N', M, N, K, 1.0, "GXXC", A.nnz, A.rows, A.cols, A.values, 
                    A.colids, A.rowptr, A.rowptr+1, b, ldb, 1.0, c, ldc);   
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
template<csr_mm_t CSR_KERNEL>
vector <double> doTiming_Acsr
(
 const CSR<INDEXTYPE, VALUETYPE> &A, 
 const INDEXTYPE M, 
 const INDEXTYPE N, 
 const INDEXTYPE K, 
 const int csKB, 
 const int nrep     /* if nrep == 0, nrep = number of wset fit in cache */
 )
{
   int i, j;
   vector <double> results; 
   double start, end;
   size_t szB, szC, ldb, ldc; 
   VALUETYPE *pb, *b, *pc, *c;
   INDEXTYPE *rowptr, *col_indx;
   VALUETYPE *values;

   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(0.0,1.0);

   ldb = ldc = N; // considering both row-major   

   szB = ((K*ldb+VLEN-1)/VLEN)*VLEN;  // szB in element
   szC = ((M*ldc+VLEN-1)/VLEN)*VLEN;  // szC in element 

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
 * Copying the data to make it same as MKL timer... 
 * We may use it later if we introduce an inspector phase 
 */
   rowptr = (INDEXTYPE*) malloc((M+1)*sizeof(INDEXTYPE));
   assert(rowptr);
   for (i=0; i < M+1; i++)
      rowptr[i] = A.rowptr[i];
   
   col_indx = (INDEXTYPE*) malloc(A.nnz*sizeof(INDEXTYPE));
   assert(col_indx);
   for (i=0; i < A.nnz; i++)
      col_indx[i] = A.colids[i]; 
   
   values = (VALUETYPE*) malloc(A.nnz*sizeof(VALUETYPE));
   assert(col_indx);
   for (i=0; i < A.nnz; i++)
      values[i] = A.values[i]; 
/*
 * NOTE: with small working set, we should not skip the first iteration 
 * (warm cache), because we want to time out of cache... 
 * We run this timer either for in-cache data or large working set
 * So we can safely skip 1st iteration... C will be in cache then
 */

   //CSR_KERNEL(M, D, N, A_csr, b, D, c, D);  // skip it's timing  
   // no setup time for our kernel so far
   results.push_back(0.0);
   //a1b1 kernel
   CSR_KERNEL('N', M, N, K, 1.0, "GXXC", A.nnz, A.rows, A.cols, values, 
              col_indx, rowptr, rowptr+1, b, ldb, 1.0, c, ldc);   
   
   start = omp_get_wtime();
   for (i=0; i < nrep; i++)
   {
      //a1b1 kernel
      CSR_KERNEL('N', M, N, K, 1.0, "GXXC", A.nnz, A.rows, A.cols, A.values, 
                 A.colids, A.rowptr, A.rowptr+1, b, ldb, 1.0, c, ldc);   
   }
   end = omp_get_wtime();
   
   results.push_back((end-start)/((double)nrep));
   
   free(pb);
   free(pc);
   
   return(results);
}
/*
 * Special timer for MKL to customize MKL's inspector-executor model
 */
vector<double> doTimingMKL_Acsr
(
 const CSR<INDEXTYPE, VALUETYPE> &A, 
 INDEXTYPE M, 
 INDEXTYPE N, 
 INDEXTYPE K, 
 int csKB, 
 int nrep     /* if nrep == 0, nrep = number of wset fit in cache */
 )
{
   int i, j;
   double start, end;
   vector <double> results;  // don't use single precision, use double  
   size_t szB, szC, ldb, ldc; 
   VALUETYPE *pb, *b, *pc, *c;
   // MKL related  
   sparse_status_t stat; 
   sparse_matrix_t Amkl = NULL; 
   struct matrix_descr Adsc; 
   MKL_INT *rowptr;
   MKL_INT *col_indx;
   VALUETYPE *values;
   const VALUETYPE alpha = 1.0, beta = 1.0; 

   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(0.0,1.0);

   // initialize B and C 
   ldb = ldc = N; // considering both row-major   

   szB = ((K*ldb+VLEN-1)/VLEN)*VLEN;  // szB in element
   szC = ((M*ldc+VLEN-1)/VLEN)*VLEN;  // szC in element 

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

/*
 *    Setup MKL's data structure 
 *    NOTE: there is no way to specify M in executor. So, we need to create csr
 *    with smaller M 
 */
   rowptr = (MKL_INT*) malloc((M+1)*sizeof(MKL_INT));
   assert(rowptr);
   for (i=0; i < M+1; i++)
      rowptr[i] = A.rowptr[i];
   
   col_indx = (MKL_INT*) malloc(A.nnz*sizeof(MKL_INT));
   assert(col_indx);
   for (i=0; i < A.nnz; i++)
      col_indx[i] = A.colids[i]; 
   
   values = (VALUETYPE*) malloc(A.nnz*sizeof(VALUETYPE));
   assert(col_indx);
   for (i=0; i < A.nnz; i++)
      values[i] = A.values[i]; 

   // timing inspector phase 
   {
      start = omp_get_wtime();
   #ifdef DREAL 
      stat = mkl_sparse_d_create_csr (&Amkl, SPARSE_INDEX_BASE_ZERO, M, A.cols, 
               rowptr, rowptr+1, col_indx, values);  
   #else
      stat = mkl_sparse_s_create_csr (&Amkl, SPARSE_INDEX_BASE_ZERO, M, A.cols, 
               rowptr, rowptr+1, col_indx, values);  
   #endif
#if 0
      if (stat != SPARSE_STATUS_SUCCESS)
      {
         cout << "creating csr for MKL failed!";
         exit(1);
      }
#endif
      end = omp_get_wtime();
      results.push_back(end-start); // setup time 
   }
   
   Adsc.type = SPARSE_MATRIX_TYPE_GENERAL;
   //Adsc.fill // no need for general matrix  
   //Adsc.diag =  SPARSE_DIAG_NON_UNIT;  // no need for general 
  
   // skiping first call 
   #ifdef DREAL 
      stat = mkl_sparse_d_mm (SPARSE_OPERATION_NON_TRANSPOSE, alpha, Amkl, Adsc, 
                              SPARSE_LAYOUT_ROW_MAJOR, b, N, ldb, beta, c, ldc); 
   #else
      stat = mkl_sparse_s_mm (SPARSE_OPERATION_NON_TRANSPOSE, alpha, Amkl, Adsc, 
                              SPARSE_LAYOUT_ROW_MAJOR, b, N, ldb, beta, c, ldc); 
   #endif
#if 0
   if (stat != SPARSE_STATUS_SUCCESS)
   {
      cout << "creating csr for MKL failed!";
      exit(1);
   }
#endif
   
   start = omp_get_wtime();
   for (i=0; i < nrep; i++)
   {
   #ifdef DREAL 
      stat = mkl_sparse_d_mm (SPARSE_OPERATION_NON_TRANSPOSE, alpha, Amkl, Adsc, 
                              SPARSE_LAYOUT_ROW_MAJOR, b, N, ldb, beta, c, ldc); 
   #else
      stat = mkl_sparse_s_mm (SPARSE_OPERATION_NON_TRANSPOSE, alpha, Amkl, Adsc, 
                              SPARSE_LAYOUT_ROW_MAJOR, b, N, ldb, beta, c, ldc); 
   #endif
   }
   end = omp_get_wtime();
   results.push_back((end-start)/((double)nrep)); // execution time 
   
   free(rowptr);
   free(col_indx);
   free(values);
   mkl_sparse_destroy(Amkl);
   free(pb);
   free(pc);
   
   return(results);
}



void GetSpeedup(string inputfile, int option, INDEXTYPE D, INDEXTYPE M, 
      int csKB, int nrep, int isTest, int skipHeader)
{
   int nerr;
   vector<double> res0, res1; 
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
   
/*
 * test the result if mandated 
 * NOTE: general notation: A->MxK B->KxN, C->MxN
 *       SDMM with D     : A->MxN, B->NxD, C->MxD 
 *                            M->M, K->N, N->D 
 *       To avoid confusion, I'm using MNK general notation routines except 
 *       this one 
 */
   assert(N && M && D);
   if (isTest)
   {
      // testing with same kernel to test the tester itself: sanity check  
      //nerr = doTesting_Acsr<dcsrmm_IKJ_a1b1, dcsrmm_IKJ_a1b1>
      //                      (A_csr0, M, D, N); 
      
      //nerr = doTesting_Acsr<dcsrmm_IKJ_a1b1,dcsrmm_IKJ_D128_a1b1>
      //                      (A_csr0, M, D, N); 
      
      nerr = doTesting_Acsr<dcsrmm_IKJ_D128_a1b1, MKL_csr_mm>
                            (A_csr0, M, D, N); 
      if (!nerr)
         fprintf(stdout, "PASSED TEST\n");
      else
      {
         fprintf(stdout, "FAILED TEST, %d ELEMENTS\n", nerr);
         exit(1); // test failed, not timed 
      }
   }

#if 0
   t1 = doTiming_Acsr_CacheFlushing<SDMM_CSR_IKJ_D128>(A_csr1, M, N, D, csKB, 
         nrep);
   fprintf(stdout, "test time = %e\n", t1); 

   t0 = doTiming_Acsr_CacheFlushing<Trusted_SDMM_CSR_IKJ>(A_csr0, M, N, D, csKB,
         nrep);
   fprintf(stdout, "Trusted time = %e\n", t0); 
#else
   // base kernel
/*
 * NOTE: general notation: A->MxK B->KxN, C->MxN
 *       SDMM with D     : A->MxN, B->NxD, C->MxD 
 */
   //t0 = doTiming_Acsr<dcsrmm_IKJ_a1b1>(A_csr0, M, D, N, csKB, nrep);
  
   //t0 = doTimingMKL_Acsr(A_csr0, M, D, N, csKB, nrep, 1);
   res0 = doTimingMKL_Acsr(A_csr0, M, D, N, csKB, nrep);
   //fprintf(stdout, "trusted time = %e\n", t0); 
  
   // optimized kernel 
   res1 = doTiming_Acsr<dcsrmm_IKJ_D128_a1b1>(A_csr1, M, D, N, csKB, nrep);
   //t1 = doTiming_Acsr<dcsrmm_IKJ_a1b1>(A_csr1, M, D, N, csKB, nrep);
   //fprintf(stdout, "test time = %e\n", t1); 
#endif
   
   //cout << "skipHeader: " << skipHeader << endl;

   if(!skipHeader) 
   {
      cout << "Filename,"
         << "NNZ,"
         << "M,"
         << "N,"
         << "D,"
         << "Trusted_inspect_time,"
         << "Trusted_exe_time,"
         << "Test_inspect_time,"
         << "Test_exe_time,"
         << "Speedup_exe_time,"
         << "Speedup_total,"
         << "Critical_point" << endl;
   }
   
   double critical_point = (res0[0]/(res1[1]-res0[1])) < 0.0 ?  -1.0 
                                             : (res0[0]/(res1[1]-res0[1])); 
   cout << inputfile << "," 
        << A_csr0.nnz << "," 
        << M << "," 
        << N << "," 
        << D << "," << std::scientific
        << res0[0] << "," 
        << res0[1] << "," 
        << res1[0] << "," 
        << res1[1] << "," 
        << std::fixed << std::showpoint
        << res0[1]/res1[1] << ","
        << ((res0[0]+res0[1])/(res1[0]+res1[1])) << ","  
        << critical_point
        << endl;
}

int main(int narg, char **argv)
{
   INDEXTYPE D, M; 
   int option, csKB, nrep, isTest, skHd;
   string inputfile; 
   GetFlags(narg, argv, inputfile, option, D, M, csKB, nrep, isTest, skHd);
   GetSpeedup(inputfile, option, D, M, csKB, nrep, isTest, skHd);
   return 0;
}

