Sparse Dense Matrix Multiplication for special shape

C = Alpha * op(A) * B + beta * C

Where A = sparse matrix (MxN), op = transpose/no-transpose/hermitian,
B = Dense matrix (NxD), C = dense matrix (MxD) 

D and M are small. 
Some supporting files (such as, converting CSC, CSR, etc) have been sourced from previous projects

