import numpy as np

# Generate a random 3x3 integer matrix with values 1-9
A = np.random.randint(1, 10, size=(3, 3)).astype(float)

print("Original random matrix A:")
print(A)
print("Product of diagonal:", np.prod(np.diag(A)))
print("Determinant:", np.linalg.det(A))
print('Product of eigenvalues:', np.prod(np.linalg.eigvals(A)))
print("\n--- Starting row reduction ---\n")

# Step 1: Make first column below pivot 0
A[1] = A[1] - (A[1,0]/A[0,0]) * A[0]
print("After R2 -> R2 - (R2[0]/R1[0])*R1")
print(A)
print("Product of diagonal:", np.prod(np.diag(A)))
print("Determinant:", np.linalg.det(A))
print('Product of eigenvalues:', np.prod(np.linalg.eigvals(A)))
print()

A[2] = A[2] - (A[2,0]/A[0,0]) * A[0]
print("After R3 -> R3 - (R3[0]/R1[0])*R1")
print(A)
print("Product of diagonal:", np.prod(np.diag(A)))
print("Determinant:", np.linalg.det(A))
print('Product of eigenvalues:', np.prod(np.linalg.eigvals(A)))
print()

# Step 2: Make second column below pivot 0
A[2] = A[2] - (A[2,1]/A[1,1]) * A[1]
print("After R3 -> R3 - (R3[1]/R2[1])*R2")
print(A)
print("Product of diagonal:", np.prod(np.diag(A)))
print("Determinant:", np.linalg.det(A))
print('Product of eigenvalues:', np.prod(np.linalg.eigvals(A)))
print()

# Final upper triangular matrix
print("Final upper triangular matrix:")
print(A)
print("Product of diagonal:", np.prod(np.diag(A)))
print("Determinant:", np.linalg.det(A))
print('Product of eigenvalues:', np.prod(np.linalg.eigvals(A)))