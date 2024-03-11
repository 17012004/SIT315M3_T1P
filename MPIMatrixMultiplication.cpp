#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 1000 // Matrix size

void matrix_multiply(int *A, int *B, int *C, int rows, int cols, int common_dim) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            C[i * cols + j] = 0;
            for (int k = 0; k < common_dim; k++) {
                C[i * cols + j] += A[i * common_dim + k] * B[k * cols + j];
            }
        }
    }
}

int main(int argc, char **argv) {
    int rank, size;
    int *A, *B, *C;
    int *subA, *subC;
    int rows_per_process, rows, cols;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        rows = N;
        cols = N;
        // Allocate memory for A, B, and C
        A = (int *)malloc(rows * cols * sizeof(int));
        B = (int *)malloc(cols * rows * sizeof(int));
        C = (int *)malloc(rows * cols * sizeof(int));

        // Initialize matrices A and B
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                A[i * cols + j] = i * cols + j;
                B[i * cols + j] = i * cols + j;
            }
        }
    }

    // Broadcast matrix dimensions
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    rows_per_process = rows / size;
    subA = (int *)malloc(rows_per_process * cols * sizeof(int));
    subC = (int *)malloc(rows_per_process * cols * sizeof(int));

    // Scatter matrix A
    MPI_Scatter(A, rows_per_process * cols, MPI_INT, subA, rows_per_process * cols, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast matrix B
    MPI_Bcast(B, rows * cols, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication
    matrix_multiply(subA, B, subC, rows_per_process, cols, cols);

    // Gather the results
    MPI_Gather(subC, rows_per_process * cols, MPI_INT, C, rows_per_process * cols, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Output or use matrix C
    }

    free(subA);
    free(subC);
    if (rank == 0) {
        free(A);
        free(B);
        free(C);
    }

    MPI_Finalize();

    return 0;
}
