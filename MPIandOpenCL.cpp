#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <CL/cl.h>

#define N 1000 // Matrix size

// Define the OpenCL kernel source code
const char *kernelSource = "__kernel void matrix_multiply(__global int* A, __global int* B, __global int* C, int rows, int cols, int common_dim) { \
                                int i = get_global_id(0); \
                                int j = get_global_id(1); \
                                int sum = 0; \
                                for (int k = 0; k < common_dim; k++) { \
                                    sum += A[i * common_dim + k] * B[k * cols + j]; \
                                } \
                                C[i * cols + j] = sum; \
                            }";

void matrix_multiply(int *A, int *B, int *C, int rows, int cols, int common_dim, cl_context context, cl_command_queue command_queue, cl_program program, cl_kernel kernel) {
    cl_mem bufferA, bufferB, bufferC;
    cl_int err;

    // Create buffer objects for matrices A, B, and C
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * rows * common_dim, A, &err);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * common_dim * cols, B, &err);
    bufferC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * rows * cols, NULL, &err);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &rows);
    clSetKernelArg(kernel, 4, sizeof(int), &cols);
    clSetKernelArg(kernel, 5, sizeof(int), &common_dim);

    // Execute the OpenCL kernel
    size_t globalWorkSize[2] = { rows, cols };
    err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);

    // Read the result back from the device
    err = clEnqueueReadBuffer(command_queue, bufferC, CL_TRUE, 0, sizeof(int) * rows * cols, C, 0, NULL, NULL);

    // Clean up
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
}

int main(int argc, char **argv) {
    int rank, size;
    int *A, *B, *C;
    int *subA, *subC;
    int rows_per_process, rows, cols, common_dim;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Initialize matrices A and B
        rows = N;
        cols = N;
        common_dim = N;
        A = (int *)malloc(rows * common_dim * sizeof(int));
        B = (int *)malloc(common_dim * cols * sizeof(int));
        C = (int *)malloc(rows * cols * sizeof(int));
        for (int i = 0; i < rows * common_dim; i++) {
            A[i] = i;
        }
        for (int i = 0; i < common_dim * cols; i++) {
            B[i] = i;
        }
    }

    // Broadcast matrix dimensions
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&common_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate number of rows per process
    rows_per_process = rows / size;

    // Allocate memory for sub-matrices
    subA = (int *)malloc(rows_per_process * common_dim * sizeof(int));
    subC = (int *)malloc(rows_per_process * cols * sizeof(int));

    // Scatter matrix A
    MPI_Scatter(A, rows_per_process * common_dim, MPI_INT, subA, rows_per_process * common_dim, MPI_INT, 0, MPI_COMM_WORLD);

    // Initialize OpenCL
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    // Create an OpenCL context
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    // Create a command queue
    command_queue = clCreateCommandQueue(context, device, 0, &err);

    // Create a program from the kernel source code
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create the OpenCL kernel
    kernel = clCreateKernel(program, "matrix_multiply", &err);

    // Perform matrix multiplication using OpenCL
    matrix_multiply(subA, B, subC, rows_per_process, cols, common_dim, context, command_queue, program, kernel);

    // Gather the results
    MPI_Gather(subC, rows_per_process * cols, MPI_INT, C, rows_per_process * cols, MPI_INT, 0, MPI_COMM_WORLD);

    // Finalize MPI
    MPI_Finalize();

    // Clean up
    free(subA);
    free(subC);
    if (rank == 0) {
        free(A);
        free(B);
        free(C);
    }

    // Clean up OpenCL resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}
