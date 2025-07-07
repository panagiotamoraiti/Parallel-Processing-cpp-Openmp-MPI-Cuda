#include <stdio.h>
#include <cuda.h>
#define N (2048 * 2048)

__global__ void upper_triangular(int n, float *a, int i)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x; 
    float ratio;
    
    if (j<n && j>=i+1)
    {	    
		ratio = a[j*(n+1)+i]/a[i*(n+1)+i];
		
	    for (int k=0; k<n+1; k++)
	    {
	    	a[j*(n+1)+k] = a[j*(n+1)+k] - ratio*a[i*(n+1)+k];
		}
	}
}

__global__ void backwardSubstitution2(int n, float *a, double *x, int i,double *x_temp)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (j<n && j>=i+1)
    {
        x_temp[j] = a[i * (n+1) + j] * x[j];

    }
}

void GaussElimination(float *a, double *x, int n, double *x_temp) 
{
    int size_1D = (n+1) * sizeof(double);
    int size_2D = (n+1) * (n+1) * sizeof(double);
    double *xd, *x_tempd, x_i;
    float *ad;
    float TotalTime, H2DTime1, KernelTime, D2HTime1, H2DTime2, D2HTime2, H2DTime3;
    float KernelTime1 = 0;
	float D2HTime2all = 0; 
	float H2DTime3all = 0; 
	float KernelTime2 = 0;
    
    int blockSize, gridSize;
    // Number of threads in each thread block
    blockSize = 1024;
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)N/blockSize);
    
    // capture start time
    cudaEvent_t  start, stop, start1, stop1, startall, stopall;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&startall);
    cudaEventCreate(&stopall);

    // allocate memory on the GPU
    cudaMalloc((void**)&ad, size_2D);
    cudaMalloc((void**)&xd, size_1D);
    cudaMalloc((void**)&x_tempd, size_1D);
    
    //Start the timer for the whole program ***
	cudaEventRecord(startall, 0);
	
    // transfer a to device memory
    cudaEventRecord(start, 0);
    cudaMemcpy(ad, a, size_2D, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&H2DTime1, start, stop);
    
    // kernel1
    for(int i=0; i<n-1; i++)
	{
		if(a[(n+1)*i+i] == 0.0)
		{
		    printf("Error, Division by zero!");
		    exit(0);
		}
		cudaEventRecord(start, 0);
		upper_triangular<<<gridSize, blockSize>>>(n, ad, i);
		cudaDeviceSynchronize(); 
    	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&KernelTime, start, stop);
    	KernelTime1 += KernelTime;
	}
    
    // transfer a from device
    cudaEventRecord(start, 0); 
    cudaMemcpy(a, ad, size_2D, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&D2HTime1, start, stop);
    
	x[n-1] = a[(n-1)*(n+1)+n]/a[(n-1)*(n+1)+n-1];
	
	// transfer x to device
	cudaEventRecord(start, 0);
	cudaMemcpy(xd, x, size_1D, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&H2DTime2, start, stop);
	
	// kernel2
	for(int i=n-2; i>=0; i--)
	{   
	    x_i = a[i*(n+1)+n];
	    
	    cudaEventRecord(start1, 0);
		backwardSubstitution2<<<gridSize, blockSize>>>(n, ad, xd, i, x_tempd);
		cudaDeviceSynchronize(); 
	    cudaEventRecord(stop1, 0);
	    cudaEventSynchronize(stop1);
	    cudaEventElapsedTime(&KernelTime, start1, stop1);
		KernelTime2 += KernelTime;
		
		// transfer x_temp from device
		cudaEventRecord(start, 0);
		cudaMemcpy(x_temp, x_tempd, size_1D, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&D2HTime2, start, stop);
    	D2HTime2all += D2HTime2;
    
		for (int k=i+1; k<n; k++)
		{
		    x_i += - x_temp[k];
		}
	    
	    x[i] = x_i/a[i*(n+1)+i];
	    
	    // transfer x to device
	    cudaEventRecord(start, 0);
        cudaMemcpy(xd, x, size_1D, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
	    cudaEventRecord(stop, 0);
	    cudaEventSynchronize(stop);
	    cudaEventElapsedTime(&H2DTime3, start, stop);
		H2DTime3all += H2DTime3;
	}
	
	// Stop the timer for the whole program
	cudaDeviceSynchronize();
    cudaEventRecord(stopall, 0);    
    cudaEventSynchronize(stopall);
    cudaEventElapsedTime(&TotalTime, startall, stopall);
 
    // display the timing results
    printf("Transfer a to device memory: %f\n", H2DTime1);
    printf("Time for kernel 1: %f\n",KernelTime1);
    printf("Transfer a from device: %f\n",D2HTime1);
    printf("Transfer x to device: %f\n",H2DTime2);
    printf("Transfer x to device all: %f\n", H2DTime3all);
    printf("Transfer x_temp from device all: %f\n", D2HTime2all);
    printf("Time for kernel 2: %f\n", KernelTime2);
    
    float Kernel_Time = KernelTime1 + KernelTime2;
    float MemCopy = H2DTime1 + H2DTime2 + H2DTime3all + D2HTime1 + D2HTime2all;
    float SerialTime = TotalTime - MemCopy - Kernel_Time;
    
    printf("\nTime for %dx%d array multiplication.\n", n, n);
	printf("MemCopy: %f sec, Kernel: %f sec, Serial: %f sec, Total: %f sec\n", MemCopy/1000, Kernel_Time/1000, SerialTime/1000, TotalTime/1000);

	/*// Print the solution
	printf("\nSolution:\n");
	for(int i=0; i<n; i++)
	{
	 	printf("x[%d] = %f\n", i, x[i]);
	}*/
	
    // free the memory allocated on the GPU
    cudaFree(ad);
    cudaFree(xd);
    cudaFree(x_tempd);

    // destroy events to free memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

main(int argc, char *argv[])
{
	// Dimensions of the matrix
	int n;
	printf("Enter number of unknowns: ");
	scanf("%d", &n);
	
    int size_1D = (n+1) * sizeof(double);
    int size_2D = (n+1) * (n+1) * sizeof(double);
    double *x;
	float *a;
	double *x_temp;

    // allocate memory on the CPU
    x = (double*)malloc(size_1D);
    a = (float*)malloc(size_2D);
    x_temp = (double*)malloc(size_1D);
    
    // initialize the matrices
    for (int i=0; i<n; i++) 
	{
	    for (int j=0; j<=n; j++)
		{
	        a[i*n + j] = rand();
	        /*printf("a[%d][%d] = ", i, j);
			scanf("%f", &a[i*(n+1)+j]);*/
	    }
    }

    GaussElimination(a, x, n,x_temp);

    // free the memory allocated on the CPU
    free(a);
    free(x);
    free(x_temp);

    return 0;
}
