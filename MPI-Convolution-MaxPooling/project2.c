#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#define EPSILON 0.000001

void convolution(int **input, double **output, int h, int w, int kernel_size, double kernel[][3]) 
{
    int i, j, x, y;
    double sum;
    int kernelCenter = kernel_size / 2.0;

    for (i = kernelCenter; i < h - kernelCenter; i++) 
    {
        for (j = kernelCenter; j < w - kernelCenter; j++) 
        {
            sum = 0;
            for (x = 0; x < kernel_size; x++) 
            {
                for (y = 0; y < kernel_size; y++) 
                {
                	sum += kernel[x][y] * (double)(input[i - kernelCenter + x][j - kernelCenter + y]);
                    //sum += kernel[x][y] * (double)(input[i - x + kernelCenter][j - y + kernelCenter]);
                }
            }
            output[i][j] = sum;
        }
    }
}

void convolution2D(int **input1, int **input2, int **input3, int h, int w, int kernel_size, double kernel1[][3], 
				   double kernel2[][3], double kernel3[][3], double bias, double **output1, double **output2, double **output3, int **output_final) 
{
    int i, j;
    double res;
    
    // Perform convolutions
    convolution(input1, output1, h, w, kernel_size, kernel1);
    convolution(input2, output2, h, w, kernel_size, kernel2);
    convolution(input3, output3, h, w, kernel_size, kernel3);

    for (i = 0; i < h; i++) 
    {
        for (j = 0; j < w; j++) 
        {
        	res = output1[i][j] + output2[i][j] + output3[i][j] + bias;
        	
        	// ReLU Activation Function
        	if(res>0) 
        	{
        		output_final[i][j] = round(res);
			}
			else
			{
				output_final[i][j] = 0;
			}
        }
    }
}

void maxpooling(int **input, int **output, int h, int w, int kernel_size, int stride) 
{
    int i, j, x, y;
    int kernelCenter = kernel_size / 2.0;

    for (i = 0; i < h; i += stride) 
    {
        for (j = 0; j < w; j += stride) 
        {
            int max = input[i][j];

            for (x = 0; x < kernel_size; x++) 
            {
                for (y = 0; y < kernel_size; y++) 
                {
                    int curr_i = i - kernelCenter + x;
                    int curr_j = j - kernelCenter + y;

                    if (curr_i >= 0 && curr_i < h && curr_j >= 0 && curr_j < w) 
                    {
                        int current = input[curr_i][curr_j];
                        if (current > max) 
                        {
                            max = current;
                        }
                    }
                }
            }

            output[i / stride][j / stride] = max;
        }
    }
}

void convolution_parallel(int *input, double *output, int h, int w, int kernel_size, double kernel[][3], int rank, int P) 
{
    int i, j, x, y;
    double sum;
    int kernelCenter = kernel_size / 2.0;
    
    int start = kernelCenter; 
    int end = start+(h/P); 
    
    if (rank == 0)
    {
	    start = kernelCenter;
        end = start+(h/P)-1; 
    }
    if (rank==P-1)
    {
    	start = kernelCenter; 
        end = start+(h/P)-1;
	}
    	
//    printf("\nRank!!!: %d\n", rank);
	for (i=start; i<end; i++) 
	{
		for (j=kernelCenter; j<w-kernelCenter; j++) 
		{
			sum = 0;
	        for (x = 0; x < kernel_size; x++) 
			{
	            for (y = 0; y < kernel_size; y++) 
				{
	                double kernelValue = kernel[x][y];
	                double inputValue = (double)(input[(i-kernelCenter+x)*w+(j-kernelCenter+y)]);

//	                printf("Multiplying kernel[%d][%d] (%lf) with input[%d] (%lf)\n", x, y, kernelValue, (i-kernelCenter+x)*w+(j-kernelCenter+y), inputValue);
	                sum += kernelValue * inputValue;
	            }
	        }
//	        if (rank==0)
//	        printf("\n");
	        
            if (rank==0)
                output[(i+1)*w+j] = sum;
            else
	            output[i*w+j] = sum;		
		}
	}
	
	// Print the output array
//    printf("\n\nRank %d - Output after convolution:\n", rank);
//    for (i = kernelCenter; i <= h/P; i++) 
//    {
//        for (j = kernelCenter; j < w-kernelCenter; j++) 
//        {
//            printf("%lf ", output[i * w + j]);
//        }
//        printf("\n");
//    }
 
}

void convolution2D_parallel(int *input1, int *input2, int *input3, int h, int w, int kernel_size, double kernel1[][3], 
		double kernel2[][3], double kernel3[][3], double bias, double *output1, double *output2, double *output3, int *output_final, int rank, int P) 
{
    int i, j;
    double res;
    int kernelCenter = kernel_size / 2.0;
    
    // Perform convolutions
    convolution_parallel(input1, output1, h, w, kernel_size, kernel1, rank, P);
    convolution_parallel(input2, output2, h, w, kernel_size, kernel2, rank, P);
    convolution_parallel(input3, output3, h, w, kernel_size, kernel3, rank, P);
    
//    printf("\n\nRank %d - Output after convolution:\n", rank);
//    for (i = kernelCenter; i <= h/P; i++) 
//    {
//        for (j = kernelCenter; j < w-kernelCenter; j++) 
//        {
//            printf("%lf ", output1[i * w + j]);
//        }
//        printf("\n");
//    }
//    printf("\n\nRank %d - Output after convolution:\n", rank);
//    for (i = kernelCenter; i <= h/P; i++) 
//    {
//        for (j = kernelCenter; j < w-kernelCenter; j++) 
//        {
//            printf("%lf ", output2[i * w + j]);
//        }
//        printf("\n");
//    }
//    printf("\n\nRank %d - Output after convolution:\n", rank);
//    for (i = 0; i <= h/P; i++) 
//    {
//        for (j = 0; j < w; j++) 
//        {
//            printf("%lf ", output3[i * w + j]);
//        }
//        printf("\n");
//    }
    
    int m = 0;
    for (i = kernelCenter; i <= h/P; i++) 
    {
        for (j = kernelCenter; j < w-kernelCenter; j++) 
        {
        	res = output1[i*w+j] + output2[i*w+j] + output3[i*w+j] + bias;
        	
        	// ReLU Activation Function
        	if(res>0) 
        	{
        		output_final[m] = round(res);
			}
			else
			{
				output_final[m] = 0;
			}
			// Print res
//            printf("Rank %d - output_final[i*w+j][%d]: %d\n", rank, m, output_final[m]);
            m++;
        }
    }
    
//    printf("\n\nRank %d - Output after convolution:\n", rank);
//    for (i = kernelCenter; i <= h/P; i++) 
//    {
//        for (j = kernelCenter; j < w-kernelCenter; j++) 
//        {
//            printf("%d ", output_final[i * w + j]);
//        }
//        printf("\n");
//    }
}

void maxpooling_parallel(int **input, int **output, int h, int w, int kernel_size, int stride) 
{
    int i, j, x, y;
    int kernelCenter = kernel_size / 2.0;

    for (i = 0; i < h; i += stride) 
    {
        for (j = 0; j < w; j += stride) 
        {
            int max = input[i][j];

            for (x = 0; x < kernel_size; x++) 
            {
                for (y = 0; y < kernel_size; y++) 
                {
                    int curr_i = i  + x;
                    int curr_j = j  + y;

                    if (curr_i >= 0 && curr_i < h && curr_j >= 0 && curr_j < w) 
                    {
                        int current = input[curr_i][curr_j];
                        if (current > max) 
                        {
                            max = current;
                        }
                    }
                }
            }

            output[i / stride][j / stride] = max;
        }
    }
    
    // Print the output matrix
//    printf("Maxpooled Output:\n");
//    for (i = 0; i < h / stride; i++) {
//        for (j = 0; j < w / stride; j++) {
//            printf("%d ", output[i][j]);
//        }
//        printf("\n");
//    }
}

void print_results(int **matrix1, int **matrix2, int **matrix3, int kernel_size, int kernelCenter, 
					int **result_serial,  int **result_parallel, int **result_serial_final, int **result_parallel_final, int h, int w)
{
	int i, j;
	
	// Print the results	
    printf("\nOriginal MatrixR:\n");
	for (i = 0; i < h; i++) 
	{
	    for (j = 0; j < w; j++) 
		{
	        printf("%d\t", matrix1[i][j]);
	    }
	    printf("\n");
	}
	
	printf("\nOriginal MatrixG:\n");
	for (i = 0; i < h; i++) 
	{
	    for (j = 0; j < w; j++) 
		{
	        printf("%d\t", matrix2[i][j]);
	    }
	    printf("\n");
	}
	
	printf("\nOriginal MatrixB:\n");
	for (i = 0; i < h; i++) 
	{
	    for (j = 0; j < w; j++) 
		{
	        printf("%d\t", matrix3[i][j]);
	    }
	    printf("\n");
	}
	
	printf("\nFeaturemap Serial:\n");
	for (i = kernelCenter; i < h - kernelCenter; i++) 
	{
	    for (j = kernelCenter; j < w - kernelCenter; j++) 
		{
	        printf("%d\t", result_serial[i][j]);
	    }
	    printf("\n");
	}
	
	printf("\nFeaturemap Parallel:\n");
	for (i = 0; i < h - kernelCenter*2; i++) 
	{
	    for (j = 0; j < w - kernelCenter*2; j++) 
		{
	        printf("%d\t", result_parallel[i][j]);
	    }
	    printf("\n");
	}
	
	printf("\nFeaturemap Serial After Pooling:\n");
	for (i = kernelCenter; i < h/2 + 1 - kernelCenter; i++) 
	{
	    for (j = kernelCenter; j < w/2 + 1 - kernelCenter; j++) 
		{
	        printf("%d\t", result_serial_final[i][j]);
	    }
	    printf("\n");
	}
	
	printf("\nFeaturemap Parallel After Pooling:\n");
	for (i = 0; i < h/2 + 1 - kernelCenter*2; i++) 
	{
	    for (j = 0; j < w/2 + 1 - kernelCenter*2; j++) 
		{
	        printf("%d\t", result_parallel_final[i][j]);
	    }
	    printf("\n");
	}
}

int check_results(int **result_serial, int **result_parallel, int h, int w, int kernelCenter) {
    int i, j;
    int correct = 1; // Assume the results are correct by default

    for (i = kernelCenter; i < h - kernelCenter; i++) {
        for (j = kernelCenter; j < w - kernelCenter; j++) {
            if (fabs(result_serial[i][j] - result_parallel[i - kernelCenter][j - kernelCenter]) > EPSILON) {
                correct = 0; // Results are not equal within the defined tolerance (EPSILON)
                return correct;
            }
        }
    }

    return correct;
}

int main (int argc, char **argv) 
{
	// MPI
	MPI_Status status;	
	int rank, P;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&P);
	
	// Variables
	int i, j, h, w, kernel_size, pooling_kernel_size, stride;
	double start_time, end_time, elapsed_time_s, tb1, tb2, tg1, tg2, elapsed_time_communication, elapsed_time, speedup;
		
	// Kernels
	kernel_size = 3;
	pooling_kernel_size = 2;
	stride = 2;
	
    double kernel1[3][3] = 
	{
        {-0.23682003, 0.1700454, 0.09052496},
        {-0.00821521, 0.1779402, -0.09653653},
        {0.07029217, -0.12574358, 0.10022978}
    };

    double kernel2[3][3] = 
	{
        {-0.07746775, -0.09846717, 0.03488337},
        {0.00120761, 0.07884538, -0.07599071},
        {0.05539479, -0.03348019, -0.07456464}
    };

    double kernel3[3][3] = 
	{
        {0.01844125, 0.20088513, -0.04941435},
        {0.13128215, -0.09104527, 0.06280853},
        {-0.05294552, 0.10650568, -0.09848029}
    };	
						
	double bias = -3.9732606;					  								
	int kernelCenter = kernel_size / 2.0;
	h = 12;
	w = 6;
	
	int **matrixR, **matrixB, **matrixG, **result_serial, **result_serial_final, **result_parallel, **result_parallel_final;
	double **outputR, **outputG, **outputB;
	int *matrixR_1D, *matrixB_1D, *matrixG_1D, *result_parallel_1D, *result_parallel_final_1D;
	
	// Allocate memory for all processes
	matrixR_1D = malloc(h * w * sizeof(int));
    matrixG_1D = malloc(h * w * sizeof(int));
    matrixB_1D = malloc(h * w * sizeof(int));
	
	
    result_parallel_1D = malloc(h * w * sizeof(int));
    result_parallel_final_1D = malloc(h * w * sizeof(int));
	
	// Malloc, Put values in matrices and Serial Code
	if(rank == 0)
	{
	    matrixR = malloc(h * sizeof(int *));
	    matrixG = malloc(h * sizeof(int *));
	    matrixB = malloc(h * sizeof(int *));
	    outputR = malloc(h * sizeof(double *));
	    outputG = malloc(h * sizeof(double *));
	    outputB = malloc(h * sizeof(double *));
	    result_serial = malloc(h * sizeof(int *));
	    result_serial_final = malloc(h * sizeof(int *));
	    result_parallel = malloc(h * sizeof(int *));
	    result_parallel_final = malloc(h * sizeof(int *));
	
	    for (i = 0; i < h; i++) 
	    {
	        matrixR[i] = malloc(w * sizeof(int));
	        matrixG[i] = malloc(w * sizeof(int));
	        matrixB[i] = malloc(w * sizeof(int));
	        outputR[i] = malloc(w * sizeof(double));
	        outputG[i] = malloc(w * sizeof(double));
	        outputB[i] = malloc(w * sizeof(double));
	        result_serial[i] = malloc(w * sizeof(int));
	        result_serial_final[i] = malloc(w * sizeof(int));
	        result_parallel[i] = malloc(w * sizeof(int));
	        result_parallel_final[i] = malloc(w * sizeof(int));
	    }
	
	    // Put the elements of the Matrices
	    for (i = 0; i < h; i++) 
	    {
	        for (j = 0; j < w; j++) 
	        {
	            // Generates random integer from 0 to 255
	            matrixR[i][j] = rand() % 256;
	            matrixG[i][j] = rand() % 256;
	            matrixB[i][j] = rand() % 256;
	            matrixR_1D[i*w+j] = matrixR[i][j];
	            matrixG_1D[i*w+j] = matrixG[i][j];
	            matrixB_1D[i*w+j] = matrixB[i][j];
			
	        }
	    }
	    
	    // Serial code
	    start_time = MPI_Wtime();
	    convolution2D(matrixR, matrixG, matrixB, h, w, kernel_size, kernel1, kernel2, kernel3, bias, outputR, outputG, outputB, result_serial);
	    maxpooling(result_serial, result_serial_final, h, w, pooling_kernel_size, stride);
	    end_time = MPI_Wtime();
	    
	    elapsed_time_s = end_time - start_time;
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime(); // count time for MPI
    
    // Send matrices to all processes
    // Broadcast
    MPI_Barrier(MPI_COMM_WORLD);
    tb1 = MPI_Wtime();
    
	MPI_Bcast(matrixR_1D, h*w, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(matrixG_1D, h*w, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(matrixB_1D, h*w, MPI_INT, 0, MPI_COMM_WORLD);
	
	MPI_Barrier(MPI_COMM_WORLD);
    tb2 = MPI_Wtime();
		
    int *localR, *localG, *localB, *local_output;
    double *localR_output, *localG_output, *localB_output;
	localR = malloc(h*w * sizeof(int));
	localG = malloc(h*w * sizeof(int));
	localB = malloc(h*w * sizeof(int));
	localR_output = malloc(h*w * sizeof(double));
	localG_output = malloc(h*w * sizeof(double));
	localB_output = malloc(h*w * sizeof(double));
	local_output = malloc(h*w * sizeof(int));
	
	// Every process convolves a part of the image
//	printf("\nRank: %d\n", rank);
	int start_line = rank*h/P-1;
	int end_line = start_line+(h/P)+1;
	
	if (rank==0)
	{
		start_line = 0;
		end_line = (h/P)+1;
	}
	if (rank==P-1)
	{
		start_line = h-(h/P)-1;
		end_line = h;
	}
	
	int m=0;
	for (i=start_line; i<=end_line; i++) // i = 0, 1
	{
		for (j=0; j<w; j++) // j = 0, 1, .., w
		{   
			localR[m*w+j] = matrixR_1D[i*w+j];
			localG[m*w+j] = matrixG_1D[i*w+j];
			localB[m*w+j] = matrixB_1D[i*w+j];
			
//			printf("localG[%d]: %d\n", m*w+j, localG[m*w+j]);			
		}
		m ++;
	}

	
//    int size = (kernel_size+kernelCenter*2)*w;
//    if (rank == 0 || rank==P-1)
//        size = size - w;
//
//	printf("\nRank!!!: %d\n", rank);
//	for (i=0; i<size; i++) 
//	{		
//		printf("localG[%d]: %d\n", i, matrixG_1D[i]);	
//	}

	convolution2D_parallel(localR, localG, localB, h, w, kernel_size, kernel1, kernel2, kernel3, bias, localR_output, localG_output, 
	              		   localB_output, local_output, rank, P);
	     
    // Gather         	
	MPI_Barrier(MPI_COMM_WORLD);
    tg1 = MPI_Wtime();	  
	 
    MPI_Gather(local_output, h*(w-2*kernelCenter)/P, MPI_INT, result_parallel_1D, h*(w-2*kernelCenter)/P, MPI_INT, 0, MPI_COMM_WORLD); // h=12, P=4
    
	MPI_Barrier(MPI_COMM_WORLD);
    tg2 = MPI_Wtime();
    
    // Print the gathered result on the root process
    // Change line every n lines
	m = 1;
	int count_elements = 1;
	int n = w - 2 * kernelCenter; // 4
	int max_el = n * (h - 2* kernelCenter); // 4*10 = 40
	
	if (rank == 0) 
	{
//        printf("Gathered result:\n");

        int serial_index = 0; // Track serial index for result_parallel_1D

        for ( i = 0; i < h; i++) 
		{
            for ( j = 0; j < w; j++) 
			{
                if (count_elements > n && count_elements <= max_el + n) 
				{
                    //printf("%d ", result_parallel_1D[i * w + j]);
                    // Store the value serially in result_parallel
                    result_parallel[serial_index / n][serial_index % n] = result_parallel_1D[i * w + j];
                    serial_index++;
                }

                if (m % n == 0) 
				{
                    //if (count_elements <= max_el + n)
                        //printf("\n");
                    m = 1; // Reset m after line break
                } 
				else 
				{
                    m++;
                }
                count_elements++;
            }
        }
        
        maxpooling_parallel(result_parallel, result_parallel_final, h, w, pooling_kernel_size, stride);
        

	    
//	    for (i=0; i < h - 2 * kernelCenter; i++)
//	    {
//	    	for (j=0; j < w - 2 * kernelCenter; j++)
//		    {
//		    	printf("%d ", result_parallel[i][j]);
//			}
//			printf("\n");
//		}
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime(); // count time for MPI
	
	// Free memory
	free(localR);
	free(localG);
	free(localB);
	free(localR_output);
	free(localG_output);
	free(localB_output);
	free(local_output);
	free(matrixR_1D);
	free(matrixG_1D);
	free(matrixB_1D);
    
	// MPI Finish
	MPI_Finalize();
	
	if (rank == 0) 
	{	
	    print_results(matrixR, matrixG, matrixB, kernel_size, kernelCenter, 
					  result_serial, result_parallel, result_serial_final, result_parallel_final, h, w);
			
	    // Check the results
	    int flag = check_results(result_serial, result_parallel, h, w, kernelCenter);
		if (flag == 1) 
		{
	    	printf("\nCorrect answer!!!!!!!!!!!!\n");
		} 
		else 
		{
		    printf("\nWrong answer!!!!!!!!!!!!\n");
		}
	    
        // Print time and speedup
        printf("\nElapsed time Serial: %fs\n", elapsed_time_s);
        
		elapsed_time_communication = tb2-tb1 + tg2-tg1;
        elapsed_time = end_time - start_time;

    	printf("Elapsed time total MPI: %fs\n", elapsed_time);
    	printf("Elapsed time MPI without communication time: %fs\n", elapsed_time-elapsed_time_communication);
    	printf("Elapsed time for MPI without broadcast time: %fs\n\n", elapsed_time-(tb2-tb1));
    	
        printf("Elapsed time communication MPI: %fs\n", elapsed_time_communication);
        printf("Elapsed time Bcast MPI: %fs\n", tb2-tb1);
        printf("Elapsed time Gather MPI: %fs\n\n", tg2-tg1);
        
    	speedup = elapsed_time_s/elapsed_time;
	    printf("Speedup: %f\n", speedup);
	    printf("Speedup without broadcast time: %f\n\n", elapsed_time_s/(elapsed_time-(tb2-tb1)));

		// Free memory
		for (i = 0; i < h; i++) 
		{
		    free(matrixR[i]);
		    free(matrixG[i]);
		    free(matrixB[i]);
		    free(outputR[i]);
		    free(outputG[i]);
		    free(outputB[i]);
		    free(result_serial[i]);
		    free(result_serial_final[i]);
		    free(result_parallel[i]);
		    free(result_parallel_final[i]);
		}
		
		free(matrixR);
		free(matrixG);
		free(matrixB);
		free(outputR);
		free(outputG);
		free(outputB);
		free(result_serial);
		free(result_serial_final);
	    free(result_parallel[i]);
	    free(result_parallel_final[i]);
	}
}
