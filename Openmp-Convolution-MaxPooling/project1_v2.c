#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<omp.h>
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

void convolution_parallel(int **input, double **output, int h, int w, int kernel_size, double kernel[][3], int P) 
{
    int i, j, x, y;
    int kernelCenter = kernel_size / 2.0;

    # pragma omp parallel for private(i, j, x, y) num_threads(P) collapse(2)
    for (i = kernelCenter; i < h - kernelCenter; i++) 
    {
        for (j = kernelCenter; j < w - kernelCenter; j++) 
        {
            double sum = 0; // sum is private
            for (x = 0; x < kernel_size; x++) 
            {
                for (y = 0; y < kernel_size; y++) 
                {
                    sum += kernel[x][y] * (double)(input[i - kernelCenter + x][j - kernelCenter + y]);
                }
            	output[i][j] = sum;
        	}
		}
    }
}

void convolution2D_parallel(int **input1, int **input2, int **input3, int h, int w, int kernel_size, double kernel1[][3], double kernel2[][3],
                            double kernel3[][3], double bias, double **output1, double **output2, double **output3, int **output_final, int P) 
{
    int i, j;
    double res;
    
    // Perform convolutions
	convolution_parallel(input1, output1, h, w, kernel_size, kernel1, P);
	convolution_parallel(input2, output2, h, w, kernel_size, kernel2, P);
	convolution_parallel(input3, output3, h, w, kernel_size, kernel3, P);

    # pragma omp parallel private(i, j, res) num_threads(P)
    {
    	# pragma omp for collapse(2)
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
}

void maxpooling_parallel(int **input, int **output, int h, int w, int kernel_size, int stride, int P) 
{
    int i, j, x, y;
    int kernelCenter = kernel_size / 2.0;

    # pragma omp parallel for private(i, j, x, y) num_threads(P) collapse(2)
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

int check_results(int **result_serial, int **result_parallel, int h, int w) 
{
    int i, j;
    int correct = 1; // Assume the results are correct by default

    for (i = 0; i < h; i++) 
	{
        for (j = 0; j < w; j++) 
		{
            if (fabs(result_serial[i][j] - result_parallel[i][j]) > EPSILON) 
			{
                correct = 0; // Results are not equal within the defined tolerance (EPSILON)
                return correct;
            }
        }
    }

    return correct;
}


int main(int argc, char *argv[])
{
	// Variables
	int i, j, h, w, kernel_size, pooling_kernel_size, stride;
	
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

	// Dimensions of the matrix
	h = 8000;
	w = 8000;
	
	// Matrices
    int **matrixR, **matrixB, **matrixG, **result_parallel, **result_serial, **result_parallel_final, **result_serial_final;
    double **outputR, **outputG, **outputB;

    matrixR = malloc(h * sizeof(int *));
    matrixG = malloc(h * sizeof(int *));
    matrixB = malloc(h * sizeof(int *));
    outputR = malloc(h * sizeof(double *));
    outputG = malloc(h * sizeof(double *));
    outputB = malloc(h * sizeof(double *));
    result_serial = malloc(h * sizeof(int *));
    result_parallel = malloc(h * sizeof(int *));
    result_serial_final = malloc(h * sizeof(int *));
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
        result_parallel[i] = malloc(w * sizeof(int));
        result_serial_final[i] = malloc(w * sizeof(int));
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
            outputR[i][j] = 0;
            outputG[i][j] = 0;
            outputB[i][j] = 0;
            result_serial[i][j] = 0;
            result_parallel[i][j] = 0;
            result_serial_final[i][j] = 0;
            result_parallel_final[i][j] = 0;
        }
    }
    

    ////// Serial code
    double start_time, end_time, elapsed_time_s;
    start_time = omp_get_wtime();
    convolution2D(matrixR, matrixG, matrixB, h, w, kernel_size, kernel1, kernel2, kernel3, bias, outputR, outputG, outputB, result_serial);
    maxpooling(result_serial, result_serial_final, h, w, pooling_kernel_size, stride);
    end_time = omp_get_wtime();
    
    elapsed_time_s = end_time - start_time;
    printf("\nElapsed time Serial: %fs\n\n", elapsed_time_s);
    
    ////// Parallel code
    int P;
	double speedup, elapsed_time, efficiency;
	for (P = 1; P <= 64; P = P*2) 
	{
		// Restore the initial values in the matrices
	    for(i=0; i<h; i++)
		{
			for(j=0; j<w; j++)
			{
				result_parallel[i][j] = 0;
				outputR[i][j] = 0;
				outputG[i][j] = 0;
				outputB[i][j] = 0;
				result_parallel_final[i][j] = 0;
			}
		}
		
		printf("P = %d\n", P);
		start_time = omp_get_wtime();
		convolution2D_parallel(matrixR, matrixG, matrixB, h, w, kernel_size, kernel1, kernel2, kernel3, bias, outputR, outputG, outputB, 
							 result_parallel, P);
		maxpooling_parallel(result_parallel, result_parallel_final, h, w, pooling_kernel_size, stride, P);
		end_time = omp_get_wtime();
		
		elapsed_time = end_time - start_time;
	    printf("Elapsed time Parallel: %fs\n", elapsed_time);
	    speedup = elapsed_time_s/elapsed_time;
	    printf("Speedup: %f\n", speedup);
	    efficiency = speedup/P;
	    printf("Efficiency: %f\n", efficiency);
		
		// Check the results
	    int flag = check_results(result_serial_final, result_parallel_final, h, w);
		
		if (flag == 1) 
		{
	    	printf("Correct answer!!!!!!!!!!!!\n\n");
		} 
		else 
		{
		    printf("Wrong answer!!!!!!!!!!!!\n\n");
		}
    }
    
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
	    free(result_parallel[i]);
	    free(result_serial_final[i]);
	    free(result_parallel_final[i]);
	}
	
	free(matrixR);
	free(matrixG);
	free(matrixB);
	free(outputR);
	free(outputG);
	free(outputB);
	free(result_serial);
	free(result_parallel);
	free(result_serial_final);
	free(result_parallel_final);
	
	return(0);
}
