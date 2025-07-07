#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<omp.h>
#define EPSILON 0.000001

// Function to read raw image data from a file
char *readRawFile(const char *file_path, int file_size)
{
    FILE *file = fopen(file_path, "rb");
    if (!file)
    {
        fprintf(stderr, "Error opening file: %s\n", file_path);
        exit(1);
    }

    char *data = (char *)malloc(file_size);
    fread(data, 1, file_size, file);
    fclose(file);

    return data;
}

// Function to write raw image data to a file
void writeRawFile(const char *file_path, const char *data, int data_size)
{
    FILE *file = fopen(file_path, "wb");
    if (!file)
    {
        fprintf(stderr, "Error opening file: %s\n", file_path);
        exit(1);
    }

    fwrite(data, 1, data_size, file);
    fclose(file);
}

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
	int i, j, kernel_size, pooling_kernel_size, stride;
	
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

	// Replace with the path to your input RGB image file
    //const char *input_rgb_path = "road.raw";
    const char *input_rgb_path = "elephant.raw";

    // Replace with the path to your output feature map image file
    //const char *output_feature_map_path = "feature_map_road.raw";
    //const char *output_feature_map_path_after_pooling = "feature_map_after_pooling_road.raw";
    const char *output_feature_map_path = "feature_map_elephant.raw";
    const char *output_feature_map_path_after_pooling = "feature_map_after_pooling_elephant.raw";

    // Replace with the width and height of your image
    //int image_width = 1920;
    //int image_height = 1080;
    int image_width = 2040;
    int image_height = 1356;

    // Read the raw data from the input file
    FILE *input_rgb_file = fopen(input_rgb_path, "rb");
    if (!input_rgb_file)
    {
        fprintf(stderr, "Error opening file: %s\n", input_rgb_path);
        return 1;
    }

    // Calculate the size of the raw data
    fseek(input_rgb_file, 0, SEEK_END);
    long file_size = ftell(input_rgb_file);
    fseek(input_rgb_file, 0, SEEK_SET);

    // Read the raw data into a buffer
    char *input_rgb_data = readRawFile(input_rgb_path, (int)file_size);

    // Separate the RGB channels
    int **red_channel = (int **)malloc(image_height * sizeof(int *));
    int **green_channel = (int **)malloc(image_height * sizeof(int *));
    int **blue_channel = (int **)malloc(image_height * sizeof(int *));

    for (i = 0; i < image_height; ++i)
    {
        red_channel[i] = (int *)malloc(image_width * sizeof(int));
        green_channel[i] = (int *)malloc(image_width * sizeof(int));
        blue_channel[i] = (int *)malloc(image_width * sizeof(int));
    }

    for (i = 0; i < image_height; ++i)
    {
        for (j = 0; j < image_width; ++j)
        {
            int index = (i * image_width + j) * 3;
            red_channel[i][j] = input_rgb_data[index];
            green_channel[i][j] = input_rgb_data[index + 1];
            blue_channel[i][j] = input_rgb_data[index + 2];
        }
    }

	// Matrices
    int **result_parallel, **result_serial, **result_parallel_final, **result_serial_final;
    double **outputR, **outputG, **outputB;

    outputR = malloc(image_height * sizeof(double *));
    outputG = malloc(image_height * sizeof(double *));
    outputB = malloc(image_height * sizeof(double *));
    result_serial = malloc(image_height * sizeof(int *));
    result_parallel = malloc(image_height * sizeof(int *));
    result_serial_final = malloc(image_height * sizeof(int *));
    result_parallel_final = malloc(image_height * sizeof(int *));

    for (i = 0; i < image_height; i++) 
    {
        outputR[i] = malloc(image_width * sizeof(double));
        outputG[i] = malloc(image_width * sizeof(double));
        outputB[i] = malloc(image_width * sizeof(double));
        result_serial[i] = malloc(image_width * sizeof(int));
        result_parallel[i] = malloc(image_width * sizeof(int));
        result_serial_final[i] = malloc(image_width * sizeof(int));
        result_parallel_final[i] = malloc(image_width * sizeof(int));
    }

    // Put the elements of the Matrices
    for (i = 0; i < image_height; i++) 
    {
        for (j = 0; j < image_width; j++) 
        {
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
    convolution2D((int **)red_channel, (int **)green_channel, (int **)blue_channel, image_height, image_width, kernel_size, 
				  kernel1, kernel2, kernel3, bias, outputR, outputG, outputB, result_serial);
	maxpooling(result_serial, result_serial_final, image_height, image_width, pooling_kernel_size, stride);
				  
    end_time = omp_get_wtime();
    
    elapsed_time_s = end_time - start_time;
    printf("\nElapsed time Serial: %fs\n\n", elapsed_time_s);
    
    ////// Parallel code
    int P;
	double speedup, elapsed_time, efficiency;
	for (P = 1; P <= 64; P = P*2) 
	{
		// Restore the initial values in the matrices
	    for(i=0; i<image_height; i++)
		{
			for(j=0; j<image_width; j++)
			{
				result_parallel[i][j] = 0;
				result_parallel_final[i][j] = 0;
				outputR[i][j] = 0;
				outputG[i][j] = 0;
				outputB[i][j] = 0;
			}
		}
		
		printf("P = %d\n", P);
		start_time = omp_get_wtime();
		convolution2D_parallel((int **)red_channel, (int **)green_channel, (int **)blue_channel, image_height, image_width, kernel_size, 
								kernel1, kernel2, kernel3, bias, outputR, outputG, outputB, result_parallel, P);
		maxpooling_parallel(result_parallel, result_parallel_final, image_height, image_width, pooling_kernel_size, stride, P);
		
		end_time = omp_get_wtime();
		
		elapsed_time = end_time - start_time;
	    printf("Elapsed time Parallel: %fs\n", elapsed_time);
	    speedup = elapsed_time_s/elapsed_time;
	    printf("Speedup: %f\n", speedup);
	    efficiency = speedup/P;
	    printf("Efficiency: %f\n", efficiency);
		
		// Check the results
	    int flag = check_results(result_serial_final, result_parallel_final, image_height, image_width);
		
		if (flag == 1) 
		{
	    	printf("Correct answer!!!!!!!!!!!!\n\n");
		} 
		else 
		{
		    printf("Wrong answer!!!!!!!!!!!!\n\n");
		}
    }
    
    // Convert the 2D array to a 1D array for writing to file
    char *feature_map = (char *)malloc(image_width * image_height);
	for (i = 0; i < image_height; ++i)
	{
	    for (j = 0; j < image_width; ++j)
	    {
	        int index = i * image_width + j;
	        feature_map[index] = (char)result_parallel[i][j]; 
	    }
	}
	
	char *feature_map_after_pooling = (char *)malloc(image_width/2 * image_height/2);
	for (i = 0; i < image_height/2; ++i)
	{
	    for (j = 0; j < image_width/2; ++j)
	    {
	        int index = i * image_width/2 + j;
	        feature_map_after_pooling[index] = (char)result_parallel_final[i][j]; 
	    }
	}

    // Write the feature map to the output file
    writeRawFile(output_feature_map_path, feature_map, image_width * image_height);
    writeRawFile(output_feature_map_path_after_pooling, feature_map_after_pooling, image_width/2 * image_height/2);
		
	// Free memory
	for (i = 0; i < image_height; i++) 
	{
	    free(outputR[i]);
	    free(outputG[i]);
	    free(outputB[i]);
	    free(result_serial[i]);
	    free(result_parallel[i]);
	    free(result_serial_final[i]);
	    free(result_parallel_final[i]);
	}
	
	free(red_channel);
	free(green_channel);
	free(blue_channel);
	free(outputR);
	free(outputG);
	free(outputB);
	free(result_serial);
	free(result_parallel);
	free(result_serial_final);
	free(result_parallel_final);
	free(feature_map);
	free(feature_map_after_pooling);
	
	return(0);
}
