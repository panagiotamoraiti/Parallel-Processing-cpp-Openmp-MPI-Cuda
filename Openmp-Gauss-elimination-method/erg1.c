#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<omp.h>

#define EPSILON 0.000001

int main(int argc, char *argv[])
{
	float ratio;
	int i, j, k, n;
	double x_i;

	 // Dimensions of the matrix
	printf("Enter number of unknowns: ");
	scanf("%d", &n);
	
	double *x, *xs;
	float **a, **A_copy;
	x = malloc((n+1)*sizeof(double));
	xs = malloc((n+1)*sizeof(double));
	a = malloc((n+1)*sizeof(double*));
	A_copy = malloc((n+1)*sizeof(double*));
	
	for (i=0; i<n+1; i++) 
	{
	    a[i] = malloc(n*sizeof(double));
	    A_copy[i] = malloc(n*sizeof(double));
	}
	 
	 // Put the elements of the Augmented Matrix 
    for (i = 0; i < n; i++)
    {
        for (j = 0; j <= n; j++)
        {
            a[i][j] = rand();
            A_copy[i][j] = a[i][j];
        }
    }

    // Gauss Elimination serial
    double start_time, end_time, elapsed_time_s;

    start_time = omp_get_wtime();
    for (i = 0; i < n-1; i++)
    {
        if (a[i][i] == 0.0)
        {
            printf("Error, Division by zero!");
            exit(0);
        }

        for (j = i+1; j < n; j++)
        {
            ratio = a[j][i] / a[i][i];

            for (k = 0; k <= n; k++)
            {
                a[j][k] = a[j][k] - ratio * a[i][k];
            }
        }
    }

    xs[n-1] = a[n-1][n] / a[n-1][n-1];

    for (i = n-2; i >= 0; i--)
    {
        xs[i] = a[i][n];
        for (j = i+1; j < n; j++)
        {
            xs[i] = xs[i] - a[i][j] * xs[j];
        }
        xs[i] = xs[i] / a[i][i];
    }
    end_time = omp_get_wtime();

    // Print the solution
    printf("\nSolution:\n");
    for (i = 0; i < n; i++)
    {
        printf("xs[%d] = %f\n", i, xs[i]);
    }

    elapsed_time_s = end_time - start_time;
    printf("\nElapsed time Serial: %fs\n\n", elapsed_time_s);
    
    // Gauss Elimination parallel
    int P;
	double speedup, elapsed_time, efficiency;
	for (P = 1; P <= 64; P = P*2) 
	{
	    // Restore the initial values in the a matrix
	    for(i=0; i<n; i++)
		{
			for(j=0; j<=n; j++)
			{
				a[i][j] = A_copy[i][j];
			}
		}
		printf("P = %d\n", P);
    
	    start_time = omp_get_wtime();
	    
		for(i=0; i<n-1; i++)
		{
		    if(a[i][i] == 0.0)
		    {
		        printf("Error, Division by zero!");
		        exit(0);
		    }
		    # pragma omp parallel for private(j,k,ratio) num_threads(P)
		    for(j=i+1; j<n; j++)
		    {
		        ratio = (a[j][i]/a[i][i]);
		        for(k=0; k<n+1; k++)
		        {
		            a[j][k] = a[j][k] - ratio*a[i][k];
		        }
		    }
		}
		
		x[n-1] = a[n-1][n]/a[n-1][n-1];
		# pragma omp parallel for private(i, x_i, j) num_threads(P)
		for(i=n-2; i>=0; i--)
		{
		    x_i = a[i][n];
		    // # pragma omp parallel for private(j) reduction(+: x_i) num_threads(P) 
		    for(j=i+1; j<n; j++)
		    {
		        x_i += -a[i][j]*x[j];
		    }
		    x[i] = x_i/a[i][i];
		}
	
		end_time = omp_get_wtime();
		 
		// Print the solution
		printf("\nSolution:\n");
		for(i=0; i<n; i++)
		{
		 	printf("x[%d] = %f\n",i, x[i]);
		}
		
		elapsed_time = end_time - start_time;
	    printf("Elapsed time Parallel: %fs\n", elapsed_time);
	    speedup = elapsed_time_s/elapsed_time;
	    printf("Speedup: %f\n", speedup);
	    efficiency = speedup/P;
	    printf("Efficiency: %f\n", efficiency);
	    
	    // Check the results
	    int flag = 0; 
		for (i=0; i<n; i++) 
		{
		    if (fabs(x[i] - xs[i]) <= EPSILON) 
			{ 
		        flag = 0; 
			}
			else
			{
				flag = 1;
				break;
			}
		}
		
		if (flag == 0) 
		{
	    	printf("Correct answer\n\n");
		} 
		else 
		{
		    printf("Wrong answer\n\n");
		}
	}
	
	// Free memory
	for(i=0; i<n+1; i++) 
	{
    	free(a[i]);
    	free(A_copy[i]);
	}
	free(x);
	free(xs);
	return(0);
}
