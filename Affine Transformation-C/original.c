#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define N 372 /* frame dimension for QCIF format */
#define M 496 /* frame dimension for QCIF format */
#define filename "cherry_496x372_444.yuv"
#define file_yuv "cherry_yuv_output.yuv"

/* code for armulator*/
#pragma arm section zidata="ram"
int current_y[N][M];
int current_u[N][M];
int current_v[N][M];
#pragma arm section

int i,j,c,x,y;
float angle_in_degrees;
float angle_in_radians;
int transformed[N][M];
int buffer[N][M];  
float affineMatrix[3][3] = {
{1.414, -1.414, 250},
{1.414, 1.414, -400},
{0, 0, 1}
};
	
void applyAffineTransform(int originalImage[N][M], float affineMatrix[3][3],int channel) 
{
    // Copy the original values to the buffer
    for (i = 0; i < N; i++) 
    {
        for (j = 0; j < M; j++) 
        {
          	buffer[i][j] = originalImage[i][j];
          	if (channel == 2 || channel == 3)
            	originalImage[i][j] = 128;
			else
				originalImage[i][j] = 0; 
        }
    }

    for (y = 0; y < N; y++) {
        for (x = 0; x < M; x++) {
            // Apply affine transformation
            int newX = (int)(affineMatrix[0][0] * x + affineMatrix[0][1] * y + affineMatrix[0][2]);
            int newY = (int)(affineMatrix[1][0] * x + affineMatrix[1][1] * y + affineMatrix[1][2]);

            if (newX >= 0 && newX < M - 1 && newY >= 0 && newY < N - 1) 
            {
              originalImage[y][x] = buffer[newY][newX];
            }
        }
    }
}

void read()
{
  FILE *frame_c;
  if((frame_c=fopen(filename,"rb"))==NULL)
  {
    printf("current frame doesn't exist\n");
    exit(-1);
  }

  for(i=0;i<N;i++)
  {
    for(j=0;j<M;j++)
    {
      current_y[i][j]=fgetc(frame_c);
    }
  }
  for(i=0;i<N;i++)
  {
    for(j=0;j<M;j++)
    {
      current_u[i][j]=fgetc(frame_c);
    }
  }
  for(i=0;i<N;i++)
  {
    for(j=0;j<M;j++)
    {
      current_v[i][j]=fgetc(frame_c);
    }
  }
  fclose(frame_c);
}

void write()
{
  FILE *frame_yuv;
  frame_yuv=fopen(file_yuv,"wb");

  for(i=0;i<N;i++)
  {
    for(j=0;j<M;j++)
    {
      fputc(current_y[i][j],frame_yuv);
    }
  }

  for(i=0;i<N;i++)
  {
    for(j=0;j<M;j++)
    {
      fputc(current_u[i][j],frame_yuv);
    }
  }

  for(i=0;i<N;i++)
  {
    for(j=0;j<M;j++)
    {
      fputc(current_v[i][j],frame_yuv);
    }
  }
  fclose(frame_yuv);
}

int main()
{
  read();
  
  applyAffineTransform(current_y, affineMatrix, 1);
  applyAffineTransform(current_u, affineMatrix, 2);
  applyAffineTransform(current_v, affineMatrix, 3);

  write();
  printf("Image Written");
    
  return 0;
}
