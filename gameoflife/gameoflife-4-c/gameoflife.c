#include "mpi.h"
#include <endian.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

#define calcIndex(width, x,y)  ((y)*(width) + (x))

long TimeSteps = 1;
int w, h;

void writeVTK2(long timestep, double *data, char prefix[1024], int lSize, int offsetX, int offsetY) {
  char filename[2048];  
  int x,y;

  float deltax=1.0;
  long  nxy = lSize * lSize * sizeof(float);  

  snprintf(filename, sizeof(filename), "%s-%05ld%s", prefix, timestep, ".vti");
  FILE* fp = fopen(filename, "w");

  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
  // fprintf(fp, "<ImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"0 0 0\" Spacing=\"%le %le %le\">\n", offsetX, (offsetX + lSize), offsetY, (offsetY + lSize), 0, 0, deltax, deltax, 0.0);
  fprintf(fp, "<ImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"0 0 0\" Spacing=\"%le %le %le\">\n", 0, lSize, 0, 1, 0, 0, deltax, deltax, 0.0);
  fprintf(fp, "<CellData Scalars=\"%s\">\n", prefix);
  fprintf(fp, "<DataArray type=\"Float32\" Name=\"%s\" format=\"appended\" offset=\"0\"/>\n", prefix);
  fprintf(fp, "</CellData>\n");
  fprintf(fp, "</ImageData>\n");
  fprintf(fp, "<AppendedData encoding=\"raw\">\n");
  fprintf(fp, "_");
  fwrite((unsigned char*)&nxy, sizeof(long), 1, fp);
  y = offsetY;
    for (x = offsetX; x < offsetX+lSize; x++) {
      float value = 1.0; //data[calcIndex(h, x, y)];
      fwrite((unsigned char*)&value, sizeof(float), 1, fp);
    }

  
  fprintf(fp, "\n</AppendedData>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose(fp);
}


void show(double* currentfield, int w, int h) {
  printf("\033[H");
  int x,y;
  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++) printf(currentfield[calcIndex(w, x,y)] ? "\033[07m  \033[m" : "  ");
    //printf("\033[E");
    printf("\n");
  }
  fflush(stdout);
}

double get_nums_neighbour(double* currentfield, int x, int y) {
  int lefter, upper, righter, downer;
  lefter = x-1;
  upper = y-1;
  righter = x+1;
  downer = y+1;

  if (x == 0) {
    lefter = w-1;
  }
  if (y == 0) {
    upper = h-1;
  }
  if (x == w-1) {
    righter = 0;
  }
  if (y == h-1) {
    downer = 0;
  }
  double neighbours_num = 0.0;
  neighbours_num += currentfield[calcIndex(w, lefter, y)];
  neighbours_num += currentfield[calcIndex(w, lefter, upper)];
  neighbours_num += currentfield[calcIndex(w, lefter, downer)];
  neighbours_num += currentfield[calcIndex(w, x, upper)];
  neighbours_num += currentfield[calcIndex(w, x, downer)];
  neighbours_num += currentfield[calcIndex(w, righter, upper)];
  neighbours_num += currentfield[calcIndex(w, righter, y)];
  neighbours_num += currentfield[calcIndex(w, righter, downer)];
  return neighbours_num;
}
 
void evolve(double* currentfield, double* newfield, int start, int lSize) {
  int x = start;
    for (int i = start; i < (start+lSize); i++) {
      x += 1;
      if(x == w) {
        x = 0;
      }
      int y = (int) (i) / w;
      double neighbours_num = get_nums_neighbour(currentfield, x, y);
      if(currentfield[calcIndex(w, x, y)] == 1.0){
      if (neighbours_num < 2.0) newfield[calcIndex(w, x, y)] = 0.0;
      if (neighbours_num > 3.0) newfield[calcIndex(w, x, y)] = 0.0;
      if (neighbours_num == 3.0 || neighbours_num == 2.0) newfield[calcIndex(w, x, y)] = 1.0;
      }
      else{
        if (neighbours_num == 3.0) newfield[calcIndex(w, x, y)] = 1.0;
        else newfield[calcIndex(w, x, y)] = 0.0;
      }
    }
    int size = sizeof(currentfield) / sizeof(double);
    printf("size: %d", size);
    /*for(int f = 0; f < size; f++){
      printf("%f ", currentfield[f]);
      if(f%9 == 0){
        printf("\n");
      }
    }*/
  
}


void filling(double* currentfield, int lSize) {
  currentfield[0] = 0.0;
  for(int i = 1; i < lSize; i++){
    currentfield[i] = !currentfield[i-1];
  }
}
 
void game(int rank, int size, MPI_Comm communicator, MPI_Status status, int lSize) {
  double *currentfield = calloc(lSize, sizeof(double));
  
  filling(currentfield, lSize);

  double *leftfield = calloc(lSize, sizeof(double));
  double *rightfield = calloc(lSize, sizeof(double));
  double *newfield     = calloc(lSize, sizeof(double));
  int rightneighbourrank, leftneighbourrank;
  MPI_Cart_shift(
    communicator, 1, 1, &leftneighbourrank, &rightneighbourrank
  );
  if (rank != 0) {
    MPI_Recv(leftfield, lSize, MPI_DOUBLE, leftneighbourrank, 0, communicator, &status);
    printf("Rank %d: Received LEFT buffer from rank %d! Content: %f\n", rank, leftneighbourrank, leftfield[0]);
    MPI_Send(currentfield, lSize, MPI_DOUBLE, rightneighbourrank, 0, communicator);
  }
  if (rank == 0) {
    MPI_Send(currentfield, lSize, MPI_DOUBLE, rightneighbourrank, 0, communicator);
    MPI_Recv(leftfield, lSize, MPI_DOUBLE, leftneighbourrank, 0, communicator, &status);
    printf("Rank %d: Received LEFT buffer from rank %d! Content: %f\n", rank, leftneighbourrank, leftfield[0]);
  }

    if (rank != 0) {
    MPI_Recv(rightfield, lSize, MPI_DOUBLE, rightneighbourrank, 0, communicator, &status);
    printf("Rank %d: Received RIGHT buffer from rank %d! Content: %d\n", rank, leftneighbourrank, rightfield[0]);
    MPI_Send(currentfield, lSize, MPI_DOUBLE, (leftneighbourrank), 0, communicator);
  }
  if (rank == 0) {
    MPI_Send(currentfield, lSize, MPI_DOUBLE, leftneighbourrank, 0, communicator);
    MPI_Recv(rightfield, lSize, MPI_DOUBLE, rightneighbourrank, 0, communicator, &status);
    printf("Rank %d: Received RIGHT buffer from rank %d! Content: %d\n", rank, leftneighbourrank, rightfield[0]);
  }
  
  
  long t;
  for (t=0;t<TimeSteps;t++) {
    // show(currentfield, w, h);
    // int start = 0;
    // evolve(currentfield, newfield, start, lSize);

    char prefix[6];
    snprintf(prefix, sizeof prefix, "%s%d%s", "P", rank, "-gol"); // replace 1 with rank later
    
    int xStart = 0;
    int yStart = (int) (rank / size )*h;
    writeVTK2(t, currentfield, prefix, lSize, xStart, yStart);
    printf("%ld timestep\n",t);
    usleep(200000);

    //SWAP
    double *temp = currentfield;
    currentfield = newfield;
    newfield = temp;
  }
  
  free(currentfield);
  free(newfield);
}
 
int main(int c, char **v) {
  int myrank, size, lSize;
  lSize = 10;
  if(c > 1) lSize = atoi(v[1]);
  if(c > 2) w = atoi(v[2]);
  if(c > 3) h = atoi(v[3]);
  MPI_Status status;
  MPI_Init( &c, &v );
  int ndims = 1;
  int *dims = malloc(sizeof(int));
  dims[0] = 10;
  int *periods = malloc(sizeof(int));
  periods[0] = true;
  
  MPI_Comm comm_1d;
  MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, false, &comm_1d );
  MPI_Comm_size(comm_1d, &size);
  MPI_Comm_rank(comm_1d, &myrank);
  printf("Rank: %d, Size:%d\n", myrank, size);
  game(myrank, size, comm_1d, status, lSize);
  MPI_Finalize();
}
