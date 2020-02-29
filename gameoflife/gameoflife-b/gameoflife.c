#include <endian.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

#define calcIndex(width, x,y)  ((y)*(width) + (x))

long TimeSteps = 100;
int w, h;

void writeVTK2(long timestep, double *data, char prefix[1024], int offsetX, int offsetY, int lSize) {
  char filename[2048];  
  int x,y;

  float deltax=1.0;
  long  nxy = lSize * lSize * sizeof(float);  

  snprintf(filename, sizeof(filename), "%s-%05ld%s", prefix, timestep, ".vti");
  FILE* fp = fopen(filename, "w");

  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
  fprintf(fp, "<ImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"0 0 0\" Spacing=\"%le %le %le\">\n", offsetX, (offsetX + lSize), offsetY, (offsetY + lSize), 0, 0, deltax, deltax, 0.0);
  fprintf(fp, "<CellData Scalars=\"%s\">\n", prefix);
  fprintf(fp, "<DataArray type=\"Float32\" Name=\"%s\" format=\"appended\" offset=\"0\"/>\n", prefix);
  fprintf(fp, "</CellData>\n");
  fprintf(fp, "</ImageData>\n");
  fprintf(fp, "<AppendedData encoding=\"raw\">\n");
  fprintf(fp, "_");
  fwrite((unsigned char*)&nxy, sizeof(long), 1, fp);

  for (y = 0; y < lSize; y++) {
    for (x = 0; x < lSize; x++) {
      float value = data[calcIndex(h, x, y)];
      fwrite((unsigned char*)&value, sizeof(float), 1, fp);
    }
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
 
void evolve(double* currentfield, double* newfield, int xStart, int yStart, int lSize) {
  for (int y = yStart; y < (yStart+lSize); y++) {
    for (int x = xStart; x < (xStart+lSize); x++) {
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
  }
  
}


void filling(double* currentfield, int w, int h) {
  int i;
  for (i = 0; i < h*w; i++) {
    currentfield[i] = 0;
    currentfield[i] = (rand() < RAND_MAX / 10) ? 1 : 0; ///< init domain randomly
  }
  /* Toad
  currentfield[144] = 1;
  currentfield[145] = 1;
  currentfield[146] = 1;
  currentfield[173] = 1;
  currentfield[174] = 1;
  currentfield[175] = 1; */
  /* Glider
  currentfield[114] = 1;
  currentfield[145] = 1;
  currentfield[173] = 1;
  currentfield[174] = 1;
  currentfield[175] = 1;*/
}
 
void game(int threadX, int threadY, int lSize) {
  w = threadX*lSize;
  h = threadY*lSize;
  double *currentfield = calloc(w*h, sizeof(double));
  double *newfield     = calloc(w*h, sizeof(double));
  
  //printf("size unsigned %d, size long %d\n",sizeof(float), sizeof(long));
  
  filling(currentfield, w, h);

  long t;
  for (t=0;t<TimeSteps;t++) {
    show(currentfield, w, h);
    int xStart, yStart;
    #pragma omp parallel firstprivate(w, h, lSize) shared(currentfield, newfield) private(xStart, yStart)
    {
      xStart = (int) (omp_get_thread_num() % threadX )*lSize;
      yStart = (int) (omp_get_thread_num() / threadX )*lSize;
      printf("threadnum: %d, xstart: %d, ystart:%d", omp_get_thread_num(), xStart, yStart);
      evolve(currentfield, newfield, xStart, yStart, lSize);

      char prefix[6];
      snprintf(prefix, sizeof prefix, "%s%d%s", "T", omp_get_thread_num(), "-gol");
      writeVTK2(t, currentfield, prefix, xStart, yStart, lSize);
    }
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
  int xThreads = 2, yThreads = 2, lSize = 10;
  if(c > 1) xThreads = atoi(v[1]);
  if(c > 2) yThreads = atoi(v[2]);
  if(c > 3) lSize = atoi(v[3]);
  printf("xThreads: %d\nyThreads: %d\nlSize: %d", xThreads, yThreads, lSize);
  omp_set_num_threads(xThreads*yThreads);
  game(xThreads, yThreads, lSize);
}
