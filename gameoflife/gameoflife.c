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

void writeVTK2(long timestep, double *data, char prefix[1024], int w, int h) {
  char filename[2048];  
  int x,y; 
  
  int offsetX=0;
  int offsetY=0;
  float deltax=1.0;
  long  nxy = w * h * sizeof(float);  

  snprintf(filename, sizeof(filename), "%s-%05ld%s", prefix, timestep, ".vti");
  FILE* fp = fopen(filename, "w");

  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
  fprintf(fp, "<ImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"0 0 0\" Spacing=\"%le %le %le\">\n", offsetX, offsetX + w, offsetY, offsetY + h, 0, 0, deltax, deltax, 0.0);
  fprintf(fp, "<CellData Scalars=\"%s\">\n", prefix);
  fprintf(fp, "<DataArray type=\"Float32\" Name=\"%s\" format=\"appended\" offset=\"0\"/>\n", prefix);
  fprintf(fp, "</CellData>\n");
  fprintf(fp, "</ImageData>\n");
  fprintf(fp, "<AppendedData encoding=\"raw\">\n");
  fprintf(fp, "_");
  fwrite((unsigned char*)&nxy, sizeof(long), 1, fp);

  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++) {
      float value = data[calcIndex(h, x,y)];
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

double get_nums_neighbour(double* currentfield, int x, int y, int w, int h) {
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
 
void evolve(double* currentfield, double* newfield, int w, int h) {
  int x,y;
  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++) {
      double neighbours_num = get_nums_neighbour(currentfield, x, y, w, h);
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
    //currentfield[i] = (rand() < RAND_MAX / 10) ? 1 : 0; ///< init domain randomly
  }
  /* Toad
  currentfield[144] = 1;
  currentfield[145] = 1;
  currentfield[146] = 1;
  currentfield[173] = 1;
  currentfield[174] = 1;
  currentfield[175] = 1; */
  // Glider
  currentfield[114] = 1;
  currentfield[145] = 1;
  currentfield[173] = 1;
  currentfield[174] = 1;
  currentfield[175] = 1;
}
 
void game(int w, int h) {
  double *currentfield = calloc(w*h, sizeof(double));
  double *newfield     = calloc(w*h, sizeof(double));
  
  //printf("size unsigned %d, size long %d\n",sizeof(float), sizeof(long));
  
  filling(currentfield, w, h);
  long t;
  for (t=0;t<TimeSteps;t++) {
    show(currentfield, w, h);
    evolve(currentfield, newfield, w, h);
    
    printf("%ld timestep\n",t);
    writeVTK2(t,currentfield,"gol", w, h);
    
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
  int w = 0, h = 0;
  if (c > 1) w = atoi(v[1]); ///< read width
  if (c > 2) h = atoi(v[2]); ///< read height
  if (w <= 0) w = 30; ///< default width
  if (h <= 0) h = 30; ///< default height
  game(w, h);
}
