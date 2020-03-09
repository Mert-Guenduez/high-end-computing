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

long TimeSteps = 100;

void writeVTK2(long timestep, double *data, char prefix[1024], int rank, int lySize, int lxSize) {
  char filename[2048];

  float deltax=1.0;
  long  nxy = lxSize * lySize * sizeof(long);  

  snprintf(filename, sizeof(filename), "%s-%05ld%s", prefix, timestep, ".vti");
  FILE* fp = fopen(filename, "w");

  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
  // fprintf(fp, "<ImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"0 0 0\" Spacing=\"%le %le %le\">\n", offsetX, (offsetX + lSize), offsetY, (offsetY + lSize), 0, 0, deltax, deltax, 0.0);
  fprintf(fp, "<ImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"%d %d 0\" Spacing=\"%le %le %le\">\n", 0, lxSize, 0, lySize, 0 ,0, 0, (rank*lySize), deltax, deltax, 0.0);
  fprintf(fp, "<CellData Scalars=\"%s\">\n", prefix);
  fprintf(fp, "<DataArray type=\"Float32\" Name=\"%s\" format=\"appended\" offset=\"0\"/>\n", prefix);
  fprintf(fp, "</CellData>\n");
  fprintf(fp, "</ImageData>\n");
  fprintf(fp, "<AppendedData encoding=\"raw\">\n");
  fprintf(fp, "_");
  fwrite((unsigned char*)&nxy, sizeof(long), 1, fp);

  for (int y = 0; y < lySize; y++) {
    for (int x = 0; x < lxSize; x++) {
      int value = data[calcIndex(lxSize, x, y)];
      fwrite((unsigned char*)&value, sizeof(int), 1, fp);
      fflush(fp);
    }
  }

  
  fprintf(fp, "\n</AppendedData>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose(fp);
}

/*double get_nums_neighbour(double* currentfield, int x, int y) {
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
}*/
/*int coutLifingsPeriodic(double* processingfield, int i, int w) {
  int n = 0;
  int yStart = (int) i/w;
  int xStart = i%w;
  for (int y = (yStart-1); y <= (yStart+1); y++)
  {   
    for (int x = (xStart-1); x <= (xStart+1); x++) {
      if (processingfield[calcIndex(w, x, y)]) {
        n++;
      }
    }
  }
  return n;
}*/
int coutLifingsPeriodic(double* processingfield, int i, int w) {
  int n = 0;
  // left
  if(i % w == 0){
    n += processingfield[i-1];
    n += processingfield[i+w-1];
    n += processingfield[i+w-1+w];
  } else {
    n+=processingfield[i-1];
    n+=processingfield[i-1+w];
    n+=processingfield[i-1-w];
  }
  // middle
  n += processingfield[i-w] + processingfield[i+w]; 
  // right
  if(i % w == w-1){
    n += processingfield[i+1];
    n += processingfield[i-w+1];
    n += processingfield[i-w+1-w];
  } else {
    n+=processingfield[i+1];
    n+=processingfield[i+1+w];
    n+=processingfield[i+1-w];
  }
  return n;
}
 

void evolve(double* processingfield, double* newfield, int w, int fieldlength, int rank) {
    for (int i = w; i < fieldlength + w; i++) {
      double neighbours_num = coutLifingsPeriodic(processingfield, i, w);
      if(rank == 0){
        printf("i: %d, neighbours:%d\n",i,neighbours_num);
      }
      if(processingfield[i] == 1.0){
      if (neighbours_num < 2.0) newfield[i-w] = 0.0;
      if (neighbours_num > 3.0) newfield[i-w] = 0.0;
      if (neighbours_num == 3.0 || neighbours_num == 2.0) newfield[i-w] = 1.0;
      }
      else{
        if (neighbours_num == 3.0) newfield[i-w] = 1.0;
        else newfield[i-w] = 0.0;
      }
    }
}


void filling(double* currentfield, int size, int rank) {
  int i = 0;
  int c;
  FILE *file;
  file = fopen("initialfield.txt", "r");
  if (file) {

    // Alles vor dem für den Prozess relevanten Teil rausschmeißen
    for (int j = 0; j < (rank*size); ) {
      c = getc(file);
      if (c == 48 || c == 49)  j++;
    }
    
    // Werte einlesen
    while (((c=getc(file)) != EOF) && (i < size)){
      if(c == 48) {
        currentfield[i] = 0;
        i++;
      }
      if(c == 49) {
        currentfield[i] = 1;
        i++;
      }
    }
    fclose(file);
  } else {
    printf("No initialfield.txt file found!");
    return 1;
  }
}

void printDebug(double* array, int size, int from, int to) {
  printf("#%d <<<< #%d:[", to, from);
  for (int i = 0; i < size; i++) {
      printf(" %f", array[i]);
    }
  printf("]\n");
}

void processCommunication(double* currentfield, double* leftfield, double* rightfield, int lxSize, int lySize, int rank, MPI_Comm communicator) {
  MPI_Status status;

  int rightrank, leftrank;
  MPI_Cart_shift(
    communicator,
    1,
    1,
    &leftrank,
    &rightrank
  );
  // Restlichen schicken Daten an rechten Nachbarn, warten dann auf Daten vom linken Nachbarn
  if (rank != 0) {
    MPI_Recv(
        leftfield,    // recv buffer
        lxSize,       // buffer size
        MPI_DOUBLE,   // buffer datatype
        leftrank,     // source rank
        0,            // tag
        communicator, // comm
        &status);     // status

    printDebug(leftfield, lxSize, leftrank, rank);

    MPI_Send(&currentfield[lxSize*(lySize-1)], lxSize, MPI_DOUBLE, rightrank, 0, communicator);
  }
  // Root wartet auf Daten vom linken Nachbarn, schickt dann Daten an seinen rechten Nachbarn
  if (rank == 0) {
    MPI_Send(&currentfield[lxSize*(lySize-1)], lxSize, MPI_DOUBLE, rightrank, 0, communicator);
    MPI_Recv(leftfield, lxSize, MPI_DOUBLE, leftrank, 0, communicator, &status);  

    printDebug(leftfield, lxSize, leftrank, rank);
  }

  // Restlichen schicken links, warten dann auf rechts
  if (rank != 0) {
    MPI_Recv(rightfield, lxSize, MPI_DOUBLE, rightrank, 0, communicator, &status);
    
    printDebug(rightfield, lxSize, rightrank, rank);

    MPI_Send(&currentfield[0], lxSize, MPI_DOUBLE, leftrank, 0, communicator);
  }
  // Root wartet auf rechts, schickt dann nach links
  if (rank == 0) {
    MPI_Send(&currentfield[0], lxSize, MPI_DOUBLE, leftrank, 0, communicator);
    MPI_Recv(rightfield, lxSize, MPI_DOUBLE, rightrank, 0, communicator, &status);
    
    printDebug(rightfield, lxSize, rightrank, rank);
  }
}
 
void game(int lxSize, int lySize, int rank, int size, MPI_Comm communicator) {
  

  // Prozess muss seine eigenen Daten lxSize*lySize und zwei zusätzliche Ränder (GhostLayer) größe lxSize verarbeiten 
  double *currentfield = calloc(lxSize*lySize, sizeof(double));
  double *leftfield = calloc(lxSize, sizeof(double));
  double *rightfield = calloc(lxSize, sizeof(double));
  double *processingfield = calloc(lxSize*lySize+lxSize+lxSize, sizeof(double));
  
  // Ergebnisse werden in diesem Feld gespeichert
  double *newfield = calloc(lxSize*lySize, sizeof(double));

  // Belegen des Feldes mit Startwerten
  filling(currentfield, lxSize*lySize, rank);

  // Präfix festlegen
  char prefix[6];
  snprintf(prefix, sizeof prefix, "%s%d%s", "P", rank, "-gol");
  
  long t;
  for (t=0;t<TimeSteps;t++) {
    processCommunication(currentfield, leftfield, rightfield, lxSize, lySize, rank, communicator);
    memcpy(&processingfield[0], leftfield, lxSize * sizeof(leftfield));
    memcpy(&processingfield[lxSize], currentfield, lxSize * lySize * sizeof(currentfield)); 
    memcpy(&processingfield[lxSize+lxSize*lySize], rightfield, lxSize * sizeof(rightfield));

    /*if (rank == 0) {
      /*printf("LEFT: ");printDebug(leftfield, lxSize, rank, rank);
      printf("MIDDLE: ");printDebug(currentfield, lxSize*lySize, rank, rank);
      printf("RIGHT: ");printDebug(rightfield, lxSize, rank, rank);
      printf("ALL: ");printDebug(processingfield, 2*lxSize+lxSize*lySize, rank, rank);
      int n = coutLifingsPeriodic(processingfield, 8, lxSize);
      printf("#0 NACHBARN: %d\n", n);
    }*/


    writeVTK2(t, currentfield, prefix, rank, lySize, lxSize);
    evolve(processingfield, newfield, lxSize, lxSize*lySize, rank);
    printf("%ld timestep\n",t);
    //usleep(200000);
    

    //SWAP
    double *temp = currentfield;
    currentfield = newfield;
    newfield = temp;
  }
  
  free(currentfield);
  free(newfield);
  free(leftfield);
  free(rightfield);
}
 
int main(int c, char **v) {
  int myrank, size, lxSize, lySize;
  
  lxSize = 5;
  lySize = 2;
  
  if(c > 1) lxSize = atoi(v[1]); // LX_SIZE
  if(c > 2) lySize = atoi(v[2]); // LY_SIZE
  
  MPI_Init( &c, &v );
  

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  if (size % 2 != 0) {
    printf("Die Anwendung kann nur mit einer geraden Anzahl an Prozessen gestartet werden!");
    return 1;
  }

  printf("Initialisierung von Prozess #%d / %d\n", myrank, size);

  int ndims = 1;
  int *dims = malloc(sizeof(int));
  dims[0] = size;
  int *periods = malloc(sizeof(int));
  periods[0] = true;
  MPI_Comm comm_1d;
  MPI_Cart_create(
    MPI_COMM_WORLD, // Kommunikator
    ndims,          // Dimension(en)
    dims,           // Anzahl Prozesse in jeder Dimension
    periods,        // Periodische Ränder?
    false,          // Prozesse in der Anwendung anders anordnen?
    &comm_1d        // neuer Kommunikator
  );
  
  // Width: lxSize
  // Height: lySize
  // Rank: Prozess rank
  // size: Totale Anzahl an Prozessoren
  // Kommunikator: MPI Kommunikator
  game(lxSize, lySize, myrank, size, comm_1d);
  MPI_Finalize();
}
