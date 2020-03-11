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

void show(double* currentfield, int size, int w, int rank) {
  
  int x;
  //printf("\nP%d:\n", rank);
  for (x = 0; x < size; x++) {
    printf(currentfield[x] ? "\033[07m  \033[m" : "  ");//"1" : "0");
    if ((x % w == (w-1))) printf("\n");
  } 
  fflush(stdout);
}

void display(double* currentfield, int size, int w, int rank, int totalProcesses, MPI_Comm comm) {
  
  if (rank == 0) printf("\033[H");
  fflush(stdout);
  for (int i = 0; i < totalProcesses; i++)
  {
    if (rank == i) {
      fflush(stdout);
      show(currentfield, size, w, rank);
    }
    MPI_Barrier(comm);
  }
  fflush(stdout);
}

void writeVTK2(long timestep, double *data, char prefix[1024], int rank, int lySize, int lxSize, MPI_Comm comm) {
  char filename[2048];

  float deltax=1.0;
  long  nxy = lxSize * lySize * sizeof(float);  

  snprintf(filename, sizeof(filename), "vti/%s-%05ld%s", prefix, timestep, ".vti");
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

  for (int x = 0; x < (lxSize*lySize); x++) {
    float value = (float) data[x];
    fwrite((unsigned char*)&value, sizeof(float), 1, fp);
    fflush(fp);
  }
  
  fprintf(fp, "\n</AppendedData>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose(fp);
}

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
 

bool evolve(double* processingfield, double* newfield, int w, int fieldlength, int rank) {
    bool changed = false;
    for (int i = w; i < fieldlength + w; i++) {
      double oldNum = processingfield[i];
      int neighbours_num = coutLifingsPeriodic(processingfield, i, w);
      if(processingfield[i] == 1.0){
      if (neighbours_num < 2) newfield[i-w] = 0.0;
      if (neighbours_num > 3) newfield[i-w] = 0.0;
      if (neighbours_num == 3 || neighbours_num == 2) newfield[i-w] = 1.0;
      }
      else{
        if (neighbours_num == 3) newfield[i-w] = 1.0;
        else newfield[i-w] = 0.0;
      }
      if(oldNum != newfield[i-w]){
        changed = true;
      }
    }
    return changed;
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

    MPI_Send(&currentfield[lxSize*(lySize-1)], lxSize, MPI_DOUBLE, rightrank, 0, communicator);
  }
  // Root wartet auf Daten vom linken Nachbarn, schickt dann Daten an seinen rechten Nachbarn
  if (rank == 0) {
    MPI_Send(&currentfield[lxSize*(lySize-1)], lxSize, MPI_DOUBLE, rightrank, 0, communicator);
    MPI_Recv(leftfield, lxSize, MPI_DOUBLE, leftrank, 0, communicator, &status);  ;
  }

  // Restlichen schicken links, warten dann auf rechts
  if (rank != 0) {
    MPI_Recv(rightfield, lxSize, MPI_DOUBLE, rightrank, 1, communicator, &status);
    MPI_Send(&currentfield[0], lxSize, MPI_DOUBLE, leftrank, 1, communicator);
  }
  // Root wartet auf rechts, schickt dann nach links
  if (rank == 0) {
    MPI_Send(&currentfield[0], lxSize, MPI_DOUBLE, leftrank, 1, communicator);
    MPI_Recv(rightfield, lxSize, MPI_DOUBLE, rightrank, 1, communicator, &status);
  }
}
void handleAbort(MPI_Comm communicator, int rank, bool changed, int timestep){
  int *localChanged = calloc(1, sizeof(int));
  localChanged[0] = changed;
  int *allChanged = calloc(1, sizeof(int));
  allChanged[0] = 0;
  MPI_Reduce(localChanged, allChanged, 1, MPI_INT, MPI_SUM, 0, communicator);
  if (rank == 0) {
    if(allChanged[0] == 0){
      printf("ABORTING ON TIMESTEP: %d, NO CHANGES SINCE LAST TIMESTEP\n", timestep);
      fflush(stdout);
      MPI_Abort(communicator, 0);
    }
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
  bool changed;
  for (t=0;t<TimeSteps;t++) {

    //display(currentfield, lxSize*lySize, lxSize, rank, size, communicator);
    if(t > 0){
      handleAbort(communicator, rank, changed, t);
    }
    usleep(100000);

    writeVTK2(t, currentfield, prefix, rank, lySize, lxSize, communicator);
    processCommunication(currentfield, leftfield, rightfield, lxSize, lySize, rank, communicator);
    memcpy(&processingfield[0], leftfield, lxSize * sizeof(leftfield));
    memcpy(&processingfield[lxSize], currentfield, lxSize * lySize * sizeof(currentfield)); 
    memcpy(&processingfield[lxSize+lxSize*lySize], rightfield, lxSize * sizeof(rightfield));

    changed = evolve(processingfield, newfield, lxSize, lxSize*lySize, rank);
    // SWAP
    double* tmp = currentfield;
    currentfield = newfield;
    newfield = tmp;
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
