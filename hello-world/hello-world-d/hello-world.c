#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>

int main(int argc, char **argv) {
  #pragma omp parallel sections num_threads(4)
  {
    #pragma omp section 
    {
      printf("Hola mundo from thread %d of %d\n",omp_get_thread_num(), omp_get_num_threads());
    }
    #pragma omp section 
    {
      printf("Hej varlden from thread %d of %d\n",omp_get_thread_num(), omp_get_num_threads());
    }
    #pragma omp section 
    {
      printf("Bonjour tout le monde from thread %d of %d\n",omp_get_thread_num(), omp_get_num_threads());
    }
    #pragma omp section 
    {
      printf("Hallo Welt from thread %d of %d\n",omp_get_thread_num(), omp_get_num_threads());
    }
    #pragma omp section 
    {
      printf("Hello World from thread %d of %d\n",omp_get_thread_num(), omp_get_num_threads());
    }
  }
  return 0;
}
