#ifndef PERFORMANCE_HEAD
#define PERFORMANCE_HEAD

#include <iostream>
#include <string>

#include "sequential/sequential_sorts.h"
#include "openmp/omp_sorts.h"
#include "cuda/cuda_sorts.h"

#define PERFORMANCE_DEBUG // if defined, prints the array after sorting it

// parallel sorts utils
namespace ps_utils {
    // algorithms
    constexpr int ID_NONE = 0;
    constexpr int ID_SEQ_INSERTION = 1;
    constexpr int ID_SEQ_BUBBLE = 2;
    constexpr int ID_SEQ_ODD_EVEN = 3;
    constexpr int ID_SEQ_MERGE = 4;
    constexpr int ID_SEQ_BITONIC = 5;
    constexpr int ID_SEQ_ODD_EVEN_MERGE = 6;
    constexpr int ID_SEQ_COUNT = 7;
    constexpr int ID_CPAR_ODD_EVEN = 8;
    constexpr int ID_CPAR_MERGE = 9;
    constexpr int ID_CPAR_BITONIC = 10;
    constexpr int ID_CPAR_ODD_EVEN_MERGE = 11;
    constexpr int ID_CPAR_COUNT = 12;
    constexpr int ID_GPAR_ODD_EVEN = 13;
    constexpr int ID_GPAR_MERGE = 14;
    constexpr int ID_GPAR_BITONIC = 15;
    constexpr int ID_GPAR_ODD_EVEN_MERGE = 16;
    constexpr int ID_GPAR_COUNT = 17;

    constexpr int ID_SEQ_BEG = 1;
    constexpr int ID_CPAR_BEG = 8;
    constexpr int ID_GPAR_BEG = 13;

    // sorts
    constexpr char INSERTION[] = "Insertion Sort";
    constexpr char BUBBLE[] = "Bubble Sort";
    constexpr char ODD_EVEN[] = "Odd-Even(Brick) Sort";
    constexpr char MERGE[] = "Merge Sort";
    constexpr char BITONIC[] = "Bitonic Sort";
    constexpr char ODD_EVEN_MERGE[] = "Batcher Odd-Even Merge Sort";
    constexpr char COUNT[] = "Counting Sort";

    // run type
    constexpr char SEQUENTIAL[] = "Sequential";
    constexpr char CPU_PAR[] = "CPU-Parallel (OpenMP)";
    constexpr char GPU_PAR[] = "GPU-Parallel (CUDA)";

    // menu
    constexpr char BACK[] = "Back";
    constexpr char EXIT[] = "Exit";

    // Constant Variables
    constexpr char TEST_PATH[] = ""; // test file path
    constexpr int NRUNS = 1; // number of runs

    // Variables
    int* iArray = nullptr;
    float* fArray = nullptr;
    int size = 0; 
    bool useFloat = false;
}

void initMenuOptions(std::string runTypes[], std::string seqSortTypes[], std::string parSortTypes[]);
int printMenu(std::string header, std::string options[], int numOfOptions);
int menu();
void loadArray();
void finish();

#endif