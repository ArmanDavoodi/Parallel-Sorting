#include "sequential/sequential_sorts.h"
#include "openmp/omp_sorts.h"
#include "cuda/cuda_sorts.h"

// #define PERFORMANCE_DEBUG // if defined, prints the array after sorting it

constexpr char TEST_PATH[] = "tests/simpleInt(2^15)"; // test file path
constexpr int NRUNS = 10; // number of runs

int* iArray = nullptr;
float* fArray = nullptr;
int* itArray = nullptr;
float* ftArray = nullptr;

int size = 0; 
int useFloat = false;

template<typename Num>
void copyArr(Num* des, Num* src, int N) {
    for (int i = 0; i < N; ++i)
        des[i] = src[i];
}

void loadArray();
void finish();

int main() {
    loadArray();

    #ifdef PERFORMANCE_DEBUG
        printf("array :\n");
        for (int i = 0; i < size; ++i) {
            if (useFloat)
                printf("%f ", fArray[i]);
            else
                printf("%i ", iArray[i]);
        }
        printf("\n\n");
    #endif

    double elapsedTime = 0;
    double pre = 0;
    int min = 0, max = 0;
    printf("sorting %s array of size %d...\n", useFloat ? "Float" : "Int", size);
    // printf("sorting %s array of size %d in range (%d, %d)...\n", useFloat ? "Float" : "Int", size, min, max);
    for (int i = 0; i < NRUNS; ++i) {
        if (useFloat) {
            copyArr(ftArray, fArray, size);
            seq::bubbleSort(ftArray, size, elapsedTime);
        }
        else {
            copyArr(itArray, iArray, size);
            seq::bubbleSort(itArray, size, elapsedTime);
        }
        printf("\tRun number %d completed in %f seconds.\n", i + 1, elapsedTime - pre);
        pre = elapsedTime;
    }
    printf("\nsorting completed in %f seconds in average.\n", elapsedTime / NRUNS);

    #ifdef PERFORMANCE_DEBUG
        printf("\nsorted array :\n");
        for (int i = 0; i < size; ++i) {
            if (useFloat)
                printf("%f ", fArray[i]);
            else
                printf("%i ", iArray[i]);
        }
        printf("\n");
    #endif

    finish();
}

void loadArray() {
    printf("Opening the file \'%s\' for reading...\n", TEST_PATH);
    FILE* fp = fopen(TEST_PATH, "r");
    
    fscanf(fp, "%d", &useFloat);
    fscanf(fp, "%d", &size);

    printf("File opend succesfully.\n \tuseFloat: %s\n\tsize: %d\n", useFloat ? "True" : "False", size);

    printf("loading the array...\n");
    if (useFloat) {
        fArray = new float[size];
        ftArray = new float[size];
    }
    else {
        iArray = new int[size];
        itArray = new int[size];
    }

	for (int i = 0; i < size; ++i)
	{
        if (useFloat) {
            fscanf(fp, "%f", fArray + i);
            ftArray[i] = fArray[i];
        }
        else {
            fscanf(fp, "%d", iArray + i);
            itArray[i] = iArray[i];
        }
	}

	fclose(fp);

	printf("\n#### Array loaded succesfully! ####\n\n");
}

void finish() {
    printf("\nfreeing memory...\n");

    if (iArray != nullptr)
        delete[] iArray;
    if (fArray != nullptr)
        delete[] fArray;
    if (itArray != nullptr)
        delete[] itArray;
    if (ftArray != nullptr)
        delete[] ftArray;
}