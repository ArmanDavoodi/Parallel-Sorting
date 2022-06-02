#include <stdio.h>
#include <cstring>
#include <random>
#include <algorithm>

// max size is 10^9


using namespace std;

int N = 10, Ni = 0, Nf = 0;
int *arri = nullptr;
float *arrf = nullptr;
bool useFloat = false;
int n_sorted = 0; // should be at least 0 and at most N -> 0 = not sorting at all, N sort the whole array
int int_min = -50000, int_max = 50000;
float float_min = -50000.0f, float_max = 50000.0f;
int repeat = 0; // should be equal or more than 0
int n_shuffle = 0; // should be equal or more than 0
char path[30];

char EXIT[30] = "exit";
char SET_SORTED[30] = "set_n_sorted";
char SET_SHUFFLE[30] = "set_n_shuffle";
char SET_REPEAT[30] = "set_repeat";
char SET_USE_FLOAT[30] = "set_use_float";
char SET_SIZE[30] = "set_size";
char SET_MINI[30] = "set_mini";
char SET_MAXI[30] = "set_maxi";
char SET_MINF[30] = "set_minf";
char SET_MAXF[30] = "set_maxf";
char GET_VARS[30] = "get_vars";
char CLEAR[30] = "clear";
char GENERATE[30] = "generate";
char SAVE[30] = "save";
char LOAD[30] = "load";
char PRINTF[30] = "printf";
char PRINTI[30] = "printi";
char HELP[30] = "help";

void clear();
void fprint();
void iprint();
void help();
void getVars();
void generate();
void save();
void load();


int main() {
    char query[30];
    int number;

    getVars();
    help();

    do {
        scanf("%s", query);
        if (!strcmp(query, SET_SORTED)) {
            scanf("%d", &n_sorted);
        }
        else if (!strcmp(query, SET_SHUFFLE)) {
            scanf("%d", &n_shuffle);
        }
        else if (!strcmp(query, SET_REPEAT)) {
            scanf("%d", &repeat);
        }
        else if (!strcmp(query, SET_USE_FLOAT)) {
            scanf("%d", &number);
            useFloat = (number != 0);
        }
        else if (!strcmp(query, SET_SIZE)) {
            scanf("%d", &N);
        }
        else if (!strcmp(query, SET_MAXI)) {
            scanf("%d", &int_max);
        }
        else if (!strcmp(query, SET_MINI)) {
            scanf("%d", &int_min);
        }
        else if (!strcmp(query, SET_MAXF)) {
            scanf("%f", &float_max);
        }
        else if (!strcmp(query, SET_MINF)) {
            scanf("%f", &float_min);
        }
        else if (!strcmp(query, GET_VARS)) {
            getVars();
        }
        else if (!strcmp(query, CLEAR)) {
            clear();
        }
        else if (!strcmp(query, GENERATE)) {
            generate();
        }
        else if (!strcmp(query, SAVE)) {
            scanf("%s", path);
            save();
        }
        else if (!strcmp(query, LOAD)) {
            scanf("%s", path);
            load();
        }
        else if (!strcmp(query, PRINTI)) {
            iprint();
        }
        else if (!strcmp(query, PRINTF)) {
            fprint();
        }
        else if (!strcmp(query, HELP)) {
            help();
        }
        else if (strcmp(query, "exit")){
            printf("\nError: unindentified query!\n\n");
        }

    } while (strcmp(query, "exit"));

    delete[] arri;
    delete[] arrf;
}

void clear() {
    system("clear");
}

void fprint() {
    printf("arrf : ... \n");
    if (arrf == nullptr) {
        printf("\nError: no array in memory!\n\n");
        return;
    }

    for (int i = 0; i < Nf; ++i) {
        printf("%.3f ", arrf[i]);
    }
    printf("\nDone!\n");
}

void iprint() {
    printf("arri : ... \n");
    if (arri == nullptr) {
        printf("\nError: no array in memory!\n\n");
        return;
    }

    for (int i = 0; i < Ni; ++i) {
        printf("%d ", arri[i]);
    }
    printf("\nDone!\n");
}

void help() {
    printf("Queries:\n");
    printf("\t\'%s\':\texits.\n", EXIT);
    printf("\t\'%s\' <NUMBER> :\tsorts each block of size <NUMBER> while creating the array\n", SET_SORTED);
    printf("\t\'%s\' <NUMBER> :\tshuffles <NUMBER> number of elements randomly at the end\n", SET_SHUFFLE);
    printf("\t\'%s\' <NUMBER> :\trepeats each element k number of times where k is between 0 and <NUMBER>\n", SET_REPEAT);
    printf("\t\'%s\' <NUMBER> :\tgenerates/loads/saves int arr if <NUMBER> is 0 else uses float arr.\n", SET_USE_FLOAT);
    printf("\t\'%s\' <NUMBER>:\tsets array size.\n", SET_SIZE);
    printf("\t\'%s\' <NUMBER>:\tsets upper bound for integers.\n", SET_MAXI);
    printf("\t\'%s\' <NUMBER>:\tsets lower bound for integers.\n", SET_MINI);
    printf("\t\'%s\' <NUMBER>:\tsets upper bound for floats.\n", SET_MAXF);
    printf("\t\'%s\' <NUMBER>:\tsets lower bound for floats.\n", SET_MINF);
    printf("\t\'%s\':\tprints the variables.\n", GET_VARS);
    printf("\t\'%s\':\tgenerates array.\n", GENERATE);
    printf("\t\'%s\' <PATH>:\tsaves array in <PATH>.\n", SAVE);
    printf("\t\'%s\' <PATH>:\tloads array from <PATH>.\n", LOAD);
    printf("\t\'%s\':\tprints the integer array.\n", PRINTI);
    printf("\t\'%s\':\tprints the float array.\n", PRINTF);
    printf("\t\'%s\':\tprints these information again.\n", HELP);
    printf("#############################\n\n");
}

void getVars() {
    printf("Variables:\n");
    printf("\t\'N\': %d\n", N);
    printf("\t\'size of current integer array\': %d\n", Ni);
    printf("\t\'size of current float array\': %d\n", Nf);
    printf("\t\'USE_FLOAT\': %d\n", (int)useFloat);
    printf("\t\'SORTED\': %d\n", n_sorted);
    printf("\t\'number of shuffles\': %d\n", n_shuffle);
    printf("\t\'Repeat\': %d\n", repeat);
    printf("\t\'integer upper bound\': %d\n", int_max);
    printf("\t\'integer lower bound\': %d\n", int_min);
    printf("\t\'float upper bound\': %f\n", float_max);
    printf("\t\'float lower bound\': %f\n", float_min);
    printf("\n");
}

void generate() {
    random_device rd;
	mt19937 gen(rd());
    uniform_real_distribution<> distribf(float_min, float_max);
	uniform_int_distribution<> distribi(int_min, int_max);

    int n_digit = 0;
    for (int i = repeat; i > 0; i /= 10)
        ++n_digit;
    uniform_int_distribution<> random_dom(0, n_digit);
    uniform_int_distribution<> random_index(0, N - 1);

    if (useFloat) {
	    arrf = new float[N];
        Nf = N;
    }
    else {
	    arri = new int[N];
        Ni = N;
    }

    // generate
	for (int i = 0; i < N; ++i)
	{
        int dom = random_dom(gen), lr = 0, ur = 0;
        if (dom != 0) {
            for (ur = 1; dom > 0; --dom)
                ur *= 10;
            lr = ur / 10;
            --lr;
            --ur;
            ur = min(ur, repeat);
        }
        uniform_int_distribution<> random_repeat(lr, ur);
        int r = random_repeat(gen); // number of times to repeat

        if (useFloat) {
            arrf[i] = distribf(gen);
            for(; i < N - 1 && r > 0; ++i, --r)
                arrf[i + 1] = arrf[i];
        }
        else {
            arri[i] = distribi(gen);
            for(; i < N - 1 && r > 0; ++i, --r)
                arri[i + 1] = arri[i];
        }
	}

    // shuffle if not uniform
    if (repeat > 0) {
        for (int i = 0; i < N; ++i)
        {
            int index = random_index(gen);
            if (useFloat) {
                arrf[i] += arrf[index];
                arrf[index] = arrf[i] - arrf[index];
                arrf[i] -= arrf[index];
            }
            else {
                arri[i] += arri[index];
                arri[index] = arri[i] - arri[index];
                arri[i] -= arri[index];
            }
        }
    }

    // sort
    for (int i = 0; i < N; ++i)
	{
        if (useFloat && n_sorted > 0 && (i + 1) % n_sorted == 0)
            sort(arrf + i + 1 - n_sorted, arrf + i + 1);
        else if (!useFloat && n_sorted > 0 && (i + 1) % n_sorted == 0)
            sort(arri + i + 1 - n_sorted, arri + i + 1);
	}

    // shuffle
    for (int i = 0; i < n_shuffle; ++i)
    {
        int i1 = random_index(gen), i2 = random_index(gen);
        if (useFloat) {
            arrf[i1] += arrf[i2];
            arrf[i2] = arrf[i1] - arrf[i2];
            arrf[i1] -= arrf[i2];
        }
        else {
            arri[i1] += arri[i2];
            arri[i2] = arri[i1] - arri[i2];
            arri[i1] -= arri[i2];
        }
    }
	
	printf("\n#### Random array generated succesfully! ####\n\n");   
}

void save() {
    FILE* fp = fopen(path, "w+");
    int n;
    if (useFloat)
	    n = Nf;
    else
        n = Ni;
    
    fprintf(fp, "%d\n", n);

	for (int i = 0; i < n; ++i)
	{
        if (useFloat)
            fprintf(fp, i == n - 1 ? "%f\n" : "%f ", arrf[i]);
        else
            fprintf(fp, i == n - 1 ? "%d\n" : "%d ", arri[i]);
		
	}

	fclose(fp);

	printf("\n#### Array saved succesfully! ####\n\n");
}

void load() {
    FILE* fp = fopen(path, "r");
    int n;
    
    fscanf(fp, "%d", &n);

	for (int i = 0; i < n; ++i)
	{
        if (useFloat)
            fscanf(fp, "%f", arrf + i);
        else
            fscanf(fp, "%d", arri + i);
	}

    if (useFloat)
	    Nf = n;
    else
        Ni = n;

	fclose(fp);

	printf("\n#### Array loaded succesfully! ####\n\n");
}