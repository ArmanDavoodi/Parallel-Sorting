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
bool uniform_sort = true;
float n_sorted = 0.0; //ratio: should be at least 0 and at most 1 -> 0 = not sorting at all, 1 sort the whole array
int int_min = -50000, int_max = 50000;
float float_min = -50000.0f, float_max = 50000.0f;
float repeat = 0.0; // ratio: should be between 0 and 1
float n_shuffle = 0.0; //ratio: should be between 0 and 1
float sort_prob = 1.0; // the probability of sorting in each iteration
float sort_asc = 1.0; // the probability of sorting in ascending order
char path[30];

char EXIT[30] = "exit";
char SET_SORTED[30] = "set_n_sorted";
char SET_EQUAL_BLOCK_SORT[30] = "set_uniform_sort";
char SET_SORT_PROB[30] = "set_sort_prob";
char SET_SORT_ASC[30] = "set_sort_asc";
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
            scanf("%f", &n_sorted);
        }
        else if (!strcmp(query, SET_SHUFFLE)) {
            scanf("%f", &n_shuffle);
        }
        else if (!strcmp(query, SET_REPEAT)) {
            scanf("%f", &repeat);
        }
        else if (!strcmp(query, SET_SORT_PROB)) {
            scanf("%f", &sort_prob);
        }
        else if (!strcmp(query, SET_SORT_ASC)) {
            scanf("%f", &sort_asc);
        }
        else if (!strcmp(query, SET_USE_FLOAT)) {
            scanf("%d", &number);
            useFloat = (number != 0);
        }
        else if (!strcmp(query, SET_EQUAL_BLOCK_SORT)) {
            scanf("%d", &number);
            uniform_sort = (number != 0);
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
    printf("\t\'%s\' <NUMBER> :\tsorts each block of size <NUMBER>  * Size while creating the array\n", SET_SORTED);
    printf("\t\'%s\' <NUMBER> :\tshuffles <NUMBER> * Size number of elements randomly at the end\n", SET_SHUFFLE);
    printf("\t\'%s\' <NUMBER> :\trepeats each element k number of times where k is between 0 and <NUMBER>  * Size\n", SET_REPEAT);
    printf("\t\'%s\' <NUMBER> :\tgenerates int arr if <NUMBER> is 0 else uses float arr.\n", SET_USE_FLOAT);
    printf("\t\'%s\' <NUMBER> :\tsorts equal sized block of data if not 0 else sorts random block sizes.\n", SET_EQUAL_BLOCK_SORT);
    printf("\t\'%s\' <NUMBER> :\tsorts each block with probability of <NUMBER>\n", SET_SORT_PROB);
    printf("\t\'%s\' <NUMBER> :\twhen sorting, sorts ascending with probability of <NUMBER>\n", SET_SORT_ASC);
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
    printf("\t\'UNIFORM_SORT\': %d\n", (int)uniform_sort);
    printf("\t\'sort probability\': %f\n", sort_prob);
    printf("\t\'sort acsending probability\': %f\n", sort_asc);
    printf("\t\'N_SORTED\': %f\n", n_sorted);
    printf("\t\'number of shuffles\': %f\n", n_shuffle);
    printf("\t\'Repeat\': %f\n", repeat);
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
    for (int i = (int)(repeat * N); i > 0; i /= 10)
        ++n_digit;
    uniform_int_distribution<> random_dom(0, n_digit);

    n_digit = 0;
    for (int i = (int)(n_sorted * N); i > 0; i /= 10)
        ++n_digit;
    uniform_int_distribution<> n_uniform_sort(0, n_digit);

    uniform_int_distribution<> random_index(0, N - 1);
    uniform_real_distribution<> prob(0.0, 1.0);

    if (useFloat) {
        if (arrf != nullptr)
            delete[] arrf;
	    arrf = new float[N];
        Nf = N;
    }
    else {
        if (arri != nullptr)
            delete[] arri;
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
            ur = min(ur, (int)(repeat * N));
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
    if (n_sorted > 0) {
        int last_index = 0;
        for (int i = 0; i < N; ++i)
        {
            int n_sort_dom = n_uniform_sort(gen), ls = 0, us = 0;
            if (n_sort_dom != 0) {
                for (us = 1; n_sort_dom > 0; --n_sort_dom)
                    us *= 10;
                ls = us / 10;
                --ls;
                --us;
                us = min(us, (int)(n_sorted * N));
            }
            uniform_int_distribution<> random_sort_block_size(ls, us);
            int size;
            int block_seen = 0;

            if (uniform_sort) {
                if (useFloat && (i + 1) % (int)(n_sorted * N) == 0 && prob(gen) <= sort_prob) {
                    if (prob(gen) <= sort_asc)
                        sort(arrf + i + 1 - (int)(n_sorted * N), arrf + i + 1, less<float>());
                    else
                        sort(arrf + i + 1 - (int)(n_sorted * N), arrf + i + 1, greater<float>());
                }
                else if (!useFloat && (i + 1) % (int)(n_sorted * N) == 0 && prob(gen) <= sort_prob) {
                    if (prob(gen) <= sort_asc)
                        sort(arri + i + 1 - (int)(n_sorted * N), arri + i + 1, less<int>());
                    else
                        sort(arri + i + 1 - (int)(n_sorted * N), arri + i + 1, greater<int>());
                }
                last_index = i + 1;
            }
            else {
                if (block_seen == 0)
                    size = random_sort_block_size(gen);
                
                if (useFloat && block_seen == size) {
                    if (prob(gen) <= sort_prob) {
                        if (prob(gen) <= sort_asc)
                            sort(arrf + last_index, arrf + i + 1, less<float>());
                        else
                            sort(arrf + last_index, arrf + i + 1, greater<float>());
                    }
                    block_seen = -1;
                    last_index = i + 1;
                }
                else if (!useFloat && block_seen == size) {
                    if (prob(gen) <= sort_prob) {
                        if (prob(gen) <= sort_asc)
                            sort(arri + last_index, arri + i + 1, less<int>());
                        else
                            sort(arri + last_index, arri + i + 1, greater<int>());
                    }
                    block_seen = -1;
                    last_index = i + 1;
                }
                
                ++block_seen;
            }
        }
        if (last_index < N && prob(gen) <= sort_prob) {
            if (useFloat) {
                if (prob(gen) <= sort_asc)
                    sort(arrf + last_index, arrf + N, less<float>());
                else
                    sort(arrf + last_index, arrf + N, greater<float>());
            }
            else {
                if (prob(gen) <= sort_asc)
                    sort(arri + last_index, arri + N, less<int>());
                else
                    sort(arri + last_index, arri + N, greater<int>());
            }
        }
    }

    // shuffle
    for (int i = 0; i < (int)(n_shuffle * N); ++i)
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
    
    fprintf(fp, "%d\n", (int)useFloat);
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
    int n, inUseFloat;
    
    fscanf(fp, "%d", &inUseFloat);
    fscanf(fp, "%d", &n);

	for (int i = 0; i < n; ++i)
	{
        if (inUseFloat)
            fscanf(fp, "%f", arrf + i);
        else
            fscanf(fp, "%d", arri + i);
	}

    if (inUseFloat)
	    Nf = n;
    else
        Ni = n;

	fclose(fp);

	printf("\n#### Array loaded succesfully! ####\n\n");
}