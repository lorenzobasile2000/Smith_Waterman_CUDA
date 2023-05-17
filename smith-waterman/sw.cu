#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define S_LEN 512
#define N 1000
#define ins -2
#define del -2
#define match -1
#define mismatch -1

#define CHECK(call)                                                     \
{                                                                       \
  const cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                           \
    printf("%s in %s at line %d\n", cudaGetErrorString(err),            \
                                    __FILE__, __LINE__);                \
    exit(EXIT_FAILURE);                                                 \
    }                                                                   \
}
#define CHECK_KERNELCALL()                                              \
{                                                                       \
    const cudaError_t err = cudaGetLastError();                         \
    if (err != cudaSuccess) {                                           \
        printf("%s in %s at line %d\n", cudaGetErrorString(err),        \
                                        __FILE__, __LINE__);            \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

double get_time() // function to get the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int max4(int n1, int n2, int n3, int n4)
{
    int tmp1, tmp2;
    tmp1 = n1 > n2 ? n1 : n2;
    tmp2 = n3 > n4 ? n3 : n4;
    tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
    return tmp1;
}

/*void backtrace(char *simple_rev_cigar, char **dir_mat, int i, int j, int max_cigar_len)
{
    int n;
    for (n = 0; n < max_cigar_len && dir_mat[i][j] != 0; n++)
    {
        int dir = dir_mat[i][j];
        if (dir == 1 || dir == 2)
        {
            i--;
            j--;
        }
        else if (dir == 3)
            i--;
        else if (dir == 4)
            j--;

        simple_rev_cigar[n] = dir;
    }
}*/

__global__ void kernel_gpu(char * query_hw, char * ref_hw, int * res_hw){
    unsigned int threadId = threadIdx.x;
    unsigned int len = SLEN + 1;
    __shared__ int * score[2*len];
    //extern __shared__ int dir[];

    //First of all, I need to initialize the score matrix
    for(int i = 0 ; i< len; i++){
        score[i*threadId] = 0;
    }
    __syncthreads();

    //Compute score alignment
    for(int j = 1; j<2*len; j++){
        if(j>len)
            threadId += len;
        if(threadId<= j){
            unsigned int index = threadId * len + j;
            unsigned int up = index - len;
            unsigned int left = index - 1;
            unsigned int upleft = index - len - 1;

            int tmp1, tmp2;
            int compar = (query_hw[n][j - 1] == ref_hw[n][j - 1]) ? match : mismatch;
            tmp1 = (score[upleft] + compar) > (score[left] +del) ? (score[upleft] + compar) : (score[left]+del);
            tmp2 = (score[up] + ins) > 0 ? (score[up] + ins) : 0;
            tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
            score[index] = tmp1;
            
        }
        __syncthreads();
    }

    //Publish resalt on global memory
    res_hw[blockIdx.x] = score[2*len];
}

int main(int argc, char *argv[])
{
    srand(time(NULL));

    char alphabet[5] = {'A', 'C', 'G', 'T', 'N'};

    char **query = (char **)malloc(N * sizeof(char *));
    for (int i = 0; i < N; i++)
        query[i] = (char *)malloc(S_LEN * sizeof(char));

    char **reference = (char **)malloc(N * sizeof(char *));
    for (int i = 0; i < N; i++)
        reference[i] = (char *)malloc(S_LEN * sizeof(char));

    int **sc_mat = (int **)malloc((S_LEN + 1) * sizeof(int *));
    for (int i = 0; i < (S_LEN + 1); i++)
        sc_mat[i] = (int *)malloc((S_LEN + 1) * sizeof(int));
    /*char **dir_mat = (char **)malloc((S_LEN + 1) * sizeof(char *));
    for (int i = 0; i < (S_LEN + 1); i++)
        dir_mat[i] = (char *)malloc((S_LEN + 1) * sizeof(char));
    */
    int *res = (int *)malloc(N * sizeof(int));
    /*char **simple_rev_cigar = (char **)malloc(N * sizeof(char *));
    for (int i = 0; i < N; i++)
        simple_rev_cigar[i] = (char *)malloc(S_LEN * 2 * sizeof(char));
    */
    int *res_sw, *res_hw;
    char *query_hw, *ref_hw, 
    //char *cigar_hw, *cigar_sw;

    int *res_sw = (int *)malloc(N * sizeof(int));


    // Device memory allocation
    CHECK(cudaMalloc(&res_hw, N*sizeof(int)));
    CHECK(cudaMalloc(&query_hw, N*S_LEN*sizeof(char)));
    CHECK(cudaMalloc(&ref_hw, N*S_LEN*sizeof(char)));
    //CHECK(cudaMalloc(&cigar_hw, N*2*S_LEN*sizeof(char)));


    // randomly generate sequences
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < S_LEN; j++)
        {
            query[i][j] = alphabet[rand() % 5];
            reference[i][j] = alphabet[rand() % 5];
        }
    }

    double start_cpu = get_time();

    for (int n = 0; n < N; n++)
    {
        int max = ins; // in sw all scores of the alignment are >= 0, so this will be for sure changed
        int maxi, maxj;
        // initialize the scoring matrix and direction matrix to 0
        for (int i = 0; i < S_LEN + 1; i++)
        {
            for (int j = 0; j < S_LEN + 1; j++)
            {
                sc_mat[i][j] = 0;
                //dir_mat[i][j] = 0;
            }
        }
        // compute the alignment
        for (int i = 1; i < S_LEN + 1; i++)
        {
            for (int j = 1; j < S_LEN + 1; j++)
            {
                // compare the sequences characters
                int comparison = (query[n][i - 1] == reference[n][j - 1]) ? match : mismatch;
                // compute the cell knowing the comparison result
                int tmp = max4(sc_mat[i - 1][j - 1] + comparison, sc_mat[i - 1][j] + del, sc_mat[i][j - 1] + ins, 0);
                /*char dir;

                if (tmp == (sc_mat[i - 1][j - 1] + comparison))
                    dir = comparison == match ? 1 : 2;
                else if (tmp == (sc_mat[i - 1][j] + del))
                    dir = 3;
                else if (tmp == (sc_mat[i][j - 1] + ins))
                    dir = 4;
                else
                    dir = 0;

                dir_mat[i][j] = dir;*/
                sc_mat[i][j] = tmp;

                if (tmp > max)
                {
                    max = tmp;
                    maxi = i;
                    maxj = j;
                }
            }
        }
        res[n] = sc_mat[maxi][maxj];
        //backtrace(simple_rev_cigar[n], dir_mat, maxi, maxj, S_LEN * 2);
    }

    double end_cpu = get_time();



    // Data transmission: CPU -> GPU
    CHECK(cudaMemcpy(query_hw, query, N*S_LEN*sizeof(char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ref_hw, reference, N*S_LEN*sizeof(char), cudaMemcpyHostToDevice));

    double start_gpu = get_time();
    //Kernel launch
    dim3 blocksPerGrid (N, 1, 1);
    dim3 threadsPerBlock (S_LEN, 1, 1);
    // Change for backtracking
    kernel_gpu<<<blocksPerGrid, threadsPerBlock>>>(query_hw, ref_hw);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
    double end_gpu = get_time();


    //Data transmission: GPU -> CPU
    CHECK(cudaMemcpy(res_sw, res_hw, N*sizeof(char), cudaMemcpyDeviceToHost));
    //CHECK(cudaMemcpy(cigar_sw, cigar_hw, N*2*S_LEN*sizeof(char), cudaMemcpyDeviceToHost));

    //Freeing memory on device
    CHECK(cudaFree(query_hw));
    CHECK(cudaFree(ref_hw));
    //CHECK(cudaFree(cigar_hw));
    CHECK(cudaFree(res_hw));

    //Freeing host memory
    free(query);
    free(reference);
    free(sc_mat);
    free(res);
    free(res_sw);

    for(int i = 0; i< S_LEN; i++)
        if(res_sw[i]!=res[i]){
            printf("GPU result error!\n");
            break;
        }

    printf("SW Time CPU: %.10lf\n", end_cpu - start_cpu);
    printf("SW Time GPU: %.10lf\n", end_gpu - start_gpu);

    return 0;
}