#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define S_LEN 512
#define N 1000
#define ins -2
#define del -2
#define match 1
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

void backtrace(char *simple_rev_cigar, char **dir_mat, int i, int j, int max_cigar_len)
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
}

__global__ void kernel_gpu(char * query_hw, char * ref_hw, int * res_hw, char * rev_cigar_hw, char *dir_mat_gpu){
    unsigned int threadId = threadIdx.x;
    __shared__ int last_score[S_LEN+1];
    __shared__ int pre_score[S_LEN+1];
    unsigned int iteration = 0;
    __shared__ int max[S_LEN];
    __shared__ int maxi[S_LEN];
    __shared__ int maxj[S_LEN];

    //First of all, I need to initialize the two matrix diagonals
    pre_score[threadId] = 0;
    last_score[threadId] = 0;
    max[threadId] = 0;
    if(threadId==0){
        pre_score[S_LEN] = 0;
        last_score[S_LEN] = 0;
    }

    for (int j = 0; j < S_LEN + 1; j++)
    {
        dir_mat_gpu[blockIdx.x * (S_LEN+1)*(S_LEN+1) + threadId * (S_LEN + 1) + j] = 0;
        if (threadId == 0)
        {
            dir_mat_gpu[blockIdx.x * (S_LEN+1)*(S_LEN+1) + S_LEN * (S_LEN + 1) + j] = 0;
        }
    }
    __syncthreads();

    //Compute score alignment
    for(int j = 0; j<2*S_LEN-1; j++){
        unsigned int tmp = 0;
        if(threadId<=j && iteration <S_LEN){
            unsigned int up = threadId+1;
            unsigned int left = threadId;
            unsigned int upleft = threadId;
            unsigned int ind_q = blockIdx.x * S_LEN + threadId;
            unsigned int ind_ref = blockIdx.x * S_LEN + iteration ;
        

            int tmp1, tmp2;
            int compar = (query_hw[ind_q] == ref_hw[ind_ref]) ? match : mismatch;
            
            //Compute the maximum
            tmp1 = (pre_score[upleft] + compar) > (last_score[left] +del) ? (pre_score[upleft] + compar) : (last_score[left]+del);
            tmp2 = (last_score[up] + ins) > 0 ? (last_score[up] + ins) : 0;
            tmp = tmp1 > tmp2 ? tmp1 : tmp2;
            
            char dir;

            if (tmp == (pre_score[upleft] + compar))
                dir = compar == match ? 1 : 2;
            else if (tmp == (last_score[left] + del))
                dir = 3;
            else if (tmp == (last_score[up] + ins))
                dir = 4;
            else
                dir = 0;

            dir_mat_gpu[blockIdx.x*(S_LEN+1)*(S_LEN+1)+(iteration)*(S_LEN+1)+(threadId)] = dir;
    
            if(tmp>max[threadId]){
                max[threadId] = tmp;
                maxi[threadId] = iteration;
                maxj[threadId] = threadId;
            }
            
            iteration++;         
        }
        __syncthreads();
        pre_score[threadId+1] = last_score[threadId+1];
        __syncthreads();
        last_score[threadId+1] = tmp;
        
        __syncthreads();
    }
    
    __syncthreads();
    //Publish result on global memory and compute backtracking
    if(threadId==0){
        int max_f = max[0];
        int maxi_f;
        int maxj_f;
        for(int i = 1; i<S_LEN; i++)
            if(max_f<max[i]){
                max_f = max[i];
                maxi_f = maxi[i];
                maxj_f = maxj[i];
            }

        res_hw[blockIdx.x] = max_f;
        
        //Backtrace
        for (int n = 0; n < 2*S_LEN && dir_mat_gpu[blockIdx.x*(S_LEN+1)*(S_LEN+1)+maxi_f*(S_LEN+1)+maxj_f] != 0; n++)
        {
            char dir = dir_mat_gpu[blockIdx.x*(S_LEN+1)*(S_LEN+1)+maxi_f*(S_LEN+1)+maxj_f];
            if (dir == 1 || dir == 2)
            {
                maxi_f--;
                maxj_f--;
            }
            else if (dir == 3)
                maxj_f--;
            else if (dir == 4)
                maxi_f--;

            rev_cigar_hw[blockIdx.x * 2*S_LEN + n] = dir;
        }
    }
        
}

int main(int argc, char *argv[])
{
    srand(time(NULL));

    char alphabet[5] = {'A', 'C', 'G', 'T', 'N'};

    char **query = (char **)malloc(N * sizeof(char *));
    char * query_gpu = (char *)malloc(N*S_LEN*sizeof(char));
    for (int i = 0; i < N; i++)
        query[i] = (char *)malloc(S_LEN * sizeof(char));

    char **reference = (char **)malloc(N * sizeof(char *));
    char * reference_gpu = (char *)malloc(N*S_LEN*sizeof(char));
    for (int i = 0; i < N; i++)
        reference[i] = (char *)malloc(S_LEN * sizeof(char));

    int **sc_mat = (int **)malloc((S_LEN + 1) * sizeof(int *));
    for (int i = 0; i < (S_LEN + 1); i++)
        sc_mat[i] = (int *)malloc((S_LEN + 1) * sizeof(int));
    char **dir_mat = (char **)malloc((S_LEN + 1) * sizeof(char *));
    for (int i = 0; i < (S_LEN + 1); i++)
        dir_mat[i] = (char *)malloc((S_LEN + 1) * sizeof(char));
    int *res = (int *)malloc(N * sizeof(int));
    char **simple_rev_cigar = (char **)malloc(N * sizeof(char *));
    for (int i = 0; i < N; i++)
        simple_rev_cigar[i] = (char *)malloc(S_LEN * 2 * sizeof(char));
    int *res_hw;
    char *query_hw, *ref_hw;
    char *cigar_hw, *dir_mat_hw;

    char * cigar_sw = (char*)malloc(N*S_LEN*2*sizeof(char));
    int *res_sw = (int *)malloc(N * sizeof(int));


    // Device memory allocation
    CHECK(cudaMalloc(&res_hw, N*sizeof(int)));
    CHECK(cudaMalloc(&query_hw, N*S_LEN*sizeof(char)));
    CHECK(cudaMalloc(&ref_hw, N*S_LEN*sizeof(char)));
    CHECK(cudaMalloc(&cigar_hw, N*2*S_LEN*sizeof(char)));
    CHECK(cudaMalloc(&dir_mat_hw,N*(S_LEN+1)*(S_LEN+1)*sizeof(char)));
    //CHECK(cudaMalloc(&max, N*sizeof(int)))

    // randomly generate sequences
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < S_LEN; j++)
        {
            query[i][j] = alphabet[rand() % 5];
            query_gpu[i*S_LEN+j] = query[i][j];
            reference[i][j] = alphabet[rand() % 5];
            reference_gpu[i*S_LEN+j] = reference[i][j];
        }
    }



    /*printf("query\n");
    for (int j = 0; j < S_LEN; j++)
        {
            printf("%c", query[0][j]);
        }
    printf("\nref\n");
    for (int j = 0; j < S_LEN; j++)
        {
            printf("%c", reference[0][j]);
        }    
    printf("\n");  */
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
                dir_mat[i][j] = 0;
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
                char dir;

                if (tmp == (sc_mat[i - 1][j - 1] + comparison))
                    dir = comparison == match ? 1 : 2;
                else if (tmp == (sc_mat[i - 1][j] + del))
                    dir = 3;
                else if (tmp == (sc_mat[i][j - 1] + ins))
                    dir = 4;
                else
                    dir = 0;

                dir_mat[i][j] = dir;
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
        backtrace(simple_rev_cigar[n], dir_mat, maxi, maxj, S_LEN * 2);
    }

    double end_cpu = get_time();



    // Data transmission: CPU -> GPU
    CHECK(cudaMemcpy(query_hw, query_gpu, N*S_LEN*sizeof(char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ref_hw, reference_gpu, N*S_LEN*sizeof(char), cudaMemcpyHostToDevice));

    double start_gpu = get_time();
    //Kernel launch
    dim3 blocksPerGrid (N, 1, 1);
    dim3 threadsPerBlock (S_LEN, 1, 1);
    // Change for backtracking
    kernel_gpu<<<blocksPerGrid, threadsPerBlock>>>(query_hw, ref_hw, res_hw, cigar_hw, dir_mat_hw);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
    double end_gpu = get_time();


    //Data transmission: GPU -> CPU
    CHECK(cudaMemcpy(res_sw, res_hw, N*sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cigar_sw, cigar_hw, N*2*S_LEN*sizeof(char), cudaMemcpyDeviceToHost));

    //Freeing memory on device
    CHECK(cudaFree(query_hw));
    CHECK(cudaFree(ref_hw));
    CHECK(cudaFree(cigar_hw));
    CHECK(cudaFree(res_hw));
    CHECK(cudaFree(dir_mat_hw));

    

    for(int i = 0; i< N; i++){
        if(res_sw[i]!=res[i]){
            printf("GPU result error on block %d, because cpu = %d and gpu = %d !\n", i, res[i], res_sw[i]);
            break;
        }
        for(int j=0; j<2*S_LEN; j++)
            if(simple_rev_cigar[i][j]!= cigar_sw[i*S_LEN*2 + j]){
                printf("GPU result error during backtracking on block %d, because cpu = %d and gpu = %d !\n", i, simple_rev_cigar[i][j], cigar_sw[i*S_LEN*2+j]);
                break;
            }
    }

    printf("SW Time CPU: %.10lf\n", end_cpu - start_cpu);
    printf("SW Time GPU: %.10lf\n", end_gpu - start_gpu);

    /*printf("CPU: \n");
    for(int i=0; i<2*S_LEN; i++)
      printf("%d ", simple_rev_cigar[0][i]);

    printf("\nGPU: \n");
    for(int i=0; i<2*S_LEN; i++)
      printf("%d ", cigar_sw[i]);*/

    //Freeing host memory
    free(query);
    free(reference);
    free(sc_mat);
    free(res);
    free(res_sw);    
    
    return 0;
}