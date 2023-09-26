// it is used to avoid VS intellisense warning for calling cuda kernel_add function 
#ifdef __INTELLISENSE__
#define CUDA_KERNEL_ARGS(...)
#define __CUDACC__
#else
#define CUDA_KERNEL_ARGS(...) <<< __VA_ARGS__ >>>
#endif


#include <iostream>
#include <iterator>
#include <algorithm>
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <assert.h>


constexpr auto required_block_dim = 20;

constexpr auto blockDIM = std::min<int>(required_block_dim, 16);

struct dp_table {
	// mark the table content as `volatile`, the volatile load/store rules will be applied
	// this situation caused a amazing problem: printing in cuda or debugging with break points can promise this program caculate out the result as right situation.
	volatile int* dev_DP_table_ptr;
	const int weight, height;
	const char* dev_str1;
	const char* dev_str2;

	__device__ bool update_when_available(int i,int j) {
		int skip_a = get_value(i - 1, j);
		int skip_b = get_value(i, j - 1);
		int skip_middle = get_value(i - 1, j - 1);
		//printf("[%d,%d]: %d %d %d\n", i, j, skip_a,skip_a,skip_middle);

		//no result
		if (skip_a == -1 || skip_b == -1 || skip_middle == -1) {
			return false;
		}

		int take_both = skip_middle + (dev_str1[i] == dev_str2[j]);
		//update
		set_value(i,j,max(take_both, max(skip_a, skip_b)));
		return true;
	}
	__device__ int get_value(int i, int j) {
		if(i < 0 || j < 0) return 0;
		assert(j < weight);
		assert(i < height);
		return dev_DP_table_ptr[i * weight + j];
	}
	__device__ void set_value(int i, int j, int v) {
		dev_DP_table_ptr[i * weight + j] = v;

	}
};


__global__ void lcs_kernel(dp_table dp) {
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < dp.height && j < dp.weight) {

		//spinning before updated
		while (true) {
			if (dp.update_when_available(i, j)) break;
		}

	}

	
}

auto main() -> int {

	unsigned int str1_len,str2_len;
	std::cin >> str1_len >> str2_len;

	char* str1_arr_0copy;
	cudaHostAlloc( (void**)&str1_arr_0copy,sizeof(char)*str1_len, cudaHostAllocMapped);

	char* str2_arr_0copy;
	cudaHostAlloc((void**)&str2_arr_0copy, sizeof(char) * str2_len, cudaHostAllocMapped);


	// When reading characters, istream_iterator skips whitespace by default 
	std::copy_n(std::istream_iterator<char>(std::cin), str1_len, str1_arr_0copy);

	std::copy_n(std::istream_iterator<char>(std::cin), str2_len, str2_arr_0copy);


	// create DP table
	int* dev_DP_table;
	cudaMalloc((void**)&dev_DP_table, str1_len * str2_len * sizeof(int));
	//fill the table with -1 to mark the unit is not result
	cudaMemset((void*)dev_DP_table, -1, str1_len * str2_len * sizeof(int));


	dim3 threads{ blockDIM,blockDIM };
	// those not aligned weight string may cannot caculate out every value,so we need  +1 to start more thread
	dim3 blocks{ str1_len / blockDIM + 1,str2_len / blockDIM + 1 };
	char*dev_str1_arr,*dev_str2_arr;
	cudaHostGetDevicePointer(&dev_str1_arr, str1_arr_0copy, 0);
	cudaHostGetDevicePointer(&dev_str2_arr, str2_arr_0copy, 0);

	dp_table dp{dev_DP_table, str1_len,str2_len , dev_str1_arr ,dev_str2_arr};


	lcs_kernel CUDA_KERNEL_ARGS(blocks, threads)(dp);

	int output;
	//copy the result in last unit from device memory to host memory
	cudaMemcpy((void*)&output, (void*)(dp.dev_DP_table_ptr + str1_len * str2_len - 1), 1 * sizeof(int), cudaMemcpyDeviceToHost);

	std::cout << output<<std::endl;
	cudaFree(dev_DP_table);
	cudaFreeHost(str1_arr_0copy);
	cudaFreeHost(str2_arr_0copy);
	return 0;
}