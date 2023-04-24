#ifndef BETA_MEM_POOL
#define BETA_MEM_POOL

//dummy mem pool for benchmarking.
//exposes the same functions as the allocator.



//inlcudes
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <poggers/allocators/alloc_utils.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/allocators/ext_veb_nosize.cuh>
#include <poggers/allocators/alloc_memory_table.cuh>
#include <poggers/allocators/one_size_allocator.cuh>

#include <poggers/allocators/offset_slab.cuh>

#include <poggers/allocators/block_storage.cuh>



#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif


namespace poggers {

namespace allocators {

#define REQUEST_BLOCK_MAX_ATTEMPTS 1

//alloc table associates chunks of memory with trees

//using uint16_t as there shouldn't be that many trees.

//register atomically inserst tree num, or registers memory from segment_tree.

using namespace poggers::utils;


template <int num_counters>
struct mem_pool {


	using my_type = mem_pool;
	
	uint64_t counters[num_counters];
	uint64_t max_allocs;
	uint64_t size;

	char * memory;



	static __host__ my_type * generate_on_device(uint64_t max_bytes, uint64_t alloc_size){


		my_type * host_version = get_host_version<my_type>();


		//plug in to get max chunks

		uint64_t max_allocs = max_bytes/alloc_size;


		uint64_t allocs_per_counter = (max_allocs-1)/num_counters+1;


		uint64_t bytes_to_allocate = allocs_per_counter*num_counters*alloc_size;

		//uint64_t max_chunks = get_max_chunks<bytes_per_segment>(max_bytes);

		//host_version->segment_tree = veb_tree::generate_on_device(max_chunks, seed);

		// one_size_allocator::generate_on_device(max_chunks, bytes_per_segment, seed);

		char * extra_memory;

		cudaMalloc((void **)&extra_memory, bytes_to_allocate);


		//uint64_t * ext_counters = get_host_version<uint64_t>(num_counters);

		for (int i = 0; i < num_counters; i++){
			host_version->counters[i] = 0;
		}

		 //= move_to_device<uint64_t>(ext_counters, num_counters);
		host_version->max_allocs = allocs_per_counter;
		host_version->memory = extra_memory;
		host_version->size = alloc_size;

		return move_to_device(host_version);

	}


	static __host__ void free_on_device(my_type * dev_version){

		//this frees dev version.
		my_type * host_version = move_to_host<my_type>(dev_version);


		cudaFree(host_version->memory);
		//cudaFree(host_version->counters);

		cudaFreeHost(host_version);

		return;


	}

	__device__ void * malloc(){


		uint64_t my_counter = threadIdx.x % num_counters;

		//uint64_t my_counter = 0;

		uint64_t my_count = atomicAdd((unsigned long long int *)&counters[my_counter], 1ULL);

		

		if (my_count >= max_allocs){

			//printf("Returning nullptr")
			return nullptr;
		}


		uint64_t relative_offset = my_counter*max_allocs+my_count;

		//printf("thread %llu, My count %llu my counter %llu, my offset %llu\n", threadIdx.x+blockIdx.x*blockDim.x, my_count, my_counter, relative_offset);

		uint64_t byte_offset = relative_offset*size;

		return (void *) (memory + byte_offset);

	}


	__device__ void free(void * alloc){
		return;
	}


	__device__ uint64_t get_offset_from_ptr(void * ext_ptr){

		char * pointer = (char *) ext_ptr;

		uint64_t raw_offset = (pointer - memory);

		return raw_offset/size;

	}

};



}

}


#endif //End of VEB guard