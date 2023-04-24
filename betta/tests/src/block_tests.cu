/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <poggers/allocators/ext_veb_nosize.cuh>
#include <poggers/allocators/alloc_memory_table.cuh>
#include <poggers/allocators/betta.cuh>

#include <poggers/beta/block.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>


#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace beta::allocators;


__global__ void alloc_all_blocks(block * blocks, uint64_t num_allocs){


   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= num_allocs) return;


   uint64_t blockID = tid/4096;


   uint64_t malloc = blocks[blockID]->block_malloc();




}

__global__ void init_blocks(block * )

//using allocator_type = buddy_allocator<0,0>;

int main(int argc, char** argv) {

   // boot_ext_tree<8ULL*1024*1024, 16ULL>();
 
   // boot_ext_tree<8ULL*1024*1024, 4096ULL>();


   // boot_alloc_table<8ULL*1024*1024, 16ULL>();


   //boot_betta_malloc_free<16ULL*1024*1024, 16ULL, 64ULL>(30ULL*1000*1000*1000);

   //not quite working - get some misses
   one_boot_betta_test_all_sizes<16ULL*1024*1024, 16ULL, 128ULL>(2000ULL*16*1024*1024);

   //betta_alloc_random<16ULL*1024*1024, 16ULL, 128ULL>(2000ULL*16*1024*1024, 100000);

   cudaDeviceReset();
   return 0;

}
