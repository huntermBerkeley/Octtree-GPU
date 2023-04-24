#ifndef BETA_ALLOC_WITH_LOCKS 
#define BETA_ALLOC_WITH_LOCKS 
//Betta, the block-based extending-tree thread allocaotor, made by Hunter McCoy (hunter@cs.utah.edu)
//Copyright (C) 2023 by Hunter McCoy

//Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
//and associated documentation files (the "Software"), to deal in the Software without restriction, 
//including without l> imitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
//and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all copies or substantial
// portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
//LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
//IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//The alloc table is an array of uint64_t, uint64_t pairs that store



//inlcudes
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/representations/representation_helpers.cuh>

#include <poggers/hash_schemes/murmurhash.cuh>

#include <poggers/allocators/alloc_utils.cuh>

#include <poggers/allocators/uint64_bitarray.cuh>

#include "stdio.h"
#include "assert.h"
#include <vector>

#include <cooperative_groups.h>

//These need to be enabled for bitarrays
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>




#define BETA_BLOCK_DEBUG 0


namespace beta {

namespace allocators {



//some phase returns ~0 as the lock?
__device__ bool alloc_with_locks(uint64_t & allocation, offset_alloc_bitarr * manager, offset_storage_bitmap * block_storage){

	uint64_t remainder = 0ULL;

	__shared__ warp_lock team_lock;

	while (true){

		cg::coalesced_group grouping = cg::coalesced_threads();

		bool ballot = false;

		if (grouping.thread_rank() == 0){	

			//one thread groups;

			ballot = team_lock.lock();

		}

		if (grouping.ballot(ballot)) break;

		//printf("Team stuck in lock?\n");

	}

	cg::coalesced_group in_lock = cg::coalesced_threads();

	__threadfence();

	// if (in_lock.thread_rank() == 0){
	// 	printf("Team of %d entering the lock\n", in_lock.size());
	// }


	//in lock is coalesced team;
	bool ballot = (block_storage->bit_malloc_v3(in_lock, allocation, remainder));




	#if SLAB_PRINT_DEBUG
	if (ballot && (allocation == ~0ULL)){
		printf("Bug in first malloc, remainder is %llu\n", remainder);
	}

	#endif


	//if 100% of requests are satisfied, we are all returning, so one thread needs to drop lock.
	if ( __popc(in_lock.ballot(ballot)) == in_lock.size()){

		if (in_lock.thread_rank() == 0){

			if (__popcll(remainder) > 0){
				block_storage->attach_buffer(allocation - (allocation % 64), remainder);
			}

			team_lock.unlock();
		}

	}

	if (ballot){
		return true;
	}



	cg::coalesced_group remaining = cg::coalesced_threads();

	// if (remaining.thread_rank() == 0){
	// 	printf("Team of size %d remaining\n", remaining.size());
	// }
	//everyone else now can access the main alloc
	//void * remainder_offset;
	//bool is_leader = false;

	//this can't partially fail - I think.
	//should only ever return teams of 64 or total bust
	bool bit_malloc_result = manager->bit_malloc_v2(remaining, allocation, remainder);



	if (bit_malloc_result){
		#if SLAB_PRINT_DEBUG
		if (!manager->belongs_to_block(allocation)){
			printf("Primary Offset bug.\n");
		}
		#endif

		//uint64_t debug_alloc_bitarr_offset = allocation/4096;

		// offset_alloc_bitarr * alt_bitarr = (offset_alloc_bitarr *) block_allocator->get_mem_from_offset(debug_alloc_bitarr_offset);

		// if (alt_bitarr != bitarr){
		// 	printf("Alt bitarr bug\n");
		// }

	}



	if (remaining.thread_rank() == 0){
	      
		  //only attempt to attach if not empty.
	      if (__popcll(remainder) > 0  && bit_malloc_result){
		      bool result = block_storage->attach_buffer(allocation - (allocation % 64), remainder);

		      #if SLAB_PRINT_DEBUG
		      if (!result){
		      	printf("Failed to attach - this is a bug\n");
		      }
		      #endif



	  		}
	      
	      team_lock.unlock();

	}

	__threadfence();


	return bit_malloc_result;


}


}

}


#endif //End of VEB guard