#ifndef BETA_THREAD_STORAGE
#define BETA_THREAD_STORAGE
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



//thread-local storage.
//one of these exists attached to each SM, and threads can reference it for the lifetime of the SM.

//Correctness precondition
//0000000000000000 is empty key
//if you create it you *will* destroy it
//so other threads don't touch blocks that show themselves as 0ULL
//This allows it to act as the intermediate state of blocks
//and allows the remove pipeline to be identical to above ^
//as we first remove and then re-add if there are leftovers.
struct thread_storage {


	//claim bits are 1 if available to claim for store
	uint64_t_bitarr claim_bits;

	//manager bits are 1 if available for malloc
	uint64_t_bitarr manager_bits;
	uint64_t_bitarr alloc_bits[64];
	uint64_t memmap[64];


	__device__ void init(){


		claim_bits.bits = ~0ULL;
		manager_bits.bits = (0ULL);
		for (int i=0; i< 64; i++){
			alloc_bits[i].bits = (0ULL);
			memmap[i] = ~0ULL;
		}


		

	}


	//move a captured buffer into the pool
	//this has to move in 3 phases
	//1) set buffer pointer from empty
	//2) set remaining allocations
	// __threadfence();
	//3) set manager bit
	__device__ bool attach_buffer(uint64_t ext_buffer, uint64_t ext_bits){


		//group
		//cg::coalesced_group active_threads = cg::coalesced_threads();

		//team shares the load
		//uint64_t_bitarr local_copy = claim_bits;

		#if SLAB_PRINT_DEBUG
		printf("Attaching buffer %llu with bits %llx\n", ext_buffer, ext_bits);
		#endif

		while (claim_bits.get_fill() != 0){

			

			
			//printf("Available for claim: %llx\n", claim_bits);


			//allocation_index_bit = local_copy.get_first_active_bit();

			int allocation_index_bit = claim_bits.get_random_active_bit();


	
			//printf("%d: Bit chosen is %d / %llx, %llx %llx\n", threadIdx.x/32, allocation_index_bit, manager_bits, alloc_bits[allocation_index_bit], memmap[allocation_index_bit]);


			//can only swap out if memory is set to 0xffff*8 ...

			if (claim_bits.unset_index(allocation_index_bit) & SET_BIT_MASK(allocation_index_bit)){


				//printf("%d claimed bit %d\n", threadIdx.x/32, allocation_index_bit);


				if (atomicCAS((unsigned long long int *)&memmap[allocation_index_bit], ~0ULL, ext_buffer) == ~0ULL){

						uint64_t swap_bits = alloc_bits[allocation_index_bit].swap_bits(ext_bits);

						if (swap_bits == 0ULL){

							__threadfence();

							if (~(manager_bits.set_index(allocation_index_bit) & SET_BIT_MASK(allocation_index_bit))){

								#if DEBUG_PRINTS
								printf("Manager bit set!\n");
								#endif


								#if SLAB_PRINT_DEBUG
								printf("Allocation bit %d set\n", allocation_index_bit);
								#endif

								return true;

							} else {
								//if you swap out you *must* succeed
								printf("Failure attaching buffer\n");
								assert(1==0);
							}


						} else {
							printf("Memory was set but buffer not empty - This is a bug\n");
							printf("Old memory is %lx\n", swap_bits);
							assert(1==0);
						}


					} else {
						printf("Memmap set failed - failure to properly reset\n");
						assert(1==0);
					}


			}




			//local_copy = manager_bits.global_load_this();
			claim_bits.global_load_this();


		}


		return false;


	}


	//exclusively malloc a section
	//this should lock the section and claim as much as possible.
	//lets just make this cycle! - atomically unset to claim and reset
	//this is important because it lets a team "claim" an allocation - with a gurantee that if they were not satisfied they have now opened a space.
	// with < 64 teams this will always work.
	__device__ bool malloc(cg::coalesced_group & active_threads, uint64_t & offset, uint64_t & remainder){


		#if SLAB_GLOBAL_LOADING
		manager_bits.global_load_this();
		#endif

		//unloading lock bit is a mistake.

		int upper_index;

		while (active_threads.thread_rank() == 0){

			upper_index = manager_bits.get_random_active_bit();

			if (upper_index == -1) break;


			if (manager_bits.unset_index(upper_index) & SET_BIT_MASK(upper_index)) break;

			// if (active_threads.thread_rank() == 0){
			// 	printf("Warp %d has failed to claim index %d!\n", threadIdx.x/32, upper_index);
			// }

			//printf("Stuck in bit malloc secure lock\n");

		}


		upper_index = active_threads.shfl(upper_index, 0);

		if (upper_index == -1) return false;

		//team succeeds or fails together, original team is fine
		alloc_bits[upper_index].global_load_this();

		uint64_t_bitarr bits;
		uint64_t mapping;

		if (active_threads.thread_rank() == 0){
			bits = alloc_bits[upper_index].swap_to_empty();
			mapping = atomicExch((unsigned long long int *)&memmap[upper_index], ~0ULL);

			#if SLAB_PRINT_DEBUG
			printf("Mapping is %llu\n", mapping);
			#endif

		}

		bits = active_threads.shfl(bits,0);
		mapping = active_threads.shfl(mapping, 0);

		int my_index = select_unique_bit(active_threads, bits);

		if (!check_indices(active_threads, my_index)){
			printf("Team %d with %d threads, Bug in select unique main storage index %d\n", threadIdx.x/32, active_threads.size(), my_index);
		}




		remainder = bits;

		offset = mapping + my_index;

		if (active_threads.thread_rank() == 0){
			//atomicExch((unsigned long long int *)&memmap[upper_index], ~0ULL);
			claim_bits.set_index(upper_index);
		}

		return (my_index != -1);

	}


	// __device__ bool bit_malloc(uint64_t & offset){


	// 	//group
	// 	//cg::coalesced_group active_threads = cg::coalesced_threads();

	// 	//team shares the load
	// 	uint64_t_bitarr local_copy = manager_bits.global_load_this();

	// 	#if DEBUG_PRINTS
	// 	if (active_threads.thread_rank() == 0){
	// 		printf("%d/%d %llx\n", active_threads.thread_rank(), active_threads.size(), local_copy);
	// 	}
	// 	#endif
		

	// 	while(local_copy.get_fill() != 0ULL){

	// 		cg::coalesced_group active_threads = cg::coalesced_threads();

	// 		int allocation_index_bit = 0;

	// 		//does removing this gate affect performance?

	// 		if (active_threads.thread_rank() == 0){

	// 			//allocation_index_bit = local_copy.get_first_active_bit();

	// 			allocation_index_bit = local_copy.get_random_active_bit();

	// 		}
			
	// 		allocation_index_bit = active_threads.shfl(allocation_index_bit, 0);
			

	// 		uint64_t_bitarr ext_bits;

	// 		bool ballot_bit_set = false;

	// 		if (active_threads.thread_rank() == 0){


	// 			if (manager_bits.unset_bit_atomic(allocation_index_bit)){


	// 				ext_bits = alloc_bits[allocation_index_bit].swap_to_empty();

	// 				ballot_bit_set = true;



	// 			}


	// 		}

	// 		//at this point, ballot_bit_set and ext_bits are set in thread 0
	// 		//so we ballot on if we can leave the loop

	// 		if (active_threads.ballot(ballot_bit_set)){


				 
	// 			ext_bits = active_threads.shfl(ext_bits, 0);

	// 			#if DEBUG_PRINTS
	// 			if (active_threads.thread_rank() == 0){
	// 				printf("%d/%d sees ext_bits for %d as %llx\n", active_threads.thread_rank(), active_threads.size(), allocation_index_bit, ext_bits);
	// 			}
	// 			#endif


	// 			if (active_threads.thread_rank()+1 <= ext_bits.get_fill()){

	// 				//next step: gather threads
	// 				cg::coalesced_group coalesced_threads = cg::coalesced_threads();

	// 				#if DEBUG_PRINTS
	// 				if (coalesced_threads.thread_rank() == 0){
	// 					printf("Leader is %d, sees %d threads coalesced.\n", active_threads.thread_rank(), coalesced_threads.size());
	// 				}
	// 				#endif

	// 				//how to sync outputs?
	// 				//everyone should pick a random lane?

	// 				//how to coalesce after lanes are picked


	// 				//options
	// 				//1) grab an allocation of the first n and try to  
	// 				//2) select the first n bits ahead of time.

	// 				//int bits_needed =  (ext_bits.get_fill() - active_threads.size());

	// 				//int my_bits = bits_before_index(active_threads.thread_rank());

	// 				// bool ballot = (bits_needeed == my_bits);

	// 				// int result = coalesced_threads.ballot(ballot);

					
	// 				int my_index;

	// 				while (true){

	// 					cg::coalesced_group searching_group = cg::coalesced_threads();

	// 					my_index = ext_bits.get_random_active_bit();

	// 					#if DEBUG_PRINTS
	// 					if (searching_group.thread_rank() == 0){
	// 						printf("Leader is %d/%d, sees ext bits as %llx\n", coalesced_threads.thread_rank(), searching_group.size(), ext_bits);
	// 					}
	// 					#endif

	// 					//any threads still searching group together
	// 					//do an exclusive scan on the OR bits 

	// 					//if the exclusive OR result doesn't contain your bit you are free to modify!

	// 					//last thread knows the true state of the system, so broadcast changes.

						

	// 					uint64_t my_mask = (1ULL) << my_index;

	// 					//now scan across the masks
	// 					uint64_t scanned_mask = cg::exclusive_scan(searching_group, my_mask, cg::bit_or<uint64_t>());

	// 					//final thread needs to broadcast updates
	// 					if (searching_group.thread_rank() == searching_group.size()-1){

	// 						//doesn't matter as the scan only adds bits
	// 						//not to set the mask to all bits not taken
	// 						uint64_t final_mask = ~(scanned_mask | my_mask);

	// 						ext_bits.apply_mask(final_mask);

	// 					}

	// 					//everyone now has an updated final copy of ext bits?
	// 					ext_bits = searching_group.shfl(ext_bits, searching_group.size()-1);


	// 					if (!(scanned_mask & my_mask)){

	// 						//I received an item!
	// 						//allocation has already been marked and index is set
	// 						//break to recoalesce for exit
	// 						break;



	// 					}


	// 				} //internal while loop

	// 				coalesced_threads.sync();

	// 				//TODO - take offset based on alloc size
	// 				//for now these are one byte allocs
	// 				allocation = (void *) (memmap[allocation_index_bit] + my_index);

	// 				//someone now has the minimum.
	// 				int my_fill = ext_bits.get_fill();

	// 				int lowest_fill = cg::reduce(coalesced_threads, my_fill, cg::less<int>());

	// 				int leader = __ffs(coalesced_threads.ballot(lowest_fill == my_fill))-1;

	// 				#if DEBUG_PRINTS
	// 				if (leader == coalesced_threads.thread_rank()){
	// 					printf("Leader reports lowest fill: %d, my_fill: %d, bits: %llx\n", lowest_fill, my_fill, ext_bits);
	// 				}
	// 				#endif
	// 				//printf("Leader is %d\n", leader, coalesced_threads.size());


	// 				if ((ext_bits.get_fill() > 0) && (leader == coalesced_threads.thread_rank())){

	// 					attach_buffer(memmap, ext_bits);

	// 				}

	// 				return true;




	// 			} //if active alloc


	// 		} //if bit set

			


	// 		//one extra inserted above this
	// 		//on failure reload local copy
	// 		local_copy = manager_bits.global_load_this();

	// 		} //current end of while loop?

	// 	return false;	

	// }

	




};

//pinned thread storage is a container
// that holds the one storage for each live thread block.

struct pinned_thread_storage {

	thread_storage * storages;



	static __host__ pinned_thread_storage * generate_on_device(int device){

		pinned_thread_storage * host_storage;

		cudaMallocHost((void **)&host_storage, sizeof(pinned_thread_storage));

		thread_storage * dev_storages;


		int num_storages = poggers::utils::get_num_streaming_multiprocessors(device);

		//printf("Booting up %d storages, %llu bytes\n", num_storages, sizeof(thread_storage)*num_storages);
		cudaMalloc((void **)&dev_storages, sizeof(thread_storage)*num_storages);

		init_storage<<<(num_storages-1)/256+1,256>>>(dev_storages, num_storages);

		cudaDeviceSynchronize();

		host_storage->storages = dev_storages;

		pinned_thread_storage * dev_ptr;

		cudaMalloc((void **)&dev_ptr, sizeof(pinned_thread_storage));

		cudaMemcpy(dev_ptr, host_storage, sizeof(pinned_thread_storage), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		cudaFreeHost(host_storage);

		return dev_ptr;


	}

	//if you don't specify we go on device 0.
	static __host__ pinned_thread_storage * generate_on_device(){

		return generate_on_device(0);
	}


	static __host__ void free_on_device(pinned_thread_storage * dev_storage){

		pinned_thread_storage * host_storage;

		cudaMallocHost((void **)&host_storage, sizeof(pinned_thread_storage));

		cudaMemcpy(host_storage, dev_storage, sizeof(pinned_thread_storage), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		cudaFree(dev_storage);

		cudaFree(host_storage->storages);

		cudaFreeHost(host_storage);

		return;


	}

	__device__ thread_storage * get_pinned_thread_storage(){

		return &storages[poggers::utils::get_smid()];

	}


};



}

}


#endif //End of VEB guard