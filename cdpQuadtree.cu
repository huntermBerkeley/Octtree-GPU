/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cooperative_groups.h>
#include <vector_functions.h>

#include <chrono>

#include <poggers/allocators/slab_one_size.cuh>

#include <poggers/allocators/one_size_allocator.cuh>


//rand
#include <random>
#include <cstdlib>

namespace cg = cooperative_groups;
#include "helper_cuda.h"

#define FAN_OUT 4
#define FLOAT_UPPER float(1<<30)

#define SANITY_CHECKS 1


using namespace std::chrono;



double elapsed(high_resolution_clock::time_point t1, high_resolution_clock::time_point t2) {
   return (duration_cast<duration<double> >(t2 - t1)).count();
}


//global vals

//using alloc_type = poggers::allocators::one_size_slab_allocator<4>;

using alloc_type = poggers::allocators::one_size_allocator;

__device__ alloc_type global_allocator;



__host__ void boot_allocator(uint64_t bytes_available, uint64_t alloc_size){

    alloc_type * local_version = alloc_type::generate_on_device(bytes_available, alloc_size, 42);

    if (local_version == nullptr){
        printf("Allocator failed to acquire memory.\n");
    }

    cudaMemcpyToSymbol(global_allocator, local_version, sizeof(alloc_type));

    cudaFree(local_version);

    cudaDeviceSynchronize();

}


__host__ void free_allocator(){

    alloc_type * local_version;

    cudaMalloc((void **)&local_version, sizeof(alloc_type));

    cudaMemcpyFromSymbol(local_version, global_allocator, sizeof(alloc_type));

    alloc_type::free_on_device(local_version);

    cudaDeviceSynchronize();

}

////////////////////////////////////////////////////////////////////////////////
// A structure of 2D points (structure of arrays).
////////////////////////////////////////////////////////////////////////////////
class Points
{
        float *m_x;
        float *m_y;
        int num_points = 8;

    public:
        // Constructor.
        __host__ __device__ Points() : m_x(NULL), m_y(NULL) {}

        // Constructor.
        __host__ __device__ Points(float *x, float *y) : m_x(x), m_y(y) {}

        // Get a point.
        __host__ __device__ __forceinline__ float2 get_point(int idx) const
        {
            return make_float2(m_x[idx], m_y[idx]);
        }

        __host__ __device__ __forceinline__ int get_num_points(){
            return num_points;
        }

        // Set a point.
        __host__ __device__ __forceinline__ void set_point(int idx, const float2 &p)
        {
            m_x[idx] = p.x;
            m_y[idx] = p.y;
        }

        // Set the pointers.
        // TODO change this to also set/update the num_points
        __host__ __device__ __forceinline__ void set(float *x, float *y)
        {
            m_x = x;
            m_y = y;
        }

};

//actually returns distance squared
//
__device__ float distance_between(float2 one_point, float2 another_point){
    return ((one_point.x - another_point.x)*(one_point.x - another_point.x)) + ((one_point.y - another_point.y) * (one_point.y - another_point.y));

}



struct point
{

    public:
    float x;
    float y;

    __host__ __device__ point(float ext_x, float ext_y): x(ext_x), y(ext_y) {}

    __host__ __device__ point(): x(0), y(0) {}

    __host__ __device__ void set_point(float ext_x, float ext_y){
        x = ext_x;
        y = ext_y;
    }

    __device__ float2 get_point(){

        return make_float2(x,y);

    }

    __device__ float distance(point alt_point){

        return sqrt( (x-alt_point.x)*(x-alt_point.x) + (y-alt_point.y)*(y-alt_point.y) );


    }

    __device__ operator uint64_t(){

            return ((uint64_t *) this)[0];

    }


};

//cast points to float2 for comparison.
//returns distance squared, should be ok as dist^2 is distance preserving.
__device__ float distance_between(point one_point, point another_point){
    return distance_between(one_point.get_point(), another_point.get_point());

}

////////////////////////////////////////////////////////////////////////////////
// A 2D bounding box
////////////////////////////////////////////////////////////////////////////////
class Bounding_box
{
        // Extreme points of the bounding box.


    public:

        float2 m_p_min;
        float2 m_p_max;

        // Constructor. Create a unit box.
        __host__ __device__ Bounding_box()
        {
            m_p_min = make_float2(0.0f, 0.0f);
            m_p_max = make_float2(1.0f, 1.0f);
        }

        // Compute the center of the bounding-box.
        __host__ __device__ void compute_center(float2 &center) const
        {
            center.x = 0.5f * (m_p_min.x + m_p_max.x);
            center.y = 0.5f * (m_p_min.y + m_p_max.y);
        }

        // The points of the box.
        __host__ __device__ __forceinline__ const float2 &get_max() const
        {
            return m_p_max;
        }

        __host__ __device__ __forceinline__ const float2 &get_min() const
        {
            return m_p_min;
        }

        // Does a box contain a point.
        __host__ __device__ bool contains(float2 p) const
        {
            return p.x >= m_p_min.x && p.x < m_p_max.x && p.y >= m_p_min.y && p.y < m_p_max.y;
        }

        __device__ bool contains(point p) const
        {
            return contains(p.get_point());
        }

        // Define the bounding box.
        __host__ __device__ void set(float min_x, float min_y, float max_x, float max_y)
        {
            m_p_min.x = min_x;
            m_p_min.y = min_y;
            m_p_max.x = max_x;
            m_p_max.y = max_y;
        }
        // Returns the minimum possible distance between the box and the point.
        // This might not even be a valid distance, but the valid distance will surely be greater than this.

};

////////////////////////////////////////////////////////////////////////////////
// A node of a quadree.
////////////////////////////////////////////////////////////////////////////////
class Quadtree_node
{
        // The identifier of the node.
        int m_id;
        // The bounding box of the tree.
        Bounding_box m_bounding_box;
        // The range of points.
        int m_begin, m_end;
        Quadtree_node* children; // Leaf has a nullptr here.
        Points maybe_some_points;


    public:
        // Constructor.
        __host__ __device__ Quadtree_node() : m_id(0), m_begin(0), m_end(0)
        {}

        // The ID of a node at its level.
        __host__ __device__ int id() const
        {
            return m_id;
        }

        // The ID of a node at its level.
        __host__ __device__ void set_id(int new_id)
        {
            m_id = new_id;
        }

        // The bounding box.
        __host__ __device__ __forceinline__ const Bounding_box &bounding_box() const
        {
            return m_bounding_box;
        }

        // Set the bounding box.
        __host__ __device__ __forceinline__ void set_bounding_box(float min_x, float min_y, float max_x, float max_y)
        {
            m_bounding_box.set(min_x, min_y, max_x, max_y);
        }

        // The number of points in the tree.
        __host__ __device__ __forceinline__ int num_points() const
        {
            return m_end - m_begin;
        }

        // The range of points in the tree.
        __host__ __device__ __forceinline__ int points_begin() const
        {
            return m_begin;
        }

        __host__ __device__ __forceinline__ int points_end() const
        {
            return m_end;
        }

        // Define the range for that node.
        __host__ __device__ __forceinline__ void set_range(int begin, int end)
        {
            m_begin = begin;
            m_end = end;
        }

        __device__ __forceinline__ Quadtree_node* get_child(float2 some_point){
            for(int citer = 0; citer < FAN_OUT; citer++){
                if(children[citer].bounding_box().contains(some_point)){
                    return &children[citer];
                }
            }
        }

};



struct quadtree_node_v2 
{

    using my_type = quadtree_node_v2;

    //data types needed
    //1) bounding box

    //8
    uint64_t metadata;

    //8
    point * my_points;

    //32
    //children are indexed in clockwise order starting from the top left 
    // 0 1
    // 3 2
    //set bounding boxes accordingly.
    my_type * children[4];

    //16
    Bounding_box my_bounding_box;

    __device__ void set_bounding_box(float min_x, float min_y, float max_x, float max_y){

        my_bounding_box.set(min_x, min_y, max_x, max_y);

    }



    __device__ float get_min_distance(cg::thread_group query_group, point comp_point, point & output){


        if (my_points == nullptr) return 2; 


    }

    // m_p_min.x = min_x;
    // m_p_min.y = min_y;
    // m_p_max.x = max_x;
    // m_p_max.y = max_y;


    __device__ Bounding_box get_child_bounding_box(int child_id){

        Bounding_box child_box = my_bounding_box;

        float2 center;
        my_bounding_box.compute_center(center);


        bool above_x = (child_id / 2) == 0;
        bool right_y = (child_id % 2) == 1;

        child_box.m_p_min.x = (right_y)*center.x + (!right_y)*child_box.m_p_min.x;
        child_box.m_p_max.x = (!right_y)*center.x + (right_y)*child_box.m_p_max.x;
        child_box.m_p_min.y = (above_x)*center.y + (!above_x)*child_box.m_p_min.y;
        child_box.m_p_max.y = (!above_x)*center.y + (above_x)*child_box.m_p_max.y;


        return child_box;

    }

    //1) child is not nullptr
    //
    __device__ bool is_correct_child(int child_id, point new_item, Bounding_box child_box){

        //do a global load
        //my_type * child = (my_type *) poggers::utils::ldca((uint64_t *)&children[child_id]);


        //float my_min_x = my_bounding_box.


        //Bounding_box child_box = get_child_bounding_box(child_id);


        return child_box.contains(new_item.get_point());

    }

    __device__ bool add_to_leaf(cg::thread_block_tile<4> insert_tile, point * my_points_ptr, point new_item){

		for (int i = insert_tile.thread_rank(); i < 8; i+= insert_tile.size()){

				bool ballot = false;

				if ((uint64_t) my_points_ptr[i] == ~0ULL){
					ballot = true;
				}

				auto ballot_result = insert_tile.ballot(ballot);

				while (ballot_result){

					ballot = false;

					const auto leader = __ffs(ballot_result) -1;

					if (leader == insert_tile.thread_rank()){
                        //  atomicCAS(int* address, int compare, int val);
						ballot = atomicCAS((unsigned long long int *) &my_points_ptr[i], ~0ULL, (unsigned long long int) new_item) == ~0ULL;
					}

					if (insert_tile.ballot(ballot)) return true;

					ballot_result ^= 1UL << leader;

				}
		}

        return false;          
    }

    __device__ bool maybe_insert_point(cg::thread_block_tile<4> insert_tile, point new_item){

            Bounding_box child_box = get_child_bounding_box(insert_tile.thread_rank());

            bool is_valid = is_correct_child(insert_tile.thread_rank(), new_item, child_box);
            auto valid = insert_tile.ballot(is_valid);
            int leader = __ffs(valid)-1;

            if (leader == -1){


                uint64_t tid = insert_tile.meta_group_rank()+blockIdx.x*insert_tile.meta_group_size();

                //printf("Tid %llu failed\n", tid);

                printf("tid %llu %llu, Failed to place point %f %f in child %f %f %f %f %d\n", threadIdx.x+blockIdx.x*blockDim.x, tid, new_item.x, new_item.y, child_box.m_p_min.x, child_box.m_p_max.x, child_box.m_p_min.y, child_box.m_p_max.y, child_box.contains(new_item));
                return false;
            }

            if (children[leader] == nullptr){
                attach_new_child(insert_tile, leader);
            }

            __threadfence();
            // either i have created a new child or someone else did, so we have a child
            children[leader]->insert(insert_tile, new_item);
    }

    __device__ bool insert(cg::thread_block_tile<4> insert_tile, point new_item){



        #if SANITY_CHECKS

        my_type * lead_node_this = insert_tile.shfl(this, 0);

        if (lead_node_this != this) printf("Thread 0 sees different node\n");

        #endif

        //printf("Inserting...\n");

        //to prevent reading from null, we cache old value.
        point * my_points_ptr = my_points;


        #if SANITY_CHECKS
        point * lead_my_points_ptr = insert_tile.shfl(my_points_ptr, 0);

        if (lead_my_points_ptr != my_points_ptr) printf("Discrepancy in my_points_ptr\n");
 
        #endif

        if (my_points_ptr != nullptr){
            // Adding to leaf
            if(add_to_leaf(insert_tile, my_points_ptr, new_item)) return true;
            
            //point* my_points_ptr = my_points;
            bool own_my_points = false;
            // Leaf is full
            if (insert_tile.thread_rank() == 0){
                
                own_my_points = (atomicCAS((unsigned long long int *) &my_points, (unsigned long long int) my_points_ptr, (unsigned long long int) nullptr) == (unsigned long long int) my_points_ptr);
                __threadfence(); // flushed to global memory.
            }
            own_my_points = insert_tile.ballot(own_my_points);
            // add the points in my_points_
            if(own_my_points){
                for(int i=0; i < 8; i++){
                    maybe_insert_point(insert_tile, my_points_ptr[i]);
                }
            }
        }
        maybe_insert_point(insert_tile, new_item);
        
       
        /*
        // first, find valid child
        bool is_valid = is_correct_child(insert_tile.thread_rank(), new_item);

        auto valid = insert_tile.ballot(is_valid);


        #if DEBUG_ASSERTS
        if (__popc(valid)!=1){
            printf("Two children to recurse to, boundaries are wrong\n");
        }
        
        #endif

        int leader = __ffs(valid)-1;

        if (children[leader] == nullptr){
            //add new child
            attach_new_child(insert_tile, leader);

        }  

        return children[leader]->insert(insert_tile, new_item);
        */
    }

    __device__ void attach_new_child(cg::thread_block_tile<4> insert_tile, int leader){

        void * allocation;

        if (insert_tile.thread_rank() < 2){

            allocation = global_allocator.malloc();

            if (allocation == nullptr){
                printf("Out of memory\n");
            }

        }

        my_type * child = (my_type * ) insert_tile.shfl(allocation, 0);

        //insert_tile.sync();

        if (insert_tile.thread_rank() == 1){

            atomicExch((unsigned long long int *)&(child->my_points), (unsigned long long int )(allocation));

        }


        __threadfence();

        //set the bounding box and null out other stuff


        if (insert_tile.thread_rank() == 0){

            child->my_bounding_box = get_child_bounding_box(leader);

            for (int i = 0; i < 4; i++){

                child->children[i] = nullptr;
            }

        }  else if (insert_tile.thread_rank() == 1){

            for (int i = 0; i < 8; i ++){


                uint64_t * def_a_ptr = &(((uint64_t *) allocation)[i]);

                //uint64_t * def_a_ptr = (uint64_t *) &(child->my_points[i]);
                *def_a_ptr = ~0ULL;

            }

        }


        __threadfence();

        if (insert_tile.thread_rank() == 1){

            if (child->my_points != allocation){
                printf("Error setting %llx instead of %llx\n", (uint64_t) child->my_points, (uint64_t) allocation);
            }

        }



        //complete init.
        bool swapped = false;

        if (insert_tile.thread_rank() == 0){
            swapped = (atomicCAS((unsigned long long int *)&children[leader], (unsigned long long int)0ULL, (unsigned long long int)child) == 0ULL);
        }

        bool success = insert_tile.ballot(swapped);

        if (!success){

            if (insert_tile.thread_rank() < 2){

                global_allocator.free(allocation);

            }

        }

    }
    __device__ float distance_bound(point query_point, cooperative_groups::thread_group g, point & result){
            if (children == nullptr){
                int num_pts = get_num_points();
                float mindist = get_minimum_distance(query_point, g, result);
                return mindist;
            } else{
                for(int citer = 0; citer < 4; citer++){
                    if (children+(citer*sizeof(my_type)) == NULL){
                        continue;
                    }
                    if(children[citer]->my_bounding_box.contains(query_point)){
                        return children[citer]->distance_bound(query_point, g, result);
                    }
                }
            }
            return -1;
    }

    __device__ int get_num_points(){
        //TODO:
        //this
        return 0;
    }

    __device__ float get_minimum_distance(point query_point, cooperative_groups::thread_group g, point & result){
        return 0;
    }


    __device__ point check_neighbouring_subtrees(point query_point, cooperative_groups::thread_group g, float bound){

        point minpoint;
        return minpoint;
        // float mindist = FLOAT_UPPER;
        // point minpoint;
        // point maybe_points[4];
        // for(int citer = 0; citer < 4; citer++){
        //     maybe_points[citer] = NULL;
        //     if(children + (citer*sizeof(my_type)) != NULL){
        //         if(is_correct_child(citer, query_point)) continue;
        //         if(get_child_bounding_box(citer).distance_between(query_point) < bound){
        //             cooperative_groups::thread_block_tile<g.size() / 2> next_tile = cooperative_groups::tiled_partition<g.size() / 2>(cooperative_groups:: this_thread_block());
        //             maybe_points[citer] = check_neighbouring_subtrees(query_point, next_tile, bound);
        //         }
        //     }
        // }
        // for(int citer = 0; citer < 4; citer++){
        //     if((maybe_points + (citer*sizeof(point)) != NULL) && !(maybe_points[citer].x == 0 && maybe_points[citer.y] == 0)){
        //         //You went down this child, and found a point that is not (0,0)
        //         float thisdist = distance_between(maybe_points[citer], query_point);
        //         if(thisdist < mindist){
        //             minpoint = maybe_points[citer];
        //         }
        //     }
        // }
        // return minpoint
    }


};

////////////////////////////////////////////////////////////////////////////////
// Algorithm parameters.
////////////////////////////////////////////////////////////////////////////////
struct Parameters
{
    // Choose the right set of points to use as in/out.
    int point_selector;
    // The number of nodes at a given level (2^k for level k).
    int num_nodes_at_this_level;
    // The recursion depth.
    int depth;
    // The max value for depth.
    const int max_depth;
    // The minimum number of points in a node to stop recursion.
    const int min_points_per_node;

    // Constructor set to default values.
    __host__ __device__ Parameters(int max_depth, int min_points_per_node) :
        point_selector(0),
        num_nodes_at_this_level(1),
        depth(0),
        max_depth(max_depth),
        min_points_per_node(min_points_per_node)
    {}

    // Copy constructor. Changes the values for next iteration.
    __host__ __device__ Parameters(const Parameters &params, bool) :
        point_selector((params.point_selector+1) % 2),
        num_nodes_at_this_level(4*params.num_nodes_at_this_level),
        depth(params.depth+1),
        max_depth(params.max_depth),
        min_points_per_node(params.min_points_per_node)
    {}
};

////////////////////////////////////////////////////////////////////////////////
// Build a quadtree on the GPU. Use CUDA Dynamic Parallelism.
//
// The algorithm works as follows. The host (CPU) launches one block of
// NUM_THREADS_PER_BLOCK threads. That block will do the following steps:
//
// 1- Check the number of points and its depth.
//
// We impose a maximum depth to the tree and a minimum number of points per
// node. If the maximum depth is exceeded or the minimum number of points is
// reached. The threads in the block exit.
//
// Before exiting, they perform a buffer swap if it is needed. Indeed, the
// algorithm uses two buffers to permute the points and make sure they are
// properly distributed in the quadtree. By design we want all points to be
// in the first buffer of points at the end of the algorithm. It is the reason
// why we may have to swap the buffer before leavin (if the points are in the
// 2nd buffer).
//
// 2- Count the number of points in each child.
//
// If the depth is not too high and the number of points is sufficient, the
// block has to dispatch the points into four geometrical buckets: Its
// children. For that purpose, we compute the center of the bounding box and
// count the number of points in each quadrant.
//
// The set of points is divided into sections. Each section is given to a
// warp of threads (32 threads). Warps use __ballot and __popc intrinsics
// to count the points. See the Programming Guide for more information about
// those functions.
//
// 3- Scan the warps' results to know the "global" numbers.
//
// Warps work independently from each other. At the end, each warp knows the
// number of points in its section. To know the numbers for the block, the
// block has to run a scan/reduce at the block level. It's a traditional
// approach. The implementation in that sample is not as optimized as what
// could be found in fast radix sorts, for example, but it relies on the same
// idea.
//
// 4- Move points.
//
// Now that the block knows how many points go in each of its 4 children, it
// remains to dispatch the points. It is straightforward.
//
// 5- Launch new blocks.
//
// The block launches four new blocks: One per children. Each of the four blocks
// will apply the same algorithm.
////////////////////////////////////////////////////////////////////////////////
template< int NUM_THREADS_PER_BLOCK >
__global__
void build_quadtree_kernel(Quadtree_node *nodes, Points *points, Parameters params)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // The number of warps in a block.
    const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warpSize;

    // Shared memory to store the number of points.
    extern __shared__ int smem[];

    // s_num_pts[4][NUM_WARPS_PER_BLOCK];
    // Addresses of shared memory.
    volatile int *s_num_pts[4];

    for (int i = 0 ; i < 4 ; ++i)
        s_num_pts[i] = (volatile int *) &smem[i*NUM_WARPS_PER_BLOCK];

    // Compute the coordinates of the threads in the block.
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    // Mask for compaction.
    int lane_mask_lt = (1 << lane_id) - 1; // Same as: asm( "mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt) );

    // The current node.
    Quadtree_node &node = nodes[blockIdx.x];

    // The number of points in the node.
    int num_points = node.num_points();

    float2 center;
    int range_begin, range_end;
    int warp_cnts[4] = {0, 0, 0, 0};
    //
    // 1- Check the number of points and its depth.
    //

    // Stop the recursion here. Make sure points[0] contains all the points.
    if (params.depth >= params.max_depth || num_points <= params.min_points_per_node)
    {
        if (params.point_selector == 1)
        {
            int it = node.points_begin(), end = node.points_end();

            for (it += threadIdx.x ; it < end ; it += NUM_THREADS_PER_BLOCK)
                if (it < end)
                    points[0].set_point(it, points[1].get_point(it));
        }

        return;
    }

    // Compute the center of the bounding box of the points.
    const Bounding_box &bbox = node.bounding_box();

    bbox.compute_center(center);

    // Find how many points to give to each warp.
    int num_points_per_warp = max(warpSize, (num_points + NUM_WARPS_PER_BLOCK-1) / NUM_WARPS_PER_BLOCK);

    // Each warp of threads will compute the number of points to move to each quadrant.
    range_begin = node.points_begin() + warp_id * num_points_per_warp;
    range_end   = min(range_begin + num_points_per_warp, node.points_end());

    //
    // 2- Count the number of points in each child.
    //

    // Input points.
    const Points &in_points = points[params.point_selector];

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    // Compute the number of points.
    for (int range_it = range_begin + tile32.thread_rank() ; tile32.any(range_it < range_end) ; range_it += warpSize)
    {
        // Is it still an active thread?
        bool is_active = range_it < range_end;

        // Load the coordinates of the point.
        float2 p = is_active ? in_points.get_point(range_it) : make_float2(0.0f, 0.0f);

        // Count top-left points.
        int num_pts = __popc(tile32.ballot(is_active && p.x < center.x && p.y >= center.y));
        warp_cnts[0] += tile32.shfl(num_pts, 0);

        // Count top-right points.
        num_pts = __popc(tile32.ballot(is_active && p.x >= center.x && p.y >= center.y));
        warp_cnts[1] += tile32.shfl(num_pts, 0);

        // Count bottom-left points.
        num_pts = __popc(tile32.ballot(is_active && p.x < center.x && p.y < center.y));
        warp_cnts[2] += tile32.shfl(num_pts, 0);

        // Count bottom-right points.
        num_pts = __popc(tile32.ballot(is_active && p.x >= center.x && p.y < center.y));
        warp_cnts[3] += tile32.shfl(num_pts, 0);
    }

    if (tile32.thread_rank() == 0)
    {
        s_num_pts[0][warp_id] = warp_cnts[0];
        s_num_pts[1][warp_id] = warp_cnts[1];
        s_num_pts[2][warp_id] = warp_cnts[2];
        s_num_pts[3][warp_id] = warp_cnts[3];
    }

    // Make sure warps have finished counting.
    cg::sync(cta);

    //
    // 3- Scan the warps' results to know the "global" numbers.
    //

    // First 4 warps scan the numbers of points per child (inclusive scan).
    if (warp_id < 4)
    {
        int num_pts = tile32.thread_rank() < NUM_WARPS_PER_BLOCK ? s_num_pts[warp_id][tile32.thread_rank()] : 0;
#pragma unroll

        for (int offset = 1 ; offset < NUM_WARPS_PER_BLOCK ; offset *= 2)
        {
            int n = tile32.shfl_up(num_pts, offset);

            if (tile32.thread_rank() >= offset)
                num_pts += n;
        }

        if (tile32.thread_rank() < NUM_WARPS_PER_BLOCK)
            s_num_pts[warp_id][tile32.thread_rank()] = num_pts;
    }

    cg::sync(cta);

    // Compute global offsets.
    if (warp_id == 0)
    {
        int sum = s_num_pts[0][NUM_WARPS_PER_BLOCK-1];

        for (int row = 1 ; row < 4 ; ++row)
        {
            int tmp = s_num_pts[row][NUM_WARPS_PER_BLOCK-1];
            cg::sync(tile32);

            if (tile32.thread_rank() < NUM_WARPS_PER_BLOCK)
                s_num_pts[row][tile32.thread_rank()] += sum;

            cg::sync(tile32);
            sum += tmp;
        }
    }

    cg::sync(cta);

    // Make the scan exclusive.
    int val = 0;
    if (threadIdx.x < 4*NUM_WARPS_PER_BLOCK)
    {
        val = threadIdx.x == 0 ? 0 : smem[threadIdx.x-1];
        val += node.points_begin();
    }

    cg::sync(cta);

    if (threadIdx.x < 4*NUM_WARPS_PER_BLOCK)
    {
        smem[threadIdx.x] = val;
    }

    cg::sync(cta);

    //
    // 4- Move points.
    //
    if (!(params.depth >= params.max_depth || num_points <= params.min_points_per_node))
    {
        // Output points.
        Points &out_points = points[(params.point_selector+1) % 2];

        warp_cnts[0] = s_num_pts[0][warp_id];
        warp_cnts[1] = s_num_pts[1][warp_id];
        warp_cnts[2] = s_num_pts[2][warp_id];
        warp_cnts[3] = s_num_pts[3][warp_id];

        const Points &in_points = points[params.point_selector];
        // Reorder points.
        for (int range_it = range_begin + tile32.thread_rank(); tile32.any(range_it < range_end) ; range_it += warpSize)
        {
            // Is it still an active thread?
            bool is_active = range_it < range_end;

            // Load the coordinates of the point.
            float2 p = is_active ? in_points.get_point(range_it) : make_float2(0.0f, 0.0f);

            // Count top-left points.
            bool pred = is_active && p.x < center.x && p.y >= center.y;
            int vote = tile32.ballot(pred);
            int dest = warp_cnts[0] + __popc(vote & lane_mask_lt);

            if (pred)
                out_points.set_point(dest, p);

            warp_cnts[0] += tile32.shfl(__popc(vote), 0);

            // Count top-right points.
            pred = is_active && p.x >= center.x && p.y >= center.y;
            vote = tile32.ballot(pred);
            dest = warp_cnts[1] + __popc(vote & lane_mask_lt);

            if (pred)
                out_points.set_point(dest, p);

            warp_cnts[1] += tile32.shfl(__popc(vote), 0);

            // Count bottom-left points.
            pred = is_active && p.x < center.x && p.y < center.y;
            vote = tile32.ballot(pred);
            dest =  warp_cnts[2] + __popc(vote & lane_mask_lt);

            if (pred)
                out_points.set_point(dest, p);

            warp_cnts[2] +=  tile32.shfl(__popc(vote), 0);

            // Count bottom-right points.
            pred = is_active && p.x >= center.x && p.y < center.y;
            vote = tile32.ballot(pred);
            dest = warp_cnts[3] + __popc(vote & lane_mask_lt);

            if (pred)
                out_points.set_point(dest, p);

            warp_cnts[3] += tile32.shfl(__popc(vote), 0);
        }
    }

    cg::sync(cta);

    if (tile32.thread_rank() == 0)
    {
        s_num_pts[0][warp_id] = warp_cnts[0];
        s_num_pts[1][warp_id] = warp_cnts[1] ;
        s_num_pts[2][warp_id] = warp_cnts[2] ;
        s_num_pts[3][warp_id] = warp_cnts[3];
    }

    cg::sync(cta);

    //
    // 5- Launch new blocks.
    //
    if (!(params.depth >= params.max_depth || num_points <= params.min_points_per_node))
    {
        // The last thread launches new blocks.
        if (threadIdx.x == NUM_THREADS_PER_BLOCK-1 )
        {
            // The children.
            Quadtree_node *children = &nodes[params.num_nodes_at_this_level - (node.id() & ~3)];

            // The offsets of the children at their level.
            int child_offset = 4*node.id();

            // Set IDs.
            children[child_offset+0].set_id(4*node.id()+0);
            children[child_offset+1].set_id(4*node.id()+1);
            children[child_offset+2].set_id(4*node.id()+2);
            children[child_offset+3].set_id(4*node.id()+3);

            const Bounding_box &bbox = node.bounding_box();
            // Points of the bounding-box.
            const float2 &p_min = bbox.get_min();
            const float2 &p_max = bbox.get_max();

            // Set the bounding boxes of the children.
            children[child_offset+0].set_bounding_box(p_min.x , center.y, center.x, p_max.y);    // Top-left.
            children[child_offset+1].set_bounding_box(center.x, center.y, p_max.x , p_max.y);    // Top-right.
            children[child_offset+2].set_bounding_box(p_min.x , p_min.y , center.x, center.y);   // Bottom-left.
            children[child_offset+3].set_bounding_box(center.x, p_min.y , p_max.x , center.y);   // Bottom-right.

            // Set the ranges of the children.

            children[child_offset+0].set_range(node.points_begin(),   s_num_pts[0][warp_id]);
            children[child_offset+1].set_range(s_num_pts[0][warp_id], s_num_pts[1][warp_id]);
            children[child_offset+2].set_range(s_num_pts[1][warp_id], s_num_pts[2][warp_id]);
            children[child_offset+3].set_range(s_num_pts[2][warp_id], s_num_pts[3][warp_id]);

            // Launch 4 children.
            build_quadtree_kernel<NUM_THREADS_PER_BLOCK><<<4, NUM_THREADS_PER_BLOCK, 4 *NUM_WARPS_PER_BLOCK *sizeof(int)>>>(&children[child_offset], points, Parameters(params, true));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Make sure a Quadtree is properly defined.
////////////////////////////////////////////////////////////////////////////////
bool check_quadtree(const Quadtree_node *nodes, int idx, int num_pts, Points *pts, Parameters params)
{
    const Quadtree_node &node = nodes[idx];
    int num_points = node.num_points();

    if (!(params.depth == params.max_depth || num_points <= params.min_points_per_node))
    {
        int num_points_in_children = 0;

        num_points_in_children += nodes[params.num_nodes_at_this_level + 4*idx+0].num_points();
        num_points_in_children += nodes[params.num_nodes_at_this_level + 4*idx+1].num_points();
        num_points_in_children += nodes[params.num_nodes_at_this_level + 4*idx+2].num_points();
        num_points_in_children += nodes[params.num_nodes_at_this_level + 4*idx+3].num_points();

        if (num_points_in_children != node.num_points())
            return false;

        return check_quadtree(&nodes[params.num_nodes_at_this_level], 4*idx+0, num_pts, pts, Parameters(params, true)) &&
               check_quadtree(&nodes[params.num_nodes_at_this_level], 4*idx+1, num_pts, pts, Parameters(params, true)) &&
               check_quadtree(&nodes[params.num_nodes_at_this_level], 4*idx+2, num_pts, pts, Parameters(params, true)) &&
               check_quadtree(&nodes[params.num_nodes_at_this_level], 4*idx+3, num_pts, pts, Parameters(params, true));
    }

    const Bounding_box &bbox = node.bounding_box();

    for (int it = node.points_begin() ; it < node.points_end() ; ++it)
    {
        if (it >= num_pts)
            return false;

        float2 p = pts->get_point(it);

        if (!bbox.contains(p))
            return false;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Parallel random number generator.
////////////////////////////////////////////////////////////////////////////////
struct Random_generator
{
  int count;

  __host__ __device__
  Random_generator() : count(0) {}
    __host__ __device__ unsigned int hash(unsigned int a)
    {
        a = (a+0x7ed55d16) + (a<<12);
        a = (a^0xc761c23c) ^ (a>>19);
        a = (a+0x165667b1) + (a<<5);
        a = (a+0xd3a2646c) ^ (a<<9);
        a = (a+0xfd7046c5) + (a<<3);
        a = (a^0xb55a4f09) ^ (a>>16);
        return a;
    }

    __host__ __device__ __forceinline__ thrust::tuple<float, float> operator()()
    {
#ifdef __CUDA_ARCH__
        unsigned seed = hash(blockIdx.x*blockDim.x + threadIdx.x + count);
        // thrust::generate may call operator() more than once per thread.
        // Hence, increment count by grid size to ensure uniqueness of seed
        count += blockDim.x * gridDim.x;
#else
        unsigned seed = hash(0);
#endif
        thrust::default_random_engine rng(seed);
        thrust::random::uniform_real_distribution<float> distrib;
        return thrust::make_tuple(distrib(rng), distrib(rng));
    }
};

////////////////////////////////////////////////////////////////////////////////
// Allocate GPU structs, launch kernel and clean up
////////////////////////////////////////////////////////////////////////////////
bool cdpQuadtree(int warp_size)
{
    // Constants to control the algorithm.
    const int num_points = 1024;
    const int max_depth  = 8;
    const int min_points_per_node = 16;

    // Allocate memory for points.
    thrust::device_vector<float> x_d0(num_points);
    thrust::device_vector<float> x_d1(num_points);
    thrust::device_vector<float> y_d0(num_points);
    thrust::device_vector<float> y_d1(num_points);

    // Generate random points.
    Random_generator rnd;
    thrust::generate(
        thrust::make_zip_iterator(thrust::make_tuple(x_d0.begin(), y_d0.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(x_d0.end(), y_d0.end())),
        rnd);

    // Host structures to analyze the device ones.
    Points points_init[2];
    points_init[0].set(thrust::raw_pointer_cast(&x_d0[0]), thrust::raw_pointer_cast(&y_d0[0]));
    points_init[1].set(thrust::raw_pointer_cast(&x_d1[0]), thrust::raw_pointer_cast(&y_d1[0]));

    // Allocate memory to store points.
    Points *points;
    checkCudaErrors(cudaMalloc((void **) &points, 2*sizeof(Points)));
    checkCudaErrors(cudaMemcpy(points, points_init, 2*sizeof(Points), cudaMemcpyHostToDevice));

    // We could use a close form...
    int max_nodes = 0;

    for (int i = 0, num_nodes_at_level = 1 ; i < max_depth ; ++i, num_nodes_at_level *= 4)
        max_nodes += num_nodes_at_level;

    // Allocate memory to store the tree.
    Quadtree_node root;
    root.set_range(0, num_points);
    Quadtree_node *nodes;
    checkCudaErrors(cudaMalloc((void **) &nodes, max_nodes*sizeof(Quadtree_node)));
    checkCudaErrors(cudaMemcpy(nodes, &root, sizeof(Quadtree_node), cudaMemcpyHostToDevice));

    // We set the recursion limit for CDP to max_depth.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth);

    // Build the quadtree.
    Parameters params(max_depth, min_points_per_node);
    std::cout << "Launching CDP kernel to build the quadtree" << std::endl;
    const int NUM_THREADS_PER_BLOCK = 128; // Do not use less than 128 threads.
    const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warp_size;
    const size_t smem_size = 4*NUM_WARPS_PER_BLOCK*sizeof(int);
    build_quadtree_kernel<NUM_THREADS_PER_BLOCK><<<1, NUM_THREADS_PER_BLOCK, smem_size>>>(nodes, points, params);
    checkCudaErrors(cudaGetLastError());

    // Copy points to CPU.
    thrust::host_vector<float> x_h(x_d0);
    thrust::host_vector<float> y_h(y_d0);
    Points host_points;
    host_points.set(thrust::raw_pointer_cast(&x_h[0]), thrust::raw_pointer_cast(&y_h[0]));

    // Copy nodes to CPU.
    Quadtree_node *host_nodes = new Quadtree_node[max_nodes];
    checkCudaErrors(cudaMemcpy(host_nodes, nodes, max_nodes *sizeof(Quadtree_node), cudaMemcpyDeviceToHost));

    // Validate the results.
    bool ok = check_quadtree(host_nodes, 0, num_points, &host_points, params);
    std::cout << "Results: " << (ok ? "OK" : "FAILED") << std::endl;

    // Free CPU memory.
    delete[] host_nodes;

    // Free memory.
    checkCudaErrors(cudaFree(nodes));
    checkCudaErrors(cudaFree(points));

    return ok;
}


__global__ void find_nn(quadtree_node_v2 * root, point query_point, point& result){
    //TODO: re-enable
    // cooperative_groups::thread_block_tile<4> some_tile = cooperative_groups::tiled_partition<4>(cooperative_groups:: this_thread_block());
    // float mindist = root->distance_bound(query_point, some_tile, result); //TODO add the second traversal
    // Quadtree_node improved_result;
    // point another_point = root->check_neighbouring_subtrees(query_point, some_tile, improved_result, mindist);
    // if(!(another_point.x == 0 && another_point.y == 0) && distance_between(query_point, another_point) < mindist){
    //     result = another_point;
    // }
}

__global__ void init_tree(quadtree_node_v2 ** root){


    uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

    if (tid != 0) return;

    quadtree_node_v2 * new_node = (quadtree_node_v2 *) global_allocator.malloc();


    void * memory = global_allocator.malloc();

    atomicExch((unsigned long long int *)&new_node->my_points, (unsigned long long int )(memory));

    if (new_node == nullptr || memory == nullptr) printf("Allocator could not request!\n");


    uint64_t * memory_as_uint = (uint64_t *) memory;

    for (int i =0; i < 8; i++){
        memory_as_uint[i] = ~0ULL;
    }

    new_node->set_bounding_box(0,0,1,1);

    root[0] = new_node;


}
    

__global__ void insert_points(quadtree_node_v2 ** head, point * points, uint64_t npoints){


    auto my_thread_block = cg::this_thread_block();

    auto my_tile = cg::tiled_partition<4>(my_thread_block);

    //precondition - make blockdim /4

    if (blockDim.x % 4 != 0){
        printf("Block dim %llu must be divisible by 4\n", blockDim.x);

        return;
    }


    uint64_t tid = my_tile.meta_group_rank() + blockIdx.x*my_tile.meta_group_size();

    if (tid >= npoints) return;


    if (!head[0]->insert(my_tile, points[tid])){
        printf("Failed insertion! %llu\n", tid);
    } else {
        //printf("%llu succeeded!\n", tid);
    }

}


__global__ void boot_many_nodes(uint64_t n_nodes){

    uint64_t tid =threadIdx.x+blockIdx.x*blockDim.x;

    if (tid >= n_nodes) return;


    void * node = global_allocator.malloc();
    void * memory = global_allocator.malloc();

}

__host__ float randomFloat()
{
    float cap = (RAND_MAX);

    float rand = std::rand();

    return rand/cap;
    //return (float)(std::rand());
}

__host__ point * generate_random_points(uint64_t npoints){

    point * host_version;

    cudaMallocHost((void **)&host_version, sizeof(point)*npoints);


    for (uint64_t i =0; i < npoints; i++){

        host_version[i].set_point(randomFloat(), randomFloat());

        //while(host_version[i])

    }

    return host_version;

}


__host__ point * get_dev_points(point * host_version, uint64_t npoints){

    point * dev_version;

    cudaMalloc((void **)&dev_version, sizeof(point)*npoints);

    cudaMemcpy(dev_version, host_version, sizeof(point)*npoints, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    return dev_version;

}  


////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{


    //get stack size

    cudaDeviceSetLimit(cudaLimitStackSize, 16384);

    size_t stack_size;

    cudaDeviceGetLimit  (&stack_size, cudaLimitStackSize);

    printf("Stack is %llu\n", stack_size);

    cudaDeviceSynchronize();

    boot_allocator(80000000+2000, 64);
    quadtree_node_v2 ** head;

    cudaMallocManaged((void **)&head, sizeof(quadtree_node_v2 *));


    init_tree<<<1,1>>>(head);


    cudaDeviceSynchronize();


    uint64_t npoints = 100000000;

    uint64_t nthreads = npoints*4;

    point * host_points = generate_random_points(npoints);

    point * dev_points = get_dev_points(host_points, npoints);

    // cudaMallocManaged((void **)&points, sizeof(point)*8);

    // cudaDeviceSynchronize();


    // for (int i = 0; i < 8; i++){

    //     points[i].set_point(1.0*i/8, 1.0-1.0*i/8);

    // }

    auto insert_start = high_resolution_clock::now();

    insert_points<<<(nthreads-1)/256+1,256>>>(head, dev_points, npoints);

    cudaDeviceSynchronize();

    auto insert_end = high_resolution_clock::now();

    std::cout << "Inserted " << npoints << " items in " << std::fixed << elapsed(insert_start, insert_end) << ", throughput " << 1.0*npoints/elapsed(insert_start, insert_end) << std::endl;


    cudaFree(head);


    cudaDeviceSynchronize();


    // uint64_t nodes_to_boot = 10000000;

    //auto malloc_start = high_resolution_clock::now();

    // boot_many_nodes<<<(nodes_to_boot-1)/512+1,512>>>(nodes_to_boot);

    // cudaDeviceSynchronize();

    // auto malloc_end = high_resolution_clock::now();

    // std::cout << "Acquired memory for " << nodes_to_boot << " in " << std::fixed << elapsed(malloc_start, malloc_end) << ", throughput " << 1.0*nodes_to_boot/elapsed(malloc_start, malloc_end) << std::endl;


    free_allocator();

}




