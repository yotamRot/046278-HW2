#include "ex2.h"
#include <cuda/atomic>
#include <iostream>
#include <cstdlib>
#include <unistd.h>

#define HISTOGRAM_SIZE 256
#define WRAP_SIZE 32
#define SHARED_MEM_USAGE 2048
#define REGISTERS_PER_THREAD 32
#define INVALID_IMAGE -1
#define KILL_IMAGE -2


__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;
    for (int stride = 1; stride < arr_size; stride *= 2) {
    if (tid >= stride && tid < arr_size) {
        increment = arr[tid - stride];
    }
    __syncthreads();
    if (tid >= stride && tid < arr_size) {
        arr[tid] += increment;
    }
    __syncthreads();
    }
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__
 void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

__device__
void process_image(uchar *in, uchar *out, uchar* maps) {
    int ti = threadIdx.x;
    int tg = ti / TILE_WIDTH;
    int workForThread = (TILE_WIDTH * TILE_WIDTH) / blockDim.x; // in bytes
    uchar imageVal;

    __shared__ int sharedHist[HISTOGRAM_SIZE]; // maybe change to 16 bit ? will be confilcits on same bank 

    int tileStartIndex;
    int insideTileIndex;
    int curIndex;
    for (int i = 0 ; i < TILE_COUNT * TILE_COUNT; i++)
    {
        // calc tile index in image buffer (shared between al threads in block)
        tileStartIndex = i % TILE_COUNT * TILE_WIDTH + (i / TILE_COUNT) * (TILE_WIDTH *TILE_WIDTH) * TILE_COUNT;
        // zero shared buffer histogram values
        if (ti < 256)
        {
            sharedHist[ti] = 0;
        }
        __syncthreads();
       for (int j = 0; j < workForThread; j++)
            {
                // calc index in tile buffer for each thread
                insideTileIndex = tg * TILE_WIDTH * TILE_COUNT + ti % TILE_WIDTH + (blockDim.x / TILE_WIDTH) * TILE_WIDTH * TILE_COUNT * j;
                // sum tile index and index inside tile to find relevant byte for thread in cur iteration
                curIndex = tileStartIndex + insideTileIndex;
                // update histogram
                imageVal = in[curIndex];
                atomicAdd(sharedHist + imageVal, 1);
        }
    
        __syncthreads();
        
        // calc CDF using prefix sumpwdon histogram buffer

        prefix_sum(sharedHist, HISTOGRAM_SIZE);

        __syncthreads();
        // calc map value for each index
        if (ti < 256)
        {
            maps[HISTOGRAM_SIZE * i + ti] = (float(sharedHist[ti]) * 255)  / (TILE_WIDTH * TILE_WIDTH);
        }
    }

    __syncthreads();
    // interpolate image using given maps buffer
    interpolate_device(maps, in, out);
    return; 
}

__global__
void process_image_kernel(uchar *in, uchar *out, uchar* maps){
    process_image(in, out, maps);
}

struct Stream_Wrap
{
    cudaStream_t stream;
    int streamImageId;
    uchar *taskMaps;
    uchar *imgIn;
    uchar *imgOut;
};

class streams_server : public image_processing_server
{
private:
    // TODO define stream server context (memory buffers, streams, etc...)
    Stream_Wrap streams[STREAM_COUNT];

public:
    streams_server()
    {
        // TODO initialize context (memory buffers, streams, etc...)
        for (int i = 0; i < STREAM_COUNT; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i].stream));
            streams[i].streamImageId = INVALID_IMAGE; // avialble
            CUDA_CHECK(cudaMalloc((void**)&(streams[i].taskMaps), TILE_COUNT * TILE_COUNT * HISTOGRAM_SIZE));
            //CUDA_CHECK(cudaMalloc((void**)&streams[i].imgIn,  IMG_WIDTH * IMG_HEIGHT));
            //CUDA_CHECK(cudaMalloc((void**)&streams[i].imgOut,IMG_WIDTH * IMG_HEIGHT));
        }
    }

    ~streams_server() override
    {
        // TODO free resources allocated in constructor
        for (int i = 0; i < STREAM_COUNT; i++) {
            CUDA_CHECK(cudaStreamDestroy(streams[i].stream));
            CUDA_CHECK(cudaFree(streams[i].taskMaps));
           // CUDA_CHECK(cudaFree(streams[i].imgIn));
           // CUDA_CHECK(cudaFree(streams[i].imgOut));
        }
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO place memory transfers and kernel invocation in streams if possible.
        for (int i = 0; i < STREAM_COUNT; i++)
        {
            if (streams[i].streamImageId == INVALID_IMAGE)
            {
                streams[i].streamImageId = img_id;
                //CUDA_CHECK(cudaMemcpyAsync(streams[i].imgIn, img_in , IMG_WIDTH * IMG_HEIGHT,cudaMemcpyHostToDevice, streams[i].stream));
                process_image_kernel<<<1, 1024, 0, streams[i].stream>>>(img_in, img_out, streams[i].taskMaps);
                //CUDA_CHECK(cudaMemcpyAsync(img_out, streams[i].imgOut, IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToHost, streams[i].stream));
                return true;
            }
        }
        return false;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) streams for any completed requests.
        for (int i = 0; i < STREAM_COUNT; i++)
        {
            if (streams[i].streamImageId != INVALID_IMAGE)
            {
                cudaError_t status = cudaStreamQuery(streams[i].stream); // TODO query diffrent stream each iteration
                switch (status) {
                case cudaSuccess:
                    // TODO return the img_id of the request that was completed.
                    *img_id = streams[i].streamImageId;
                    streams[i].streamImageId = INVALID_IMAGE;
                    return true;
                case cudaErrorNotReady:
		    continue;
                default:
                    CUDA_CHECK(status);
                    return false;
                }
            }
            
        }

        return false;
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

typedef cuda::atomic<int, cuda::thread_scope_device> gpu_atomic_int;
__device__ gpu_atomic_int* push_lock;
__device__ gpu_atomic_int* pop_lock;

__global__
void init_locks() 
{ 
    push_lock = new gpu_atomic_int(0); 
    pop_lock = new gpu_atomic_int(0); 
}

__global__
void free_locks() 
{ 
    delete push_lock; 
    delete pop_lock; 
}
__device__
 void lock(gpu_atomic_int * l) 
{
   do 
    {
        while (l->load(cuda::memory_order_relaxed)) continue;
    } while (l->exchange(1, cuda::memory_order_relaxed)); // actual atomic locking
  
   // while (l->exchange(1, cuda::memory_order_acq_rel));
}
__device__
 void unlock(gpu_atomic_int * l) 
{
    l->store(0, cuda::memory_order_release);
}

// TODO implement a lock
// TODO implement a MPMC queue
// TODO implement the persistent kernel
// TODO implement a function for calculating the threadblocks count


struct request
{
	    int imgID;	
    	uchar *imgIn;
    	uchar *imgOut;
};
class ring_buffer {
	private:
		int N;
		request* _mailbox;
		cuda::atomic<int> _head, _tail;
	public:

		ring_buffer(); // def contructor
		~ring_buffer()
        {
            CUDA_CHECK(cudaFreeHost(_mailbox));
		} 
		ring_buffer(int size)
        {
			 N = size;
			_head = 0, _tail = 0;
            CUDA_CHECK( cudaMallocHost(&_mailbox, sizeof(request)*N ));
		}

		__device__ __host__
		bool push(const request data) 
        {
	 		int tail = _tail.load(cuda::memory_order_relaxed);
            // printf("push function - tail is: %d img id is - %d\n" , tail, data.imgID);
	 		if (tail - _head.load(cuda::memory_order_acquire) != N){
				_mailbox[tail % N] = data;
	 			_tail.store(tail + 1, cuda::memory_order_release);
				return true;
			} else{
				return false;
			}
	 	}

		__device__ __host__
	 	request pop() 
         {
	 		int head = _head.load(cuda::memory_order_relaxed);
            // printf("pop function - head is: %d \n" , head);
			request item;
	 		if (_tail.load(cuda::memory_order_acquire) != head){
	 			item = _mailbox[head % N];
	 			_head.store(head + 1, cuda::memory_order_release);
			} else{
				item.imgID = INVALID_IMAGE;//item is not valid
			}
	 		return item;
	 	}
};

__global__
void process_image_kernel_queue(ring_buffer* cpu_to_gpu, ring_buffer* gpu_to_cpu, uchar* maps)
{
	__shared__ request req_i;
    int tid = threadIdx.x;
    uchar* block_maps = maps + blockIdx.x * TILE_COUNT * TILE_COUNT * HISTOGRAM_SIZE;
	while(1)
    {

        if (tid == 0)
        {   
            lock(pop_lock);
            req_i = cpu_to_gpu->pop();
            unlock(pop_lock);
        }

        // got request to stop
        __syncthreads();
        if (req_i.imgID == KILL_IMAGE)
        {
          return;
        }
		else if (req_i.imgID != INVALID_IMAGE && req_i.imgID != KILL_IMAGE) 
        {


            //  printf("image id poped by gpu = %d\n",req_i.imgID);

            __syncthreads();
             process_image(req_i.imgIn, req_i.imgOut, block_maps);
             __syncthreads();

            if(tid == 0) {
                // printf("gpu proccess - befor push image id : %d\n", req_i.imgID);
                lock(push_lock);
                while(!gpu_to_cpu->push(req_i));
                unlock(push_lock);
                // printf("gpu proccess - affter push image id : %d\n", req_i.imgID);

            }
		}	
	}
}



int calc_max_thread_blocks(int threads)
    {
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));


          //constraints
        // int max_tb_sm = deviceProp.maxBlocksPerMultiProcessor;
        int max_shared_mem_sm = deviceProp.sharedMemPerMultiprocessor;
        int max_regs_per_sm = deviceProp.regsPerMultiprocessor;
        int max_threads_per_sm = deviceProp.maxThreadsPerMultiProcessor;
        // int max_block_per_sm = deviceProp.maxBlocksPerMultiProcessor;

        int max_tb_mem_constraint = max_shared_mem_sm / SHARED_MEM_USAGE;
        int max_tb_reg_constraint = max_regs_per_sm / (REGISTERS_PER_THREAD * threads);
        int max_tb_threads_constraint = max_threads_per_sm / threads;

        int max_tb = std::min(max_tb_mem_constraint,std::min(max_tb_reg_constraint, max_tb_threads_constraint));
        int max_num_sm = deviceProp.multiProcessorCount;
        return max_num_sm * max_tb;

    }

class queue_server : public image_processing_server
{
private:
	ring_buffer* cpu_to_gpu;
	ring_buffer* gpu_to_cpu;

	char* cpu_to_gpu_buf;
	char* gpu_to_cpu_buf;
	uchar* server_maps;
    int tb_num;
	
public:
    queue_server(int threads)
    {
        tb_num = calc_max_thread_blocks(threads);//TODO calc from calc_max_thread_blocks
        int ring_buf_size = std::pow(2, std::ceil(std::log(16*tb_num)/std::log(2)));//TODO - calc 2^celling(log2(16*tb_num)/log2(2))
        
        // printf("tb_num %d\n", tb_num);
        // printf("ring_buf_size %d\n", ring_buf_size);

        CUDA_CHECK(cudaMalloc((void**)&server_maps, tb_num * TILE_COUNT * TILE_COUNT * HISTOGRAM_SIZE));

        CUDA_CHECK(cudaMallocHost(&cpu_to_gpu_buf, sizeof(ring_buffer)));
        CUDA_CHECK(cudaMallocHost(&gpu_to_cpu_buf, sizeof(ring_buffer)));  
        cpu_to_gpu = new (cpu_to_gpu_buf) ring_buffer(ring_buf_size);
        gpu_to_cpu = new (gpu_to_cpu_buf) ring_buffer(ring_buf_size);

            //  launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
        
        init_locks<<<1, 1>>>();
        CUDA_CHECK(cudaDeviceSynchronize());
        process_image_kernel_queue<<<tb_num, threads>>>(cpu_to_gpu, gpu_to_cpu, server_maps);
    }

    ~queue_server() override
    {
        //Kill kernel
        for (int i = 0 ; i<tb_num; i++)
        {
            // send enough kills to kill all tb
            this->enqueue(KILL_IMAGE, NULL, NULL);
        }
        CUDA_CHECK(cudaDeviceSynchronize()); 
        free_locks<<<1, 1>>>();
        CUDA_CHECK(cudaDeviceSynchronize()); 
        cpu_to_gpu->~ring_buffer();
        gpu_to_cpu->~ring_buffer();
        CUDA_CHECK(cudaFreeHost(cpu_to_gpu_buf));
        CUDA_CHECK(cudaFreeHost(gpu_to_cpu_buf));
        CUDA_CHECK(cudaFree(server_maps));

    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // printf("started enqueue, img ID = %d \n", img_id);
        request request_i;
	    request_i.imgID = img_id;
	    request_i.imgIn = img_in;
	    request_i.imgOut = img_out;

        if(cpu_to_gpu->push(request_i)){
		    return true;
	    } else{

		    return false;
	    }
        return false;
    }

    bool dequeue(int *img_id) override
    {
        // printf("cpu started dequeue\n");
        request request_i = gpu_to_cpu->pop();
        // printf("cpu dequeue - after pop, img ID = %d , taskmaps ptr = %p \n", request_i.imgID,request_i.taskMaps);
        if(request_i.imgID == INVALID_IMAGE){
            return false;// queue is empty
        } else {
            *img_id = request_i.imgID;
            return true;	
        }
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
