#include "ex2.h"
#include <cuda/atomic>

#define HISTOGRAM_SIZE 256
#define NUM_OF_THREADS 1024
#define WRAP_SIZE 32
#define SHARED_MEM_USAGE 3072

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
    int bi = blockIdx.x;
    int workForThread = (TILE_WIDTH * TILE_WIDTH) / NUM_OF_THREADS; // in bytes
    uchar imageVal;

    __shared__ int sharedHist[HISTOGRAM_SIZE]; // maybe change to 16 bit ? will be confilcits on same bank 

    int imageStartIndex = 0;// bi * IMG_HEIGHT * IMG_WIDTH;
    int mapStartIndex = 0;// bi * TILE_COUNT * TILE_COUNT * HISTOGRAM_SIZE;
    int tileStartIndex;
    int insideTileIndex;
    int curIndex;
    for (int i = 0 ; i < TILE_COUNT * TILE_COUNT; i++)
    {
        // calc tile index in image buffer (shared between al threads in block)
        tileStartIndex = imageStartIndex + i % TILE_COUNT * TILE_WIDTH + (i / TILE_COUNT) * (TILE_WIDTH *TILE_WIDTH) * TILE_COUNT;
        // zero shared buffer histogram values
        if (ti < 256)
        {
            sharedHist[ti] = 0;
        }
        __syncthreads();
       for (int j = 0; j < workForThread; j++)
            {
                // calc index in tile buffer for each thread
                insideTileIndex = tg * TILE_WIDTH * TILE_COUNT + ti % TILE_WIDTH + (NUM_OF_THREADS / TILE_WIDTH) * TILE_WIDTH * TILE_COUNT * j;
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
            maps[mapStartIndex + HISTOGRAM_SIZE * i + ti] = (float(sharedHist[ti]) * 255)  / (TILE_WIDTH * TILE_WIDTH);
        }
    }

    __syncthreads();
    // interpolate image using given maps buffer
    interpolate_device(maps + mapStartIndex, in + imageStartIndex, out + imageStartIndex);
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
            streams[i].streamImageId = -1; // avialble
            CUDA_CHECK(cudaMalloc((void**)&streams[i].taskMaps, TILE_COUNT * TILE_COUNT * HISTOGRAM_SIZE));
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
            CUDA_CHECK(cudaFree(streams[i].imgIn));
            CUDA_CHECK(cudaFree(streams[i].imgOut));
        }
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO place memory transfers and kernel invocation in streams if possible.
        for (int i = 0; i < STREAM_COUNT; i++)
        {
            if (streams[i].streamImageId == -1)
            {
                streams[i].streamImageId = img_id;
                //CUDA_CHECK(cudaMemcpyAsync(streams[i].imgIn, img_in , IMG_WIDTH * IMG_HEIGHT,cudaMemcpyHostToDevice, streams[i].stream));
                process_image_kernel<<<1, NUM_OF_THREADS, 0, streams[i].stream>>>(img_in, img_out, streams[i].taskMaps);
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
            if (streams[i].streamImageId != -1)
            {
                cudaError_t status = cudaStreamQuery(streams[i].stream); // TODO query diffrent stream each iteration
                switch (status) {
                case cudaSuccess:
                    // TODO return the img_id of the request that was completed.
                    // printf("bla");
                    *img_id = streams[i].streamImageId;
                    streams[i].streamImageId = -1;
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




// TODO implement a lock
// TODO implement a MPMC queue
// TODO implement the persistent kernel
// TODO implement a function for calculating the threadblocks count

int calc_max_thread_blocks()
    {
        int register_per_thread = 32;
        int threads_per_thread_block = NUM_OF_THREADS;
        
        //constraints
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
        int max_shared_mem_sm = deviceProp.sharedMemPerMultiprocessor;
        int max_regs_per_sm = deviceProp.regsPerMultiprocessor;
        int max_wraps_per_sm = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
        // int max_block_per_sm = deviceProp.maxBlocksPerMultiProcessor;
;

        int max_warps_per_block = deviceProp.maxThreadsPerBlock / deviceProp.warpSize;
    }

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
//	ring_buffer cpu_to_gpu;
//	ring_buffer gpu_to_cpu;
public:
    queue_server(int threads)
    {
//        int tb_num = 3;//TODO calc from calc_max_thread_blocks
//	int ring_buf_size = 16*3 ;//TODO - calc 2^celling(log2(16*tb_num)/log2(2))
//
//	   
//	// TODO initialize host state
//	cpu_to_gpu = ring_buffer(ring_buf_size);
//	gpu_to_cpu = ring_buffer(ring_buf_size);
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks

    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
  //      // TODO push new task into queue if possible
  //          request request_i;
  //          CUDA_CHECK(cudaMemcpyAsync(streams[i].imgIn, img_in , IMG_WIDTH * IMG_HEIGHT,cudaMemcpyHostToDevice, streams[i].stream));
  //          cpu_to_gpu.push(request_i);
        return false;
    }

    bool dequeue(int *img_id) override
    {

//	    request request_i = gpu_to_cpu.pop();
//	    cpu_to_gpu.push(request_i);
//        // TODO query (don't block) the producer-consumer queue for any responses.
//        return false;
//
//        // TODO return the img_id of the request that was completed.
//        //*img_id = ... 
//        return true;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}

struct request
{
	int imgID;	
    	uchar *taskMaps;
    	uchar *imgIn;
    	uchar *imgOut;
};

//class ring_buffer {
//	private:
//		static int N;
//		request _mailbox[N];
//		cuda::atomic<int> _head, _tail;
//	public:
//		ring_buffer(int size){
//			 N = 1 << size;
//			_head = 0, _tail = 0;
//		}
//		void push(const request &data) {
//	 		int tail = _tail.load(memory_order_relaxed);
//	 		while (tail - _head.load(memory_order_acquire) == N);
//			 _mailbox[_tail % N] = data;
//	 		_tail.store(tail + 1, memory_order_release);
//	 	}
//	 	request pop() {
//	 		int head = _head.load(memory_order_relaxed);
//	 		while (_tail.load(memory_order_acquire) == _head);
//	 		request item = _mailbox[_head % N];
//	 		_head.store(head + 1, memory_order_release);
//	 		return item;
//	 	}
/
