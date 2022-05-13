#include "ex2.h"
#include <cuda/atomic>

#define HISTOGRAM_SIZE 256
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
    int workForThread = (TILE_WIDTH * TILE_WIDTH) / blockDim.x; // in bytes
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
           // CUDA_CHECK(cudaFree(streams[i].imgIn));
           // CUDA_CHECK(cudaFree(streams[i].imgOut));
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


struct request
{
	int imgID;	
    	uchar *taskMaps;
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
		~ring_buffer(){
            		CUDA_CHECK(cudaFreeHost(_mailbox));
		} 
		ring_buffer(int size){
			 N = size;
			_head = 0, _tail = 0;
    			CUDA_CHECK( cudaMallocHost(&_mailbox, sizeof(request)*N ));
		}
		__device__ __host__
		bool push(const request &data) {
	 		int tail = _tail.load(cuda::memory_order_relaxed);
	 		if (tail - _head.load(cuda::memory_order_acquire) != N){
				_mailbox[_tail % N] = data;
	 			_tail.store(tail + 1, cuda::memory_order_release);
				return true;
			} else{
				return false;
			}
	 	}
		__device__ __host__
	 	request pop() {
	 		int head = _head.load(cuda::memory_order_relaxed);
			request item;
	 		if (_tail.load(cuda::memory_order_acquire) != _head){
	 			item = _mailbox[_head % N];
	 			_head.store(head + 1, cuda::memory_order_release);
			} else{
				item.imgID = -1;//item is not valid
			}
	 		return item;
	 	}
};

__global__
void process_image_kernel_queue(ring_buffer* cpu_to_gpu, ring_buffer* gpu_to_cpu){
	request req_i;
	while(1){
		req_i = cpu_to_gpu->pop();	
		if(req_i.imgID != -1){
    			process_image(req_i.imgIn, req_i.imgOut, req_i.taskMaps);
			while(!gpu_to_cpu->push(req_i));
		}	
	}
}



int calc_max_thread_blocks(int threads)
    {
        int register_per_thread = 32;
        int threads_per_thread_block = threads;
        
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
	ring_buffer* cpu_to_gpu;
	ring_buffer* gpu_to_cpu;

	char* cpu_to_gpu_buf;
	char* gpu_to_cpu_buf;
	
public:
    queue_server(int threads)
    {
        int tb_num = 1;//TODO calc from calc_max_thread_blocks
	int ring_buf_size = 16*1 ;//TODO - calc 2^celling(log2(16*tb_num)/log2(2))

       	CUDA_CHECK(cudaMallocHost(&cpu_to_gpu_buf,sizeof(ring_buffer)));
       	CUDA_CHECK(cudaMallocHost(&gpu_to_cpu_buf,sizeof(ring_buffer)));

	cpu_to_gpu = new (cpu_to_gpu_buf) ring_buffer(ring_buf_size);
	gpu_to_cpu = new (gpu_to_cpu_buf) ring_buffer(ring_buf_size);

        //  launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
	process_image_kernel_queue<<<tb_num,threads>>>(cpu_to_gpu,gpu_to_cpu);
    }

    ~queue_server() override
    {
	cpu_to_gpu->~ring_buffer();
	gpu_to_cpu->~ring_buffer();
	cudaFreeHost(cpu_to_gpu_buf);
	cudaFreeHost(gpu_to_cpu_buf);
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
            request request_i;
	    request_i.imgID = img_id;
	    CUDA_CHECK(cudaMalloc((void**)&request_i.taskMaps,  TILE_COUNT * TILE_COUNT * HISTOGRAM_SIZE));
	    request_i.imgIn = img_in;
	    request_i.imgOut = img_out;

	    //CUDA_CHECK(cudaMemcpyAsync(streams[i].imgIn, img_in , IMG_WIDTH * IMG_HEIGHT,cudaMemcpyHostToDevice, streams[i].stream));
            if(cpu_to_gpu->push(request_i)){
		return true;
	    } else{
            	CUDA_CHECK(cudaFree(request_i.taskMaps));
		return false;
	    }
        return false;
    }

    bool dequeue(int *img_id) override
    {

	request request_i = gpu_to_cpu->pop();
	if(request_i.imgID == -1){
		return false;// queue is empty
	} else {
		*img_id = request_i.imgID;
            	CUDA_CHECK(cudaFree(request_i.taskMaps));
		return true;	
	}
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
