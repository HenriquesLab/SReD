//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define image_width $WIDTH$
#define image_height $HEIGHT$
#define image_depth $DEPTH$
#define block_size $BLOCK_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define bRZ $BRZ$
#define EPSILON $EPSILON$

kernel void kernelGetLocalStatistics3D(
global float* ref_pixels,
global float* local_means,
global float* local_stds
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int gz = get_global_id(2);

    // Bound check (avoids borders dynamically based on block dimensions)
    if(gx<bRW || gx>=image_width-bRW || gy<bRH || gy>=image_height-bRH || gz<bRZ || gz>=image_depth-bRZ){
        local_means[image_width*image_height*gz+gy*image_width+gx] = 0.0f;
        local_stds[image_width*image_height*gz+gy*image_width+gx] = 0.0f;
        return;
    }

    // Get block pixels
    float block[block_size];
    int index = 0;
    for(int z=gz-bRZ; z<=gz+bRZ; z++){
        for(int y=gy-bRH; y<=gy+bRH; y++){
            for(int x=gx-bRW; x<=gx+bRW; x++){
                // Extract only pixels within the inbound spheroid
                float dx = (float)(x-gx);
                float dy = (float)(y-gy);
                float dz = (float)(z-gz);
                if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH))+((dz*dz)/(float)(bRZ*bRZ)) <= 1.0f){
                    block[index] = ref_pixels[image_width*image_height*z+y*image_width+x];
                    index++;
                }
            }
        }
    }

    // Calculate block mean
    float mean = 0.0f;
    for(int i=0; i<block_size; i++){
        mean += block[i];
    }
    mean /= (float)block_size;
    local_means[image_width*image_height*gz+gy*image_width+gx] = mean;

    // Calculate block StdDev
    float var = 0.0f;
    for(int i=0; i<block_size; i++){
        var += (block[i] - mean) * (block[i] - mean);
    }

    local_stds[image_width*image_height*gz+gy*image_width+gx] = (float)sqrt(var/(float)(block_size-1));
}
