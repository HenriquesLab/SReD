//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define image_width $WIDTH$
#define image_height $HEIGHT$
#define block_size $BLOCK_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define EPSILON $EPSILON$

kernel void kernelGetLocalStatistics2D(
global float* ref_pixels,
global float* local_means,
global float* local_stds
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on block dimensions)
    if(gx<bRW || gx>=image_width-bRW || gy<bRH || gy>=image_height-bRH){
        local_means[gy*image_width+gx] = 0.0f;
        local_stds[gy*image_width+gx] = 0.0f;
        return;
    }

    // Get block pixels
    float block[block_size];
    int index = 0;
    for(int y=gy-bRH; y<=gy+bRH; y++){
        for(int x=gx-bRW; x<=gx+bRW; x++){
            // Extract only pixels within the inbound circle/ellipse
            float dx = (float)(x-gx);
            float dy = (float)(y-gy);
            if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){
                block[index] = ref_pixels[y*image_width+x];
                index++;
            }
        }
    }

    // Calculate block mean
    float mean = 0.0f;
    for(int i=0; i<block_size; i++){
        mean += block[i];
    }
    mean /= (float)block_size;
    local_means[gy*image_width+gx] = (float)mean;

    // Calculate block StdDev
    float var = 0.0f;
    for(int i=0; i<block_size; i++){
        var += (block[i] - mean) * (block[i] - mean);
    }
    var /= (float)(block_size-1);

    local_stds[gy*image_width+gx] = (float)sqrt(var);
}
