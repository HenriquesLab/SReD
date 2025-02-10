//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define image_width $WIDTH$
#define image_height $HEIGHT$
#define image_depth $DEPTH$
#define block_size $BLOCK_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define bRZ $BRZ$
#define EPSILON $EPSILON$

kernel void kernelGetBlockNrmse3D(
    global float* block_pixels,
    global float* ref_pixels,
    global float* local_means,
    global float* rmse_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int gz = get_global_id(2);

    // Bound check (avoids borders dynamically based on block dimensions)
    if(gx<bRW || gx>=image_width-bRW || gy<bRH || gy>=image_height-bRH || gz<bRZ || gz>=image_depth-bRZ){
        rmse_map[image_width*image_height*gz+gy*image_width+gx] = 0.0f;
        return;
    }


    // Get mean-subtracted reference block
    __local float ref_block[block_size]; // Make a local copy to avoid slower reads from global memory

    for(int i=0; i<block_size; i++){
        ref_block[i] = block_pixels[i]; // Block is mean-subtracted in the host Java class
    }


    // Get comparison block pixels
    float comp_block[block_size] = {0.0f};
    int index = 0;
    for(int z=gz-bRZ; z<=gz+bRZ; z++){
        for(int y=gy-bRH; y<=gy+bRH; y++){
            for(int x=gx-bRW; x<=gx+bRW; x++){
                float dx = (float)(x-gx);
                float dy = (float)(y-gy);
                float dz = (float)(z-gz);
                if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH))+((dz*dz)/(float)(bRZ*bRZ)) <= 1.0f){
                    comp_block[index] = ref_pixels[image_width*image_height*z+y*image_width+x];
                    index++;
                }
            }
        }
    }

    // Mean-subtract comparison block
    float comp_mean = local_means[image_width*image_height*gz+gy*image_width+gx];
    for(int i=0; i<block_size; i++){
        comp_block[i] = comp_block[i] - comp_mean;
    }

    // Calculate NRMSE
    float mse = 0.0f;
    for(int i=0; i<block_size; i++){
        mse += (ref_block[i]-comp_block[i])*(ref_block[i]-comp_block[i]);
    }
    mse /= (float) block_size;
    rmse_map[image_width*image_height*gz+gy*image_width+gx] = sqrt(mse);
}