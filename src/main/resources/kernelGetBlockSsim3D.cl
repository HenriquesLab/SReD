//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define imageWidth $WIDTH$
#define imageHeight $HEIGHT$
#define imageDepth $DEPTH$
#define block_size $BLOCK_SIZE$
#define bW $BW$
#define bH $BH$
#define bZ $BZ$
#define bRW $BRW$
#define bRH $BRH$
#define bRZ $BRZ$
#define ref_mean $BLOCK_MEAN$
#define ref_std $BLOCK_STD$
#define EPSILON $EPSILON$

kernel void kernelGetBlockSsim3D(
    global float* block_pixels,
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* ssim_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int gz = get_global_id(2);

    // Bound check (avoids borders dynamically based on block dimensions)
    if(gx<bRW || gx>=imageWidth-bRW || gy<bRH || gy>=imageHeight-bRH || gz<bRZ || gz>=imageDepth-bRZ){
        return;
    }


    // --------------------------------------------- //
    // ---- Get mean-subtracted reference block ---- //
    // --------------------------------------------- //

    __local float ref_block[block_size]; // Make a local copy to avoid slower reads from global memory

    for(int i=0; i<block_size; i++){
        ref_block[i] = block_pixels[i]; // Block is mean-subtracted in the host Java class
    }


    // ------------------------------------- //
    // ---- Get comparison block pixels ---- //
    // ------------------------------------- //

    float comp_block[block_size] = {0.0f};
    int index = 0;
    for(int z=gz-bRZ; z<=gz+bRZ; z++){
        for(int y=gy-bRH; y<=gy+bRH; y++){
            for(int x=gx-bRW; x<=gx+bRW; x++){
                float dx = (float)(x-gx);
                float dy = (float)(y-gy);
                float dz = (float)(z-gz);
                if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)+((dz*dz)/(float)(bRZ*bRZ))) <= 1.0f){
                    comp_block[index] = ref_pixels[imageWidth*imageHeight*z+y*imageWidth+x];
                    index++;
                }
            }
        }
    }

    // Mean-subtract comparison block
    float comp_mean = local_means[imageWidth*imageHeight*gz+gy*imageWidth+gx];
    for(int i=0; i<block_size; i++){
        comp_block[i] = comp_block[i] - comp_mean;
    }


    // ------------------------------- //
    // ---- Calculate Covariance ----- //
    // ------------------------------- //

    float covariance = 0.0f;
    for(int i=0; i<block_size; i++){
        covariance += ref_block[i] * comp_block[i];
    }
    covariance /= (float)(block_size-1);


    // ------------------------ //
    // ---- Calculate SSIM ---- //
    // ------------------------ //

    //float c1 = (0.01f * 1.0f) * (0.01f * 1.0f);
    float c1 = 0.0001f;
    //float c2 = (0.03f * 1.0f) * (0.03f * 1.0f);
    float c2 = 0.0009f;
    //float c3 = c2/2.0f;
    float c3 = 0.00045f;
    float comp_std = local_stds[imageWidth*imageHeight*gz+gy*imageWidth+gx];

    if(ref_std == 0.0 && comp_std == 0.0){
        ssim_map[imageWidth*imageHeight*gz+gy*imageWidth+gx] = 1.0f; // Special case when both blocks are flat, correlation is 1
    }else if(ref_std == 0.0 || comp_std == 0.0){
        ssim_map[imageWidth*imageHeight*gz+gy*imageWidth+gx] = 0.0f; // Special case when one block is flat, correlation is 0
    }else{
        float ssim = (float) ((2.0f*ref_mean*comp_mean+c1)*(2.0f*covariance+c2))/(((ref_mean*ref_mean)+(comp_mean*comp_mean)+c1)*((ref_std*ref_std)+(comp_std*comp_std)+c2));
        ssim_map[imageWidth*imageHeight*gz+gy*imageWidth+gx] = (float) fmax(0.0f, ssim);
    }
}
