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

kernel void kernelGetBlockPearson3D(
    global float* block_pixels,
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* pearson_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int gz = get_global_id(2);

    // Bound check (avoids borders dynamically based on block dimensions)
    if(gx<bRW || gx>=imageWidth-bRW || gy<bRH || gy>=imageHeight-bRH || gz<bRZ || gz>=imageDepth-bRZ){
        return;
    }


    // ------------------------------------------------------------------------ //
    // ---- Get mean-subtracted and normalized reference block from buffer ---- //
    // ------------------------------------------------------------------------ //

    __local float ref_block[block_size]; // Make a local copy to avoid slower reads from global memory

    for(int i=0; i<block_size; i++){
        ref_block[i] = block_pixels[i];
    }


    // ------------------------------------- //
    // ---- Get comparison block pixels ---- //
    // ------------------------------------- //

    float comp_block[block_size];
    int index = 0;
    for(int z=gz-bRZ; z<=gz+bRZ; z++){
        for(int y=gy-bRH; y<=gy+bRH; y++){
            for(int x=gx-bRW; x<=gx+bRW; x++){
                float dx = (float)(x-gx);
                float dy = (float)(y-gy);
                float dz = (float)(z-gz);
                if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH))+((dz*dz)/(float)(bRZ*bRZ)) <= 1.0f){
                    comp_block[index] = ref_pixels[imageWidth*imageHeight*z+y*imageWidth+x];
                    index++;
                }
            }
        }
    }


    // ------------------------------------ //
    // ---- Normalize comparison block ---- //
    // ------------------------------------ //
    float min_intensity = FLT_MAX;
    float max_intensity = -FLT_MAX;

    for(int i=0; i<block_size; i++){
        float pixel_value = comp_block[i];
        min_intensity = min(min_intensity, pixel_value);
        max_intensity = max(max_intensity, pixel_value);
    }

    // Remap pixel values
    for(int i=0; i<block_size; i++){
        comp_block[i] = (comp_block[i] - min_intensity) / (max_intensity - min_intensity + EPSILON);
    }


    // ---------------------------------------- //
    // ---- Mean-subtract comparison block ---- //
    // ---------------------------------------- //

    float comp_mean = local_means[imageWidth*imageHeight*gz+gy*imageWidth+gx];
    for(int i=0; i<block_size; i++){
        comp_block[i] = comp_block[i] - comp_mean;
    }


    // ------------------------------------------ //
    // ---- Normalize comparison block again ---- //
    // ------------------------------------------ //

    min_intensity = FLT_MAX;
    max_intensity = -FLT_MAX;

    for(int i=0; i<block_size; i++){
        float pixel_value = comp_block[i];
        min_intensity = min(min_intensity, pixel_value);
        max_intensity = max(max_intensity, pixel_value);
    }

    // Remap pixel values
    for(int i=0; i<block_size; i++){
        comp_block[i] = (comp_block[i] - min_intensity) / (max_intensity - min_intensity + EPSILON);
    }


    // ------------------------- //
    // ---- Get Covariance ----- //
    // ------------------------- //

    float covariance = 0.0f;
    for(int i=0; i<block_size; i++){
        covariance += ref_block[i] * comp_block[i];

    }
    covariance /= (float)(block_size-1);

    // Calculate Pearson correlation coefficient REF vs. COMP and add it to the sum at REF
    float comp_std = local_stds[imageWidth*imageHeight*gz+gy*imageWidth+gx];

    if(ref_std == 0.0 && comp_std == 0.0){
        pearson_map[imageWidth*imageHeight*gz+gy*imageWidth+gx] = 1.0f; // Special case when both blocks are flat (correlation would be NaN but we want 1 because textures are the same)
    }else if(ref_std==0.0 || comp_std==0.0){
        pearson_map[imageWidth*imageHeight*gz+gy*imageWidth+gx] = 0.0f; // Special case when only one block is flat, correlation would be NaN but we want 0
    }else{
        pearson_map[imageWidth*imageHeight*gz+gy*imageWidth+gx] = (float)fmax(0.0f, (float)(covariance / ((ref_std * comp_std) + EPSILON))); // Truncate anti-correlations
    }
}
