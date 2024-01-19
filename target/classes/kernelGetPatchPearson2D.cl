#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define patch_size $PATCH_SIZE$
#define bW $BW$
#define bH $BH$
#define bRW $BRW$
#define bRH $BRH$
#define ref_mean $PATCH_MEAN$
#define ref_std $PATCH_STD$
#define EPSILON $EPSILON$

kernel void kernelGetPatchPearson2D(
    global float* patch_pixels,
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* pearson_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }


    // --------------------------------------------- //
    // ---- Get mean-subtracted reference block ---- //
    // --------------------------------------------- //

    __local double ref_patch[patch_size]; // Make a local copy to avoid slower reads from global memory

    for(int i=0; i<patch_size; i++){
        ref_patch[i] = (double)patch_pixels[i];
    }


    // ------------------------------------- //
    // ---- Get comparison patch pixels ---- //
    // ------------------------------------- //

    double comp_patch[patch_size] = {0.0};
    int index = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            float dx = (float)(i-gx);
            float dy = (float)(j-gy);
            if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){
                comp_patch[index] = (double)ref_pixels[j*w+i];
                index++;
            }
        }
    }


    // ---------------------------------------- //
    // ---- Mean-subtract comparison patch ---- //
    // ---------------------------------------- //

    double comp_mean = (double)local_means[gy*w+gx];
    for(int i=0; i<patch_size; i++){
        comp_patch[i] = comp_patch[i] - comp_mean;
    }


    // ------------------------------- //
    // ---- Calculate covariance ----- //
    // ------------------------------- //

    double covariance = 0.0;
    for(int i=0; i<patch_size; i++){
        covariance += ref_patch[i] * comp_patch[i];
    }
    covariance /= (double)(patch_size-1);


    // ----------------------------------------------------- //
    // ---- Calculate Pearson's correlation coefficient ---- //
    // ----------------------------------------------------- //

    double ref_std_d = (double)ref_std;
    double comp_std_d = (double)local_stds[gy*w+gx];

    if(ref_std_d == 0.0 && comp_std_d == 0.0){
        pearson_map[gy*w+gx] = 1.0f; // Special case when both patches are flat (correlation would be NaN but we want 1 because textures are the same)
    }else if(ref_std_d==0.0 || comp_std_d==0.0){
        pearson_map[gy*w+gx] = 0.0; // Special case when only one patch is flat, correlation would be NaN but we want 0
    }else{
        pearson_map[gy*w+gx] = (float) fmax(0.0, (covariance / ((ref_std_d * comp_std_d) + (double)EPSILON))); // Truncate anti-correlations
    }
}
