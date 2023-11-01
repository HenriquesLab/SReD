#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define z $DEPTH$
#define patch_size $PATCH_SIZE$
#define bW $BW$
#define bH $BH$
#define bZ $BZ$
#define bRW $BRW$
#define bRH $BRH$
#define bRZ $BRZ$
#define ref_mean $PATCH_MEAN$
#define ref_std $PATCH_STD$
#define EPSILON $EPSILON$

kernel void kernelGetPatchPearson3D(
    global float* patch_pixels,
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* pearson_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int gz = get_global_id(2);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH || gz<bRZ || gz>=z-bRZ){
        return;
    }


    // ------------------------------------------------------------------------ //
    // ---- Get mean-subtracted and normalized reference patch from buffer ---- //
    // ------------------------------------------------------------------------ //

    __local double ref_patch[patch_size]; // Make a local copy to avoid slower reads from global memory

    for(int i=0; i<patch_size; i++){
        ref_patch[i] = (double)patch_pixels[i];
    }


    // ------------------------------------- //
    // ---- Get comparison patch pixels ---- //
    // ------------------------------------- //

    double comp_patch[patch_size];
    int index = 0;
    for(int n=gz-bRZ; n<=gz+bRZ; n++){
        for(int j=gy-bRH; j<=gy+bRH; j++){
            for(int i=gx-bRW; i<=gx+bRW; i++){
                float dx = (float)(i-gx);
                float dy = (float)(j-gy);
                float dz = (float)(n-gz);
                if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH))+((dz*dz)/(float)(bRZ*bRZ)) <= 1.0f){
                    comp_patch[index] = (double)ref_pixels[w*h*n+j*w+i];
                    index++;
                }
            }
        }
    }


    // ------------------------------------ //
    // ---- Normalize comparison patch ---- //
    // ------------------------------------ //
    double min_intensity = DBL_MAX;
    double max_intensity = -DBL_MAX;

    for(int i=0; i<patch_size; i++){
        double pixel_value = comp_patch[i];
        min_intensity = min(min_intensity, pixel_value);
        max_intensity = max(max_intensity, pixel_value);
    }

    // Remap pixel values
    for(int i=0; i<patch_size; i++){
        comp_patch[i] = (comp_patch[i] - min_intensity) / (max_intensity - min_intensity + (double)EPSILON);
    }


    // ---------------------------------------- //
    // ---- Mean-subtract comparison patch ---- //
    // ---------------------------------------- //

    double comp_mean = (double)local_means[w*h*gz+gy*w+gx];
    for(int i=0; i<patch_size; i++){
        comp_patch[i] = comp_patch[i] - comp_mean;
    }


    // ------------------------------------------ //
    // ---- Normalize comparison patch again ---- //
    // ------------------------------------------ //

    min_intensity = DBL_MAX;
    max_intensity = -DBL_MAX;

    for(int i=0; i<patch_size; i++){
        double pixel_value = comp_patch[i];
        min_intensity = min(min_intensity, pixel_value);
        max_intensity = max(max_intensity, pixel_value);
    }

    // Remap pixel values
    for(int i=0; i<patch_size; i++){
        comp_patch[i] = (comp_patch[i] - min_intensity) / (max_intensity - min_intensity + (double)EPSILON);
    }


    // ------------------------- //
    // ---- Get Covariance ----- //
    // ------------------------- //

    double covar = 0.0;
    for(int i=0; i<patch_size; i++){
        covar += ref_patch[i] * comp_patch[i];

    }
    covar /= (double)(patch_size-1);

    // Calculate Pearson correlation coefficient REF vs. COMP and add it to the sum at REF (avoiding division by zero)
    double ref_std_d = (double)ref_std;
    double comp_std_d = (double)local_stds[w*h*gz+gy*w+gx];

    if(ref_std_d == 0.0 && comp_std_d == 0.0){
        pearson_map[w*h*gz+gy*w+gx] = 1.0f; // Special case when both patches are flat (correlation would be NaN but we want 1 because textures are the same)
    }else if(ref_std_d==0.0 || comp_std_d==0.0){
        pearson_map[w*h*gz+gy*w+gx] = 0.0; // Special case when only one patch is flat, correlation would be NaN but we want 0
    }else{
        pearson_map[w*h*gz+gy*w+gx] = (float) fmax(0.0, (covar / ((ref_std_d * comp_std_d) + EPSILON))); // Truncate anti-correlations
    }
}
