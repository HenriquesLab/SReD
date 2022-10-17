#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define center_x $CENTER_X$
#define center_y $CENTER_Y$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define EPSILON $EPSILON$
float getExpDecayWeight(float ref, float comp);

kernel void kernelGetPatchPearson(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* pearson_map,
    global float* gaussian_kernel
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    // Get mean_subtracted reference patch
    float ref_patch[patch_size] = {0.0f};
    float ref_mean = local_means[center_y*w+center_x];
    float ref_std = local_stds[center_y*w+center_x];

    int counter = 0;
    for(int j=center_y-bRH; j<=center_y+bRH; j++){
            for(int i=center_x-bRW; i<=center_x+bRW; i++){
                ref_patch[counter] = ref_pixels[j*w+i]*gaussian_kernel[counter]  - ref_mean;
                counter++;
        }
    }

    // For each comparison pixel...
    // Get mean_subtracted comparison patch
    float comp_patch[patch_size] = {0.0f};
    float comp_mean = local_means[gy*w+gx];
    float comp_std = local_stds[gy*w+gx];

    counter = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            comp_patch[counter] = ref_pixels[j*w+i]*gaussian_kernel[counter] - comp_mean;
            counter++;
        }
    }

    // Calculate Pearson's correlation coefficient
     float covar = 0.0f;
     for(int i=0; i<patch_size; i++){
         covar += ref_patch[i] * comp_patch[i];
     }
    covar /= patch_size;

    // Calculate weight
    float weight = 0.0f;
    weight = getExpDecayWeight(ref_std, comp_std);

    // Calculate Pearson correlation coefficient X,Y and add it to the sum at X (avoiding division by zero)
    if(ref_std == 0.0f && comp_std == 0.0f){
        pearson_map[gy*w+gx] = 0.0f; // Special case when both patches are flat (correlation would be NaN but we want 1 because textures are the same)
    }else{
        pearson_map[gy*w+gx] = (1.0f - ((float) fmax(0.0f, (float)(covar / ((ref_std * comp_std) + EPSILON))))); // Pearson distance, Truncate anti-correlations
    }
}

// ---- USER FUNCTIONS ----
float getExpDecayWeight(float ref, float comp){
    // Gaussian weight, see https://en.wikipedia.org/wiki/Non-local_means#Common_weighting_functions
    // Alternative: exponential decay function: 1 - abs(mean_x - mean_y / abs(mean_x + abs(mean_y)))
    float weight = 0;

    if(ref == comp){
        weight = 1;
    }else{
        weight = 1 - (fabs(ref - comp) / (ref + comp));
    }
    return weight;

}
