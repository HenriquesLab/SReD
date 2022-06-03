#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define center_x $CENTER_X$
#define center_y $CENTER_Y$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$

float getExpDecayWeight(float ref, float comp);

kernel void kernelGetPatchSsim(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* ssim_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    float EPSILON = 0.0000001f;

    // Get reference patch
    float ref_patch[patch_size] = {0.0f};
    float ref_mean = local_means[center_y*w+center_x];
    float ref_var = local_stds[center_y*w+center_x] * local_stds[center_y*w+center_x];

    int counter = 0;
    for(int j=center_y-bRH; j<=center_y+bRH; j++){
        for(int i=center_x-bRW; i<=center_x+bRW; i++){
            ref_patch[counter] = ref_pixels[j*w+i] - ref_mean;
            counter++;
        }
    }

    // For each comparison pixel...
    // Get mean-subtracted comparison patch and some variables for SSIM calculation
    float comp_patch[patch_size] = {0.0f};
    float comp_mean = local_means[gy*w+gx];
    float comp_var = local_stds[gy*w+gx] * local_stds[gy*w+gx];
    float covar = 0.0f;

    counter = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            comp_patch[counter] = ref_pixels[j*w+i] - comp_mean;
            covar += ref_patch[counter] * comp_patch[counter];
            counter++;
        }
    }
    covar /= patch_size;

    // Calculate SSIM
    float c1 = (0.01f * 1.0f) * (0.01f * 1.0f); // constant1 * float dynamic range
    float c2 = (0.03f * 1.0f) * (0.03f * 1.0f); // constant2 * float dynamic range
    float ref_mean_sq = ref_mean * ref_mean;
    float comp_mean_sq = comp_mean * comp_mean;

    ssim_map[gy*w+gx] = (2.0f * covar + c2) / (ref_var + comp_var + c2); // Removed the luminance component to remove intensity-variant component
}

// ---- USER FUNCTIONS ----
float getExpDecayWeight(float ref, float comp){
    // Gaussian weight, see https://en.wikipedia.org/wiki/Non-local_means#Common_weighting_functions
    // Alternative: exponential decay function: 1-abs(mean_x-mean_y/abs(mean_x+abs(mean_y)))
    float weight = 0;

    if(ref == comp){
        weight = 1;
    }else{
        weight = 1-(fabs(ref-comp)/(ref+comp));
    }
    return weight;
}