#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define center_x $CENTER_X$
#define center_y $CENTER_Y$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define std_x $STD_X$

float getExpDecayWeight(float ref, float comp);

kernel void kernelGetPatchPearson(
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

    // Get mean_subtracted reference patch
    float ref_patch[patch_size] = {0.0f};
    float ref_mean = local_means[center_y*w+center_x];

    int counter = 0;
    for(int j=center_y-bRH; j<=center_y+bRH; j++){
            for(int i=center_x-bRW; i<=center_x+bRW; i++){
                ref_patch[counter] = ref_pixels[j*w+i]-ref_mean;
                counter++;
        }
    }

    // For each comparison pixel...
    // Get mean_subtracted comparison patch
    float comp_patch[patch_size] = {0.0f};
    counter = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            comp_patch[counter] = ref_pixels[j*w+i] - local_means[gy*w+gx];
            counter++;
        }
    }

    // Calculate Pearson's correlation coefficient
     float sum_XY = 0.0f;
     for(int i=0; i<patch_size; i++){
         sum_XY += ref_patch[i]*comp_patch[i];
     }

    // Calculate weight
    float std_y = local_stds[gy*w+gx];
    float weight = 0.0f;
    weight = getExpDecayWeight(std_x, std_y);

    // Calculate Pearson correlation coefficient X,Y and add it to the sum at X (avoiding division by zero)
    if(std_x == 0.0f && std_y == 0.0f){
        pearson_map[gy*w+gx] = 1.0f; // Special case when both patches are flat (correlation would be NaN but we want 1 because textures are the same)
    }else{
        pearson_map[gy*w+gx] = (float) fmax(0.0f, (float)(sum_XY / ((patch_size*std_x*std_y) + 0.00001f))); // Truncate anti-correlations
    }
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
