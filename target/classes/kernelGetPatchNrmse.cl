//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define center_x $CENTER_X$
#define center_y $CENTER_Y$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define std_x $STD_X$

float getExpDecayWeight(float ref, float comp);

kernel void kernelGetPatchNrmse(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* nrmse_map,
    global float* mae_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    // Get reference patch
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
    // Get comparison patch minimum and maximum
    float min_y = ref_pixels[gy*w+gx];
    float max_y = ref_pixels[gy*w+gx];

    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            if(ref_pixels[j*w+i] < min_y){
               min_y = ref_pixels[j*w+i];
            }
            if(ref_pixels[j*w+i] > max_y){
                max_y = ref_pixels[j*w+i];
            }
        }
    }
    float comp_range = max_y-min_y;

    // Get mean-subtracted comparison patch
    float comp_patch[patch_size];
    float comp_mean = local_means[gy*w+gx];
    counter = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            comp_patch[counter] = ref_pixels[j*w+i] - comp_mean;
            counter++;
        }
    }

    // Calculate NRMSE and MAE
    float nmrse = 0.0f;
    float mae = 0.0f;
    for(int i=0; i<patch_size; i++){
        nmrse += (ref_patch[i] - comp_patch[i]) * (ref_patch[i] - comp_patch[i]);
        mae += fabs(ref_patch[i] - comp_patch[i]);
    }

    nrmse_map[gy*w+gx] = sqrt((nmrse/patch_size)) / (ref_mean + 0.0000001f);
    mae_map[gy*w+gx] = mae/patch_size;
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