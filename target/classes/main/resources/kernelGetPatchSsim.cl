#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define ref_mean $MEAN_X$
#define std_x $STD_X$
#define ref_var $VAR_X$

float getExpDecayWeight(float ref, float comp);
float getSsim(float mean_x, float mean_y, float var_x, float var_y, float cov_xy, int n);

kernel void kernelGetPatchSsim(
    global float* ref_patch,
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

    // For each comparison pixel...
    float weight = 0.0f;

    // Get comparison patch Y
    float comp_patch[patch_size] = {0.0f};
    float meanSub_y[patch_size] = {0.0f};
    float var_y = 0.0f;
    float cov_xy = 0.0f;
    int comp_counter = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            comp_patch[comp_counter] = ref_pixels[j*w+i];
            meanSub_y[comp_counter] = comp_patch[comp_counter] - local_means[gy*w+gx];
            cov_xy += ref_patch[comp_counter]*meanSub_y[comp_counter];
            comp_counter++;
        }
    }
    var_y = local_stds[gy*w+gx] * local_stds[gy*w+gx];
    cov_xy /= patch_size;

    // Calculate weight
    weight = getExpDecayWeight(std_x, local_stds[gy*w+gx]);

    // Calculate SSIM and add it to the sum at X
    ssim_map[gy*w+gx] = getSsim(ref_mean, local_means[gy*w+gx], ref_var, var_y, cov_xy, patch_size);
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

float getSsim(float mean_x, float mean_y, float var_x, float var_y, float cov_xy, int n){
    float ssim = 0;
    float c1 = (0.01*4294967295)*(0.01*4294967295); // constant1*dynamic range
    float c2 = (0.03*4294967295)*(0.03*4294967295); // constant2*dynamic range
    float mean_x_sq = mean_x*mean_x;
    float mean_y_sq = mean_y*mean_y;

    ssim = (2*mean_x*mean_y+c1)*(2*cov_xy+c2)/((mean_x_sq+mean_y_sq+c1)*(var_x+var_y+c2));
    return ssim;
}