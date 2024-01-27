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

kernel void kernelGetPatchSsim2D(
    global float* patch_pixels,
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


    // ------------------------------------------------------------------------ //
    // ---- Get mean-subtracted and normalized reference patch from buffer ---- //
    // ------------------------------------------------------------------------ //

    __local float ref_patch[patch_size]; // Make a local copy to avoid slower reads from global memory

    for(int i=0; i<patch_size; i++){
        ref_patch[i] = patch_pixels[i];
    }


    // ------------------------------------- //
    // ---- Get comparison patch pixels ---- //
    // ------------------------------------- //

    float comp_patch[patch_size];
    int index = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            float dx = (float)(i-gx);
            float dy = (float)(j-gy);
            if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){
                comp_patch[index] = ref_pixels[j*w+i];
                index++;
            }
        }
    }


    // ------------------------------------ //
    // ---- Normalize comparison patch ---- //
    // ------------------------------------ //

    float min_intensity = FLT_MAX;
    float max_intensity = -FLT_MAX;

    for(int i=0; i<patch_size; i++){
        float pixel_value = comp_patch[i];
        min_intensity = min(min_intensity, pixel_value);
        max_intensity = max(max_intensity, pixel_value);
    }

    // Remap pixel values
    for(int i=0; i<patch_size; i++){
        comp_patch[i] = (comp_patch[i] - min_intensity) / (max_intensity - min_intensity + EPSILON);
    }


    // ---------------------------------------- //
    // ---- Mean-subtract comparison patch ---- //
    // ---------------------------------------- //

    float comp_mean = local_means[gy*w+gx];

    for(int i=0; i<patch_size; i++){
        comp_patch[i] = comp_patch[i] - comp_mean;
    }


    // ------------------------------------------ //
    // ---- Normalize comparison patch again ---- //
    // ------------------------------------------ //

    min_intensity = FLT_MAX;
    max_intensity = -FLT_MAX;

    for(int i=0; i<patch_size; i++){
        float pixel_value = comp_patch[i];
        min_intensity = min(min_intensity, pixel_value);
        max_intensity = max(max_intensity, pixel_value);
    }

    // Remap pixel values
    for(int i=0; i<patch_size; i++){
        comp_patch[i] = (comp_patch[i] - min_intensity) / (max_intensity - min_intensity + EPSILON);
    }


    // ------------------------- //
    // ---- Get Covariance ----- //
    // ------------------------- //

    float covar = 0.0;
    for(int i=0; i<patch_size; i++){
        covar += ref_patch[i] * comp_patch[i];
    }
    covar /= (float)(patch_size-1);


    // ----------------------------------- //
    // ---- Calculate (modified) SSIM ---- //
    // ----------------------------------- //

    float c1 = (0.01f * 1.0f) * (0.01f * 1.0f);
    //float c2 = (0.03f * 1.0f) * (0.03f * 1.0f);
    float c2 = 0.0000001f;
    float c3 = c2/2.0f;

    float comp_std = local_stds[gy*w+gx];

    //ssim_map[gy*w+gx] = (float)fmax(0.0, (2.0 * ref_mean * comp_mean + c1)/((ref_mean*ref_mean)+(comp_mean*comp_mean)+c1)); // Luminance
    //ssim_map[gy*w+gx] = (float)fmax(0.0, (2.0 * ref_std * comp_std + c1)/((ref_std*ref_std)+(comp_std*comp_std)+c1)); // Contrast

    if(ref_std == 0.0 && comp_std == 0.0){
        ssim_map[gy*w+gx] = 1.0f; // Special case when both patches are flat, correlation is 1
    }else if(ref_std == 0.0 || comp_std == 0.0){
        ssim_map[gy*w+gx] = 0.0f; // Special case when one patch is flat, correlation is 0
    }else{
        //ssim_map[gy*w+gx] = (float)fmax(0.0, (covar+c3)/(ref_std*comp_std+c3)); // Structure
        ssim_map[gy*w+gx] = (float)fmax(0.0, (2.0*covar+c3)/((ref_std*ref_std)+(comp_std*comp_std)+c3)); // Structure

    }





    //ssim_map[gy*w+gx] = (float)fmax(0.0, ((2.0*ref_std*comp_std+c2)/((ref_std*ref_std)+(comp_std*comp_std)+c2)));


    //ssim_map[gy*w+gx] = (float)fmax(0.0, () * ((2.0*ref_std*comp_std+c2)/((ref_std*ref_std)+(comp_std*comp_std)+c2)) * ((covar+c3)/(ref_std*comp_std)));

    //ssim_map[gy*w+gx] = (float)fmax(0.0, ((2.0*ref_std*comp_std+c2)/((ref_std*ref_std)+(comp_std*comp_std)+c2)) * ((covar+c3)/(ref_std*comp_std)));
}
