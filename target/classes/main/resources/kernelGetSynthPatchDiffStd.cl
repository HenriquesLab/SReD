#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define patch_size $PATCH_SIZE$
#define bW $BW$
#define bH $BH$
#define bRW $BRW$
#define bRH $BRH$
#define ref_std $PATCH_STD$
#define EPSILON $EPSILON$

kernel void kernelGetSynthPatchDiffStd(
    global float* patch_pixels,
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* diff_std_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    // Get mean_subtracted reference patch from buffer
    __local float ref_patch[patch_size];

    int counter = 0;
    float r2 = bRW*bRW;
    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            float dx = (float)(i-bRW);
            float dy = (float)(j-bRH);
            if(dx*dx + dy*dy <= r2){
                ref_patch[counter] = ref_pixels[j*bW+i];
                counter++;
            }
        }
    }

    // For each comparison pixel...
    // Get mean_subtracted comparison patch
    float comp_patch[patch_size];
    float comp_mean = local_means[gy*w+gx];
    float comp_std = local_stds[gy*w+gx];

    counter = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            float dx = (float)(i-gx);
            float dy = (float)(j-gy);
            if(dx*dx+dy*dy <= r2){
                comp_patch[counter] = ref_pixels[j*w+i] - comp_mean;
                counter++;
            }
        }
    }

    // Calculate absolute difference of standard deviations
    diff_std_map[gy*w+gx] = fabs(ref_std - comp_std);
}
