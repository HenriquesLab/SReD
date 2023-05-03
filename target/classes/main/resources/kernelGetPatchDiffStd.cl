#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define center_x $CENTER_X$
#define center_y $CENTER_Y$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define EPSILON $EPSILON$

kernel void kernelGetPatchDiffStd(
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

    // Get mean_subtracted reference patch
    __local float ref_patch[patch_size];

    float ref_mean = local_means[center_y*w+center_x];
    float ref_std = local_stds[center_y*w+center_x];

    int counter = 0;
    float r2 = bRW*bRW;
    for(int j=center_y-bRH; j<=center_y+bRH; j++){
        for(int i=center_x-bRW; i<=center_x+bRW; i++){
            float dx = (float)(i-center_x);
            float dy = (float)(j-center_y);
            if(dx*dx + dy*dy <= r2){
                ref_patch[counter] = ref_pixels[j*w+i] - ref_mean;
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
