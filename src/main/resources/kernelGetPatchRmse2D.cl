//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define patch_size $PATCH_SIZE$
#define bW $BW$
#define bH $BH$
#define bRW $BRW$
#define bRH $BRH$
#define ref_mean $PATCH_MEAN$
#define EPSILON $EPSILON$

kernel void kernelGetPatchRmse2D(
    global float* patch_pixels,
    global float* ref_pixels,
    global float* local_means,
    global float* rmse_map
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

    __local float ref_patch[patch_size]; // Make a local copy to avoid slower reads from global memory

    for(int i=0; i<patch_size; i++){
        ref_patch[i] = patch_pixels[i]; // Block is mean-subtracted in the host Java class
    }


    // ------------------------------------- //
    // ---- Get comparison patch pixels ---- //
    // ------------------------------------- //

    float comp_patch[patch_size] = {0.0f};
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


    // Mean-subtract comparison patch
    float comp_mean = local_means[gy*w+gx];
    for(int i=0; i<patch_size; i++){
        comp_patch[i] = comp_patch[i] - comp_mean;
    }


    // ------------------------- //
    // ---- Calculate NRMSE ---- //
    // ------------------------- //

    float mse = 0.0f;
    for(int i=0; i<patch_size; i++){
        mse += (ref_patch[i]-comp_patch[i])*(ref_patch[i]-comp_patch[i]);
    }
    mse /= (float) patch_size;
    rmse_map[gy*w+gx] = sqrt(mse);
}
