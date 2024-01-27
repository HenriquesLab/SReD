#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define EPSILON $EPSILON$

kernel void kernelGetLocalMeans(
global float* ref_pixels,
global float* local_means,
global float* local_stds
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }


    // -------------------------- //
    // ---- Get patch pixels ---- //
    // -------------------------- //

    float patch[patch_size];
    int index = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            // Extract only pixels within the inbound circle/ellipse
            float dx = (float)(i-gx);
            float dy = (float)(j-gy);
            if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){
                patch[index] = ref_pixels[j*w+i];
                index++;
            }
        }
    }


    // ------------------------------ //
    // ---- Calculate patch mean ---- //
    // ------------------------------ //

    float mean = 0.0;
    for(int i=0; i<patch_size; i++){
        mean += patch[i];
    }
    mean /= (float)patch_size;
    local_means[gy*w+gx] = (float)mean;


    // -------------------------------- //
    // ---- Calculate patch StdDev ---- //
    // -------------------------------- //

    float var = 0.0;
    for(int i=0; i<patch_size; i++){
        var += (patch[i] - mean) * (patch[i] - mean);
    }

    var /= (float)(patch_size-1);
    local_stds[gy*w+gx] = (float)sqrt(var);
}
