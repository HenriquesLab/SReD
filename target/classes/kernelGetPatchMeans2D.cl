#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define EPSILON $EPSILON$

kernel void kernelGetPatchMeans2D(
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

    double patch[patch_size];
    int index = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            // Extract only pixels within the inbound circle/ellipse
            float dx = (float)(i-gx);
            float dy = (float)(j-gy);
            if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){
                patch[index] = (double)ref_pixels[j*w+i];
                index++;
            }
        }
    }


    // ------------------------- //
    // ---- Normalize patch ---- //
    // ------------------------- //

    // Find min and max
    double min_intensity = DBL_MAX;
    double max_intensity = -DBL_MAX;
    for(int i=0; i<patch_size; i++){
        double pixel_value = patch[i];
        min_intensity = min(min_intensity, pixel_value);
        max_intensity = max(max_intensity, pixel_value);
    }

    // Remap pixels
    for(int i=0; i<patch_size; i++){
        patch[i] = (patch[i] - min_intensity) / (max_intensity - min_intensity + (double)EPSILON);
    }


    // ------------------------------ //
    // ---- Calculate patch mean ---- //
    // ------------------------------ //

    double mean = 0.0;
    for(int i=0; i<patch_size; i++){
        mean += patch[i];
    }
    local_means[gy*w+gx] = (float)(mean/(double)patch_size);


    // -------------------------------- //
    // ---- Calculate patch StdDev ---- //
    // -------------------------------- //
    double var = 0.0;
        for(int i=0; i<patch_size; i++){
            var += (patch[i] - mean) * (patch[i] - mean);
        }

    local_stds[gy*w+gx] = (float)sqrt(var/(double)(patch_size-1));
}
