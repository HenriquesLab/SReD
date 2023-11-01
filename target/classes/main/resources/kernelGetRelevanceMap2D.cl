#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define EPSILON $EPSILON$

kernel void kernelGetRelevanceMap2D(
global float* ref_pixels,
global float* relevance_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }


    // ----------------------------------------- //
    // ---- Get patch pixels and patch mean ---- //
    // ----------------------------------------- //

    double patch[patch_size];
    double mean = 0.0;
    int index = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            // Extract only pixels within the inbound circle/ellipse
            float dx = (float)(i-gx);
            float dy = (float)(j-gy);
            if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){
                double pixel_value = (double)ref_pixels[j*w+i];
                patch[index] = pixel_value;
                mean += pixel_value;
                index++;
            }
        }
    }
    mean /= (double)patch_size;


    // -------------------------------- //
    // ---- Calculate patch StdDev ---- //
    // -------------------------------- //

    double var = 0.0;
        for(int i=0; i<patch_size; i++){
            var += (patch[i] - mean) * (patch[i] - mean);
        }
    var /= (double)(patch_size-1);

    relevance_map[gy*w+gx] = (float) var;
}
