#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define z $DEPTH$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define bRZ $BRZ$
#define EPSILON $EPSILON$

kernel void kernelGetPatchMeans3D(
global float* ref_pixels,
global float* local_means,
global float* local_stds
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int gz = get_global_id(2);
    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH || gz<bRZ || gz>=z-bRZ){
        return;
    }


    // -------------------------- //
    // ---- Get patch pixels ---- //
    // -------------------------- //

    float patch[patch_size];
    int index = 0;
    for(int n=gz-bRZ; n<=gz+bRZ; n++){
        for(int j=gy-bRH; j<=gy+bRH; j++){
            for(int i=gx-bRW; i<=gx+bRW; i++){
                // Extract only pixels within the inbound circle/ellipse
                float dx = (float)(i-gx);
                float dy = (float)(j-gy);
                float dz = (float)(n-gz);
                if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH))+((dz*dz)/(float)(bRZ*bRZ)) <= 1.0f){
                    patch[index] = ref_pixels[w*h*n+j*w+i];
                    index++;
                }
            }
        }
    }

    // ------------------------- //
    // ---- Normalize patch ---- //
    // ------------------------- //

    // Find min and max
    float min_intensity = FLT_MAX;
    float max_intensity = -FLT_MAX;
    for(int i=0; i<patch_size; i++){
        float pixel_value = patch[i];
        min_intensity = min(min_intensity, pixel_value);
        max_intensity = max(max_intensity, pixel_value);
    }

    // Remap pixels
    for(int i=0; i<patch_size; i++){
        patch[i] = (patch[i] - min_intensity) / (max_intensity - min_intensity + EPSILON);
    }


    // ------------------------------ //
    // ---- Calculate patch mean ---- //
    // ------------------------------ //

    float mean = 0.0f;
    for(int i=0; i<patch_size; i++){
        mean += patch[i];
    }

    local_means[w*h*gz+gy*w+gx] = (float)(mean/(float)patch_size);

    // -------------------------------- //
    // ---- Calculate patch StdDev ---- //
    // -------------------------------- //
    float var = 0.0;
        for(int i=0; i<patch_size; i++){
            var += (patch[i] - mean) * (patch[i] - mean);
        }

    local_stds[w*h*gz+gy*w+gx] = (float)sqrt(var/(float)(patch_size-1));
}
