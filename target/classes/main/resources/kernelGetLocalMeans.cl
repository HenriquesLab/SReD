#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define patch_size $PATCH_SIZE$
#define offset_x $OFFSET_X$
#define offset_y $OFFSET_Y$

kernel void kernelGetLocalMeans(
global float* ref_pixels,
global float* local_means,
global float* local_stds
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int bRW = bW/2;
    int bRH = bH/2;

    // Get patch's center pixel
    for (gy=offset_y; gy<h-offset_y; gy++) {
        for (gx=offset_x; gx<w-offset_x; gx++) {

        // Get patch and calculate local mean
        float mean = 0.0f;
        for(int j=gy-bRH; j<=gy+bRH; j++){
            for(int i=gx-bRW; i<=gx+bRW; i++){
                mean += ref_pixels[j*w+i];
            }
        }
        local_means[gy*w+gx] = mean/patch_size;

        // Calculate standard deviation
        float std = 0.0f;
        for(int j=gy-bRH; j<=gy+bRH; j++){
            for(int i=gx-bRW; i<=gx+bRW; i++){
                std += (ref_pixels[j*w+i]-local_means[gy*w+gx]) * (ref_pixels[j*w+i]-local_means[gy*w+gx]);
            }
        }
        local_stds[gy*w+gx] = sqrt(std/patch_size);
        }
    }


}
