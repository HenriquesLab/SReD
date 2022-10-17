#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$

kernel void kernelGetLocalMeans(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* gaussian_kernel
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    // Get patch and calculate local mean
    //float gauss_kernel[patch_size] = {0.011f, 0.084f, 0.011f, 0.084f, 0.619f, 0.084f, 0.011f, 0.084f, 0.011f};

    double value = 0.0f;
    double sum = 0.0f;
    double sq_sum = 0.0f;

    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            //value = ref_pixels[j*w+i] * gaussian_kernel[(j-gy)*patch_size+(i-gx)];
            value = ref_pixels[j*w+i];

            sum += value;
            sq_sum += value * value;
        }
    }

    double mean = sum / patch_size;
    double variance = fabs(sq_sum / (double) patch_size - mean * mean); // fabs() avoids negative values; solves a bug

    local_means[gy*w+gx] = (float) mean;
    local_stds[gy*w+gx] = (float) sqrt(variance);
}