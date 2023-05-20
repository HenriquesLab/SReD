#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define center_x $CENTER_X$
#define center_y $CENTER_Y$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define filter_param $FILTER_PARAM$
#define EPSILON $EPSILON$

kernel void kernelGetWeightsMap(
    global float* local_stds,
    global float* weights_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    float ref_std = local_stds[center_y*w+center_x];
    float comp_std = local_stds[gy*w+gx];


    // Calculate weights
    weights_map[gy*w+gx] = exp((-1.0f)*(fabs(ref_std - comp_std)*fabs(ref_std - comp_std))/(10.0f*filter_param));
}
