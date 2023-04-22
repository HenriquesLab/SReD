#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bRW $BRW$
#define bRH $BRH$
#define filter_param $FILTERPARAM$
#define EPSILON $EPSILON$
float getGaussianWeight(float ref, float comp, float h2);

kernel void kernelGetWeightsSumMap(
    global float* local_stds,
    global float* weights_sum_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    float ref_std = local_stds[gy*w+gx];

    // For each comparison pixel...
    for(int y=bRH; y<h-bRH; y++){
        for(int x=bRW; x<w-bRW; x++){
            // Get comparison StdDev
            float comp_std = local_stds[y*w+x];

            // Calculate weight
            weights_sum_map[gy*w+gx] += 1.0f - getGaussianWeight(ref_std, comp_std, filter_param);
        }
    }
}

// ---- USER FUNCTIONS ----
float getGaussianWeight(float ref, float comp, float h2){
    float weight = (-1) * (((fabs(comp-ref)) * (fabs(comp-ref))) / (h2 + EPSILON));
    weight = exp(weight);
    weight = fmax(weight, 0.0f);
    return weight;
}