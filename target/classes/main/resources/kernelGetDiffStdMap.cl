//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bRW $BRW$
#define bRH $BRH$
#define filter_param $FILTERPARAM$
#define filter_constant $FILTERCONSTANT$
#define EPSILON $EPSILON$

kernel void kernelGetDiffStdMap(
    global float* ref_pixels,
    global float* local_stds,
    global float* weights_sum_map,
    global float* diff_std_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check to avoids borders dynamically based on patch dimensions
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    // Check if reference pixel is an estimated noise pixel, and if so, kill the thread
    double ref_std = (double)local_stds[gy*w+gx];
    double threshold = (double)(filter_param*filter_constant);
    if((ref_std*ref_std)<threshold){
        diff_std_map[gy*w+gx] = 0.0f; // Set pixel to zero to avoid retaining spurious values already in memory
        return;
    }

    // Iterate over all pixels (excluding borders)
    for(int y=bRH; y<h-bRH; y++){
        for(int x=bRW; x<w-bRW; x++){

            // Check if comparison pixel is an estimated noise pixel, and if so, kill the thread
            double comp_std = (double)local_stds[y*w+x];
            if((comp_std*comp_std)<threshold){
                diff_std_map[gy*w+gx] += 0.0f;
                return;
            }

            double weight = exp((-1.0)*(((ref_std-comp_std)*(ref_std-comp_std))/((double)filter_param+(double)EPSILON)));
            weights_sum_map[gy*w+gx] += (float)weight;

            float similarity = (float)(ref_std*comp_std) / (float)(sqrt(ref_std*ref_std)*sqrt(comp_std*comp_std)+(double)EPSILON); // Based on cosine similarity, ranges between -1 and 1, same interpretation as PEarson
            diff_std_map[gy*w+gx] += (float)fmax(0.0f, similarity)*(float)weight;
            //printf("%f\n", (float)weight);

            //diff_std_map[gy*w+gx] += (float)(1.0f - (float)fabs(ref_std-comp_std));// * (float)weight;
            //printf("%f\n", (1.0f-(float)fabs(ref_std-comp_std))*(float)weight);
            //printf("%f\n", weight);
        }
    }
}