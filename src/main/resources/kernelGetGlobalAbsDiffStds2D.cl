//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bRW $BRW$
#define bRH $BRH$
#define filter_param $FILTER_PARAM$
#define threshold $THRESHOLD$
#define EPSILON $EPSILON$

kernel void kernelGetGlobalAbsDiffStds2D(
    global float* ref_pixels,
    global float* local_stds,
    global float* weights_sum_map,
    global float* diff_std_map
){

    // Get global indexes
    int gx = get_global_id(0);
    int gy = get_global_id(1);


    // Bound check to avoid borders
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    // Check to avoid blocks with no structural relevance
    float ref_std = local_stds[gy*w+gx];
    float ref_var = (float)ref_std*(float)ref_std;

    if(ref_var<threshold){
        diff_std_map[gy*w+gx] = 0.0f; // Set pixel to zero to avoid retaining spurious values already in memory
        return;
    }

    // Calculate similarity between the reference and test blocks
    for(int y=bRH; y<h-bRH; y++){
        for(int x=bRW; x<w-bRW; x++){

            // Check if test pixel is an estimated noise pixel, and if so, kill the thread
            float test_std = local_stds[y*w+x];
            float test_var = (float)test_std*(float)test_std;

            if(test_var<threshold){
                diff_std_map[gy*w+gx] += 0.0f;
            }else{
                float weight = exp((-1.0f)*(((ref_std-test_std)*(ref_std-test_std))/(filter_param+EPSILON)));
                weights_sum_map[gy*w+gx] += (float)weight;

                diff_std_map[gy*w+gx] += fabs(ref_std-test_std)*weight;
            }
        }
    }
}