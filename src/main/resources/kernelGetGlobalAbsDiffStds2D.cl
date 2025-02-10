//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define image_width $WIDTH$
#define image_height $HEIGHT$
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
    if(gx<bRW || gx>=image_width-bRW || gy<bRH || gy>=image_height-bRH){
        diff_std_map[gy*image_width+gx] = 0.0f;
        return;
    }

    // Check to avoid blocks with no structural relevance
    float ref_std = local_stds[gy*image_width+gx];
    float ref_var = (float)ref_std*(float)ref_std;

    if(ref_var<threshold){
        diff_std_map[gy*image_width+gx] = 0.0f; // Set pixel to zero to avoid retaining spurious values already in memory
        return;
    }

    // Calculate similarity between the reference and test blocks
    for(int y=bRH; y<image_height-bRH; y++){
        for(int x=bRW; x<image_width-bRW; x++){

            // Check if test pixel is an estimated noise pixel, and if so, kill the thread
            float test_std = local_stds[y*image_width+x];
            float test_var = (float)test_std*(float)test_std;

            if(test_var<threshold){
                diff_std_map[gy*image_width+gx] += 0.0f;
            }else{
                float weight = exp((-1.0f)*(((ref_std-test_std)*(ref_std-test_std))/(filter_param+EPSILON)));
                weights_sum_map[gy*image_width+gx] += (float)weight;

                diff_std_map[gy*image_width+gx] += fabs(ref_std-test_std)*weight;
            }
        }
    }
}