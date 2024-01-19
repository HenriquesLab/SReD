//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bRW $BRW$
#define bRH $BRH$
#define filter_param $FILTERPARAM$
#define threshold $THRESHOLD$
#define EPSILON $EPSILON$

kernel void kernelGetDiffStdMap(
    global float* ref_pixels,
    global float* local_stds,
    global float* weights_sum_map,
    global float* diff_std_map
){

    // ---------------------------- //
    // ---- Get global indexes ---- //
    // ---------------------------- //

    int gx = get_global_id(0);
    int gy = get_global_id(1);


    // -------------------------------------- //
    // ---- Bound check to avoid borders ---- //
    // -------------------------------------- //

    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }


    // ------------------------------------------------------------ //
    // ---- Check to avoid blocks with no structural relevance ---- //
    // ------------------------------------------------------------ //

    double ref_std = (double)local_stds[gy*w+gx];
    float ref_var = (float)ref_std*(float)ref_std;

    if(ref_var<threshold || ref_var==0.0f){
        diff_std_map[gy*w+gx] = 0.0f; // Set pixel to zero to avoid retaining spurious values already in memory
        return;
    }


    // -------------------------------------------------------------------- //
    // ---- Calculate similarity between the reference and test blocks ---- //
    // -------------------------------------------------------------------- //

    for(int y=bRH; y<h-bRH; y++){
        for(int x=bRW; x<w-bRW; x++){

            // Check if test pixel is an estimated noise pixel, and if so, kill the thread
            double test_std = (double)local_stds[y*w+x];
            float test_var = (float)test_std*(float)test_std;

            if(test_var<threshold || test_var==0.0f){
                diff_std_map[gy*w+gx] += 0.0f;
            }else{
                double weight = exp((-1.0)*(((ref_std-test_std)*(ref_std-test_std))/((double)filter_param+(double)EPSILON)));
                weights_sum_map[gy*w+gx] += (float)weight;

                float similarity = (float)(ref_std*test_std) / (float)(sqrt(ref_std*ref_std)*sqrt(test_std*test_std)+(double)EPSILON); // Based on cosine similarity, ranges between -1 and 1, same interpretation as PEarson
                diff_std_map[gy*w+gx] += (float)fmax(0.0f, similarity)*(float)weight;
            }
        }
    }
}