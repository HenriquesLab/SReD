//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define z $DEPTH$
#define bRW $BRW$
#define bRH $BRH$
#define bRZ $BRZ$
#define filter_param $FILTERPARAM$
#define threshold $THRESHOLD$
#define EPSILON $EPSILON$

kernel void kernelGetCosineSimMap3D(
    global float* ref_pixels,
    global float* local_stds,
    global float* weights_sum_map,
    global float* cosine_sim_map
){

    // ---------------------------- //
    // ---- Get global indexes ---- //
    // ---------------------------- //

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int gz = get_global_id(2);


    // -------------------------------------- //
    // ---- Bound check to avoid borders ---- //
    // -------------------------------------- //

    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH || gz<bRZ || gz>=z-bRZ){
        return;
    }


    // ------------------------------------------------------------ //
    // ---- Check to avoid blocks with no structural relevance ---- //
    // ------------------------------------------------------------ //

    float ref_std = local_stds[w*h*gz+gy*w+gx];
    float ref_var = ref_std*ref_std;

    if(ref_var<threshold || ref_var==0.0f){
        cosine_sim_map[w*h*gz+gy*w+gx] = 0.0f; // Set pixel to zero to avoid retaining spurious values already in memory
        return;
    }


    // -------------------------------------------------------------------- //
    // ---- Calculate similarity between the reference and test blocks ---- //
    // -------------------------------------------------------------------- //

    for(int n=bRZ; n<z-bRZ; n++){
        for(int y=bRH; y<h-bRH; y++){
            for(int x=bRW; x<w-bRW; x++){

                // Check if test pixel is an estimated noise pixel, and if so, kill the thread
                float test_std = local_stds[w*h*n+y*w+x];
                float test_var = test_std*test_std;

                if(test_var<threshold || test_var==0.0f){
                    cosine_sim_map[w*h*gz+gy*w+gx] += 0.0f;
                }else{
                    float weight = exp((-1.0)*(((ref_std-test_std)*(ref_std-test_std))/(filter_param+EPSILON)));
                    weights_sum_map[w*h*gz+gy*w+gx] += weight;

                    float similarity = (ref_std*test_std) / (float)(sqrt(ref_std*ref_std)*sqrt(test_std*test_std)+EPSILON); // Based on cosine similarity, ranges between -1 and 1, same interpretation as PEarson
                    cosine_sim_map[w*h*gz+gy*w+gx] += (float)fmax(0.0f, similarity) * weight;
                }
            }
        }
    }
}