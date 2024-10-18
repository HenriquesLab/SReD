//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define image_width $WIDTH$
#define image_height $HEIGHT$
#define image_depth $DEPTH$
#define bRW $BRW$
#define bRH $BRH$
#define bRZ $BRZ$
#define filter_param $FILTER_PARAM$
#define threshold $THRESHOLD$
#define EPSILON $EPSILON$

kernel void kernelGetGlobalCosineSimilarity3D(
    global float* ref_pixels,
    global float* local_stds,
    global float* weights_sum_map,
    global float* cosine_sim_map
){

    // Get global indexes
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int gz = get_global_id(2);


    // Bound check to avoid borders
    if(gx<bRW || gx>=image_width-bRW || gy<bRH || gy>=image_height-bRH || gz<bRZ || gz>=image_depth-bRZ){
        return;
    }

    // Check to avoid blocks with no structural relevance
    int ref_index = image_width*image_height*gz+gy*image_width+gx;
    float ref_std = local_stds[ref_index];
    float ref_var = ref_std*ref_std;

    if(ref_var<threshold || ref_var==0.0f){
        cosine_sim_map[ref_index] = 0.0f; // Set pixel to zero to avoid retaining spurious values already in memory
        return;
    }

    // Calculate similarity between the reference and test blocks
    float weight_sum = 0.0f;
    float cosine_similarity_sum = 0.0f;

    for(int z=bRZ; z<image_depth-bRZ; z++){
        for(int y=bRH; y<image_height-bRH; y++){
            for(int x=bRW; x<image_width-bRW; x++){

                // Check if test pixel is an estimated noise pixel, and if so, kill the thread
                float test_std = local_stds[image_width*image_height*z+y*image_width+x];
                float test_var = test_std*test_std;

                if(test_var<threshold || test_var==0.0f){
                    cosine_similarity_sum += 0.0f;
                }else{
                    float weight = exp((-1.0f)*(((ref_std-test_std)*(ref_std-test_std))/(filter_param+EPSILON)));
                    weight_sum += weight;

                    float similarity = (ref_std*test_std) / (float)(sqrt(ref_std*ref_std)*sqrt(test_std*test_std)+EPSILON); // Based on cosine similarity, ranges between -1 and 1, same interpretation as PEarson
                    cosine_similarity_sum += (float)fmax(0.0f, similarity) * weight;
                }
            }
        }
    }
    cosine_sim_map[ref_index] = cosine_similarity_sum;
    weights_sum_map[ref_index] = weight_sum;
}