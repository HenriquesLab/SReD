//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bRW $BRW$
#define bRH $BRH$
#define ref_std $BLOCK_STD$
#define EPSILON $EPSILON$

kernel void kernelGetBlockCosineSimilarity2D(
    global float* local_stds,
    global float* cosine_similarity_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on block dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        cosine_similarity_map[gy*w+gx] = 0.0f;
        return;
    }


    // ------------------------------------- //
    // ---- Calculate cosine similarity ---- //
    // ------------------------------------- //

    float test_std = local_stds[gy*w+gx];

    float similarity = (ref_std*test_std) / (sqrt((float)ref_std*(float)ref_std)*sqrt((float)test_std*(float)test_std)+EPSILON);
    cosine_similarity_map[gy*w+gx] = (float)fmax(0.0f, similarity);
}
