//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define imageWidth $WIDTH$
#define imageHeight $HEIGHT$
#define imageDepth $DEPTH$
#define bRW $BRW$
#define bRH $BRH$
#define bRZ $BRZ$
#define ref_std $BLOCK_STD$
#define EPSILON $EPSILON$

kernel void kernelGetBlockCosineSimilarity3D(
    global float* local_stds,
    global float* cosine_similarity_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int gz = get_global_id(2);

    // Bound check (avoids borders dynamically based on block dimensions)
    if(gx<bRW || gx>=imageWidth-bRW || gy<bRH || gy>=imageHeight-bRH || gz<bRZ || gz>=imageDepth-bRZ){
        cosine_similarity_map[imageWidth*imageHeight*gz+gy*imageWidth+gx] = 0.0f;
        return;
    }


    // Calculate cosine similarity
    float test_std = local_stds[imageWidth*imageHeight*gz+gy*imageWidth+gx];
    float similarity = (ref_std*test_std) / (sqrt(ref_std*ref_std)*sqrt(test_std*test_std)+EPSILON);
    cosine_similarity_map[imageWidth*imageHeight*gz+gy*imageWidth+gx] = (float)fmax(0.0f, similarity);
}
