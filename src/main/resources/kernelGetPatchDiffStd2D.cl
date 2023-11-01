#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bRW $BRW$
#define bRH $BRH$
#define ref_std $PATCH_STD$
#define EPSILON $EPSILON$
kernel void kernelGetSynthPatchDiffStd(
    global float* local_stds,
    global float* diff_std_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    // -------------------------------------------------------------- //
    // ---- Calculate absolute difference of standard deviations ---- //
    // -------------------------------------------------------------- //

    float comp_std = local_stds[gy*w+gx];

    // We use the reciprocal to get a measure of similarity
    // We cap the max value to Float.MAX to avoid "inf"

    float similarity = 1.0f / (fabs(ref_std - comp_std)+EPSILON);
    similarity = fmin(similarity, FLT_MAX);
    diff_std_map[gy*w+gx] = similarity;
}
