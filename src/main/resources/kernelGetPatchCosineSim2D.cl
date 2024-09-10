//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bRW $BRW$
#define bRH $BRH$
#define ref_std $PATCH_STD$
#define EPSILON $EPSILON$
kernel void kernelGetPatchCosineSim2D(
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

    float test_std = local_stds[gy*w+gx];

    float similarity = (ref_std*test_std) / (sqrt(ref_std*ref_std)*sqrt(test_std*test_std)+EPSILON);
    diff_std_map[gy*w+gx] = (float)fmax(0.0f, similarity);
}
