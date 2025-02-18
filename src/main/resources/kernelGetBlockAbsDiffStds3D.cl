//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define imageWidth $WIDTH$
#define imageHeight $HEIGHT$
#define imageDepth $DEPTH$
#define bRW $BRW$
#define bRH $BRH$
#define bRZ $BRZ$
#define ref_std $BLOCK_STD$
#define EPSILON $EPSILON$

kernel void kernelGetBlockAbsDiffStds3D(
    global float* local_stds,
    global float* diff_std_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int gz = get_global_id(2);

    // Bound check (avoids borders dynamically based on block dimensions)
    if(gx<bRW || gx>=imageWidth-bRW || gy<bRH || gy>=imageHeight-bRH || gz<bRZ || gz>=imageDepth-bRZ){
        diff_std_map[imageWidth*imageHeight*gz+gy*imageWidth+gx] = 0.0f;
        return;
    }

    // Calculate abslute difference of standard deviations
    float test_std = local_stds[imageWidth*imageHeight*gz+gy*imageWidth+gx];
    diff_std_map[imageWidth*imageHeight*gz+gy*imageWidth+gx] = fabs((float)ref_std - (float)test_std);
}
