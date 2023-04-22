#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define center_x $CENTER_X$
#define center_y $CENTER_Y$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define EPSILON $EPSILON$
float getExpDecayWeight(float ref, float comp);
float getHausdorffDistance(float* ref_patch, float* comp_patch, int bL, int width);

kernel void kernelGetPatchHausdorff(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* hausdorff_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    // Get mean_subtracted reference patch
    float ref_patch[patch_size] = {0.0f};
    int counter = 0;
    for(int j=center_y-bRH; j<=center_y+bRH; j++){
            for(int i=center_x-bRW; i<=center_x+bRW; i++){
                ref_patch[counter] = j*w+i;
                counter++;
        }
    }

    // For each comparison pixel...
    // Get comparison patch
    float comp_patch[patch_size] = {0.0f};

    counter = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            comp_patch[counter] = j*w+i;
            counter++;
        }
    }

    // Calculate weight
    //float weight = 0.0f;
    //weight = getExpDecayWeight(ref_std, comp_std);

    // Calculate Hausdorff distance
    hausdorff_map[gy*w+gx] = getHausdorffDistance(ref_patch, comp_patch, bRW, w);

}

// ---- USER FUNCTIONS ----
float getExpDecayWeight(float ref, float comp){
    // Gaussian weight, see https://en.wikipedia.org/wiki/Non-local_means#Common_weighting_functions
    // Alternative: exponential decay function: 1 - abs(mean_x - mean_y / abs(mean_x + abs(mean_y)))
    float weight = 0;

    if(ref == comp){
        weight = 1;
    }else{
        weight = 1 - (fabs(ref - comp) / (ref + comp));
    }
    return weight;
}

float getHausdorffDistance(float* ref_patch, float* comp_patch, int bL, int width){
    float max_distance = 0.0f;

    for(int j=0; j<bL; j++){
        for(int i=0; i<bL; i++){
            float min_distance = FLT_MAX;
            for(int jj=0; jj<bL; jj++){
                for(int ii=0; ii<bL; ii++){
                    int ref_patch_id = j*bL+i;
                    int comp_patch_id = jj*bL+ii;

                    int ref_patch_x = (int) ref_patch[ref_patch_id] % width;
                    int ref_patch_y = (int) ref_patch[ref_patch_id] / width;

                    int comp_patch_x = (int) comp_patch[comp_patch_id] % width;
                    int comp_patch_y = (int) comp_patch[comp_patch_id] / width;

                    float distance = (float) sqrt((((float)ref_patch_x-(float)comp_patch_x)*((float)ref_patch_x-(float)comp_patch_x)) + (((float)ref_patch_y-(float)comp_patch_y)*((float)ref_patch_y-(float)comp_patch_y)));
                    min_distance = min(min_distance, distance);
                }
            }
            max_distance = fmax(max_distance, min_distance)+EPSILON;
            max_distance = 1.0f / max_distance;
            max_distance = fmin(max_distance, 1.0f);
            //max_distance = fmax(1.0f/(fmax(max_distance, min_distance)+EPSILON), 1.0f);
        }
    }
    return max_distance;

}
