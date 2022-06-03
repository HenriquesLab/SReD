#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define patch_size $PATCH_SIZE$
#define center_x $CENTER_X$
#define center_y $CENTER_Y$
#define bRW $BRW$
#define bRH $BRH$

float getInvariant(float* patch, int patch_w, int patch_h, int p, int q);

kernel void kernelGetPatchHu(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* hu_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    // Get reference patch
    float ref_patch[patch_size] = {0.0f};
    float ref_mean = local_means[center_y*w+center_x];

    int counter = 0;
    for(int j=center_y-bRH; j<=center_y+bRH; j++){
        for(int i=center_x-bRW; i<=center_x+bRW; i++){
            ref_patch[counter] = ref_pixels[j*w+i]-ref_mean;
            counter++;
        }
    }

    // Get reference Hu moment 2
    float ref_invariant_20 = 0.0f;
    float ref_invariant_02 = 0.0f;

    ref_invariant_20 = getInvariant(ref_patch, bW, bH, 2, 0);
    ref_invariant_02 = getInvariant(ref_patch, bW, bH, 0, 2);
    float ref_hu = ref_invariant_20 + ref_invariant_02;

    // Get mean-subtracted comparison patch
    float comp_patch[patch_size] = {0.0f};
    float comp_mean = local_means[gy*w+gx];

    counter = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            comp_patch[counter] = ref_pixels[j*w+i] - comp_mean;
            counter++;
        }
    }

    // Get comparison Hu moment 2
    float comp_invariant_20 = 0.0f;
    float comp_invariant_02 = 0.0f;

    comp_invariant_20 = getInvariant(comp_patch, bW, bH, 2, 0);
    comp_invariant_02 = getInvariant(comp_patch, bW, bH, 0, 2);
    float comp_hu = comp_invariant_20 + comp_invariant_02;

    // Calculate Euclidean distance between Hu moments and add to Hu map
    //hu_map[gy*w+gx] = ref_hu - comp_hu;
    hu_map[gy*w+gx] = fabs((float) ref_hu - (float) comp_hu);

}

// ---- USER FUNCTIONS ----
float getInvariant(float* patch, int patch_w, int patch_h, int p, int q){
    float moment_10 = 0.0f;
    float moment_01 = 0.0f;
    float moment_00 = 0.0f;
    float centroid_x = 0.0f;
    float centroid_y = 0.0f;
    float mu_pq = 0.0f;
    float invariant = 0.0f;

    // Get centroids x and y
    for(int j=0; j<patch_h; j++){
        for(int i=0; i<patch_w; i++){
            moment_10 += patch[j*patch_w+i] * (float) pown((float) i+1, (int) 1);
            moment_01 += patch[j*patch_w+i] * (float ) pown((float) j+1, (int) 1);
            moment_00 += patch[j*patch_w+i];
        }
    }

    centroid_x = moment_10 / (moment_00 + 0.0000001f);
    centroid_y = moment_01 / (moment_00 + 0.0000001f);

    for(int j=0; j<patch_h; j++){
        for(int i=0; i<patch_w; i++){
            mu_pq += patch[j*patch_w+i] * (float) pown((float) i+1-centroid_x, (int) p) * (float) pown((float) j+1-centroid_y, (int) q);
        }
    }

    invariant = mu_pq / (float) (pow(moment_00, (float) (1+(p+q/2))) + 0.0000001f);
    return invariant;
}