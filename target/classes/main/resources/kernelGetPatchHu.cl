#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define std_x $STD_X$
#define hu_x $HU_X$

float getExpDecayWeight(float ref, float comp);
float getInvariant(float* patch, int patch_w, int patch_h, int p, int q);

kernel void kernelGetPatchHu(
    global float* ref_patch,
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

    // For each comparison pixel...
    float weight = 0.0f;
    float invariant_20_y = 0.0f;
    float invariant_02_y = 0.0f;
    float hu_y = 0.0f;

    for(int gy=bRH; gy<h-bRH; gy++){
        for(int gx=bRW; gx<w-bRW; gx++){

            weight = 0.0f;
            invariant_20_y = 0.0f;
            invariant_02_y = 0.0f;
            hu_y = 0.0f;

            // Get comparison patch Y
            float comp_patch[patch_size] = {0.0f};
            float meanSub_y[patch_size] = {0.0f};
            int comp_counter = 0;

            for(int j=gy-bRH; j<=gy+bRH; j++){
                for(int i=gx-bRW; i<=gx+bRW; i++){
                    comp_patch[comp_counter] = ref_pixels[j*w+i];
                    //comp_patch[comp_counter] = (ref_pixels[j*w+i] - min_y) / (max_y - min_y + 0.00001f); // Normalize patch to [0,1]
                    meanSub_y[comp_counter] = comp_patch[comp_counter] - local_means[gy*w+gx];
                    comp_counter++;
                }
            }

            // Calculate Hu moment 2 for comparison patch
            invariant_20_y = getInvariant(meanSub_y, bW, bH, 2, 0);
            invariant_02_y = getInvariant(meanSub_y, bW, bH, 0, 2);
            hu_y = invariant_20_y + invariant_02_y;

            // Calculate weight
            weight = getExpDecayWeight(std_x, local_stds[gy*w+gx]);

            // Calculate Euclidean distance between Hu moments and add to Hu map
            hu_map[gy*w+gx] = fabs((float) hu_y - (float) hu_x);
        }
    }
}

// ---- USER FUNCTIONS ----
float getExpDecayWeight(float ref, float comp){
    // Gaussian weight, see https://en.wikipedia.org/wiki/Non-local_means#Common_weighting_functions
    // Alternative: exponential decay function: 1-abs(mean_x-mean_y/abs(mean_x+abs(mean_y)))
    float weight = 0;

    if(ref == comp){
        weight = 1;
    }else{
        weight = 1-(fabs(ref-comp)/(ref+comp));
    }
    return weight;

}

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
            moment_10 += patch[j*patch_w+i] * pown((float) i+1, (int) 1);
            moment_01 += patch[j*patch_w+i] * pown((float) j+1, (int) 1);
            moment_00 += patch[j*patch_w+i];
        }
    }

    // Avoid division by zero
    if(moment_00 < 0.00001f){
        moment_00 += 0.00001f;
    }

    centroid_x = moment_10/moment_00;
    centroid_y = moment_01/moment_00;

    for(int j=0; j<patch_h; j++){
            for(int i=0; i<patch_w; i++){
                mu_pq += patch[j*patch_w+i] * pown((float) i+1-centroid_x, (int) p) * pown((float) j+1-centroid_y, (int) q);
            }
    }

    invariant = mu_pq / pow(moment_00, (float) (1+(p+q/2)));
    return invariant;
}