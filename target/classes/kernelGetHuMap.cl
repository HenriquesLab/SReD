//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define filter_param_sq $FILTER_PARAM_SQ$
#define patch_size $PATCH_SIZE$
#define offset_x $OFFSET_X$
#define offset_y $OFFSET_Y$
float getGaussianWeight(float ref, float comp);
float getExpDecayWeight(float ref, float comp);
float getInvariant(float* patch, int patch_w, int patch_h, int p, int q);

kernel void kernelGetHuMap(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* hu_map
){

    int x0 = get_global_id(0);
    int y0 = get_global_id(1);
    int bRW = bW/2;
    int bRH = bH/2;

    float invariant_20_x = 0;
    float invariant_02_x = 0;
    float hu_x = 0;

    // Get reference patch
    float ref_patch[patch_size];
    float meanSub_x[patch_size];
    int ref_counter = 0;
    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            ref_patch[ref_counter] = ref_pixels[j0*w+i0];
            meanSub_x[ref_counter] = ref_patch[ref_counter] - local_means[y0*w+x0];
            ref_counter++;
        }
    }

    // Calculate Hu moment 2 for reference patch
    invariant_20_x = getInvariant(ref_patch, bW, bH, 2, 0);
    invariant_02_x = getInvariant(ref_patch, bW, bH, 0, 2);
    hu_x = invariant_20_x + invariant_02_x;

    // For each comparison pixel...
    float weight;
    float invariant_20_y;
    float invariant_02_y;
    float hu_y;

    for(int y1=offset_y; y1<h-offset_y; y1++){
        for(int x1=offset_x; x1<w-offset_x; x1++){

            weight = 0;
            invariant_20_y = 0;
            invariant_02_y = 0;
            hu_y = 0;

            // Get comparison patch Y
            float comp_patch[patch_size];
            float meanSub_y[patch_size];
            int comp_counter = 0;
            for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                    comp_patch[comp_counter] = ref_pixels[j1*w+i1];
                    meanSub_y[comp_counter] = comp_patch[comp_counter] - local_means[y1*w+x1];
                    comp_counter++;
                }
            }

            // Calculate weight
            weight = getGaussianWeight(local_stds[y0*w+x0], local_stds[y1*w+x1]);

            // Calculate Hu moment 2 for comparison patch
            invariant_20_y = getInvariant(comp_patch, bW, bH, 2, 0);
            invariant_02_y = getInvariant(comp_patch, bW, bH, 0, 2);
            hu_y = invariant_20_y + invariant_02_y;

            // Calculate Euclidean distance between Hu moments and add to Hu map
            hu_map[y0*w+x0] += fabs(hu_y - hu_x) * weight;
        }
    }
}

float getGaussianWeight(float mean_x, float mean_y){
    // Gaussian weight, see https://en.wikipedia.org/wiki/Non-local_means#Common_weighting_functions
    // Alternative: exponential decay function: 1-abs(mean_x-mean_y/abs(mean_x+abs(mean_y)))

    float weight = 0;
    weight = mean_y - mean_x;
    weight = fabs(weight);
    weight = weight*weight;
    weight = weight/filter_param_sq;
    weight = (-1) * weight;
    weight = exp(weight);
    return weight;
}

float getExpDecayWeight(float ref, float comp){
    // Gaussian weight, see https://en.wikipedia.org/wiki/Non-local_means#Common_weighting_functions
    // Alternative: exponential decay function: 1-abs(mean_x-mean_y/abs(mean_x+abs(mean_y)))

    float weight = 0;


    weight = 1-(fabs(ref-comp)/fabs(ref+fabs(comp)));
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
    if(moment_00 < 0.000001f){
        moment_00 += 0.000001f;
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
