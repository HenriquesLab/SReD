#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define EPSILON $EPSILON$
#define nUnique $NUNIQUE$
#define speedUp $SPEEDUP$
float getExpDecayWeight(float ref, float comp);
float getInvariant(float* patch, int patch_w, int patch_h, int p, int q);

kernel void kernelGetHuMap(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global int* uniqueStdCoords,
    global float* hu_map
){

    int x0 = get_global_id(0);
    int y0 = get_global_id(1);

    // Check if reference pixel belongs to the unique list, and if not, kill the thread
    if(speedUp == 1){
        int isUnique = 0;
        for(int i=0; i<nUnique; i++){
            if(y0*w+x0 == uniqueStdCoords[i]){
                isUnique = 1;
                break;
            }
        }

        if(isUnique == 0){
            return;
        }
    }

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(x0<bRW || x0>=w-bRW || y0<bRH || y0>=h-bRH){
        return;
    }

    // ---- Get reference patch ----
    // Get reference patch minimum and maximum
    float min_x = ref_pixels[y0*w+x0];
    float max_x = ref_pixels[y0*w+x0];

    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            if(ref_pixels[j0*w+i0] < min_x){
                min_x = ref_pixels[j0*w+i0];
            }
            if(ref_pixels[j0*w+i0] > max_x){
                max_x = ref_pixels[j0*w+i0];
            }
        }
    }

    // Get mean-subtracted reference patch
    float ref_patch[patch_size] = {0.0f};
    float ref_mean = local_means[y0*w+x0];

    int ref_counter = 0;
    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            ref_patch[ref_counter] = ref_pixels[j0*w+i0] - ref_mean;
            ref_counter++;
        }
    }

    // Calculate Hu moment 2 for reference patch
    float ref_invariant_20 = 0.0f;
    float ref_invariant_02 = 0.0f;

    ref_invariant_20 = getInvariant(ref_patch, bW, bH, 2, 0);
    ref_invariant_02 = getInvariant(ref_patch, bW, bH, 0, 2);
    float ref_hu = ref_invariant_20 + ref_invariant_02;

    // For each comparison pixel...
    float weight = 0.0f;
    float comp_invariant_20 = 0.0f;
    float comp_invariant_02 = 0.0f;

    for(int y1=bRH; y1<h-bRH; y1++){
        for(int x1=bRW; x1<w-bRW; x1++){

            weight = 0.0f;
            comp_invariant_20 = 0.0f;
            comp_invariant_02 = 0.0f;

            // Get comparison patch minimum and maximum
            float min_y = ref_pixels[y1*w+x1];
            float max_y = ref_pixels[y1*w+x1];

            for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                    if(ref_pixels[j1*w+i1] < min_y){
                        min_y = ref_pixels[j1*w+i1];
                    }
                    if(ref_pixels[j1*w+i1] > max_y){
                        max_y = ref_pixels[j1*w+i1];
                    }
                }
            }

            // Get mean-subtracted comparison patch
            float comp_patch[patch_size] = {0.0f};
            float comp_mean = local_means[y1*w+x1];

            int comp_counter = 0;
            for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                    comp_patch[comp_counter] = ref_pixels[j1*w+i1] - comp_mean;
                    comp_counter++;
                }
            }

            // Calculate comparison patch's Hu moment 1
            comp_invariant_20 = getInvariant(comp_patch, bW, bH, 2, 0);
            comp_invariant_02 = getInvariant(comp_patch, bW, bH, 0, 2);
            float comp_hu = comp_invariant_20 + comp_invariant_02;

            // Calculate weight
            weight = getExpDecayWeight(local_stds[y0*w+x0], local_stds[y1*w+x1]);

            // Calculate Euclidean distance between Hu moments and add to Hu map
            //hu_map[y0*w+x0] += (1 - (0.5 * ((fabs((float) comp_hu - (float) ref_hu)) / (fabs((float) comp_hu + (float) ref_hu) + EPSILON)))) * weight;
            hu_map[y0*w+x0] += fabs((float) comp_hu - (float) ref_hu) * weight;
        }
    }
}

float getExpDecayWeight(float ref, float comp){
    // Gaussian weight, see https://en.wikipedia.org/wiki/Non-local_means#Common_weighting_functions
    // Alternative: exponential decay function: 1-abs(mean_x-mean_y/abs(mean_x+abs(mean_y)))
    float weight = 0;
    float similarity = 1.0f - (fabs(ref-comp)/(fabs(ref)+fabs(comp) + EPSILON));
    weight =  ((float) pow(100, similarity) - 1) / 99;

    return weight;
}

float getInvariant(float* patch, int patch_w, int patch_h, int p, int q){
    float x_avg = 0.0f;
    for(int i=1; i<=patch_w; i++){
        x_avg += (float)i;
    }
    x_avg /= (float)patch_w;

    float y_avg = 0.0f;
    for(int i=1; i<=patch_h; i++){
        y_avg += (float) i;
    }
    y_avg /= (float) patch_h;

    float mu_pq = 0.0f;
    float mu_00 = 0.0f;

    for(int j=0; j<patch_h; j++){
        for(int i=0; i<patch_w; i++){
            mu_pq += (float) pow((float) i + 1.0f - x_avg, (float) p) * ((float) pow((float) j + 1.0f - y_avg, (float) q)) * patch[j*patch_w+i];
            mu_00 += patch[j*patch_w+i];
        }
    }

    mu_00 = (float) pow(mu_00, 1.0f + (p+q)/2.0f);
    float invariant = mu_pq / (mu_00 + EPSILON);
    return invariant;
}
