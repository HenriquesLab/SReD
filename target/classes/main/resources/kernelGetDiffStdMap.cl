//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define filter_param $FILTERPARAM$
#define EPSILON $EPSILON$
#define nUnique $NUNIQUE$
#define speedUp $SPEEDUP$
float getExpDecayWeight(float ref, float comp);
float getGaussianWeight(float ref, float comp, float h2);

kernel void kernelGetDiffStdMap(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global int* uniqueStdCoords,
    global float* diff_std_map
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

    // Bound check to avoids borders dynamically based on patch dimensions
    if(x0<bRW || x0>=w-bRW || y0<bRH || y0>=h-bRH){
        return;
    }

    // ---- Reference patch ----
    // Get mean-subtracted patch
    float ref_patch[patch_size] = {0.0f};
    float ref_mean = local_means[y0*w+x0];

    int ref_counter = 0;
    float r2 = bRW*bRW;
    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            float dx = (float)(i0-x0);
            float dy = (float)(j0-y0);
            if(dx*dx+dy*dy <= r2){
                ref_patch[ref_counter] = (ref_pixels[j0*w+i0] - ref_mean);
                ref_counter++;
            }
        }
    }

    // For each comparison pixel...
    for(int y1=bRH; y1<h-bRH; y1++){
        for(int x1=bRW; x1<w-bRW; x1++){
            // Get mean-subtracted patch and local standard deviation
            float comp_patch[patch_size] = {0.0f};
            float comp_mean = local_means[y1*w+x1];
            int comp_counter = 0;
            for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                    float dx = (float)(i1-x1);
                    float dy = (float)(j1-y1);
                    if(dx*dx+dy*dy <= r2){
                        comp_patch[comp_counter] = (ref_pixels[j1*w+i1] - comp_mean);
                        comp_counter++;
                    }
                }
            }

            // Calculate absolute difference of standard deviations add it to the sum at X (avoiding division by zero)
            float std_x = local_stds[y0*w+x0];
            float std_y = local_stds[y1*w+x1];
            float weight = 1.0f - getGaussianWeight(std_x, std_y, filter_param);
            diff_std_map[y0*w+x0] += ((fabs(std_x - std_y)) * weight);
        }
    }
}

// ---- USER FUNCTIONS ----
float getExpDecayWeight(float ref, float comp){
    // Gaussian weight, see https://en.wikipedia.org/wiki/Non-local_means#Common_weighting_functions
    // Alternative: exponential decay function: 1-abs(mean_x-mean_y/abs(mean_x+abs(mean_y)))
    float weight = 0;
    float similarity = (-fabs(ref-comp)/(fabs(ref)+fabs(comp) + EPSILON));
    weight = ((float) pow(100, similarity) - 1) / 99;

    return weight;
}

float getGaussianWeight(float ref, float comp, float h2){
    float weight = (-1) * (((fabs(comp-ref)) * (fabs(comp-ref))) / (h2 + EPSILON));
    weight = exp(weight);
    weight = fmax(weight, 0.0f);
    return weight;
}