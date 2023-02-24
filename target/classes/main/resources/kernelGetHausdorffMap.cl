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
float getHausdorffDistance(float* ref_patch, float* comp_patch, int bL, int width);

kernel void kernelGetHausdorffMap(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global int* uniqueStdCoords,
    global float* hausdorff_map,
    global float* gaussian_kernel,
    global float* weight_sum
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
    // Get array of patch coordinates
    float ref_patch[patch_size] = {0.0f};

    int ref_counter = 0;
    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            ref_patch[ref_counter] = j0*w+i0;
            ref_counter++;
        }
    }

    // For each comparison pixel...
    float weight = 0.0f;

    for(int y1=bRH; y1<h-bRH; y1++){
        for(int x1=bRW; x1<w-bRW; x1++){

            weight = 0.0f;

            // Get values, subtract the mean, and get local standard deviation
            float comp_patch[patch_size] = {0.0f};
            int comp_counter = 0;
            for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                    comp_patch[comp_counter] = j1*w+i1;
                    comp_counter++;
                }
            }

            // Calculate weight
            float ref_std = local_stds[y0*w+x0];
            float comp_std = local_stds[y1*w+x1];
            weight = getGaussianWeight(ref_std, comp_std, filter_param);
            weight_sum[y0*w+x0] += weight;

            // Calculate Hausdorff distance and add it to the sum
            hausdorff_map[y0*w+x0] += getHausdorffDistance(ref_patch, comp_patch, bRW, w) * weight;
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
