//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
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

kernel void kernelGetPearsonMap(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global int* uniqueStdCoords,
    global float* pearson_map
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

    // TODO: WTF IS THIS?
    if(min_x == max_x){
        min_x = 0.0f;
        max_x = 1.0f;
    }

    // Get mean-subtracted patch
    float ref_patch[patch_size] = {0.0f};
    float ref_mean = local_means[y0*w+x0];

    int ref_counter = 0;
    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            ref_patch[ref_counter] = ref_pixels[j0*w+i0] - ref_mean;
            //ref_patch[ref_counter] = (ref_pixels[j0*w+i0] - min_x) / (max_x - min_x + EPSILON); // Normalize patch to [0,1]
            ref_counter++;
        }
    }

    // For each comparison pixel...
    float weight = 0.0f;

    for(int y1=bRH; y1<h-bRH; y1++){
        for(int x1=bRW; x1<w-bRW; x1++){

            weight = 0.0f;

            // Get patch minimum and maximum
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

            // Get values, subtract the mean, and get local standard deviation
            float comp_patch[patch_size] = {0.0f};
            float comp_mean = local_means[y1*w+x1];
            float covar = 0.0f;

            int comp_counter = 0;
            for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                    comp_patch[comp_counter] = ref_pixels[j1*w+i1] - comp_mean;
                    //comp_patch[comp_counter] = (ref_pixels[j1*w+i1] - min_y) / (max_y - min_y + EPSILON); // Normalize patch to [0,1]
                    covar += ref_patch[comp_counter] * comp_patch[comp_counter];
                    comp_counter++;
                }
            }
            covar /= patch_size;

            // Calculate weight
            float std_x = local_stds[y0*w+x0];
            float std_y = local_stds[y1*w+x1];
            weight = getExpDecayWeight(std_x, std_y);
            float similarity = 1.0f - (fabs(std_x-std_y)/(fabs(std_x)+fabs(std_y) + EPSILON));

            // Calculate Pearson correlation coefficient X,Y and add it to the sum at X (avoiding division by zero)
            if(std_x == 0.0f && std_y == 0.0f){
                pearson_map[y0*w+x0] += 0.0f * weight; // Special case when both patches are flat (correlation would be NaN but we want 1 because textures are the same, so 1-PEarson = 1-1 = 0)
            }else{
                pearson_map[y0*w+x0] += (1.0f - ((float) fmax(0.0f, (float) (covar / ((std_x * std_y) + EPSILON))))) * weight; // Truncate anti-correlations to zero
            }
        }
    }
}

// ---- USER FUNCTIONS ----
float getExpDecayWeight(float ref, float comp){
    // Gaussian weight, see https://en.wikipedia.org/wiki/Non-local_means#Common_weighting_functions
    // Alternative: exponential decay function: 1-abs(mean_x-mean_y/abs(mean_x+abs(mean_y)))
    float weight = 0;
    float similarity = 1.0f - (fabs(ref-comp)/(fabs(ref)+fabs(comp) + EPSILON));
    weight =  ((float) pow(100, similarity) - 1) / 99;

    return weight;
}