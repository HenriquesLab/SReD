//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define filter_param_sq $FILTER_PARAM_SQ$
#define patch_size $PATCH_SIZE$
#define offset_x $OFFSET_X$
#define offset_y $OFFSET_Y$
float getWeight(float ref, float comp);
float getEntropy(float* patch, int n);

kernel void kernelGetEntropyMap(
    global short* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* entropy_map
){

    int x0 = get_global_id(0);
    int y0 = get_global_id(1);
    int bRW = bW/2;
    int bRH = bH/2;

    float entropy_x = 0.0;

    // Get reference patch
    float ref_patch[patch_size];
    float meanSub_x[patch_size];
    int ref_counter = 0;
    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            ref_patch[ref_counter] = (float) ref_pixels[j0*w+i0];
            meanSub_x[ref_counter] = ref_patch[ref_counter] - local_means[y0*w+x0];
            ref_counter++;
        }
    }

    // Calculate Hu moment 2 for reference patch
    entropy_x = getEntropy(ref_patch, patch_size);

    // For each comparison pixel...
    float weight;
    float entropy_y;

    for(int y1=offset_y; y1<h-offset_y; y1++){
        for(int x1=offset_x; x1<w-offset_x; x1++){

            weight = 0.0;
            entropy_y = 0.0;

            // Get comparison patch Y
            float comp_patch[patch_size];
            float meanSub_y[patch_size];
            int comp_counter = 0;
            for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                    comp_patch[comp_counter] = (float) ref_pixels[j1*w+i1];
                    meanSub_y[comp_counter] = comp_patch[comp_counter] - local_means[y1*w+x1];
                    comp_counter++;
                }
            }

            // Calculate weight
            weight = getWeight(local_means[y0*w+x0], local_means[y1*w+x1]);

            // Calculate Hu moment 2 for comparison patch
            entropy_y = getEntropy(comp_patch, patch_size);

            // Calculate Euclidean distance between Hu moments and add to Hu map
            entropy_map[y0*w+x0] += fabs(entropy_y - entropy_x) * weight;
        }
    }
}

float getWeight(float mean_x, float mean_y){
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

float getEntropy(float* patch, int n){
    float entropy = 0.0f;
    for (int depth=0; depth<255; depth++){
        float p = 0.0f;
        for(int length=0; length<n; length++){
            if(patch[length] == depth){
                p += 1.0f;
            }
        }
        p = p/n;
        entropy += p*log2((float) p);
    }
    entropy = entropy*(-1);
    return entropy;
}
