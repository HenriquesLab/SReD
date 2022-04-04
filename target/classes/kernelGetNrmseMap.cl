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
float getNrmse(float* ref_patch, float* comp_patch, float mean_y, int n);
float getMae(float* ref_patch, float* comp_patch, int n);

kernel void kernelGetNrmseMap(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* nrmse_map,
    global float* mae_map
){

    int x0 = get_global_id(0);
    int y0 = get_global_id(1);
    int bRW = bW/2;
    int bRH = bH/2;

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

    // For each comparison pixel...
    float weight;
    for(int y1=offset_y; y1<h-offset_y; y1++){
        for(int x1=offset_x; x1<w-offset_x; x1++){

        weight = 0;

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

            // Calculate NRMSE(X,Y) and add it to the sum at X
            nrmse_map[y0*w+x0] += getNrmse(meanSub_x, meanSub_y, local_means[y1*w+x1], patch_size) * weight;
            mae_map[y0*w+x0] += getMae(meanSub_x, meanSub_y, patch_size) * weight;
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

float getNrmse(float* ref_patch, float* comp_patch, float mean_y, int n){
    float foo = 0;
    float nrmse = 0;
    for(int i=0; i<n; i++){
        foo = ref_patch[i] - comp_patch[i];
        foo = foo*foo;
        nrmse += foo;
    }
    nrmse = nrmse/n;
    nrmse = sqrt(nrmse);
    nrmse = nrmse/(mean_y+0.000001f);

    return nrmse;
}

float getMae(float* ref_patch, float* comp_patch, int n){
    float foo = 0;
    float mae = 0;
    for(int i=0; i<n; i++){
        foo = ref_patch[i] - comp_patch[i];
        foo = fabs(foo);
        mae += foo;
    }
    mae = mae/n;
    return mae;
}