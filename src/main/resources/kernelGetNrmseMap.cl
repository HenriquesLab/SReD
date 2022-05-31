//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define filter_param_sq $FILTER_PARAM_SQ$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
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

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(x0<bRW || x0>=w-bRW || y0<bRH || y0>=h-bRH){
        return;
    }

    float EPSILON = 0.0000001f;

    // Get reference patch max and min
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

    // Get reference patch
    float ref_patch[patch_size] = {0.0f};
    float meanSub_x[patch_size] = {0.0f};
    int ref_counter = 0;
    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            ref_patch[ref_counter] = ref_pixels[j0*w+i0];
            meanSub_x[ref_counter] = (ref_patch[ref_counter] - local_means[y0*w+x0]) / (max_x + EPSILON);
            ref_counter++;
        }
    }

    // For each comparison pixel...
    float weight = 0.0f;
    for(int y1=bRH; y1<h-bRH; y1++){
        for(int x1=bRW; x1<w-bRW; x1++){

            weight = 0.0f;

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

            // Get comparison patch Y
            float comp_patch[patch_size];
            float meanSub_y[patch_size];
            int comp_counter = 0;
            for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                    comp_patch[comp_counter] = ref_pixels[j1*w+i1];
                    meanSub_y[comp_counter] = (comp_patch[comp_counter] - local_means[y1*w+x1]) / (max_y + EPSILON);
                    comp_counter++;
                }
            }

            // Calculate weight
            //weight = getGaussianWeight(local_stds[y0*w+x0], local_stds[y1*w+x1]);
            weight = getExpDecayWeight(local_stds[y0*w+x0], local_stds[y1*w+x1]);
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
    if(ref == comp){
            weight = 1;
        }else{
            weight = 1-(fabs(ref-comp)/(ref+comp));
        }

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
    nrmse = nrmse/(mean_y+0.00001f);
    nrmse = nrmse;

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
    //mae = 1-mae;
    return mae;
}