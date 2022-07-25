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

kernel void kernelGetNrmseMap(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global int* uniqueStdCoords,
    global float* nrmse_map,
    global float* mae_map,
    global float* psnr_map
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

    // Get mean-subtracted reference patch
    float ref_patch[patch_size] = {0.0f};
    float ref_mean = local_means[y0*w+x0];
    float ref_std = local_stds[y0*w+x0];

    int ref_counter = 0;
    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            ref_patch[ref_counter] = (ref_pixels[j0*w+i0] - ref_mean) / (ref_std + EPSILON);
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

            // Get comparison patch MEAN SUBTRACTION
            float comp_patch[patch_size];
            float comp_mean = local_means[y1*w+x1];
            float comp_std = local_stds[y1*w+x1];
            int comp_counter = 0;
            for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                    comp_patch[comp_counter] = (ref_pixels[j1*w+i1] - comp_mean) / (comp_std + EPSILON);
                    comp_counter++;
                }
            }

            // Calculate weight
            weight = getExpDecayWeight(local_stds[y0*w+x0], local_stds[y1*w+x1]);

            // Calculate NRMSE and MAE
            float nrmse = 0.0f;
            float mae = 0.0f;

            for(int i=0; i<patch_size; i++){
                nrmse += (ref_patch[i] - comp_patch[i]) * (ref_patch[i] - comp_patch[i]);
                mae += fabs(ref_patch[i] - comp_patch[i]);
            }

            nrmse = sqrt(nrmse / patch_size);

            nrmse_map[y0*w+x0] += nrmse * weight;
            mae_map[y0*w+x0] += (mae/patch_size) * weight;
            psnr_map[y0*w+x0] += ((float) 20.0 * (float) log10((float) 1.0f / (float) (nrmse + EPSILON))) * weight;
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

