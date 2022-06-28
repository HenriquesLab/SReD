//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define filter_param_sq $FILTER_PARAM_SQ$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
float getExpDecayWeight(float ref, float comp);

kernel void kernelGetSsimMap(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* ssim_map
){

    int x0 = get_global_id(0);
    int y0 = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(x0<bRW || x0>=w-bRW || y0<bRH || y0>=h-bRH){
        return;
    }

    // Get reference patch
    float ref_patch[patch_size] = {0.0f};
    float ref_mean = local_means[y0*w+x0];
    float ref_std = local_stds[y0*w+x0];
    float ref_var = ref_std * ref_std;

    int ref_counter = 0;
    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            ref_patch[ref_counter] = ref_pixels[j0*w+i0] - ref_mean;
            ref_counter++;
        }
    }

    // For each comparison pixel...
    float weight = 0.0f;

    for(int y1=bRH; y1<h-bRH; y1++){
        for(int x1=bRW; x1<w-bRW; x1++){

            weight = 0.0f;

            // Get comparison patch Y
            float comp_patch[patch_size] = {0.0f};
            float comp_mean = local_means[y1*w+x1];
            float comp_std = local_stds[y1*w+x1];
            float comp_var = comp_std * comp_std;
            float covar = 0.0f;

            int comp_counter = 0;
            for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                    comp_patch[comp_counter] = ref_pixels[j1*w+i1] - comp_mean;
                    covar += ref_patch[comp_counter] * comp_patch[comp_counter];
                    comp_counter++;
                }
            }
            covar /= patch_size;

            // Calculate weight
            weight = getExpDecayWeight(ref_std, comp_std);

            // Calculate SSIM and add it to the sum at X
            float c1= (0.01f * 1) * (0.01f * 1); // constant1 * float dynamic range
            float c2 = (0.03f * 1) * (0.03f * 1); // constant2 * float dynamic range

            //ssim_map[y0*w+x0] += ((2.0f * ref_std * comp_std + c2) / ((ref_std * ref_std) + (comp_std * comp_std) + c1)) * weight; // Contrast
            ssim_map[y0*w+x0] += ((2.0f * local_stds[y0*w+x0] * local_stds[y1*w+x1] + c2)/((local_stds[y0*w+x0]*local_stds[y0*w+x0])+(local_stds[y0*w+x0]*local_stds[y0*w+x0]) + c2)) * weight; // Removed the luminance component to remove intensity-variant component
            //ssim_map[y0*w+x0] += ((2.0f * covar + c2) / (ref_var + comp_var + c2)) * weight; // Removed the luminance component to remove intensity-variant component
        }
    }
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
