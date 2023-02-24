//TODO: if only the contrast component is kept, delete covariance calculations because it is only needed for the structure component.
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

kernel void kernelGetSsimMap(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global int* uniqueStdCoords,
    global float* ssim_map,
    global float* luminance_map,
    global float* contrast_map,
    global float* structure_map

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
            float c1 = 0.001f; // Only correct for float images
            float c2 = 0.0009f; // Only correct for float images
            float c3 = c2 / 2; // Only correct for float images

            luminance_map[y0*w+x0] = ((2.0f * ref_mean * comp_mean + c1) / ((ref_mean * ref_mean) + (comp_mean * comp_mean) + c1)) * weight;
            contrast_map[y0*w+x0] = ((2.0f * ref_std * comp_std + c2) / ((ref_std * ref_std) * (comp_std * comp_std) + c2)) * weight;
            structure_map[y0*w+x0] = ((covar + c3) / (ref_std * comp_std + c3)) * weight;

            //ssim_map[y0*w+x0] += ((1.0f - ((float) fmax(0.0f, (((2.0f * ref_mean * comp_mean + c1) * (2.0f * covar + c2)) / (((ref_mean * ref_mean) + (comp_mean * comp_mean) + c1) * ((ref_std * ref_std) + (comp_std * comp_std) + c2)))))) / 2) * weight;
            //ssim_map[y0*w+x0] += ((1.0f - ((float) fmax(0.0f, ((2.0f * ref_std * comp_std + c2) / (ref_std * ref_std + comp_std * comp_std + c2))))) / 2) * weight; // Contrast component (distance, not similarity)
            float ref_comp_std = ref_std * comp_std;
            ssim_map[y0*w+x0] += ((1.0f - fmax(0.0f, (((2.0f * ref_comp_std + c2) / ((ref_std * ref_std) + (comp_std * comp_std) + c2)) * ((covar + c3) / (ref_comp_std + c3))))) / 2) * weight;
        }
    }
}

float getExpDecayWeight(float ref, float comp){
    // Exponential weighting function
    float weight = 0;
    float similarity = 1.0f - (fabs(ref-comp)/(fabs(ref)+fabs(comp) + EPSILON));
    weight =  ((float) pow(100, similarity) - 1) / 99;

    return weight;
}
