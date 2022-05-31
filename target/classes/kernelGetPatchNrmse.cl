#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define std_x $STD_X$

float getExpDecayWeight(float ref, float comp);
float getNrmse(float* ref, float* comp, float mean_y, int n);
float getMae(float* ref, float* comp, int n);

kernel void kernelGetPatchNrmse(
    global float* ref_patch,
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* nrmse_map,
    global float* mae_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    // Get reference patch
    float refPatch[patch_size] = {0.0f};
    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            refPatch[j*bW+i] = ref_patch[j*bW+i];
        }
    }

    // For each comparison pixel...
    float weight = 0.0f;

    // Get comparison patch minimum and maximum
    float min_y = ref_pixels[gy*w+gx];
    float max_y = ref_pixels[gy*w+gx];

    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            if(ref_pixels[j*w+i] < min_y){
               min_y = ref_pixels[j*w+i];
            }
            if(ref_pixels[j*w+i] > max_y){
                max_y = ref_pixels[j*w+i];
            }
        }
    }

    // Get comparison patch Y
    float comp_patch[patch_size];
    float meanSub_y[patch_size];
    int comp_counter = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            comp_patch[comp_counter] = ref_pixels[j*w+i];
            meanSub_y[comp_counter] = comp_patch[comp_counter] - local_means[gy*w+gx];
            comp_counter++;
        }
    }

    // Calculate weight
    weight = getExpDecayWeight(std_x, local_stds[gy*w+gx]);

    // Calculate NRMSE(X,Y) and add it to the sum at X
    nrmse_map[gy*w+gx] = getNrmse(refPatch, meanSub_y, local_means[gy*w+gx], patch_size) * weight;
    mae_map[gy*w+gx] = getMae(refPatch, meanSub_y, patch_size);
}




// ---- USER FUNCTIONS ----
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

float getNrmse(float* ref, float* comp, float mean_y, int n){
    float foo = 0;
    float nrmse = 0;
    for(int i=0; i<n; i++){
        foo = ref[i] - comp[i];
        foo = foo*foo;
        nrmse += foo;
    }
    nrmse = nrmse/n;
    nrmse = sqrt(nrmse);
    nrmse = nrmse/(mean_y+0.0000001f);
    nrmse = nrmse;

    return nrmse;
}

float getMae(float* ref, float* comp, int n){
    float foo = 0;
    float mae = 0;
    for(int i=0; i<n; i++){
        foo = ref[i] - comp[i];
        foo = fabs(foo);
        mae += foo;
    }
    mae = mae/n;
    return mae;
}