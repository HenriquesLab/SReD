#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define filter_param_sq $FILTER_PARAM_SQ$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define EPSILON $EPSILON$
float getExpDecayWeight(float ref, float comp);

kernel void kernelGetEntropyMap(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* entropy_map
){

    int x0 = get_global_id(0);
    int y0 = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(x0<bRW || x0>=w-bRW || y0<bRH || y0>=h-bRH){
        return;
    }

    // Get mean-subtracted reference patch
    float ref_patch[patch_size] = {0.0f};
    float ref_mean = local_means[y0*w+x0];

    int ref_counter = 0;
    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            ref_patch[ref_counter] = (ref_pixels[j0*w+i0] - ref_mean) / (local_stds[y0*w+x0] + EPSILON);
            ref_counter++;
        }
    }

    // ---- Get reference patch's entropy
    // Get number of unique values
    int ref_unique_size = 0;
    bool isUnique = true;

    for(int i=0; i<patch_size; i++){
        isUnique = true;
        for(int j=i+1; j<patch_size; j++){
            if(ref_patch[i] == ref_patch[j]){
                isUnique = false;
            }
        }
        if(isUnique == true){
            ref_unique_size += 1;
        }
    }

    // Get reference patch's unique values
    float ref_unique[patch_size] = {0.0}; // Cannot initialize with variable length
    int ref_index = 0;
    for(int i=0; i<patch_size; i++){
        isUnique = true;
        for(int j=i+1; j<patch_size; j++){
            if(ref_patch[i] == ref_patch[j]){
                isUnique = false;
            }
        }
        if(isUnique == true){
            ref_unique[ref_index] = ref_patch[i];
            ref_index++;
        }
    }

    // Get reference probabilities and entropy
    float p = 0.0;
    float ref_entropy = 0.0;
    for(int i=0; i<ref_unique_size; i++){
        p= 0.0;
        for(int j=0; j<patch_size; j++){
            if(ref_unique[i] == ref_patch[j]){
                p += 1.0;
            }
        }
        p /= patch_size;
        if(p != 0.0){
            ref_entropy += p * (float) log2((float) 1.0 / p);
        }else{
            continue;
        }
    }

    // For each comparison pixel...
    float weight = 0.0f;

    for(int y1=bRH; y1<h-bRH; y1++){
        for(int x1=bRW; x1<w-bRW; x1++){

            weight = 0.0f;

            // Get mean-subtracted comparison patch
            float comp_patch[patch_size] = {0.0f};
            float comp_mean = local_means[y1*w+x1];

            int comp_counter = 0;
            for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                    comp_patch[comp_counter] = (ref_pixels[j1*w+i1] - comp_mean) / (local_stds[y1*w+x1] + EPSILON);
                    comp_counter++;
                }
            }

            // ---- Get comparison patch's entropy ----
            // Get number of unique values
            int comp_unique_size = 0.0;
            isUnique = true;

            for(int i=0; i<patch_size; i++){
                isUnique = true;
                for(int j=i+1; j<patch_size; j++){
                    if(comp_patch[i] == comp_patch[j]){
                        isUnique = false;
                    }
                }
                if(isUnique == true){
                    comp_unique_size += 1;
                }
            }

            // Get comparison patch's unique values
            float comp_unique[patch_size] = {0.0}; // Cannot initialize with variable length
            int comp_index = 0;
            for(int i=0; i<patch_size; i++){
                isUnique = true;
                for(int j=i+1; j<patch_size; j++){
                    if(comp_patch[i] == comp_patch[j]){
                        isUnique = false;
                    }
                }
                if(isUnique == true){
                    comp_unique[comp_index] = comp_patch[i];
                    comp_index++;
                }
            }

            // Get comparison probabilities and entropy
            p = 0.0;
            float comp_entropy = 0.0;
            for(int i=0; i<comp_unique_size; i++){
                p = 0.0;
                for(int j=0; j<patch_size; j++){
                    if(comp_unique[i] == comp_patch[j]){
                        p += 1.0;
                    }
                }
                p /= patch_size;

                if(p != 0.0){
                    comp_entropy += p * (float) log2((float) 1.0 / p);
                }else{
                    continue;
                }
            }

            // Calculate weight
            weight = getExpDecayWeight(local_stds[y0*w+x0], local_stds[y1*w+x1]);

            // Calculate delta entropy and add it to the sum at reference coordinates
            entropy_map[y0*w+x0] += fabs((float)((-1.0)*ref_entropy) - ((-1.0)*comp_entropy)) * weight;
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
