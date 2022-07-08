#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define center_x $CENTER_X$
#define center_y $CENTER_Y$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$

kernel void kernelGetPatchEntropy(
    global float* ref_pixels,
    global float* local_means,
    global float* entropy_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    float EPSILON = 0.0000001f;

    // Get mean-subtracted reference patch
    float ref_patch[patch_size] = {0.0f};
    float ref_mean = local_means[center_y*w+center_x];

    int counter = 0;
    for(int j=center_y-bRH; j<=center_y+bRH; j++){
        for(int i=center_x-bRW; i<=center_x+bRW; i++){
            ref_patch[counter] = ref_pixels[j*w+i] - ref_mean;
            counter++;
        }
    }

    // ---- Calculate reference patch's entropy ----
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

    // Get array of unique values
    float ref_unique[patch_size] = {0.0}; // Cannot initialize array with variable length
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
        p = 0.0;
        for(int j=0; j<patch_size; j++){
            if(ref_unique[i] == ref_patch[j]){
                p += 1.0;
            }
        }
        p /= patch_size;
        if(p != 0.0){
            ref_entropy += p * (float) log2((float) 1.0/p);
        }else{
            continue;
        }
    }

    // For each comparison pixel...
    // Get mean-subtracted comparison patch
    float comp_patch[patch_size] = {0.0f};
    float comp_mean = local_means[gy*w+gx];

    counter = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            comp_patch[counter] = ref_pixels[j*w+i] - comp_mean;
            counter++;
        }
    }

    // ---- Calculate comparison patch's entropy ----
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

    // Get array of unique values
    float comp_unique[patch_size] = {0.0}; // Cannot initialize array with variable length
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
            comp_entropy += p * (float) log2((float) 1.0/p);
        }else{
            continue;
        }
    }

    // Calculate delta entropy and store in entropy map
    entropy_map[gy*w+gx] = fabs(((-1)*ref_entropy) - ((-1)*comp_entropy)); // Delta entropy
}
