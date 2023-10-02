#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define patch_size $PATCH_SIZE$
#define bW $BW$
#define bH $BH$
#define bRW $BRW$
#define bRH $BRH$
#define EPSILON $EPSILON$

kernel void kernelGetSynthPatchHu(
    global float* patch_pixels,
    global float* ref_pixels,
    global float* local_means,
    global float* hu_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // ---------------------------------------------------------------------------- //
    // ---- Bound check (avoids borders dynamically based on patch dimensions) ---- //
    // ---------------------------------------------------------------------------- //

    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }


    // ---------------------------------------------------------------------------------------------------------------- //
    // ---- Get mean-subtracted reference patch from buffer (already mean_subtracted and normalized in the buffer) ---- //
    // ---------------------------------------------------------------------------------------------------------------- //

    __local double ref_patch[bW*bH];

    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            float dx = (float)(i-bRW)/bRW;
            float dy = (float)(j-bRH)/bRH;
            if(dx*dx + dy*dy <= 1.0f){
                ref_patch[j*bW+i] = (double)patch_pixels[j*bW+i];
            }
        }
    }


    // ------------------------------------------------------ //
    // ---- Calculate Hu moments for the reference patch ---- //
    // ------------------------------------------------------ //

    // Calculate raw moments
    double ref_M00 = 0.0;
    double ref_M10 = 0.0;
    double ref_M01 = 0.0;

    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            double dx = (double)(i-bRW)/bRW;
            double dy = (double)(j-bRH)/bRH;
            if(dx*dx + dy*dy <= 1.0){
                double pixel_value = ref_patch[j*bW+i];
                ref_M00 += pixel_value;
                ref_M10 += (double)(i+1) * pixel_value;
                ref_M01 += (double)(j+1) * pixel_value;
            }
        }
    }

    // Calculate central moments
    double ref_centroid_x = ref_M10 / (ref_M00 + (double)EPSILON);
    double ref_centroid_y = ref_M01 / (ref_M00 + (double)EPSILON);
    double ref_mu20 = 0.0;
    double ref_mu02 = 0.0;
    double ref_mu11 = 0.0;
    double ref_mu30 = 0.0;
    double ref_mu03 = 0.0;
    double ref_mu12 = 0.0;
    double ref_mu21 = 0.0;

    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            double dx = (double)(i-bRW)/bRW;
            double dy = (double)(j-bRH)/bRH;
            if(dx*dx + dy*dy <= 1.0){
                double pixel_value = ref_patch[j*bW+i];
                ref_mu20 += pow(((double)i-ref_centroid_x), 2.0) * pixel_value;
                ref_mu02 += pow(((double)j-ref_centroid_y), 2.0) * pixel_value;
                ref_mu11 += ((double)i-ref_centroid_x) * ((double)j-ref_centroid_y) * pixel_value;
                ref_mu30 += pow(((double)i-ref_centroid_x), 3.0) * pixel_value;
                ref_mu03 += pow(((double)j-ref_centroid_y), 3.0) * pixel_value;
                ref_mu12 += ((double)i-ref_centroid_x) * pow(((double)j-ref_centroid_y), 2.0) * pixel_value;
                ref_mu21 += pow(((double)i-ref_centroid_x), 2.0) * ((double)j-ref_centroid_y) * pixel_value;

            }
        }
    }

    // Calculate Hu invariant 1
    double ref_nu20 = ref_mu20 / (pow(ref_M00, 2.0) + (double)EPSILON);
    double ref_nu02 = ref_mu02 / (pow(ref_M00, 2.0) + (double)EPSILON);
    double ref_hu1 = ref_nu20 + ref_nu02;

    // Calculate Hu invariant 2
    double ref_nu11 = ref_mu11 / (pow(ref_M00, 2.0) + (double)EPSILON);
    double ref_hu2 = (ref_nu20 - ref_nu02) * (ref_nu20 - ref_nu02) + (4.0*pow(ref_nu11, 2.0));

    // Calculate Hu invariant 3
    double ref_nu30 = ref_mu30 / (pow(ref_M00, 2.5) + (double)EPSILON);
    double ref_nu03 = ref_mu03 / (pow(ref_M00, 2.5) + (double)EPSILON);
    double ref_nu12 = ref_mu12 / (pow(ref_M00, 2.5) + (double)EPSILON);
    double ref_nu21 = ref_mu21 / (pow(ref_M00, 2.5) + (double)EPSILON);

    double ref_hu3 = pow(ref_nu30-3.0*ref_nu12, 2.0) + pow(3.0*ref_nu21-ref_nu03, 2.0);

    // Calculate Hu invariant 4
    double ref_hu4 = pow(ref_nu30+ref_nu12, 2.0) + pow(ref_nu21-ref_nu03, 2.0);

    // Calculate Hu invariant 5
    double ref_hu5 = (ref_nu30-3.0*ref_nu12) * (ref_nu30+ref_nu12) * (pow(ref_nu30+ref_nu12, 2.0) - 3.0*pow(ref_nu21+ref_nu03, 2.0)) + (3.0*ref_nu21-ref_nu03) * (ref_nu21+ref_nu03) * (3.0*pow(ref_nu30+ref_nu12, 2.0) - pow(ref_nu21+ref_nu03, 2.0));


    // ---------------------------------------------- //
    // ---- Get mean-subtracted comparison patch ---- //
    // ---------------------------------------------- //

    // Get patch and patch mean
    double comp_patch[bW*bH];
    double comp_mean = 0.0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            float dx = (float)((i-gx)/bRW);
            float dy = (float)((j-gy)/bRH);
            if(dx*dx+dy*dy <= 1.0f){
                double pixel_value = (double)ref_pixels[j*w+i];
                comp_patch[(j-(gy-bRH)) * bW + (i-(gx-bRW))] = pixel_value;
                comp_mean += pixel_value;
            }
        }
    }
    comp_mean /= (double)patch_size;

    // Mean-subtract patch
    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            float dx = (float)(i-bRW)/bRW;
            float dy = (float)(j-bRH)/bRH;
            if(dx*dx+dy*dy <= 1.0f){
                comp_patch[j*bW+i] -= comp_mean;
            }
        }
    }

    // ------------------------------------ //
    // ---- Normalize comparison patch ---- //
    // ------------------------------------ //

    // Find min and max
    double comp_min_intensity = DBL_MAX;
    double comp_max_intensity = -DBL_MAX;

    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            float dx = (float)(i-bRW)/bRW;
            float dy = (float)(j-bRH)/bRH;
            if(dx*dx+dy*dy <= 1.0f){
                double pixel_value = comp_patch[j*bW+i];
                comp_min_intensity = min(comp_min_intensity, pixel_value);
                comp_max_intensity = max(comp_max_intensity, pixel_value);
            }
        }
    }

    // Remap pixel values
    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            float dx = (float)(i-bRW)/bRW;
            float dy = (float)(j-bRH)/bRH;
            if(dx*dx+dy*dy <= 1.0f){
                comp_patch[j*bW+i] = (comp_patch[j*bW+i] - comp_min_intensity) / (comp_max_intensity - comp_min_intensity + (double)EPSILON);
            }
        }
    }


    // ------------------------------------------------------- //
    // ---- Calculate Hu moments for the comparison patch ---- //
    // ------------------------------------------------------- //

    // Calculate raw moments
    double comp_M00 = 0.0;
    double comp_M10 = 0.0;
    double comp_M01 = 0.0;

    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            float dx = (float)(i-bRW)/bRW;
            float dy = (float)(j-bRH)/bRH;
            if(dx*dx + dy*dy <= 1.0f){
                double pixel_value = comp_patch[j*bW+i];
                comp_M00 += pixel_value;
                comp_M10 += (double)(i+1) * pixel_value;
                comp_M01 += (double)(j+1) * pixel_value;
            }
        }
    }

    // Calculate central moments
    double comp_centroid_x = comp_M10 / (comp_M00 + (double)EPSILON);
    double comp_centroid_y = comp_M01 / (comp_M00 + (double)EPSILON);

    double comp_mu20 = 0.0;
    double comp_mu02 = 0.0;
    double comp_mu11 = 0.0;
    double comp_mu30 = 0.0;
    double comp_mu03 = 0.0;
    double comp_mu12 = 0.0;
    double comp_mu21 = 0.0;

    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            float dx = (float)(i-bRW)/bRW;
            float dy = (float)(j-bRH)/bRH;
            if(dx*dx + dy*dy <= 1.0f){
                double pixel_value = comp_patch[j*bW+i];
                comp_mu20 += pow(((double)i-comp_centroid_x), 2.0) * pixel_value;
                comp_mu02 += pow(((double)j-comp_centroid_y), 2.0) * pixel_value;
                comp_mu11 += ((double)i-comp_centroid_x) * ((double)j-comp_centroid_y) * pixel_value;
                comp_mu30 += pow(((double)i-comp_centroid_x), 3.0) * pixel_value;
                comp_mu03 += pow(((double)j-comp_centroid_y), 3.0) * pixel_value;
                comp_mu12 += ((double)i-comp_centroid_x) * pow(((double)j-comp_centroid_y), 2.0) * pixel_value;
                comp_mu21 += pow(((double)i-comp_centroid_x), 2.0) * ((double)j-comp_centroid_y) * pixel_value;
            }
        }
    }

    // Calculate Hu invariant 1
    double comp_nu20 = comp_mu20 / (pow(comp_M00, 2.0) + (double)EPSILON);
    double comp_nu02 = comp_mu02 / (pow(comp_M00, 2.0) + (double)EPSILON);
    double comp_hu1 = comp_nu20 + comp_nu02;

    // Calculate Hu invariant 2
    double comp_nu11 = comp_mu11 / (pow(comp_M00, 2.0) + (double)EPSILON);
    double comp_hu2 = pow(comp_nu20 - comp_nu02, 2.0) + (4.0*pow(comp_nu11, 2.0));

    // Calculate Hu invariant 3
    double comp_nu30 = comp_mu30 / (pow(comp_M00, 2.5) + (double)EPSILON);
    double comp_nu03 = comp_mu03 / (pow(comp_M00, 2.5) + (double)EPSILON);
    double comp_nu12 = comp_mu12 / (pow(comp_M00, 2.5) + (double)EPSILON);
    double comp_nu21 = comp_mu21 / (pow(comp_M00, 2.5) + (double)EPSILON);
    double comp_hu3 = pow(comp_nu30-3.0*comp_nu12, 2.0) + pow(3.0*comp_nu21-comp_nu03, 2.0);

    // Calculate Hu invariant 4
    double comp_hu4 = pow(comp_nu30+comp_nu12, 2.0) + pow(comp_nu21-comp_nu03, 2.0);

    // Calculate Hu invariant 5
    double comp_hu5 = (comp_nu30-3.0*comp_nu12) * (comp_nu30+comp_nu12) * (pow(comp_nu30+comp_nu12, 2.0) - 3.0*pow(comp_nu21+comp_nu03, 2.0)) + (3.0*comp_nu21-comp_nu03) * (comp_nu21+comp_nu03) * (3.0*pow(comp_nu30+comp_nu12, 2.0) - pow(comp_nu21+comp_nu03, 2.0));



    // ------------------------------------------------- //
    // ---- Calculate the Bhattacharyya coefficient ---- //
    // ------------------------------------------------- //

    // Hu 1
    hu_map[gy*w+gx] = (float) comp_hu1;

    //hu_map[gy*w+gx] = (float) sqrt(ref_hu1*comp_hu1);

    // Hu 2
    //hu_map[gy*w+gx] = (float) sqrt(ref_hu2*comp_hu2);

    // Hu 3
    //hu_map[gy*w+gx] = (float) sqrt(ref_hu3*comp_hu3);

    // Hu 4
    //hu_map[gy*w+gx] = (float) sqrt(ref_hu4*comp_hu4);

    // Hu 5
    //hu_map[gy*w+gx] = (float) sqrt(ref_hu5*comp_hu5);

    //hu_map[gy*w+gx] = (float) sqrt(ref_hu1*comp_hu1) + sqrt(ref_hu2*comp_hu2) + sqrt(ref_hu3*comp_hu3) + sqrt(ref_hu4*comp_hu4) + sqrt(ref_hu5*comp_hu5);
}
