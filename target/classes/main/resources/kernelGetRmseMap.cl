#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define filterParamSq $FILTER_PARAM_SQ$
#define patchSize $PATCH_SIZE$
#define offset_x $OFFSET_X$
#define offset_y $OFFSET_Y$
float getWeight(float ref, float comp);
float getRmse(float* ref_patch[], float* comp_patch[], int n);
float getMae(float* ref_patch[], float* comp_patch[], int n);

kernel void kernelGetRmseMap(
    global float* refPixels,
    global float* localMeans,
    global float* rmseMap,
    global float* maeMap
){
    // Calculate weight (based on the Gaussian weight function used in non-local means
    // (see https://en.wikipedia.org/wiki/Non-local_means#Common_weighting_functions)
    // TODO: Check division by zero - also the function is missing the filtering parameter
    // Java expression: exp((-1)*pow(abs(patchStats1[0]-patchStats0[0]),2)/pow(0.4F*sigma,2))
    // Can also try exponential decay function: 1-abs(patchStats0[0]-patchStats1[0]/abs(patchStats0[0]+abs(patchStats1[0])))

    int x0 = get_global_id(0);
    int y0 = get_global_id(1);
    int bRW = bW/2;
    int bRH = bH/2;

    // Get reference patch
    float refPatch[patchSize];
    float meanSub_x[patchSize];
    int refCounter = 0;
    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            refPatch[refCounter] = refPixels[j0*w+i0];
            meanSub_x[refCounter] = refPatch[refCounter] - localMeans[y0*w+x0];
            refCounter++;
        }
    }

    // For each comparison pixel...
    float weight;
    for(int y1=offset_y; y1<h-offset_y; y1++){
        for(int x1=offset_x; x1<w-offset_x; x1++){

        weight = 0;

            // Get comparison patch Y
            float compPatch[patchSize];
            float meanSub_y[patchSize];
            int compCounter = 0;
            for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                    compPatch[compCounter] = refPixels[j1*w+i1];
                    meanSub_y[compCounter] = compPatch[compCounter] - localMeans[y1*w+x1];
                    compCounter++;
                }
            }

            // Calculate weight
            weight = getWeight(localMeans[y0*w+x0], localMeans[y1*w+x1]);

            // Calculate RMSE(X,Y) and add it to the sum at X
            rmseMap[y0*w+x0] += getRmse(meanSub_x, meanSub_y, patchSize) * weight;
            maeMap[y0*w+x0] += getMae(meanSub_x, meanSub_y, patchSize) * weight;
        }
    }
}

float getWeight(float mean_x, float mean_y){
    float weight = 0;
    weight = mean_y - mean_x;
    weight = fabs(weight);
    weight = weight*weight;
    weight = weight/filterParamSq;
    weight = (-1) * weight;
    weight = exp(weight);
    return weight;
}

float getRmse(float* ref_patch[], float* comp_patch[], int n){
    float foo = 0;
    float rmse = 0;
    for(int i=0; i<n; i++){
        foo = ref_patch[i] - comp_patch[i];
        foo = foo*foo;
        rmse += foo;
    }
    rmse = rmse/n;
    rmse = sqrt(rmse);
    return rmse;
}

float getMae(float* ref_patch[], float* comp_patch[], int n){
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