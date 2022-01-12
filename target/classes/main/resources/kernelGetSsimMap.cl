//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define filterParamSq $FILTER_PARAM_SQ$
#define patchSize $PATCH_SIZE$
#define offset_x $OFFSET_X$
#define offset_y $OFFSET_Y$
float getWeight(float ref, float comp);
float getSsim(float mean_x, float mean_y, float var_x, float var_y, float cov_xy, int n);

kernel void kernelGetSsimMap(
    global float* refPixels,
    global float* localMeans,
    global float* ssimMap
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
    float var_x = 0;
    int refCounter = 0;
    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            refPatch[refCounter] = refPixels[j0*w+i0];
            meanSub_x[refCounter] = refPatch[refCounter] - localMeans[y0*w+x0];
            var_x += meanSub_x[refCounter]*meanSub_x[refCounter];
            refCounter++;
        }
    }
    var_x /= patchSize;

    // For each comparison pixel...
    float weight;
    for(int y1=offset_y; y1<h-offset_y; y1++){
        for(int x1=offset_x; x1<w-offset_x; x1++){

            weight = 0;

            // Get comparison patch Y
            float compPatch[patchSize];
            float meanSub_y[patchSize];
            float var_y = 0;
            float cov_xy = 0;
            int compCounter = 0;
            for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                    compPatch[compCounter] = refPixels[j1*w+i1];
                    meanSub_y[compCounter] = compPatch[compCounter] - localMeans[y1*w+x1];
                    var_y += meanSub_y[compCounter]*meanSub_y[compCounter];
                    cov_xy += meanSub_x[compCounter]*meanSub_y[compCounter];
                    compCounter++;
                }
            }
            var_y /= patchSize;
            cov_xy /= patchSize;

            // Calculate weight
            weight = getWeight(localMeans[y0*w+x0], localMeans[y1*w+x1]);

            // Calculate RMSE(X,Y) and add it to the sum at X
            ssimMap[y0*w+x0] += getSsim(localMeans[y0*w+x0], localMeans[y1*w+x1], var_x, var_y, cov_xy, patchSize) * weight;
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

float getSsim(float mean_x, float mean_y, float var_x, float var_y, float cov_xy, int n){
    float ssim = 0;
    float c1 = (0.01*255)*(0.01*255);
    float c2 = (0.03*255)*(0.03*255);
    float mean_x_sq = mean_x*mean_x;
    float mean_y_sq = mean_y*mean_y;

    ssim = (2*mean_x*mean_y+c1)*(2*cov_xy+c2)/((mean_x_sq+mean_y_sq+c1)*(var_x+var_y+c2));
    return ssim;
}
