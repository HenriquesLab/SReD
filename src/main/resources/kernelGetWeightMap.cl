#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define sigma $SIGMA$
#define filterParamSq $FILTER_PARAM_SQ$
#define patchSize $PATCH_SIZE$
float getWeight(float ref, float comp);

kernel void kernelGetWeightMap(
    global float* refPixels,
    global float* localMeans,
    global float* weightMap,
    global float* pearsonMap,
    local float* tempImage,
    local float* tempMeansMap,
    local float* tempWeightMap,
    local float* tempPearsonMap
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

    // Make local copy of the reference image and the local means
    for(int a=0; a<w*h; a++) {
            tempImage[a] = refPixels[a];
            tempMeansMap[a] = localMeans[a];
    }

    // For each reference pixel
    for(y0=1; y0<=1; y0++){
        for(x0=1; x0<=1; x0++){

            // Get reference patch
            float refPatch[patchSize];
            float meanSub_x[patchSize];
            float std_x = 0;
            int refCounter = 0;
            for(int j0=y0-bRH; j0<=y0+bRH; j0++){
                for(int i0=x0-bRW; i0<=x0+bRW; i0++){
                    refPatch[refCounter] = refPixels[j0*w+i0];
                    meanSub_x[refCounter] = refPatch[refCounter] - localMeans[y0*w+x0];
                    std_x += meanSub_x[refCounter]*meanSub_x[refCounter];
                    refCounter++;
                }
            }
            std_x = sqrt(std_x);

            // For each comparison pixel
            for(int y1=1; y1<h-1; y1++){
                for(int x1=1; x1<w-1; x1++){

                    // Get comparison patch
                    float compPatch[patchSize];
                    float meanSub_y[patchSize];
                    float std_y = 0;
                    float meanSub_xy = 0;
                    int compCounter = 0;
                    for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                        for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                            compPatch[compCounter] = refPixels[j1*w+i1];
                            meanSub_y[compCounter] = compPatch[compCounter] - localMeans[y1*w+x1];
                            std_y += meanSub_y[compCounter]*meanSub_y[compCounter];
                            meanSub_xy += meanSub_x[compCounter] * meanSub_y[compCounter];
                            compCounter++;

                        }
                    }
                    std_y = sqrt(std_y);

                    // Calculate weight and store in a temp weight map
                    weightMap[y1*w+x1] = getWeight(localMeans[y0*w+x0], localMeans[y1*w+x1]);

                    // Calculate Pearson correlation coefficient
                    pearsonMap[y1*w+x1] = fmax((float) 0, meanSub_xy/(std_x*std_y));
                }

            }
        }




    }
}


float getWeight(float ref, float comp){
    float weight = 0;
    weight = comp - ref;
    weight = fabs(weight);
    weight = weight*weight;
    weight = weight/filterParamSq;
    weight = (-1) * weight;
    weight = exp(weight);
    return weight;
}