#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define sigma $SIGMA$
#define filterParamSq $FILTER_PARAM_SQ$
#define patchSize $PATCH_SIZE$
float getWeight(float ref, float comp);
//float getPearson(float x_patch[], float x_mean, float x_std, float y_patch[], float y_mean, float y_std, int patch_size);

kernel void kernelGetWeightMap(
    global float* refPixels,
    global float* localMeans,
    //global float* localDeviations,
    global float* weightMap,
    //global float* pearsonMap,
    local float* tempImage,
    local float* tempMeansMap,
    local float* tempWeightMap
    //local float* tempDeviationsMap,
    //local float* tempPearsonMap
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
    for(int b=0; b<h; b++) {
        for(int a=0; a<w; a++) {
            tempImage[b*w+a] = refPixels[b*w+a];
            tempMeansMap[b*w+a] = localMeans[b*w+a];
            //tempDeviationsMap[b*w+a] = localDeviations[b*w+a];
        }
    }

    // For each reference pixel
    for(y0=207; y0<=207; y0++){
        for(x0=218; x0<=218; x0++){

            // Get reference patch
            float refPatch[bW*bH];
            //float sum_X = 0;
            //float squareSum_X = 0;
            int refCounter = 0;
            for(int j0=y0-bRH; j0<=y0+bRH; j0++){
                for(int i0=x0-bRW; i0<=x0+bRW; i0++){
                    refPatch[refCounter] = tempImage[j0*w+i0];
                    //sum_X += refPatch[refCounter];
                    //squareSum_X += refPatch[refCounter]*refPatch[refCounter];
                    refCounter++;
                }
            }

            // For each comparison pixel
            for(int y1=1; y1<h-1; y1++){
                for(int x1=1; x1<w-1; x1++){

                    // Get comparison patch
                    float compPatch[bW*bH];
                    //float sum_Y = 0;
                    //float squareSum_Y = 0;
                    //float sum_XY = 0;
                    int compCounter = 0;
                    for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                        for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                            compPatch[compCounter] = tempImage[j1*w+i1];
                            //sum_Y += compPatch[compCounter];
                            //squareSum_Y += compPatch[compCounter]*compPatch[compCounter];
                            //sum_XY += refPatch[compCounter]*compPatch[compCounter];
                            compCounter++;
                        }
                    }

                    // Calculate weight and store in a temp weightmap
                    weightMap[y1*w+x1] = getWeight(localMeans[y0*w+x0], localMeans[y1*w+x1]);

                    // Calculate Pearson correlation coefficient
                    //pearsonMap[y1*w+x1] = getPearson(refPatch, localMeans[y0*w+x0], localDeviations[y0*w+x0], compPatch, localMeans[y1*w+x1], localDeviations[y1*w+x1], patchSize);
                    //printf("%f\n", pearsonMap[y1*w+x1]);



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

/*float getPearson(float x_patch[], float x_mean, float x_std, float y_patch[], float y_mean, float y_std, int patch_size){
    float numerator = 0;
    float denominator = x_std*y_std;
    float pearson = 0;

    for(int i=0; i<patch_size; i++){
        numerator += (x_patch[i]-x_mean)*(y_patch[i]-y_mean);
    }

    numerator *= patch_size;
    pearson = numerator/denominator;

}
*/