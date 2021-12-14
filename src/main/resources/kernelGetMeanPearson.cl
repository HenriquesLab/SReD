#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define sigma $SIGMA$
float getArrayWeightedMean(float a[], float b[], int n);

// kernel: Get mean Pearson's correlation coefficient
kernel void kernelGetMeanPearson(
	global float* refPixels,
	global float* localMeans,
	global float* localDeviations,
	global float* meanPearsonMap
){
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int bRW = bW/2; // half of the block width
    int bRH = bH/2; // half of the block height

    // Create arrays to store each Pearson's correlation coefficient and its corresponding weight
	float currentPearsonList[w*h];
	float currentWeightList[w*h];

    // Get reference patch pixels
    for (gy=1; gy<h-1; gy++) {
        for (gx=1; gx<w-1; gx++) {
            float refPatch[bW*bH];
            int refCounter = 0;
            for (int j=gy-bRH; j<gy+bRH; j++) {
                for(int i=gx-bRW; i<gx+bRW; i++) {
                    refPatch[refCounter] = refPixels[j*w+i];
                    refCounter++;
                }
            }

            // Mean-subtract reference patch
            float refPatchMeanSub[bW*bH];

            for (int a=0; a<bW*bH; a++) {
                refPatchMeanSub[a] = refPatch[a] - localMeans[gy*w+gx];
            }

            // Loop across comparison pixels
            for (int y = 1; y < h - 1; y++) {
                for (int x = 1; x < w - 1; x++) {

                // Get patch pixels
                float compPatch[bW*bH];

                int compCounter = 0;
                for (int j = y-bRH; j<y+bRH; j++){
                    for (int i = x-bRW; i<x+bRW; x++) {
                        compPatch[compCounter] = refPixels[j*w+i];
                        compCounter++;
                    }
                }

                // Pre-compare patches' standard deviation to decide whether it's worth calculating Pearson's
                float preDiff = localDeviations[gy*w+gx] - localDeviations[y*w+x];
                float preComparison = fabs(preDiff);
                float threshold = localDeviations[gy*w+gx]*2; // Used in the pre-comparison

                // Mean-subtract comparison patch (needed for the next step)
                float compPatchMeanSub[bW*bH];

                for (int b = 0; b < bW*bH; b++) {
                    compPatchMeanSub[b] = compPatch[b] - localMeans[y*w+x];
                }

                // Pre-compare, and proceed to calculate Pearson's if pre-comparison crosses the redundancy threshold
                float pearson;
                float weight;
                float filteringParamSquared;

                if (preComparison <= threshold) {

                    // Calculate Pearson's correlation coefficient and truncate
                    float num = 0;

                    for (int c=0; c<bW*bH; c++) {
                        num += refPatchMeanSub[c] * compPatchMeanSub[c];
                    }

                    pearson = num / localDeviations[gy*w+gx] * localDeviations[y*w+x];
                    pearson = fmax((float) 0, pearson);

                    // Calculate weight
                    // Non-local means Gaussian weight function;
                    // https://en.wikipedia.org/wiki/Non-local_means#Common_weighting_functions
                    // TODO:Check division by zero
                    // Java expression: exp((-1)*pow(abs(patchStats1[0]-patchStats0[0]),2)/pow(0.4F*sigma,2))
                    // Can also try exponential decay: 1-abs(patchStats0[0]-patchStats1[0]/abs(patchStats0[0]+abs(patchStats1[0])))
                    filteringParamSquared = (float) pow((float) 0.4 * (float) sigma, (float) 2.0);
                    weight = localMeans[y*w+x] - localMeans[gy*w+gx];
                    weight = fabs(weight);
                    weight = pow(weight, 2);
                    weight = (-1) * weight;
                    weight = exp(weight);

                    }else{
                        // Store an arbitrary Pearson's and weight (defaulted to zero, representing the lowest Pearson's possible)
                        pearson = 0;
                        weight = 0;
                    }

                    // Store values
                    currentPearsonList[y*w+x] = pearson;
                    currentWeightList[y*w+x] = weight;

                }
            }

            // Get the (weighted) mean Pearson's correlation coefficient for this reference pixel
            int finalSize = sizeof(meanPearsonMap) / sizeof(meanPearsonMap[0]);
            meanPearsonMap[gy*w+gx] = getArrayWeightedMean(currentPearsonList, currentWeightList, finalSize);
        }
    }
}

// ---- User functions ----
float getArrayWeightedMean(float a[], float b[], int n) {
    float weightedMean = 0;

    for(int i=0; i<n; i++){
        weightedMean += a[i] * b[i];
    }

    weightedMean /= n;
    return weightedMean;
}