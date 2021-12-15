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
	global float* weightMap,
	global float* meanPearsonMap
){
    int gx = 1;
    int gy = 1;
    int bRW = bW/2; // half of the block width
    int bRH = bH/2; // half of the block height

    // Create arrays to store each Pearson's correlation coefficient and its corresponding weight
	float currentPearsonList[w*h];

    // Get reference patch pixels
            float refPatch[bW*bH];
            int refCounter = 0;
            for (int j=gy-bRH; j<gy+bRH; j++) {
                for(int i=gx-bRW; i<gx+bRW; i++) {
                    refPatch[refCounter] = refPixels[j*w+i];
                    refCounter++;

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

                // Calculate Pearson's correlation coefficient and truncate
                float pearson;
                float numerator;
                for (int c=0; c<bW*bH; c++) {
                    numerator += refPatchMeanSub[c] * compPatchMeanSub[c];
                }

                pearson = numerator / localDeviations[gy*w+gx] * localDeviations[y*w+x];
                pearson = fmax((float) 0, pearson);

                // Store values
                currentPearsonList[y*w+x] = pearson;
                }
            }

            // Get the (weighted) mean Pearson's correlation coefficient for this reference pixel
            int finalSize = sizeof(meanPearsonMap) / sizeof(meanPearsonMap[0]);
            float weights[w*h];
            for (int q=0; q<h; q++){
                for (int p=0; p<w; p++) {
                    weights[q*w+p] = weightMap[q*w+p];
                }
            }
            meanPearsonMap[gy*w+gx] = getArrayWeightedMean(currentPearsonList, weights, finalSize);
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