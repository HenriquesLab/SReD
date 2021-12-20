#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
float getSum(float a[], int n);
float getSqSum(float a[], int n);
float getMean(float sum, int n);
float getVariance(float sqSum, float mean, int n);
float getStdDev(float variance);

kernel void kernelGetStats(
    global float* refPixels,
    global float* localSums,
    global float* localSqSums,
    global float* localMeans,
    global float* localVariances,
    global float* localDeviations
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int bRW = bW/2;
    int bRH = bH/2;
    int patchSize = bW*bH;

    for (gy=1; gy<h-1; gy++){
        for (gx=1; gx<w-1; gx++){

            // Get reference patch pixels
             float refPatch[patchSize];
             int refCounter = 0;
             for (int j=gy-bRH; j<=gy+bRH; j++){
                 for (int i=gx-bRW; i<=gx+bRW; i++){
                     refPatch[refCounter] = refPixels[j*w+i];
                     refCounter++;
                 }
             }

              // Get patch stats
              localSums[gy*w+gx] = getSum(refPatch, patchSize);
              localSqSums[gy*w+gx] = getSqSum(refPatch, patchSize);
              localMeans[gy*w+gx] = getMean(localSums[gy*w+gx], patchSize);
              localVariances[gy*w+gx] = getVariance(localSqSums[gy*w+gx], localMeans[gy*w+gx], patchSize);
              localDeviations[gy*w+gx] = getStdDev(localVariances[gy*w+gx]);

        }
    }
}

float getSum(float a[], int n) {
    float sum = 0;
    for (int i=0; i<n; i++){
        sum += a[i];
    }
    return sum;
}

float getSqSum(float a[], int n) {
    float sqSum = 0;
    for (int i=0; i<n; i++){
        sqSum += a[i]*a[i];
    }
    return sqSum;
}

float getMean(float sum, int n) {
    float mean = sum / n;
    return mean;
}

float getVariance(float sqSum, float mean, int n) {
    float variance = 0;
    if (n == 0) {
        return 0.0;
    }else{
        variance = sqSum / n - mean * mean;
    }
}

float getStdDev(float variance) {
    float stdDev = sqrt(variance);
    return stdDev;
}
