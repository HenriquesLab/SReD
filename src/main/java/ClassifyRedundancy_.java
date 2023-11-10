import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.NonBlockingGenericDialog;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;

import java.util.*;

public class ClassifyRedundancy_ implements PlugIn {


    @Override
    public void run(String s) {

        // ---- Dialog box ----
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("Classify Redundancy");
        gd.addNumericField("Border size (in pixels): ", 1, 2);
        gd.addNumericField("Number of classes: ", 3, 3);
        gd.addCheckbox("Use optimal K (elbow method)?", false);
        gd.addNumericField("Maximum number of classes (elbow method): ", 10, 3);
        gd.showDialog();
        if (gd.wasCanceled()) return;

        // Get input image (a.k.a. redundancy map)
        ImagePlus imp = WindowManager.getCurrentImage();
        if (imp == null) {
            IJ.error("No image found. Please open an image and try again.");
            return;
        }

        IJ.log("Redundancy classification started...");

        FloatProcessor fp = imp.getProcessor().convertToFloatProcessor();
        float[] redMap = (float[]) fp.getPixels();

        // Get image dimensions
        int w = fp.getWidth();
        int h = fp.getHeight();

        // Get border size
        int border = (int) gd.getNextNumber();

        // Get number of classes
        int k = (int) gd.getNextNumber();
        int maxK = (int) gd.getNextNumber();

        // Get feature vectors (adjust array's second dimension according to number of features)
        int wWOBorder = w-2*border;
        int hWOBorder = h-2*border;
        int whWOBorder = wWOBorder*hWOBorder;

        double[][] featureVectors = new double[whWOBorder][1];

        int counter = 0;
        for(int y=border; y<h-border; y++){
            for(int x=border; x<w-border; x++){
                featureVectors[counter][0] = redMap[y*w+x];
                //featureVectors[counter][1] = x/(w+0.000000001f);
                //featureVectors[counter][2] = y/(h+0.000000001f);

                counter++;
            }
        }

        // Get max K (in case user wants to automatically calculate optimal K)
        if(gd.getNextBoolean()==true){
            IJ.log("Calculating optimal K value...");
            k = getOptimalK(featureVectors, maxK);
        }

        IJ.log("K = " + k);

        // Cluster redundancy vector with KMeans
        IJ.log("Calculating K Means...");
        int[] clustered = getKMeans(featureVectors, k);

        // Create output array
        int[] output = new int[w*h];
        counter = 0;
        for(int y=border; y<h-border; y++){
            for(int x=border; x<w-border; x++){
                output[y*w+x] = clustered[counter];
                counter++;
            }
        }

        // Create cluster map
        FloatProcessor ipOut = new FloatProcessor(w, h, output);
        ImagePlus imp1 = new ImagePlus("Redundancy clusters", ipOut);
        imp1.show();

        // Apply LUT
        IJ.run(imp1, "mpl-inferno", "");
        IJ.run(imp1, "Invert LUT", "");

        IJ.log("Finished classifying redundancy!");
    }

    // ------------------------ //
    // ---- USER FUNCTIONS ---- //
    // ------------------------ //

    public static int getOptimalK(double[][] featureVectors, int maxK){
        // Convert feature vectors array into List
        List<DoublePoint> featureVectorList = new ArrayList<>();
        for(double[] featureVector : featureVectors){
            featureVectorList.add(new DoublePoint(featureVector));
        }

        int optimalK = 1;
        double previousWCSS = Double.MAX_VALUE;
        for(int k=1; k<=maxK; k++){
            // Update progress bar
            float progress = (k / maxK) * 100;
            IJ.showStatus("Calculating optimal K... " + progress + "%");

            // Calculate WCSS for this K
            KMeansPlusPlusClusterer<DoublePoint> kMeans = new KMeansPlusPlusClusterer<>(k, 1000, new EuclideanDistance());
            List<CentroidCluster<DoublePoint>> clusters = kMeans.cluster(featureVectorList);
            double wcss = 0.0;
            for (CentroidCluster<DoublePoint> cluster : clusters) {
                for (DoublePoint featureVector : cluster.getPoints()) {
                    double distance = new EuclideanDistance().compute(cluster.getCenter().getPoint(), featureVector.getPoint());
                    wcss += distance * distance;
                }
            }
            if (wcss < previousWCSS) {
                previousWCSS = wcss;
                optimalK = k;
            }else{
                break;
            }
        }
        return optimalK;
    }

    public static int[] getKMeans(double[][] featureVectors, int k){
        // Convert feature vectors array into list
        List<DoublePoint> featureVectorList = new ArrayList<>();
        for(double[] featureVector : featureVectors){
            featureVectorList.add(new DoublePoint(featureVector));
        }

        // Create fit and predict
        KMeansPlusPlusClusterer<DoublePoint> kMeans = new KMeansPlusPlusClusterer<>(k, 1000, new EuclideanDistance());
        List<CentroidCluster<DoublePoint>> clusters = kMeans.cluster(featureVectorList);

        // Initialize output array
        int[] output = new int[featureVectors.length];

        // Map each cluster to an id, and order the clusters based on the number of elements
        Map<CentroidCluster<DoublePoint>, Integer> clusterIdMap = new HashMap<>();
        List<CentroidCluster<DoublePoint>> sortedClusters = new ArrayList<>(clusters);
        Collections.sort(sortedClusters, new Comparator<CentroidCluster<DoublePoint>>() {
            @Override
            public int compare(CentroidCluster<DoublePoint> c1, CentroidCluster<DoublePoint> c2) {
                return Integer.compare(c2.getPoints().size(), c1.getPoints().size());
            }
        });
        for(int i=0; i<sortedClusters.size(); i++){
            clusterIdMap.put(sortedClusters.get(i), i);
        }

        // Fill result array with cluster ids
        for (int i = 0; i < featureVectors.length; i++) {
            DoublePoint featureVector = featureVectorList.get(i);
            double minDistance = Double.MAX_VALUE;
            int closestClusterId = 0;
            CentroidCluster<DoublePoint> closestCluster = null;

            for (CentroidCluster<DoublePoint> cluster : clusters) {
                double[] centroid = cluster.getCenter().getPoint();
                double[] feature = featureVector.getPoint();
                double distance = new EuclideanDistance().compute(centroid, feature);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestCluster = cluster;
                    closestClusterId = clusterIdMap.get(cluster);
                }
            }
            output[i] = clusterIdMap.get(closestCluster);

        }

        return output;
    }
}
