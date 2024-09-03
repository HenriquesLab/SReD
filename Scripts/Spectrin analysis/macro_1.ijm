/*
 * // --------------------------------- //
 * // ---- IMPORTANT - PLEASE READ ---- //
 * // --------------------------------- //
 * 
 * This macro calculates block repetition maps for each reference block provided.
 * 
 * In Mendes et al. 2024, this macro was used to calculate repetition maps for reference blocks comprising lines at different orientations using skeletonized neuron images.
 * 
 * The input image must be open in ImageJ - the "input" variable must be defined using the name of the image and its extension (e.g., "input.tif").
 * 
 * The reference blocks should be provided as a stack and opened in ImageJ before running - the "blockStack" variable must be defined using the name of the file and its extension (e.g., "blocks.tif").
 * 
 * The path where the macro stores the results must be defined in the "outputDir" variable. The path provided should not end with a file separator (e.g., "/" or "\"), otherwise saving will fail.
 * 
 * @author Afonso Mendes
 *
 */

// -------------------------- //
// ---- Define variables ---- //
// -------------------------- //

input = "input_gauss15_otsu_skeletonize.tif";
blockStack = "block_stack.tif";
outputDir = File.separator + "Users/ammendes/Library/CloudStorage/OneDrive-igc.gulbenkian.pt/Mendes_2023/Data_Leterrier_spectrin/Analysis/CTRL/1/sred_orientations";
 

// ---------------------------------------- //
// ---- Get number of reference blocks ---- //
// ---------------------------------------- //

selectWindow(blockStack);
n = nSlices;


// ----------------------- //
// ---- Define angles ---- //
// ----------------------- //

angles=newArray(n);
angle=90; // If first block is vertical
angle_step = 180/n;

for(i=0; i<n; i++){
	angles[i] = angle;
	angle -= angle_step;
}


// ----------------------------- //
// ---- Get repetition maps ---- //
// ----------------------------- //

for (i=0; i<n; i++) { 
    print(i+1 + "/" + n);
    
    // Open file
    selectWindow(blockStack);
    setSlice(i+1);
    
    // Run SReD
    run("Find block repetition (2D)", "block=" + blockStack + " image=" + input + " filter=1.000000000");
	title = "Block Redundancy Map";
	
	// Save
	selectWindow(title);
	saveAs("TIFF", outputDir + File.separator + "sred_orientations_"+angles[i]+".tif");
	close("sred_orientations_"+angles[i]+".tif");
}

print("Done");

