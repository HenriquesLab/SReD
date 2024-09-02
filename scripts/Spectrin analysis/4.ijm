/*
 * // --------------------------------- //
 * // ---- IMPORTANT - PLEASE READ ---- //
 * // --------------------------------- //
 * 
 * This macro generates repetition maps for multiple input images using a predefined reference block.
 * 
 * The input images' directory must be defined and should not end with a file separator (e.g., "/" or "\").
 * 
 * The reference blocks' stack should be open in ImageJ before running - the "blockStack" variable must be defined using the name of the file and its extension (e.g., "blocks.tif").
 * 
 * The path where the macro stores the results must be defined in the "outputDir" variable. The path provided should not end with a file separator (e.g., "/" or "\"), otherwise saving will fail.
 * 
 * @author Afonso Mendes
 *
 */


// --------------------------------------- //
// ---- Define paths and get filelist ---- //
// --------------------------------------- //

inputDir = File.separator + "Users/ammendes/Library/CloudStorage/OneDrive-igc.gulbenkian.pt/Mendes_2023/Data_Leterrier_spectrin/Analysis/CTRL/6/inputs_padded_rotated";
block = "block.tif";
outputDir = File.separator + "Users/ammendes/Library/CloudStorage/OneDrive-igc.gulbenkian.pt/Mendes_2023/Data_Leterrier_spectrin/Analysis/CTRL/6/sred_rings";


// ----------------------------------- //
// ---- Calculate repetition maps ---- //
// ----------------------------------- //

filelist = getFileList(inputDir);
for(i=0; i<filelist.length; i++){
	// Open input
	open(inputDir + File.separator + filelist[i]);
	tempTitle1=getTitle();
	
	// Run SReD
	run("Find block repetition (2D)", "patch=" + block + " image=" + filelist[i] + " filter=1.0");
	tempTitle2 = "Block Redundancy Map";
	
	// save and close
	saveAs("TIFF", outputDir + File.separator + "sred_"+filelist[i]);
	close(tempTitle1);
	close("sred_"+filelist[i]);
}

print("done");