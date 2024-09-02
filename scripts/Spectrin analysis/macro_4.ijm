/*
 * // --------------------------------- //
 * // ---- IMPORTANT - PLEASE READ ---- //
 * // --------------------------------- //
 * 
 * This macro rotates the repetition maps back to the original orientation of the input image.
 * 
 * The repetition maps' directory must be defined in the "inputDir" variable and should not end with a file separator (e.g., "/" or "\").
 * 
 * The original non-padded non-rotated image must be open - the "input" variable should be named accordingly (e.g., "input.tif").
 * 
 * The path where the macro stores the results must be defined in the "outputDir" variables. The path provided should not end with a file separator (e.g., "/" or "\"), otherwise saving will fail.
 * 
 * The uncropped result will be stored in the "outputDir_raw" path. The cropped result will be stored in the "outputDir_cropped" path.
 * 
 * @author Afonso Mendes
 *
 */


// -------------------------- //
// ---- Define variables ---- //
// -------------------------- //
input = "input_normalize.tif";
inputDir = File.separator + "Users/ammendes/Library/CloudStorage/OneDrive-igc.gulbenkian.pt/Mendes_2023/Data_Leterrier_spectrin/Analysis/SWIN/6/sred_rings";
outputDir_raw = File.separator +"Users/ammendes/Library/CloudStorage/OneDrive-igc.gulbenkian.pt/Mendes_2023/Data_Leterrier_spectrin/Analysis/SWIN/6/sred_rings_derotate";
outputDir_cropped = File.separator + "Users/ammendes/Library/CloudStorage/OneDrive-igc.gulbenkian.pt/Mendes_2023/Data_Leterrier_spectrin/Analysis/SWIN/6/sred_rings_derotate_crop";


// ----------------------- //
// ---- Get file list ---- //
// ----------------------- //

filelist = getFileList(inputDir);


// ----------------------- //
// ---- Define angles ---- //
// ----------------------- //

//Array.show(filelist);
n = filelist.length; // number of angles
angles = newArray(-10, -20, -30, -40, -50, -60, -70, -80, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90); // Make sure the order of the angles here matches the order at which the images are shown in the filelist

// Invert the angles (so that we rotate back to the original)
for(i=0; i<n; i++){
	angles[i] = angles[i]*(-1);
}


// ------------------------------------------ //
// ---- Get dimensions of original input ---- //
// ------------------------------------------ //
selectWindow(input);
ogW = getWidth();
ogH = getHeight();


// ------------------------ //
// ---- Process images ---- //
// ------------------------ //

for(i=0; i<n; i++){
	// Open image
	open(inputDir + File.separator + filelist[i]);
	
	// Rotate and save
	run("Rotate... ", "angle="+ angles[i] +" grid=1 interpolation=Bicubic");
	saveAs("TIFF", outputDir_raw + File.separator + filelist[i]);
	
	// Crop and save separately
	run("Canvas Size...", "width="+ ogW +" height="+ ogH +" position=Center");
	saveAs("TIFF", outputDir_cropped + File.separator + filelist[i]);
	close(filelist[i]);
}

print("Done");
