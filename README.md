# SReD: Structural Repetition Detector Plugin for ImageJ and Fiji

<img src="https://github.com/HenriquesLab/SReD/blob/main/Docs/Logo/logo_small.png" align="right" width="200"/>

SReD (Structural Repetition Detector) is an ImageJ and Fiji plugin for analyzing structural repetition in microscopy images. It introduces a novel approach utilising a custom sampling scheme and correlation metrics to compute Repetition Maps. These maps highlight and quantify the degree of structural repetition present within images, providing valuable insights for image analysis and structural studies.
| Starter's Guide |
|:-:|
| [![](https://github.com/user-attachments/assets/abec46ad-4723-4f04-bf5a-f34bc6030838)](https://youtu.be/85coxW4H7f4) | 

<img width="1330" alt="image" src="https://github.com/user-attachments/assets/5e0e84c5-4d12-4304-83ec-0f9b09f95634">

## Installation

### Installing from GitHub releases

1. Ensure you have ImageJ or Fiji installed on your system. If not, download and install it from:
   - ImageJ: https://imagej.nih.gov/ij/download.html
   - Fiji: https://fiji.sc/

2. Download the latest SReD_.jar file from the [releases page](https://github.com/HenriquesLab/SReD/releases).

3. Place the SReD_.jar file in the "plugins" folder of your ImageJ/Fiji installation:
   - For ImageJ: `ImageJ/plugins/`
   - For Fiji: `Fiji.app/plugins/`

4. Restart ImageJ/Fiji if it is already running.

5. The SReD plugin should now appear in the Plugins menu of ImageJ/Fiji.

### Installing from ImageJ Update Site

1. Ensure you have ImageJ or Fiji installed on your system. If not, download and install it from:
   - ImageJ: https://imagej.nih.gov/ij/download.html
   - Fiji: https://fiji.sc/

2. Click on Help > Update... > Manage update sites

3. Tick the "SReD_Update_Site" box

4. Click on Apply Changes and restart ImageJ/Fiji

## System Requirements

- Java 8 or higher
- OpenCL-capable GPU (recommended for faster processing)

## Usage

After installation, you can access SReD functions from the Plugins > SReD menu in ImageJ/Fiji. 

For detailed usage instructions, please refer to the [user guide](link_to_user_guide).

## Troubleshooting

If you encounter any issues during installation or usage, please check the [FAQ section](link_to_FAQ) or open an issue on our [GitHub page](https://github.com/HenriquesLab/SReD/issues).

## License

SReD is distributed under the [MIT](LICENSE).

## Citation

If you use SReD in your research, please cite our paper:

[Citation details]

