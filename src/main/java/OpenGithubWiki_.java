import ij.IJ;
import ij.plugin.PlugIn;
import java.awt.Desktop;
import java.net.URI;

public class OpenGithubWiki_ implements PlugIn {
    @Override
    public void run(String arg) {
        try {
            // Specify the URL
            String url = "https://github.com/HenriquesLab/SReD/wiki";
            Desktop desktop = Desktop.getDesktop();
            desktop.browse(new URI(url));
        } catch (Exception e) {
            // Handle any exceptions (e.g., if the browser cannot be opened)
            IJ.error("Could not open browser. Please head to https://github.com/HenriquesLab/SReD/wiki");
        }
    }
}
