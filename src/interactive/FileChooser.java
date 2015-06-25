package interactive;

import javax.swing.JFileChooser;

public class FileChooser {
	
	public static void main(String[] args) {
		
		JFileChooser f = new JFileChooser();
		f.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES); 
		f.showOpenDialog(null);

		System.out.println(f.getCurrentDirectory());
		System.out.println(f.getSelectedFile());
		
	}
	
}