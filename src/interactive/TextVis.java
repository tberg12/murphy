package interactive;

import java.awt.Font;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

public class TextVis {
	
	public static void textPopup(String text, int height, int width) {
		textPopup("", text, height, width);
	}
	
	public static void textPopup(String name, String text, int height, int width) {
		JTextArea textArea = new JTextArea(text, height, width);
		textArea.setFont(new Font("Times New Roman", Font.PLAIN, 18));
		JPanel textPanel = new JPanel();
		textPanel.add(new JScrollPane(textArea, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED, JScrollPane.HORIZONTAL_SCROLLBAR_NEVER));
		JFrame frame = new JFrame(name);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().add(textPanel);
		frame.pack();
		frame.setLocationRelativeTo(null);
		frame.setVisible(true);
	}
	
	public static void main(String[] args) {
		
		StringBuffer text = new StringBuffer();
		for (int i=0; i<10; i++) {
			text.append("This is a text. \n");
		}
		
		textPopup("output", text.toString(), 30, 60);
		
	}
	
}
