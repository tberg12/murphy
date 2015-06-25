package regressor;

import java.util.Arrays;

public class ConstantRegressor implements Regressor {

	float c;
	
	public ConstantRegressor(float c) {
		this.c = c;
	}
	
	public void train(float[][] x, float[][] y) {
	}

	public float[][] predict(float[][] x) {
		float[][] result = new float[x.length][x[0].length];
		for (float[] vect : result) Arrays.fill(vect, c);
		return result;
	}

}
