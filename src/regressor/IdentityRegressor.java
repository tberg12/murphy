package regressor;

import arrays.a;

public class IdentityRegressor implements Regressor {
	public void train(float[][] x, float[][] y) {
	}
	public float[][] predict(float[][] x) {
		return a.copy(x);
	}
}
