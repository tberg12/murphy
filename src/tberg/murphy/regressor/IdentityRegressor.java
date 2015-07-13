package tberg.murphy.regressor;

import tberg.murphy.arrays.a;

public class IdentityRegressor implements Regressor {
	public void train(float[][] x, float[][] y) {
	}
	public float[][] predict(float[][] x) {
		return a.copy(x);
	}
}
