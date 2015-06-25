package regressor;

import arrays.a;

public class ScaleAndShiftRegressor implements Regressor {

	float[] xMean;
	float[] yMean;
	float[] xStd;
	float[] yStd;
	
	public void train(float[][] x, float[][] y) {
		xMean = rowMean(x);
		yMean = rowMean(y);
		xStd = rowStd(x);
		yStd = rowStd(y);
	}

	public float[][] predict(float[][] x) {
		float[][] result = new float[x.length][];
		for (int t=0; t<result.length; ++t) {
			result[t] = a.comb(a.pointwiseMult(a.pointwiseDiv(yStd, xStd), a.comb(x[t], 1.0f, xMean, -1.0f)), 1.0f, yMean, 1.0f);
		}
		return result;
	}
	
	private static float[] rowMean(float[][] x) {
		float[] rowMean = new float[x[0].length];
		for (int i=0; i<x.length; ++i) {
			a.combi(rowMean, 1.0f, x[i], 1.0f / x.length);
		}
		return rowMean;
	}
	
	private static float[] rowStd(float[][] x) {
		float[] rowMean = rowMean(x);
		float[] std = new float[rowMean.length];
		for (float[] r : x) {
			float[] diff = a.comb(r, 1.0f, rowMean, -1.0f);
			a.sqri(diff);
			a.combi(std, 1.0f, diff, 1.0f / x.length);
		}
		a.sqrti(std);
		return std;
	}

}
