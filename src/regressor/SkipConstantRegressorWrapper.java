package regressor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SkipConstantRegressorWrapper implements Regressor {

	Regressor regressor;
	int sdim;
	int edim;
	int outDim;
	float c;
	
	public SkipConstantRegressorWrapper(Regressor regressor, int sdim, int edim, float c) {
		this.regressor = regressor;
		this.sdim = sdim;
		this.edim = edim;
		this.c = c;
	}
	
	public void train(float[][] xWithZeros, float[][] yWithZeros) {
		this.outDim = yWithZeros[0].length; 
		List<float[]> xList = new ArrayList<float[]>();
		List<float[]> yList = new ArrayList<float[]>();
		for (int i=0; i<xWithZeros.length; ++i) {
			float[] xr = xWithZeros[i];
			float[] yr = yWithZeros[i];
			if (!allConstant(xr, sdim, edim, c) && !allConstant(yr, sdim, edim, c)) {
				xList.add(xr);
				yList.add(yr);
			}
		}
		float[][] x = new float[xList.size()][];
		for (int i=0; i<xList.size(); ++i) {
			x[i] = xList.get(i);
		}
		float[][] y = new float[yList.size()][];
		for (int i=0; i<yList.size(); ++i) {
			y[i] = yList.get(i);
		}
		this.regressor.train(x, y);
	}
	
	public float[][] predict(float[][] x) {
		float[][] result = regressor.predict(x);
		for (int t=0; t<result.length; ++t) {
			if (allConstant(x[t], sdim, edim, c)) {
				float[] y = new float[outDim];
				Arrays.fill(y, c);
				result[t] = y;
			}
		}
		return result;
	}
	
	private static boolean allConstant(float[] x, int sdim, int edim, float c) {
		for (int i=sdim; i<edim; ++i) {
			if (x[i] != c) return false;
		}
		return true;
	}
	
}
