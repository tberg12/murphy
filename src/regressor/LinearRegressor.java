package regressor;

import org.jblas.FloatMatrix;
import org.jblas.Solve;

import arrays.a;

public class LinearRegressor implements Regressor {

	float reg;
	FloatMatrix weights;
	
	public LinearRegressor(float reg) {
		this.reg = reg;
	}
	
	public void train(float[][] xraw, float[][] yraw) {
		float[][] x = new float[xraw.length][];
		for (int i=0; i<xraw.length; ++i) {
			x[i] = a.append(1.0f, xraw[i]);
		}
		float[][] y = yraw;
		
		FloatMatrix yMat = new FloatMatrix(y);
		FloatMatrix xMat = new FloatMatrix(x);
		FloatMatrix xTrMat = xMat.transpose();
		
		weights = Solve.solvePositive(xTrMat.mmul(xMat).add(FloatMatrix.eye(x[0].length).mmul(reg)), xTrMat.mmul(yMat));
	}
	
	public float[][] predict(final float[][] xinraw) {
		float[][] xin = new float[xinraw.length][];
		for (int i=0; i<xinraw.length; ++i) {
			xin[i] = a.append(1.0f, xinraw[i]);
		}
		FloatMatrix xinMat = new FloatMatrix(xin);
		return xinMat.mmul(weights).toArray2();
	}
	
}
