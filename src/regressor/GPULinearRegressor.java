package regressor;

import java.util.ArrayList;
import java.util.List;

import gpu.CublasUtil;
import gpu.CublasUtil.Matrix;

import org.jblas.FloatMatrix;
import org.jblas.Solve;

import arrays.a;

public class GPULinearRegressor implements Regressor {

	float reg;
	Matrix weights;
	
	public GPULinearRegressor(float reg) {
		this.reg = reg;
	}
	
	public void train(float[][] xraw, float[][] yraw) {
		float[][] x = new float[xraw.length][];
		for (int i=0; i<xraw.length; ++i) {
			x[i] = a.append(1.0f, xraw[i]);
		}
		float[][] y = yraw;
		
		Matrix yMat = Matrix.build(y);
		Matrix xMat = Matrix.build(x);
		Matrix xTrMat = xMat.transpose();
		
		List<Matrix> A = new ArrayList<Matrix>();
		A.add(xTrMat.mmul(xMat).diagAdd(reg));
		List<Matrix> B = new ArrayList<Matrix>();
		B.add(xTrMat.mmul(yMat));
		
		this.weights = Matrix.build(Solve.solvePositive(new FloatMatrix(xTrMat.mmul(xMat).diagAdd(reg).toArray2()), new FloatMatrix(xTrMat.mmul(yMat).toArray2())).toArray2());
		this.weights.setDontFree(true);
		
		CublasUtil.freeAll();
	}
	
	public float[][] predict(final float[][] xinraw) {
		float[][] xin = new float[xinraw.length][];
		for (int i=0; i<xinraw.length; ++i) {
			xin[i] = a.append(1.0f, xinraw[i]);
		}
		Matrix xinMat = Matrix.build(xin);
		float[][] result = xinMat.mmul(weights).toArray2();
		CublasUtil.freeAll();
		return result;
	}
	
}
