package regressor;

import gpu.CublasUtil;

import org.jblas.FloatMatrix;
import org.jblas.Solve;

public class KernelLinearRegressor implements Regressor {

	float[][] x;
	KernelMatrixBuilder kernelBuilder;
	float reg;
	FloatMatrix alpha;
	FloatMatrix Kcolsum;
	float Ksum;
	
	public KernelLinearRegressor(KernelMatrixBuilder kernelBuilder, float reg) {
		this.reg = reg;
		this.kernelBuilder = kernelBuilder;
	}
	
	public void train(float[][] x, float[][] y) {
		this.x = x;
		FloatMatrix K = new FloatMatrix(kernelBuilder.build(x, x));
		CublasUtil.freeAll();
		this.Kcolsum = K.rowSums();
		this.Ksum = Kcolsum.sum();
		K.addiColumnVector(Kcolsum.mul(-1.0f/x.length));
		K.addiRowVector(Kcolsum.transpose().mul(-1.0f/x.length));
		K.addi(Ksum/(x.length*x.length));
		
		K.addi(FloatMatrix.eye(x.length).muli(reg));
		alpha = Solve.solvePositive(K, new FloatMatrix(y)).transpose();
	}

	public float[][] predict(float[][] xin) {
		FloatMatrix k = new FloatMatrix(kernelBuilder.build(x, xin));
		CublasUtil.freeAll();
		k.addiColumnVector(Kcolsum.mul(-1.0f/x.length));
		k.addiRowVector(k.columnSums().mul(-1.0f/x.length));
		k.addi(Ksum/(x.length*x.length));
		
		return alpha.mmul(k).transpose().toArray2();
	}

}
