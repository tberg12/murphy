package regressor;

import gpu.CublasUtil;
import gpu.CublasUtil.Matrix;

public class GPULinearRBFKernelMatrixBuilder implements KernelMatrixBuilder {
	
	float var;
	float a;
	float b;
	float c;
	
	public GPULinearRBFKernelMatrixBuilder(float var, float a, float b, float c) {
		this.var = var;
		this.a = a;
		this.b = b;
		this.c = c;
	}

	public float[][] build(float[][] x, float[][] y) {
		Matrix X = Matrix.build(x);
		Matrix Y = Matrix.build(y);
		Matrix K = X.mmul(Y.transpose());
		K.muli(-2.0f);
		K.colAddi(X.sqr().colSum());
		K.rowAddi(Y.sqr().colSum());
		Matrix Klinear = K.copy();
		K.muli(-0.5f / var);
		K.expi();
		K.muli(a);
		K.addi(b);
		K.addi(Klinear.muli(c));
		float[][] result = K.toArray2();
		CublasUtil.freeAll();
		return result;
	}

}
