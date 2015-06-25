package regressor;

import gpu.CublasUtil;
import gpu.CublasUtil.Matrix;

public class GPURBFKernelMatrixBuilder implements KernelMatrixBuilder {
	
	float var;
	
	public GPURBFKernelMatrixBuilder(float var) {
		this.var = var;
	}

	public float[][] build(float[][] x, float[][] y) {
		Matrix X = Matrix.build(x);
		Matrix Y = Matrix.build(y);
		Matrix K = X.mmul(Y.transpose());
		K.muli(-2.0f);
		K.colAddi(X.sqr().colSum());
		K.rowAddi(Y.sqr().colSum());
		K.muli(-0.5f / var);
		K.expi();
		float[][] result = K.toArray2();
		CublasUtil.freeAll();
		return result;
	}

}
