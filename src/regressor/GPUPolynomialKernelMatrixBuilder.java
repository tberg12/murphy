package regressor;

import gpu.CublasUtil;
import gpu.CublasUtil.Matrix;

public class GPUPolynomialKernelMatrixBuilder implements KernelMatrixBuilder {
	
	float pow;
	float c;
	
	public GPUPolynomialKernelMatrixBuilder(float pow, float c) {
		this.pow = pow;
		this.c = c;
	}

	public float[][] build(float[][] x, float[][] y) {
		Matrix X = Matrix.build(x);
		Matrix Y = Matrix.build(y);
		Matrix K = X.mmul(Y.transpose());
		if (c != 0.0f) K.addi(c);
		if (pow == 2.0f)
			K.sqri();
		else if (pow != 0.0f)
			K.powi(pow);
		float[][] result = K.toArray2();
		CublasUtil.freeAll();
		return result;
	}

}
