package regressor;

import org.jblas.FloatMatrix;
import org.jblas.Solve;

public class KernelPLSRegressor implements Regressor {
	
	float[][] x;
	KernelMatrixBuilder kernelBuilder;
	float reg;
	FloatMatrix alpha;
	FloatMatrix Kcolsum;
	float Ksum;
	int k;
	float tol;
	
	public KernelPLSRegressor(KernelMatrixBuilder kernelBuilder, int k, float tol) {
		this.k = k;
		this.tol = tol;
		this.kernelBuilder = kernelBuilder;
	}
	
	public void train(float[][] x, float[][] y) {
		this.x = x;
		FloatMatrix K = new FloatMatrix(kernelBuilder.build(x, x));
		this.Kcolsum = K.rowSums();
		this.Ksum = Kcolsum.sum();
		K.addiColumnVector(Kcolsum.mul(-1.0f/x.length));
		K.addiRowVector(Kcolsum.transpose().mul(-1.0f/x.length));
		K.addi(Ksum/(x.length*x.length));
		
		FloatMatrix Kj = K.dup();
		FloatMatrix Y = new FloatMatrix(y);
		FloatMatrix Yj = Y.dup();
		
		FloatMatrix B = FloatMatrix.zeros(y.length, k);
		FloatMatrix T = FloatMatrix.zeros(x.length, k);
		
		for (int j=0; j<k; ++j) {
			FloatMatrix bj = Yj.getColumn(0);
			bj.divi(bj.norm2());
			
			if (Y.columns > 1) {
				FloatMatrix YYK = Yj.mmul(Yj.transpose().mmul(Kj));
				FloatMatrix bjprev = null;
				while (bjprev == null || bjprev.distance2(bj) > tol) {
					bjprev = bj;
					bj = YYK.mmul(bj);
					bj.divi(bj.norm2());
				}
			}
			
			FloatMatrix tj = Kj.mmul(bj);
			float tjsqrnorm = tj.norm2();
			tjsqrnorm = tjsqrnorm*tjsqrnorm;
			FloatMatrix cj = Yj.transpose().mmul(tj).divi(tjsqrnorm);
			Yj.subi(tj.mmul(cj.transpose()));
			FloatMatrix Itt = tj.mmul(tj.transpose()).divi(-tjsqrnorm).addi(FloatMatrix.eye(x.length));
			Kj = Itt.mmul(Kj).mmul(Itt);
			
			B.putColumn(j, bj);
			T.putColumn(j, tj);
		}
		
		FloatMatrix TKB = T.transpose().mmul(K).mmul(B);
		FloatMatrix TY = T.transpose().mmul(Y);
		alpha = B.mmul(Solve.solve(TKB, TY)).transpose();
	}

	public float[][] predict(float[][] xin) {
		FloatMatrix k = new FloatMatrix(kernelBuilder.build(x, xin));
		k.addiColumnVector(Kcolsum.mul(-1.0f/x.length));
		k.addiRowVector(k.columnSums().mul(-1.0f/x.length));
		k.addi(Ksum/(x.length*x.length));
		
		return alpha.mmul(k).transpose().toArray2();
	}


}
