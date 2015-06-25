package regressor;

import gpu.CublasUtil;
import gpu.CublasUtil.Matrix;

import org.jblas.FloatMatrix;
import org.jblas.Solve;

public class GPUKernelPLSRegressor implements Regressor {
	
	float[][] x;
	KernelMatrixBuilder kernelBuilder;
	float reg;
	Matrix alpha;
	Matrix Kcolsum;
	float Ksum;
	int k;
	float tol;
	
	public GPUKernelPLSRegressor(KernelMatrixBuilder kernelBuilder, int k, float tol) {
		this.k = k;
		this.tol = tol;
		this.kernelBuilder = kernelBuilder;
	}
	
	public void train(float[][] x, float[][] y) {
		this.x = x;
		Matrix K = Matrix.build(kernelBuilder.build(x, x));
		
		CublasUtil.freeAllBut(K);
		
		this.Kcolsum = K.colSum();
		this.Kcolsum.setDontFree(true);
		this.Ksum = Kcolsum.rowSum().toArray()[0];
		K.colAddi(Kcolsum.mul(-1.0f/x.length));
		K.rowAddi(Kcolsum.transpose().mul(-1.0f/x.length));
		K.addi(Ksum/(x.length*x.length));
		
		CublasUtil.freeAllBut(K);

		Matrix Kj = K.copy();
		Matrix Y = Matrix.build(y);
		Matrix Yj = Y.copy();
		
		Matrix B = Matrix.zeros(y.length, k);
		Matrix T = Matrix.zeros(x.length, k);
		
		for (int j=0; j<k; ++j) {
			Matrix bj = Yj.copyCol(0);
			bj.muli(1.0f / bj.norm2());
			
			if (Y.cols() > 1) {
				Matrix YYK = Yj.mmul(Yj.transpose().mmul(Kj));
				
				CublasUtil.freeAllBut(bj, YYK, Kj, K, Yj, Y, B, T);
				
				Matrix bjprev = null;
				while (bjprev == null || bjprev.distance2(bj) > tol) {
					bjprev = bj;
					bj = YYK.mmul(bj);
					bj.muli(1.0f / bj.norm2());
				}
				
				CublasUtil.freeAllBut(bj, Kj, K, Yj, Y, B, T);
				
			}
			
			B.setCol(j, bj);
			Matrix tj = Kj.mmul(bj);
			T.setCol(j, tj);

			float tjsqrnorm = tj.norm2();
			tjsqrnorm = tjsqrnorm*tjsqrnorm;
			Matrix cj = Yj.transpose().mmul(tj).muli(1.0f / tjsqrnorm);
			Yj.subi(tj.mmul(cj.transpose()));
			
			CublasUtil.freeAllBut(tj, Kj, K, Yj, Y, B, T);
			
//			Matrix Itt = tj.mmul(tj.transpose()).muli(-1.0f / tjsqrnorm).diagAddi(Matrix.ones(x.length,1));
			Matrix Itt = tj.mmul(tj.transpose()).muli(-1.0f / tjsqrnorm).addi(Matrix.eye(x.length));

			CublasUtil.freeAllBut(Itt, Kj, K, Yj, Y, B, T);
			
			
			Kj = Itt.mmul(Kj).mmul(Itt);
			
			
			CublasUtil.freeAllBut(Kj, K, Yj, Y, B, T);
		}
		
		Matrix TKB = T.transpose().mmul(K).mmul(B);
		Matrix TY = T.transpose().mmul(Y);
		this.alpha = B.mmul(Matrix.build(Solve.solve(new FloatMatrix(TKB.toArray2()), new FloatMatrix(TY.toArray2())).toArray2())).transpose();
		this.alpha.setDontFree(true);
		
		CublasUtil.freeAll();
	}

	public float[][] predict(float[][] xin) {
		Matrix k = Matrix.build(kernelBuilder.build(x, xin));
		k.colAddi(Kcolsum.mul(-1.0f/x.length));
		k.rowAddi(k.rowSum().mul(-1.0f/x.length));
		k.addi(Ksum/(x.length*x.length));
		float[][] result = alpha.mmul(k).transpose().toArray2();
		CublasUtil.freeAll();
		return result;
	}


}
