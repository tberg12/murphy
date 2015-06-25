package regressor;

import org.jblas.FloatMatrix;
import org.jblas.Solve;

import arrays.a;

public class PLSRegressor implements Regressor {
	
	float[][] x;
	float[][] y;
	float[] xMean;
	int k;
	float tol;
	FloatMatrix W;
	
	public PLSRegressor(int k, float tol) {
		this.k = k;
		this.tol = tol;
	}
	
	public void train(float[][] xraw, float[][] yraw) {
		this.x = new float[xraw.length][];
		for (int i=0; i<xraw.length; ++i) {
			this.x[i] = a.append(1.0f, xraw[i]);
		}
		this.y = yraw;
		this.xMean = a.scale(a.sum(a.transpose(x)), 1.0f / x.length);
		for (float[] vect : x) a.combi(vect, 1.0f, xMean, -1.0f);
		
		FloatMatrix U = FloatMatrix.zeros(x[0].length, k);
		FloatMatrix P = FloatMatrix.zeros(x[0].length, k);
		FloatMatrix C = FloatMatrix.zeros(y[0].length, k);
		
		FloatMatrix Xj = new FloatMatrix(x);
		FloatMatrix Y = new FloatMatrix(y);
		for (int j=0; j<k; ++j) {
			FloatMatrix YX = Y.transpose().mmul(Xj);
			FloatMatrix XY = YX.transpose();

			FloatMatrix uj = XY.getColumn(0);
			uj.divi(uj.norm2());
			
			if (Y.columns > 1) {
				FloatMatrix ujprev = null;
				while (ujprev == null || ujprev.distance2(uj) > tol) {
					ujprev = uj;
					uj = YX.mmul(uj);
					uj = YX.transpose().mmul(uj);
					uj.divi(uj.norm2());
				}
			}
			
			FloatMatrix t = Xj.mmul(uj);
			FloatMatrix tt = t.transpose().mmul(t);
			FloatMatrix cj = Y.transpose().mmul(t).div(tt);
			FloatMatrix pj = Xj.transpose().mmul(t).div(tt);
			
			U.putColumn(j, uj);
			C.putColumn(j, cj);
			P.putColumn(j, pj);
			
			Xj.subi(t.mmul(pj.transpose()));
		}
		
		this.W = U.mmul(Solve.solve(P.transpose().mmul(U), C.transpose()));
	}

	public float[][] predict(float[][] xinraw) {
		float[][] xin = new float[xinraw.length][];
		for (int i=0; i<xinraw.length; ++i) {
			xin[i] = a.append(1.0f, xinraw[i]);
		}
		for (float[] vect : xin) a.combi(vect, 1.0f, xMean, -1.0f);
		
		FloatMatrix Xin = new FloatMatrix(xin);
		return Xin.mmul(W).toArray2();
	}

}
