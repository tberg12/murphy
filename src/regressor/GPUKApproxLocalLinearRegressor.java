package regressor;

import gpu.CublasUtil;
import gpu.CublasUtil.Matrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.jblas.FloatMatrix;
import org.jblas.Singular;

import arrays.a;

public class GPUKApproxLocalLinearRegressor implements Regressor {
	
	public static final double MAX_BATCH_ELEMENTS = 2.5e9;
//	public static final double MAX_BATCH_ELEMENTS = 1.2e9;
	public static final boolean LOWER_MEM_FOOTPRINT = true;
	
	public static final int K = 2;

	float reg;
	float std;
	int numNeighbors;
	
	float[][] xOrig;
	float[][] yOrig;
	float[] xMean;
	float[][][] x;
	float[][][] y;
	float[][] projDirection;
	float[][] proj;
	
	Matrix[] x_d;
	Matrix[] xSqrNorms_d;
	Matrix[] y_d;
	Matrix xMean_d;
	Matrix[] projDirection_d;
	Matrix[] proj_d;
	
	public GPUKApproxLocalLinearRegressor(float reg, float std, int numNeighbors) {
		this.reg = reg;
		this.std = std;
		this.numNeighbors = numNeighbors;
	}
	
	public void train(float[][] xraw, float[][] yraw) {
		this.xOrig = new float[xraw.length][];
		for (int i=0; i<xraw.length; ++i) {
			this.xOrig[i] = a.append(1.0f, xraw[i]);
		}
		this.yOrig = yraw;
		
		long start = System.nanoTime();
		this.xMean = a.scale(a.sum(a.transpose(xOrig)), 1.0f / xOrig.length);
		FloatMatrix meanMat = new FloatMatrix(xMean);
		FloatMatrix XMat = new FloatMatrix(xOrig);
		FloatMatrix XMatSubMean = XMat.subRowVector(meanMat);
		FloatMatrix covXMat = XMatSubMean.transpose().mmul(XMatSubMean);
		FloatMatrix WTransMat = Singular.fullSVD(covXMat)[0].transpose();
		System.out.println("PCA time: "+(System.nanoTime() - start) / 1e6);

		start = System.nanoTime();
		this.projDirection = Arrays.copyOfRange(WTransMat.toArray2(), 0, K);
		this.proj = new float[K][];
		this.x = new float[K][][];
		this.y = new float[K][][];
		for (int k=0; k<K; ++k) {
			final int kfinal = k;
			this.proj[k] = new float[xOrig.length];
			for (int i=0; i<xOrig.length; ++i) {
				proj[k][i] = a.innerProd(projDirection[k], a.comb(xOrig[i], 1.0f, xMean, -1.0f));
			}
			Integer[] indices = new Integer[xOrig.length];
			for (int i=0; i<xOrig.length; ++i) indices[i] = i;
			Arrays.sort(indices, new Comparator<Integer>() {
				public int compare(Integer i1, Integer i2) {
					if (proj[kfinal][i1] < proj[kfinal][i2]) {
						return -1;
					} else if (proj[kfinal][i1] > proj[kfinal][i2]) {
						return 1;
					} else {
						return 0;
					}
				}
			});
			float[] projOld = proj[k];
			x[k] = new float[xOrig.length][];
			y[k] = new float[yOrig.length][];
			proj[k] = new float[projOld.length];
			for (int i=0; i<xOrig.length; ++i) {
				x[k][i] = xOrig[indices[i]];
				y[k][i] = yOrig[indices[i]];
				proj[k][i] = projOld[indices[i]];
			}
		}
		System.out.println("Project / sort time: "+(System.nanoTime() - start) / 1e6);
		
		this.xMean_d = Matrix.build(1, xMean.length, xMean);
		this.xMean_d.setDontFree(true);
		this.x_d = new Matrix[K];
		this.y_d = new Matrix[K];
		this.xSqrNorms_d = new Matrix[K];
		this.projDirection_d = new Matrix[K];
		this.proj_d = new Matrix[K];
		for (int k=0; k<K; ++k) {
			this.x_d[k] = Matrix.build(x[k]);
			this.x_d[k].setDontFree(true);
			this.y_d[k] = Matrix.build(y[k]);
			this.y_d[k].setDontFree(true);
			this.xSqrNorms_d[k] = x_d[k].sqr().colSum();
			this.xSqrNorms_d[k].setDontFree(true);
			this.projDirection_d[k] = Matrix.build(projDirection[k].length, 1, projDirection[k]);
			this.projDirection_d[k].setDontFree(true);
			this.proj_d[k] = Matrix.build(proj[k].length, 1, proj[k]);
			this.proj_d[k].setDontFree(true);
		}
	}
	
	private static int binarySearch(float[] sorted, float val) {
		int start = 0;
		int end = sorted.length;
		while (end > start+1) {
			int mid = (end+start)/2;
			if (val > sorted[mid]) {
				start = mid;
			} else {
				end = mid;
			}
		}
		return end;
	}
	
	public float[][] predict(float[][] xinraw) {
		System.out.println("predicting...");
		long outerStart = System.nanoTime();
		float[][] xin = new float[xinraw.length][];
		for (int i=0; i<xinraw.length; ++i) {
			xin[i] = a.append(1.0f, xinraw[i]);
		}
		int batchSize = (int) (MAX_BATCH_ELEMENTS / ((double) numNeighbors * x[0].length));
		int numBatches = (int) Math.ceil((double) xin.length / batchSize);
		List<float[]> resultList = new ArrayList<float[]>();
		for (int b=0; b<numBatches; ++b) {
			float[][] batch = predictBatch(Arrays.copyOfRange(xin, b*batchSize, Math.min(xin.length, (b+1)*batchSize)));
			for (float[] vect : batch) resultList.add(vect);
		}
		float[][] result = new float[xin.length][];
		for (int i=0; i<xin.length; ++i) result[i] = resultList.get(i);
		System.out.println("total time seconds: "+(System.nanoTime() - outerStart) / 1e9);
		return result;
	}
	
	public float[][] predictBatch(float[][] xin) {
		int numNeighbors = Math.min(this.numNeighbors, x.length);
		
		int xdim = xOrig[0].length;
		int ydim = yOrig[0].length;
		
		long start = System.nanoTime();
		Matrix xin_d = Matrix.build(xin);
		System.out.println("copy time: "+(System.nanoTime() - start)/1e9);
		
		start = System.nanoTime();
		Matrix xinSqrNorms_d = xin_d.sqr().colSum();
		System.out.println("sqr norms time: "+(System.nanoTime() - start)/1e9);
		
		start = System.nanoTime();
		List<Matrix> Blist_d = new ArrayList<Matrix>();
		List<Matrix> yNeighborslist_d = new ArrayList<Matrix>();
		List<Matrix> xinSinglelist_d = new ArrayList<Matrix>();
		List<Matrix> wlist_d = new ArrayList<Matrix>();
		for (int f=0; f<xin.length; ++f) {
			xinSinglelist_d.add(xin_d.copySubmatrix(f, f+1, 0, xdim));
			Blist_d.add(new Matrix(numNeighbors, xdim));
			yNeighborslist_d.add(new Matrix(numNeighbors, ydim));
			wlist_d.add(new Matrix(numNeighbors, 1));
		}
		System.out.println("allocate time: "+(System.nanoTime() - start)/1e9);
		
		for (int k=0; k<K; ++k) {
			start = System.nanoTime();
			Matrix xinProj_d = xin_d.rowSub(xMean_d).mmul(projDirection_d[k]);
			float[] xinProj = xinProj_d.toArray();
			System.out.println("proj time: "+(System.nanoTime() - start)/1e9);

			start = System.nanoTime();
			Matrix sqrDists_d = x_d[k].mmul(xin_d.transpose());
			sqrDists_d.muli(-2.0f);
			sqrDists_d.colAddi(xSqrNorms_d[k]);
			sqrDists_d.rowAddi(xinSqrNorms_d);
			Matrix xWeights_d = sqrDists_d.muli(-0.5f / (std*std)).expi();
			System.out.println("compute weights time: "+(System.nanoTime() - start)/1e9);
			
			start = System.nanoTime();
			for (int f=0; f<xin.length; ++f) {
				int centerIndex = binarySearch(proj[k], xinProj[f]);
				centerIndex = Math.max(numNeighbors/(2*K), centerIndex);
				centerIndex = Math.min(xOrig.length-numNeighbors/(2*K), centerIndex);

				Blist_d.get(f).setSubmatrix(k*(numNeighbors/K), 0, x_d[k], centerIndex - numNeighbors/(2*K), centerIndex + numNeighbors/(2*K), 0, xdim);
				yNeighborslist_d.get(f).setSubmatrix(k*(numNeighbors/K), 0, y_d[k], centerIndex - numNeighbors/(2*K), centerIndex + numNeighbors/(2*K), 0, ydim);
				wlist_d.get(f).setSubmatrix(k*(numNeighbors/K), 0, xWeights_d, centerIndex - numNeighbors/(2*K), centerIndex + numNeighbors/(2*K), f, f+1);
			}
			System.out.println("extract time: "+(System.nanoTime() - start)/1e9);
		}

		List<Matrix> BWTrlist_d = new ArrayList<Matrix>();
		List<Matrix> BBlist_d = new ArrayList<Matrix>();
		if (LOWER_MEM_FOOTPRINT) {
			start = System.nanoTime();
			for (int f=0; f<xin.length; ++f) {
				Matrix BWTr_d = Blist_d.get(f).transpose();
				BWTr_d.rowMuli(wlist_d.get(f));
				BBlist_d.add(BWTr_d.mmul(Blist_d.get(f)));
				BWTrlist_d.add(BWTr_d);
				Blist_d.get(f).free();
				wlist_d.get(f).free();
			}
			System.out.println("compute weights / mmul time: "+(System.nanoTime() - start)/1e9);
		} else {
			start = System.nanoTime();
			for (int f=0; f<xin.length; ++f) {
				Matrix BWTr_d = Blist_d.get(f).transpose();
				BWTr_d.rowMuli(wlist_d.get(f));
				BWTrlist_d.add(BWTr_d);
			}
			System.out.println("compute weights time: "+(System.nanoTime() - start)/1e9);

			start = System.nanoTime();
			BBlist_d = Matrix.mmul(BWTrlist_d, Blist_d);
			System.out.println("mmul time: "+(System.nanoTime() - start)/1e9);
		}
		
		start = System.nanoTime();
		for (Matrix BB_d : BBlist_d) {
			BB_d.diagAddi(reg);
		}
		System.out.println("add reg time: "+(System.nanoTime() - start)/1e9);
		
		start = System.nanoTime();
		List<Matrix> invlist_d = Matrix.invert(BBlist_d);
		System.out.println("invert time: "+(System.nanoTime() - start)/1e9);
		
		start = System.nanoTime();
		List<Matrix> resultlist_d = Matrix.mmul(xinSinglelist_d, invlist_d);
		resultlist_d = Matrix.mmul(resultlist_d, BWTrlist_d);
		resultlist_d = Matrix.mmul(resultlist_d, yNeighborslist_d);
		System.out.println("mmul time: "+(System.nanoTime() - start)/1e9);
		
		start = System.nanoTime();
		float[][] yout = new float[xin.length][];
		for (int f=0; f<xin.length; ++f) {
			yout[f] = resultlist_d.get(f).toArray();
		}
		System.out.println("copy time: "+(System.nanoTime() - start)/1e9);
		
		
//		if (a.hasinf(yout)) System.out.println("INF TROUBLE");
//		if (a.hasnan(yout)) System.out.println("NAN TROUBLE");
		
		CublasUtil.freeAll();

		return yout;
	}
		

}
