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

public class GPUApproxLocalLinearRegressor implements Regressor {

	public static final double MAX_BATCH_ELEMENTS = 2.5e9;
//	public static final double MAX_BATCH_ELEMENTS = 1.2e9;
	public static final boolean LOWER_MEM_FOOTPRINT = true;
	
	float reg;
	float std;
	int numNeighbors;
	
	float[][] x;
	float[][] y;
	float[] xMean;
	float[] projDirection;
	float[] proj;
	
	Matrix x_d;
	Matrix xSqrNorms_d;
	Matrix y_d;
	Matrix xMean_d;
	Matrix projDirection_d;
	Matrix proj_d;
	
	public GPUApproxLocalLinearRegressor(float reg, float std, int numNeighbors) {
		this.reg = reg;
		this.std = std;
		this.numNeighbors = numNeighbors;
	}
	
	public void train(float[][] xraw, float[][] yraw) {
		this.x = new float[xraw.length][];
		for (int i=0; i<xraw.length; ++i) {
			this.x[i] = a.append(1.0f, xraw[i]);
		}
		this.y = yraw;
		
		long start = System.nanoTime();
		this.xMean = a.scale(a.sum(a.transpose(x)), 1.0f / x.length);
		FloatMatrix meanMat = new FloatMatrix(xMean);
		FloatMatrix XMat = new FloatMatrix(x);
		FloatMatrix XMatSubMean = XMat.subRowVector(meanMat);
		FloatMatrix covXMat = XMatSubMean.transpose().mmul(XMatSubMean);
		FloatMatrix WTransMat = Singular.fullSVD(covXMat)[0].transpose();
		System.out.println("PCA time: "+(System.nanoTime() - start) / 1e6);

		start = System.nanoTime();
		this.projDirection = WTransMat.toArray2()[0];
		this.proj = new float[x.length];
		for (int i=0; i<x.length; ++i) {
			proj[i] = a.innerProd(projDirection, a.comb(x[i], 1.0f, xMean, -1.0f));
		}
		Integer[] indices = new Integer[x.length];
		for (int i=0; i<x.length; ++i) indices[i] = i;
		Arrays.sort(indices, new Comparator<Integer>() {
			public int compare(Integer i1, Integer i2) {
				if (proj[i1] < proj[i2]) {
					return -1;
				} else if (proj[i1] > proj[i2]) {
					return 1;
				} else {
					return 0;
				}
			}
		});
		float[][] xold = x;
		float[][] yold = y;
		float[] projOld = proj;
		x = new float[xold.length][];
		y = new float[yold.length][];
		proj = new float[projOld.length];
		for (int i=0; i<x.length; ++i) {
			x[i] = xold[indices[i]];
			y[i] = yold[indices[i]];
			proj[i] = projOld[indices[i]];
		}
		System.out.println("Project / sort time: "+(System.nanoTime() - start) / 1e6);

		this.x_d = Matrix.build(x);
		this.y_d = Matrix.build(y);
		this.xSqrNorms_d = x_d.sqr().colSum();
		this.xMean_d = Matrix.build(1, xMean.length, xMean);
		this.projDirection_d = Matrix.build(projDirection.length, 1, projDirection);
		this.proj_d = Matrix.build(proj.length, 1, proj);
		
		this.x_d.setDontFree(true);
		this.y_d.setDontFree(true);
		this.xSqrNorms_d.setDontFree(true);
		this.xMean_d.setDontFree(true);
		this.projDirection_d.setDontFree(true);
		this.proj_d.setDontFree(true);
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
		
		int xdim = x[0].length;
		int ydim = y[0].length;
		
		long start = System.nanoTime();
		Matrix xin_d = Matrix.build(xin);
		System.out.println("copy time: "+(System.nanoTime() - start)/1e9);
		
		start = System.nanoTime();
		Matrix xinProj_d = xin_d.rowSub(xMean_d).mmul(projDirection_d);
		float[] xinProj = xinProj_d.toArray();
		System.out.println("proj time: "+(System.nanoTime() - start)/1e9);
		
		start = System.nanoTime();
		Matrix xinSqrNorms_d = xin_d.sqr().colSum();
		Matrix sqrDists_d = x_d.mmul(xin_d.transpose());
		sqrDists_d.muli(-2.0f);
		sqrDists_d.colAddi(xSqrNorms_d);
		sqrDists_d.rowAddi(xinSqrNorms_d);
		Matrix xWeights_d = sqrDists_d.muli(-0.5f / (std*std)).expi();
		List<Matrix> Blist_d = new ArrayList<Matrix>();
		List<Matrix> yNeighborslist_d = new ArrayList<Matrix>();
		List<Matrix> xinSinglelist_d = new ArrayList<Matrix>();
		List<Matrix> wlist_d = new ArrayList<Matrix>();
		for (int f=0; f<xin.length; ++f) {
			int centerIndex = binarySearch(proj, xinProj[f]);
			centerIndex = Math.max(numNeighbors/2, centerIndex);
			centerIndex = Math.min(x.length-numNeighbors/2, centerIndex);
			
			Matrix xinSingle_d = xin_d.copySubmatrix(f, f+1, 0, xdim);
			Matrix B_d = x_d.copySubmatrix(centerIndex - numNeighbors/2, centerIndex + numNeighbors/2, 0, xdim);
			Matrix yNeighbors_d = y_d.copySubmatrix(centerIndex - numNeighbors/2, centerIndex + numNeighbors/2, 0, ydim);
			Matrix w_d = xWeights_d.copySubmatrix(centerIndex - numNeighbors/2, centerIndex + numNeighbors/2, f, f+1);
			
			xinSinglelist_d.add(xinSingle_d);
			Blist_d.add(B_d);
			yNeighborslist_d.add(yNeighbors_d);
			wlist_d.add(w_d);
		}
		
		List<Matrix> BWTrlist_d = new ArrayList<Matrix>();
		List<Matrix> BBlist_d = new ArrayList<Matrix>();
		if (LOWER_MEM_FOOTPRINT) {
			for (int f=0; f<xin.length; ++f) {
				Matrix BWTr_d = Blist_d.get(f).transpose();
				BWTr_d.rowMuli(wlist_d.get(f));
				BBlist_d.add(BWTr_d.mmul(Blist_d.get(f)));
				BWTrlist_d.add(BWTr_d);
				Blist_d.get(f).free();
				wlist_d.get(f).free();
			}
			System.out.println("extract / compute weights / mmul time: "+(System.nanoTime() - start)/1e9);
		} else {
			for (int f=0; f<xin.length; ++f) {
				Matrix BWTr_d = Blist_d.get(f).transpose();
				BWTr_d.rowMuli(wlist_d.get(f));
				BWTrlist_d.add(BWTr_d);
			}
			System.out.println("extract / compute weights time: "+(System.nanoTime() - start)/1e9);

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
