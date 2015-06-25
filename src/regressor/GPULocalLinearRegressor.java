package regressor;

import gpu.CublasUtil;
import gpu.CublasUtil.Matrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import threading.BetterThreader;
import util.PriorityQueue;
import arrays.a;

public class GPULocalLinearRegressor implements Regressor {
	
	public static final double MAX_BATCH_ELEMENTS = 1.4e9;

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
	
	public GPULocalLinearRegressor(float reg, float std, int numNeighbors) {
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
		
		this.x_d = Matrix.build(x);
		this.x_d.setDontFree(true);
		this.y_d = Matrix.build(y);
		this.y_d.setDontFree(true);
		this.xSqrNorms_d = x_d.sqr().colSum();
		this.xSqrNorms_d.setDontFree(true);
		
		CublasUtil.freeAll();
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
	
	public float[][] predictBatch(final float[][] xin) {
		final int numNeighbors = Math.min(this.numNeighbors, x.length);
		
		int xdim = x[0].length;
		
		long start = System.nanoTime();
		Matrix xin_d = Matrix.build(xin);
		System.out.println("copy time: "+(System.nanoTime() - start)/1e9);
		
		start = System.nanoTime();
		Matrix xinSqrNorms_d = xin_d.sqr().colSum();
		Matrix sqrDists_d = x_d.mmul(xin_d.transpose());
		sqrDists_d.transposei();
		sqrDists_d.muli(-2.0f);
		sqrDists_d.rowAddi(xSqrNorms_d);
		sqrDists_d.colAddi(xinSqrNorms_d);
		final float[][] sqrDists = sqrDists_d.toArray2();
		System.out.println("dists time: "+(System.nanoTime() - start)/1e9);
		
		CublasUtil.freeAllBut(xin_d, sqrDists_d);
		
		start = System.nanoTime();
		final int[][] indices = new int[xin.length][numNeighbors];
		BetterThreader.Function<Integer,Object> func = new BetterThreader.Function<Integer,Object>() { public void call(Integer f, Object ignore) {
			PriorityQueue<Integer> queue = new PriorityQueue<Integer>(numNeighbors);
			for (int i=0; i<x.length; ++i) {
				queue.add(i, sqrDists[f][i]);
				while (queue.size() > numNeighbors) queue.next();
			}
			for (int i=0; i<numNeighbors; ++i) {
				indices[f][i] = queue.next();
			}
		}};
		BetterThreader<Integer,Object> threader = new BetterThreader<Integer,Object>(func, 8);
		for (int f=0; f<xin.length; ++f) threader.addFunctionArgument(f); 
		threader.run();
		System.out.println("sort time: "+(System.nanoTime() - start) / 1e6);
		
		start = System.nanoTime();
		List<Matrix> Blist_d = new ArrayList<Matrix>();
		List<Matrix> yNeighborslist_d = new ArrayList<Matrix>();
		List<Matrix> xinSinglelist_d = new ArrayList<Matrix>();
		for (int f=0; f<xin.length; ++f) {
			Matrix xinSingle_d = xin_d.copySubmatrix(f, f+1, 0, xdim);
			
//			Matrix B_d = new Matrix(numNeighbors, xdim);
//			for (int ii=0; ii<numNeighbors; ++ii) {
//				int i = indices[f][ii];
//				B_d.setSubmatrix(ii, 0, x_d, i, i+1, 0, xdim);
//			}
			float[][] B = new float[numNeighbors][];
			for (int ii=0; ii<numNeighbors; ++ii) {
				int i = indices[f][ii];
				B[ii] = x[i];
			}
			Matrix B_d = Matrix.build(B);
			
//			Matrix yNeighbors_d = new Matrix(numNeighbors, ydim);
//			for (int ii=0; ii<numNeighbors; ++ii) {
//				int i = indices[f][ii];
//				yNeighbors_d.setSubmatrix(ii, 0, y_d, i, i+1, 0, ydim);
//			}
			float[][] yNeighbors = new float[numNeighbors][];
			for (int ii=0; ii<numNeighbors; ++ii) {
				int i = indices[f][ii];
				yNeighbors[ii] = y[i];
			}
			Matrix yNeighbors_d = Matrix.build(yNeighbors);
			
			Blist_d.add(B_d);
			yNeighborslist_d.add(yNeighbors_d);
			xinSinglelist_d.add(xinSingle_d);
		}
		System.out.println("extract time: "+(System.nanoTime() - start) / 1e6);
		
		start = System.nanoTime();
		Matrix xWeights_d = sqrDists_d.mul(-0.5f / (std*std)).expi();
		float[][] xWeights = xWeights_d.toArray2();
		List<Matrix> BWTrlist_d = new ArrayList<Matrix>();
		for (int f=0; f<xin.length; ++f) {
			float[] w = new float[numNeighbors];
			for (int ii=0; ii<numNeighbors; ++ii) {
				int i = indices[f][ii];
				w[ii] = xWeights[f][i];
			}
			Matrix w_d = Matrix.build(numNeighbors, 1, w);
			
			Matrix BWTr_d = Blist_d.get(f).transpose();
			BWTr_d.rowMuli(w_d);
			BWTrlist_d.add(BWTr_d);
		}
		System.out.println("compute weights time: "+(System.nanoTime() - start)/1e9);
		
		start = System.nanoTime();
		List<Matrix> BBlist_d = Matrix.mmul(BWTrlist_d, Blist_d);
		System.out.println("mmul time: "+(System.nanoTime() - start)/1e9);
		
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
