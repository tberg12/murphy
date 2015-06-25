package regressor;

import org.jblas.FloatMatrix;
import org.jblas.Solve;

import threading.BetterThreader;
import util.PriorityQueue;
import arrays.a;

public class LocalLinearRegressor implements Regressor {

	float reg;
	float std;
	int numNeighbors;
	int numThreads;
	
	float[][] x;
	float[][] y;
	
	public LocalLinearRegressor(float reg, float std, int numNeighbors, int numThreads) {
		this.reg = reg;
		this.std = std;
		this.numNeighbors = numNeighbors;
		this.numThreads = numThreads;
	}
	
	public void train(float[][] xraw, float[][] yraw) {
		this.x = new float[xraw.length][];
		for (int i=0; i<xraw.length; ++i) {
			this.x[i] = a.append(1.0f, xraw[i]);
		}
		this.y = yraw;
	}
	
	public float[][] predict(final float[][] xinraw) {
		System.out.println("predicting...");
		long outerStart = System.nanoTime();
		
		final float[][] xin = new float[xinraw.length][];
		for (int i=0; i<xinraw.length; ++i) {
			xin[i] = a.append(1.0f, xinraw[i]);
		}
		final float[][] yout = new float[xin.length][];
		BetterThreader.Function<Integer,Object> func = new BetterThreader.Function<Integer,Object>() { public void call(Integer i, Object ignore) {
			yout[i] = predict(xin[i]);
		}};
		BetterThreader<Integer,Object> threader = new BetterThreader<Integer,Object>(func, numThreads);
		for (int i=0; i<xin.length; ++i) threader.addFunctionArgument(i); 
		threader.run();
		
		System.out.println("total time seconds: "+(System.nanoTime() - outerStart) / 1e9);
		return yout;
	}
		
	private float[] predict(float[] xin) {
		System.out.println("predicting...");
		long outerStart = System.nanoTime();
		
		int numNeighbors = Math.min(this.numNeighbors, x.length);
		
		long start = System.nanoTime();
		float[] sqrDists = new float[x.length];
		for (int i=0; i<x.length; ++i) {
			sqrDists[i] = (float) a.sum(a.sqr(a.comb(xin, 1.0f, x[i], -1.0f)));
		}
		System.out.println("dist time: "+(System.nanoTime() - start) / 1e6);
		
		start = System.nanoTime();
		int[] indices = new int[x.length];
		PriorityQueue<Integer> queue = new PriorityQueue<Integer>(numNeighbors);
		for (int i=0; i<x.length; ++i) {
			queue.add(i, sqrDists[i]);
			while (queue.size() > numNeighbors) queue.next();
		}
		for (int i=0; i<numNeighbors; ++i) {
			indices[i] = queue.next();
		}
		System.out.println("sort time: "+(System.nanoTime() - start) / 1e6);
		
		start = System.nanoTime();
		float[][] B = new float[numNeighbors][];
		float[][] yNeighbors = new float[numNeighbors][];
		float[] w = new float[numNeighbors];
		
		for (int ii=0; ii<numNeighbors; ++ii) {
			int i = indices[ii];
			double sqrDist = sqrDists[i];
			float weight = (float) Math.exp(-0.5*(sqrDist/(std*std)));
			B[ii] = a.copy(x[i]);
			yNeighbors[ii] = a.copy(y[i]);
			w[ii] = weight;
		}
		
		FloatMatrix xinMat = new FloatMatrix(xin);
		FloatMatrix yNeighborsMat = new FloatMatrix(yNeighbors);
		FloatMatrix BMat = new FloatMatrix(B);
		FloatMatrix wMat = new FloatMatrix(w);
		FloatMatrix BWMatTr = BMat.transpose().muliRowVector(wMat);
		
		FloatMatrix invBWMatTr = Solve.solvePositive(BWMatTr.mmul(BMat).add(FloatMatrix.eye(x[0].length).mmul(reg)), BWMatTr);
		float[] result = (xinMat.transpose().mmul(invBWMatTr)).mmul(yNeighborsMat).toArray2()[0];

		System.out.println("mmul / invert time: "+(System.nanoTime() - start) / 1e6);

		System.out.println("total time seconds: "+(System.nanoTime() - outerStart) / 1e9);
		
		return result;
	}

}
