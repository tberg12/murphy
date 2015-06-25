package regressor;

import java.util.ArrayList;
import java.util.List;

import arrays.a;
import counter.CounterInterface;
import counter.IntCounter;
import tuple.Pair;
import classifier.Classifier;
import classifier.LibLinearWrapper;
import classifier.LogisticRegressionClassifier;
import de.bwaldvogel.liblinear.SolverType;

public class BinaryRegressor implements Regressor {

	Classifier[] classifiers;
	float thresh;
	float c0;
	float c1;
	float reg;
	
	public BinaryRegressor(float reg, float thresh, float c0, float c1) {
		this.thresh = thresh;
		this.c0 = c0;
		this.c1 = c1;
		this.reg = reg;
	}
	
	public void train(float[][] xraw, float[][] yraw) {
		float[][] x = new float[xraw.length][];
		for (int i=0; i<xraw.length; ++i) {
			x[i] = a.append(10.0f, xraw[i]);
		}
		float[][] y = yraw;
		
		this.classifiers = new Classifier[y[0].length];
		for (int c=0; c<classifiers.length; ++c) {
			classifiers[c] = new LogisticRegressionClassifier(reg, 1e-20, 100, 8);
//			classifiers[c] = new LibLinearWrapper(SolverType.L2R_L2LOSS_SVC, 1e1, 1e-8);
			List<Pair<CounterInterface<Integer>,Integer>> trainSet = new ArrayList<Pair<CounterInterface<Integer>,Integer>>();
			for (int i=0; i<x.length; ++i) {
//				CounterInterface<Integer> features = IntCounter.wrapArray(a.toDouble(x[i]), x[i].length);
				CounterInterface<Integer> features = new IntCounter();
				for (int j=0; j<x[i].length; ++j) {
					features.setCount(j, (double) x[i][j]);
				}
				int label;
				if (y[i][c] > thresh) {
					label = 1;
				} else {
					label = 0;
				}
				trainSet.add(Pair.makePair(features, label));
			}
			classifiers[c].train(trainSet);
		}
	}

	public float[][] predict(float[][] xinraw) {
		float[][] xin = new float[xinraw.length][];
		for (int i=0; i<xinraw.length; ++i) {
			xin[i] = a.append(1.0f, xinraw[i]);
		}
		
		float[][] result = new float[xinraw.length][classifiers.length];
		for (int i=0; i<result.length; ++i) {
//			CounterInterface<Integer> features = IntCounter.wrapArray(a.toDouble(xin[i]), xin[i].length);
			CounterInterface<Integer> features = new IntCounter();
			for (int j=0; j<xin[i].length; ++j) {
				features.setCount(j, (double) xin[i][j]);
			}
			for (int c=0; c<classifiers.length; ++c) {
				int label = classifiers[c].predict(features);
				if (label == 0) {
					result[i][c] = c0;
				} else {
					result[i][c] = c1;
				}
			}
		}
		return result;
	}

}
