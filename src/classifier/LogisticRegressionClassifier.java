package classifier;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import opt.DifferentiableFunction;
import opt.LBFGSMinimizer;
import opt.Minimizer;
import arrays.a;
import threading.BetterThreader;
import tuple.Pair;
import counter.Counter;
import counter.CounterInterface;

public class LogisticRegressionClassifier implements Classifier {
	
	private double tol;
	private double regConstant;
	private int maxIters;
	private int numLabels;
	private int numFeatures;
	private int numThreads;
	private float[] weights;
	
	public LogisticRegressionClassifier(double regConstant, double tol, int maxIters, int numThreads) {
		this.regConstant = regConstant;
		this.tol = tol;
		this.maxIters = maxIters;
		this.numThreads = numThreads;
	}

	public void train(final List<Pair<CounterInterface<Integer>, Integer>> trainSet) {
		int maxLabel = 0;
		int maxFeatureIndex = 0;
		for (Pair<CounterInterface<Integer>, Integer> datum : trainSet) {
			maxLabel = Math.max(maxLabel, datum.getSecond());
			for (Integer featureIndex : datum.getFirst().keySet()) {
				maxFeatureIndex = Math.max(maxFeatureIndex, featureIndex);
			}
		}
		this.numLabels = maxLabel+1;
		this.numFeatures = maxFeatureIndex+1;
		this.weights = new float[numFeatures*numLabels];
		DifferentiableFunction objective = new DifferentiableFunction() {
				public Pair<Double, double[]> calculate(double[] x) {
					final float[] weights = a.toFloat(x);
					final float[] vals = new float[numThreads];
					final float[][] grads = new float[numThreads][weights.length];
					BetterThreader.Function<Pair<CounterInterface<Integer>, Integer>,Integer> func = new BetterThreader.Function<Pair<CounterInterface<Integer>, Integer>,Integer>(){public void call(Pair<CounterInterface<Integer>, Integer> datum, Integer threadId){
						float[] scores = getScores(weights, datum.getFirst());
						float[] probs = a.exp(scores);
						a.normalizei(probs);
						vals[threadId] -= Math.log(probs[datum.getSecond()]);
						for (Map.Entry<Integer, Double> featurePair : datum.getFirst().entries()) {
							int featureIndex = featurePair.getKey();
							float featureVal = (float) (double) featurePair.getValue();
							for (int label=0; label<numLabels; ++label) {
								grads[threadId][featureIndex*numLabels + label] += featureVal * probs[label];
							}
							grads[threadId][featureIndex*numLabels + datum.getSecond()] -= featureVal;
						}
					}};
					BetterThreader<Pair<CounterInterface<Integer>, Integer>,Integer> threader = new BetterThreader<Pair<CounterInterface<Integer>, Integer>,Integer>(func, numThreads);
					for (int threadId=0; threadId<numThreads; ++threadId) threader.setThreadArgument(threadId, threadId);
					for (Pair<CounterInterface<Integer>, Integer> datum : trainSet) threader.addFunctionArgument(datum);
					threader.run();
					float[] grad = a.scale(weights, 2.0f * (float) regConstant);
					for (int threadId=0; threadId<numThreads; ++threadId) {
						a.combi(grad, 1.0f, grads[threadId], 1.0f);
					}
					return Pair.makePair(a.sum(vals) + regConstant * a.sum(a.sqr(weights)), a.toDouble(grad));
				}
		};
		Minimizer minimizer = new LBFGSMinimizer(tol, maxIters, true);
		this.weights = a.toFloat(minimizer.minimize(objective, a.toDouble(weights), true, null));
	}

	private float[] getScores(float[] weights, CounterInterface<Integer> features) {
		float[] scores = new float[numLabels];
		for (Map.Entry<Integer, Double> featurePair : features.entries()) {
			int feature = featurePair.getKey();
			float value = (float) (double) featurePair.getValue();
			for (int label=0; label<numLabels; ++label) {
				scores[label] += value * weights[feature*numLabels + label];
			}
		}
		return scores;
	}
	
	public Map<Integer, CounterInterface<Integer>> getWeights() {
		Map<Integer,CounterInterface<Integer>> result = new HashMap<Integer,CounterInterface<Integer>>();
		for (int label=0; label<numLabels; ++label) {
			for (int feature=0; feature<numFeatures; ++feature) {
				CounterInterface<Integer> labelWeights = result.get(label);
				if (labelWeights == null) {
					labelWeights = new Counter<Integer>();
					result.put(label, labelWeights);
				}
				labelWeights.setCount(feature, weights[feature*numLabels + label]);
			}
		}
		return result;
	}

	public Integer predict(CounterInterface<Integer> toPredict) {
		float[] labelScores = getScores(weights, toPredict);
		return a.argmax(labelScores);
	}

}
