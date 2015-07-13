package tberg.murphy.structpred;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import tberg.murphy.counter.CounterInterface;
import tberg.murphy.counter.IntCounter;
import tberg.murphy.opt.AdaGradL1Minimizer;
import tberg.murphy.opt.AdaGradL2Minimizer;
import tberg.murphy.opt.DifferentiableFunction;
import tberg.murphy.opt.OnlineMinimizer;
import tberg.murphy.tuple.Pair;

public class PrimalSubgradientSVMLearner<D> implements LossAugmentedLearner<D> {
	
	int numFeatures;
	double C;
	double delta;
	double stepSize;
	boolean L1reg;
	int batchSize;
	
	public PrimalSubgradientSVMLearner(double C, double delta, double stepSize, int numFeatures, boolean L1reg) {
		this(C, delta, stepSize, numFeatures, L1reg, 1);
	}
	
	public PrimalSubgradientSVMLearner(double C, double delta, double stepSize, int numFeatures, boolean L1reg, int batchSize) {
		this.C = C;
		this.delta = delta;
		this.stepSize = stepSize;
		this.numFeatures = numFeatures;
		this.L1reg = L1reg;
		this.batchSize = batchSize;
	}
	
	public CounterInterface<Integer> train(CounterInterface<Integer> initWeights, final LossAugmentedLinearModel<D> model, List<D> data, int iters) {
		double[] denseInitWeights = dense(initWeights, numFeatures);
		List<DifferentiableFunction> objs = new ArrayList<DifferentiableFunction>();

		int numBatches = (int) Math.ceil(data.size() / (double) batchSize);
		for (int b=0; b<numBatches; ++b) {
			final List<D> batch = data.subList(b*batchSize, Math.min(data.size(), (b+1)*batchSize));
			objs.add(new DifferentiableFunction() {
				public Pair<Double, double[]> calculate(double[] x) {
					CounterInterface<Integer> weights = sparse(x);
					model.setWeights(weights);
					List<UpdateBundle> ubBatch = model.getLossAugmentedUpdateBundleBatch(batch, 1.0);
					double valBatch = 0.0;
					CounterInterface<Integer> deltaBatch = new IntCounter();
					for (UpdateBundle ub : ubBatch) {
						CounterInterface<Integer> delta = new IntCounter();
						delta.incrementAll(ub.gold, -1.0);
						delta.incrementAll(ub.guess, 1.0);
						double val = ub.loss + delta.dotProduct(weights);
						if (val > 0.0) {
							valBatch += val;
							deltaBatch.incrementAll(delta);
						}
					}
					return Pair.makePair(valBatch, dense(deltaBatch, numFeatures));
				}
			});
		}
		
		OnlineMinimizer minimizer = (L1reg? new AdaGradL1Minimizer(stepSize, delta, C, iters) : new AdaGradL2Minimizer(stepSize, delta, C, iters));
		return sparse(minimizer.minimize(objs, denseInitWeights, true, null));
	}
	
//	public CounterInterface<Integer> train(CounterInterface<Integer> initWeights, final LossAugmentedLinearModel<D> model, List<D> data, int iters) {
//		double[] denseInitWeights = dense(initWeights, numFeatures);
//		List<DifferentiableFunction> objs = new ArrayList<DifferentiableFunction>();
//		
//		for (final D datum : data) {
//			objs.add(new DifferentiableFunction() {
//				public Pair<Double, double[]> calculate(double[] x) {
//					CounterInterface<Integer> weights = sparse(x);
//					model.setWeights(weights);
//					UpdateBundle ub = model.getLossAugmentedUpdateBundle(datum, 1.0);
//					CounterInterface<Integer> delta = new IntCounter();
//					delta.incrementAll(ub.gold, -1.0);
//					delta.incrementAll(ub.guess, 1.0);
//					double val = ub.loss + delta.dotProduct(weights);
//					if (val <= 0.0) {
//						return Pair.makePair(0.0, new double[numFeatures]);
//					} else {
//						return Pair.makePair(val, dense(delta, numFeatures));
//					}
//					
//				}
//			});
//		}
//		
//		OnlineMinimizer minimizer = (L1reg? new AdaGradL1Minimizer(stepSize, delta, C, iters) : new AdaGradL2Minimizer(stepSize, delta, C, iters));
//		return sparse(minimizer.minimize(objs, denseInitWeights, true, null));
//	}
	
	public static double[] dense(CounterInterface<Integer> vect, int dim) {
		double[] result = new double[dim];
		for(Map.Entry<Integer,Double> entry : vect.entries()) {
			result[entry.getKey()] = entry.getValue();
		}
		return result;
	}
	
	public static CounterInterface<Integer> sparse(double[] vect) {
		return IntCounter.wrapArray(vect, vect.length);
	}
	
}
