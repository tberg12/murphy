package tberg.murphy.faststructpred;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import tberg.murphy.counter.CounterInterface;
import tberg.murphy.counter.IntCounter;
import tberg.murphy.fastopt.AdaGradL1Minimizer;
import tberg.murphy.fastopt.AdaGradL2Minimizer;
import tberg.murphy.fastopt.DifferentiableFunction;
import tberg.murphy.fastopt.OnlineMinimizer;
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
	
	public float[] train(float[] initWeights, final LossAugmentedLinearModel<D> model, List<D> data, int iters) {
		List<DifferentiableFunction> objs = new ArrayList<DifferentiableFunction>();

		int numBatches = (int) Math.ceil(data.size() / (double) batchSize);
		for (int b=0; b<numBatches; ++b) {
			final List<D> batch = data.subList(b*batchSize, Math.min(data.size(), (b+1)*batchSize));
			objs.add(new DifferentiableFunction() {
				public Pair<Double, CounterInterface<Integer>> calculate(float[] x) {
					model.setWeights(x);
					List<UpdateBundle> ubBatch = model.getLossAugmentedUpdateBundleBatch(batch, 1.0);
					double valBatch = 0.0;
					CounterInterface<Integer> deltaBatch = new IntCounter();
					for (UpdateBundle ub : ubBatch) {
						CounterInterface<Integer> delta = new IntCounter();
						delta.incrementAll(ub.gold, -1.0);
						delta.incrementAll(ub.guess, 1.0);
						float dotProd = 0.0f;
						for (Map.Entry<Integer,Double> entry : delta.entries()) {
						  final int key = entry.getKey();
						  final float val = entry.getValue().floatValue();
						  dotProd += val * x[key];
						}
						double val = ub.loss + dotProd;
						if (val > 0.0) {
							valBatch += val;
							deltaBatch.incrementAll(delta);
						}
					}
					return Pair.makePair(valBatch, deltaBatch);
				}
			});
		}
		
		OnlineMinimizer minimizer = (L1reg? new AdaGradL1Minimizer(stepSize, delta, C, iters) : new AdaGradL2Minimizer(stepSize, delta, C, iters));
		return minimizer.minimize(objs, initWeights, true, null);
	}
	
}
