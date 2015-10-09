package tberg.murphy.lazystructpred;

import java.util.ArrayList;
import java.util.List;

import tberg.murphy.counter.CounterInterface;
import tberg.murphy.counter.IntCounter;
import tberg.murphy.lazyopt.AdaGradL1Minimizer;
import tberg.murphy.lazyopt.AdaGradL2Minimizer;
import tberg.murphy.lazyopt.DifferentiableFunction;
import tberg.murphy.lazyopt.OnlineMinimizer;
import tberg.murphy.structpred.SummingLossAugmentedLearner;
import tberg.murphy.structpred.SummingLossAugmentedLinearModel;
import tberg.murphy.structpred.UpdateBundle;
import tberg.murphy.tuple.Pair;

public class SubgradientSoftMaxMarginLearner<D> implements SummingLossAugmentedLearner<D> {
	
	int numFeatures;
	double C;
	double delta;
	double stepSize;
	boolean L1reg;
	int batchSize;
	
	public SubgradientSoftMaxMarginLearner(double C, double delta, double stepSize, int numFeatures, boolean L1reg) {
		this(C, delta, stepSize, numFeatures, L1reg, 1);
	}
	
	public SubgradientSoftMaxMarginLearner(double C, double delta, double stepSize, int numFeatures, boolean L1reg, int batchSize) {
		this.C = C;
		this.delta = delta;
		this.stepSize = stepSize;
		this.numFeatures = numFeatures;
		this.L1reg = L1reg;
		this.batchSize = batchSize;
	}
	
  public CounterInterface<Integer> train(CounterInterface<Integer> initWeights, final SummingLossAugmentedLinearModel<D> model, List<D> data, int iters) {
    List<DifferentiableFunction> objs = new ArrayList<DifferentiableFunction>();

    int numBatches = (int) Math.ceil(data.size() / (double) batchSize);
    for (int b=0; b<numBatches; ++b) {
      final List<D> batch = data.subList(b*batchSize, Math.min(data.size(), (b+1)*batchSize));
      objs.add(new DifferentiableFunction() {
        public Pair<Double, CounterInterface<Integer>> calculate(CounterInterface<Integer> x) {
          model.setWeights(x);
          List<UpdateBundle> ubBatch = model.getExpectedLossAugmentedUpdateBundleBatch(batch, 1.0);
          double valBatch = 0.0;
          CounterInterface<Integer> deltaBatch = new IntCounter();
          for (UpdateBundle ub : ubBatch) {
            deltaBatch.incrementAll(ub.gold, -1.0);
            deltaBatch.incrementAll(ub.guess, 1.0);
            valBatch += -ub.gold.dotProduct(x) + ub.loss;
          }
          return Pair.makePair(valBatch, deltaBatch);
        }
      });
    }
    
    OnlineMinimizer minimizer = (L1reg? new AdaGradL1Minimizer(stepSize, delta, C, iters) : new AdaGradL2Minimizer(stepSize, delta, C, iters));
    return minimizer.minimize(objs, IntCounter.convertToFloatArray(initWeights, numFeatures), true, null);
  }

}
