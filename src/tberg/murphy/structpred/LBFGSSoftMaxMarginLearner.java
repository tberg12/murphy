package tberg.murphy.structpred;

import java.util.List;
import java.util.Map;

import tberg.murphy.arrays.a;
import tberg.murphy.counter.CounterInterface;
import tberg.murphy.counter.IntCounter;
import tberg.murphy.opt.DifferentiableFunction;
import tberg.murphy.opt.LBFGSMinimizer;
import tberg.murphy.opt.Minimizer;
import tberg.murphy.tuple.Pair;

public class LBFGSSoftMaxMarginLearner<D> implements SummingLossAugmentedLearner<D> {
	
	int numFeatures;
	double C;
	double tolerance;
	
	public LBFGSSoftMaxMarginLearner(double C, double tolerance, int numFeatures) {
    this.C = C;
    this.numFeatures = numFeatures;
    this.tolerance = tolerance;
	}
	
  public CounterInterface<Integer> train(CounterInterface<Integer> initWeights, final SummingLossAugmentedLinearModel<D> model, final List<D> data, int iters) {
    DifferentiableFunction obj = new DifferentiableFunction() {
      public Pair<Double, double[]> calculate(double[] x) {
        CounterInterface<Integer> weights = IntCounter.wrapArray(x, numFeatures);
        model.setWeights(weights);
        List<UpdateBundle> ubs = model.getExpectedLossAugmentedUpdateBundleBatch(data, 1.0);
        double valTotal = 0.0;
        double[] deltaTotal = new double[numFeatures];
        for (UpdateBundle ub : ubs) {
          for (Map.Entry<Integer,Double> entry : ub.gold.entries()) {
            deltaTotal[entry.getKey()] -= entry.getValue();
          }
          for (Map.Entry<Integer,Double> entry : ub.guess.entries()) {
            deltaTotal[entry.getKey()] += entry.getValue();
          }
          valTotal += -ub.gold.dotProduct(weights) + ub.loss;
        }
        a.combi(deltaTotal, 1.0, x, 2*C);
        valTotal += C * a.sum(a.sqr(x));
        return Pair.makePair(valTotal, deltaTotal);
      }
    };
    
    Minimizer minimizer = new LBFGSMinimizer(tolerance, iters);
    double[] initWeightsArray = new double[numFeatures];
    for (Map.Entry<Integer, Double> entry : initWeights.entries()) {
      initWeightsArray[entry.getKey()] = entry.getValue();
    }
    return IntCounter.wrapArray(minimizer.minimize(obj, initWeightsArray, true, null), numFeatures);
  }

}
