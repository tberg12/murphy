package tberg.murphy.lazystructpred;

import java.util.List;

import tberg.murphy.counter.CounterInterface;

public interface LossAugmentedLearner<T> {

	public CounterInterface<Integer> train(float[] initWeights, LossAugmentedLinearModel<T> model, List<T> data, int maxIters);

}