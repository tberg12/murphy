package tberg.murphy.structpred;

import java.util.List;

import tberg.murphy.counter.CounterInterface;

public interface SummingLossAugmentedLearner<T> {

	public CounterInterface<Integer> train(CounterInterface<Integer> initWeights, SummingLossAugmentedLinearModel<T> model, List<T> data, int maxIters);

}