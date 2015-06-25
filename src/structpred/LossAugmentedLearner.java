package structpred;

import java.util.List;

import counter.CounterInterface;

public interface LossAugmentedLearner<T> {

	public CounterInterface<Integer> train(CounterInterface<Integer> initWeights, LossAugmentedLinearModel<T> model, List<T> data, int maxIters);

}