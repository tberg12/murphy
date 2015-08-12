package tberg.murphy.faststructpred;

import java.util.List;

public interface LossAugmentedLearner<T> {

	public float[] train(float[] initWeights, LossAugmentedLinearModel<T> model, List<T> data, int maxIters);

}