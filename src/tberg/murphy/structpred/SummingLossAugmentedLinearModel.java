package tberg.murphy.structpred;

import java.util.List;

public interface SummingLossAugmentedLinearModel<T> extends LossAugmentedLinearModel<T> {
	
    public UpdateBundle getExpectedLossAugmentedUpdateBundle(T datum, double lossWeight);

    public List<UpdateBundle> getExpectedLossAugmentedUpdateBundleBatch(List<T> datum, double lossWeight);
    
}
