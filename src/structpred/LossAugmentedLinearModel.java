package structpred;

import java.util.List;

public interface LossAugmentedLinearModel<T> extends LinearModel<T> {
	
    public UpdateBundle getLossAugmentedUpdateBundle(T datum, double lossWeight);

    public List<UpdateBundle> getLossAugmentedUpdateBundleBatch(List<T> datum, double lossWeight);
    
}
