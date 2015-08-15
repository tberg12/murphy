package tberg.murphy.floatstructpred;

import java.util.List;

import tberg.murphy.counter.IntCounter;

public interface LinearModel<T> {

	public UpdateBundle getUpdateBundle(T datum);

	public List<UpdateBundle> getUpdateBundleBatch(List<T> datum);

	public void setWeights(float[] weights);

	public void updateWeights(IntCounter weightsDelta, double scale);

	public float[] getWeights();

	public void startIteration(int t);

}