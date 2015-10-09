package example;

import java.util.ArrayList;
import java.util.List;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.Map;

import tberg.murphy.counter.CounterInterface;
import tberg.murphy.counter.IntCounter;
import tberg.murphy.structpred.LossAugmentedLinearModel;
import tberg.murphy.structpred.UpdateBundle;

public class StructpredInterface implements LossAugmentedLinearModel<DataSample> {
  public CounterInterface<Integer> weights = null;

  public StructpredInterface(CounterInterface<Integer> initWeights) {
    this.weights = initWeights;
  }

	public void startIteration(int t) { }

	public CounterInterface<Integer> getWeights() { return null; }

	public void updateWeights(IntCounter weightsDelta, double scale) {
    Iterator<Map.Entry<Integer, Double>> iterator = weightsDelta.entries().iterator();
    while (iterator.hasNext()) {
      Map.Entry<Integer, Double> pair = iterator.next();
      Integer key = pair.getKey();
      Double weight = pair.getValue();
      Double prev = weights.getCount(key);
      weights.setCount(key, prev + weight * scale);
    }
	}

	public void setWeights(CounterInterface<Integer> weights) {
    Iterator<Map.Entry<Integer, Double>> iterator = weights.entries().iterator();
    while (iterator.hasNext()) {
      Map.Entry<Integer, Double> pair = iterator.next();
      Integer key = pair.getKey();
      Double weight = pair.getValue();
      this.weights.setCount(key, weight);
    }
  }

  public UpdateBundle getLossAugmentedUpdateBundle(DataSample datum, double lossWeight) {
    // Inference
    double score0 = 0.0;
    double score1 = 0.0;
    CounterInterface<Integer> counter0 = new IntCounter();
    CounterInterface<Integer> counter1 = new IntCounter();
    for (int i = 0; i < datum.features.length; i++) {
      if (datum.features[i] == 1) {
        score0 += weights.getCount(i * 2);
        score1 += weights.getCount(i * 2 + 1);
        counter0.setCount(i * 2, 1);
        counter1.setCount(i * 2 + 1, 1);
      }
    }
    int label = 0;
    if (score0 < score1) label = 1;

    // Create the result
    int goldLabel = datum.gold;
    UpdateBundle ans = new UpdateBundle();
    // Active features for gold
    if (goldLabel == 0) ans.gold = counter0;
    else ans.gold = counter1;
    // Active features for guess
    if (label == 0) ans.guess = counter0;
    else ans.guess = counter1;
    // Loss
    if (label == goldLabel) ans.loss = 0.0;
    else ans.loss = 1.0;

    return ans;
  }

  public List<UpdateBundle> getLossAugmentedUpdateBundleBatch(List<DataSample> data, double lossWeight) {
    // Note - This is the point at which parallel processing would typically be
    // used to save time.
    List<UpdateBundle> ans = new ArrayList<UpdateBundle>();
    Iterator<DataSample> iterator = data.iterator();
    while (iterator.hasNext()) {
      DataSample datum = iterator.next();
      ans.add(getLossAugmentedUpdateBundle(datum, lossWeight));
    }
    return ans;
  }

	public UpdateBundle getUpdateBundle(DataSample datum) {
    return getLossAugmentedUpdateBundle(datum, 0.0);
  }

	public List<UpdateBundle> getUpdateBundleBatch(List<DataSample> data) {
    return getLossAugmentedUpdateBundleBatch(data, 0.0);
  }
}

