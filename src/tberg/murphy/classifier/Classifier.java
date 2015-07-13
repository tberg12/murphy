package tberg.murphy.classifier;

import java.util.List;
import java.util.Map;

import tberg.murphy.counter.CounterInterface;
import tberg.murphy.tuple.Pair;

public interface Classifier {
	
	public void train(List<Pair<CounterInterface<Integer>,Integer>> trainSet);
	
	public Map<Integer,CounterInterface<Integer>> getWeights();
	
	public Integer predict(CounterInterface<Integer> toPredict);

}
