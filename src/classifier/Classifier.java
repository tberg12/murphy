package classifier;

import java.util.List;
import java.util.Map;

import tuple.Pair;
import counter.CounterInterface;

public interface Classifier {
	
	public void train(List<Pair<CounterInterface<Integer>,Integer>> trainSet);
	
	public Map<Integer,CounterInterface<Integer>> getWeights();
	
	public Integer predict(CounterInterface<Integer> toPredict);

}
