package tberg.murphy.floatopt;

import tberg.murphy.counter.CounterInterface;
import tberg.murphy.tuple.Pair;

public interface DifferentiableFunction {
	public abstract Pair<Double, CounterInterface<Integer>> calculate(float[] x);
}
