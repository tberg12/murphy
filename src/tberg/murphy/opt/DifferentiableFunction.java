package tberg.murphy.opt;

import tberg.murphy.tuple.Pair;

public interface DifferentiableFunction {
	public abstract Pair<Double, double[]> calculate(double[] x);
}
