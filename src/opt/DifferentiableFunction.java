package opt;

import tuple.Pair;

public interface DifferentiableFunction {
	public abstract Pair<Double, double[]> calculate(double[] x);
}
