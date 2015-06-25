package opt;

import java.util.List;

import opt.Minimizer.Callback;

public interface OnlineMinimizer {
	public abstract double[] minimize(List<DifferentiableFunction> functions, double[] initial, boolean verbose, Callback iterCallbackFunction);
}