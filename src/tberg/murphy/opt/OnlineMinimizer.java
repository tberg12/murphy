package tberg.murphy.opt;

import java.util.List;

import tberg.murphy.opt.Minimizer.Callback;

public interface OnlineMinimizer {
	public abstract double[] minimize(List<DifferentiableFunction> functions, double[] initial, boolean verbose, Callback iterCallbackFunction);
}