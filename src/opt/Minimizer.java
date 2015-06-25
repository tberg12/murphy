package opt;

public interface Minimizer {
	public static interface Callback {
		public void callback(double[] guess, int iter, double val, double[] grad) ;
		public static class NullCallback implements Callback {
			public void callback(double[] guess, int iter, double val, double[] grad) {}
		}
	}
	public abstract double[] minimize(DifferentiableFunction function, double[] initial, boolean verbose, Callback iterCallbackFunction);
}