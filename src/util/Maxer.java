package util;

public class Maxer<T> {
    private double max = Double.NEGATIVE_INFINITY;
    private T argMax = null;
    private boolean verbose = false;

    @Override
	public String toString() {
		return (argMax == null ? "null " : argMax.toString()) + ": " + max;
    }

    public void observe(T t, double val) {
        if (verbose) System.out.println("Observing " + t.toString() + " @ " + val);
        if (val > max) {
            max = val;
            argMax = t;
			if (verbose) System.out.println(t.toString() + " is new max");
        }
    }

    public double getMax() {
        return max;
    }

    public T argMax() {
        return argMax;
    }

    public void setVerbose(boolean b) {
        verbose = b;
    }
}
