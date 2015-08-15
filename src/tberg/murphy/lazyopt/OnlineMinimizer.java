package tberg.murphy.lazyopt;

import java.util.List;

import tberg.murphy.counter.CounterInterface;

public interface OnlineMinimizer {
  public static interface Callback {
    public void callback(float[] guess, int iter, double val) ;
    public static class NullCallback implements Callback {
      public void callback(float[] guess, int iter, double val) {}
    }
  }
	public abstract CounterInterface<Integer> minimize(List<DifferentiableFunction> functions, float[] initial, boolean verbose, Callback iterCallbackFunction);
}