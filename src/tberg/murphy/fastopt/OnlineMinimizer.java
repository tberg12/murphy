package tberg.murphy.fastopt;

import java.util.List;

public interface OnlineMinimizer {
  public static interface Callback {
    public void callback(float[] guess, int iter, double val) ;
    public static class NullCallback implements Callback {
      public void callback(float[] guess, int iter, double val) {}
    }
  }
	public abstract float[] minimize(List<DifferentiableFunction> functions, float[] initial, boolean verbose, Callback iterCallbackFunction);
}