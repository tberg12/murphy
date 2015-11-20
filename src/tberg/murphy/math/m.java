package tberg.murphy.math;

public class m {

	public final static double LOG_INV_SQRT_2_PI = -Math.log(Math.sqrt(2*Math.PI));

	public static double gaussianProb(double mean, double var, double x) {
		return Math.exp(gaussianLogProb(mean, var, x));
	}
	
	public static double gaussianProb(double distSqr, double stdDev) {
		return Math.exp(gaussianLogProb(distSqr, stdDev));
	}

	public static double gaussianLogProb(double mean, double var, double x) {
		return gaussianLogProb((x-mean)*(x-mean), var);
	}
	
	public static double gaussianLogProb(double distSqr, double var) {
		return -0.5*distSqr/var + LOG_INV_SQRT_2_PI - 0.5*Math.log(var);
	}

    public static double poissonLogProb(double k, double lambda) {
        return -lambda + k * Math.log(lambda) - logGamma(1 + k);
    }

    // from http://introcs.cs.princeton.edu/java/91float/Gamma.java.html
    // should probably use a standard impl
    static double logGamma(double x) {
        double tmp = (x - 0.5) * Math.log(x + 4.5) - (x + 4.5);
        double ser = 1.0
                   + 76.18009173 / (x + 0)
                   - 86.50532033 / (x + 1)
                   + 24.01409822 / (x + 2)
                   - 1.231739516 / (x + 3)
                   + 0.00120858003 / (x + 4)
                   - 0.00000536382 / (x + 5);
        return tmp + Math.log(ser * Math.sqrt(2 * Math.PI));
    }

    public static float logAdd(float a, float b) {
      if (a == Float.NEGATIVE_INFINITY) return b;
      else if (b == Float.NEGATIVE_INFINITY) return a;
      else if (a < b) return b + (float) Math.log(1.0 + (float) Math.exp(a - b));
      else return a + (float) Math.log(1.0 + (float) Math.exp(b - a));
    }

    public static double logAdd(double a, double b) {
      if (a == Double.NEGATIVE_INFINITY) return b;
      else if (b == Double.NEGATIVE_INFINITY) return a;
      else if (a < b) return b + Math.log(1.0 + Math.exp(a - b));
      else return a + Math.log(1.0 + Math.exp(b - a));
    }

}
