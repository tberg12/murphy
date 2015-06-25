package opt;

import tuple.Pair;
import arrays.a;

public class EmpiricalGradientTester {
	private static final double EPS = 1e-10;

	public static void test(DifferentiableFunction func, double[] x, double relEps, double delInitial, double delMin) {
		double[] nextX = a.copy(x);
		Pair<Double,double[]> valAndGrad = func.calculate(x);
		double baseVal = valAndGrad.getFirst();
		double[] grad = valAndGrad.getSecond();
		for (int i=0; i<x.length; ++i) {
			double delta = delInitial;
			boolean ok = false;
			double empDeriv = 0.0;
			while (delta > delMin && !ok) {
				nextX[i] += delta;
				double nextVal = func.calculate(nextX).getFirst();
				empDeriv = (nextVal - baseVal) / delta;
				if (close(empDeriv, grad[i], relEps)) {
					System.out.printf("Gradient ok for dim %d, delta %f, calculated %f, empirical: %f\n", i, delta, grad[i], empDeriv);
					ok = true;
				}
				nextX[i] -= delta;
				if (!ok) delta /= 2.0;
			}
			if (!ok) System.out.printf("Empirical gradient step-size underflow dim %d, delta %.12f, calculated %.12f, empirical: %.12f\n", i, delta, grad[i], empDeriv);
		}
	}
	
	public static void test(DifferentiableFunction func, double[] x, double relEps, double delInitial, double delMin, int i) {
		double[] nextX = a.copy(x);
		Pair<Double,double[]> valAndGrad = func.calculate(x);
		double baseVal = valAndGrad.getFirst();
		double[] grad = valAndGrad.getSecond();
		double delta = delInitial;
		boolean ok = false;
		double empDeriv = 0.0;
		while (delta > delMin && !ok) {
			nextX[i] += delta;
			double nextVal = func.calculate(nextX).getFirst();
			empDeriv = (nextVal - baseVal) / delta;
			if (close(empDeriv, grad[i], relEps)) {
				System.out.printf("Gradient ok for dim %d, delta %f, calculated %f, empirical: %f\n", i, delta, grad[i], empDeriv);
				ok = true;
			}
			nextX[i] -= delta;
			if (!ok) delta /= 2.0;
		}
		if (!ok) System.out.printf("Empirical gradient step-size underflow dim %d, delta %.12f, calculated %.12f, empirical: %.12f\n", i, delta, grad[i], empDeriv);
	}

	public static boolean close(double x, double y, double relEps) {
		if (Math.abs(x - y) < EPS) return true;
		double avgMag = (Math.abs(x) + Math.abs(y)) / 2.0;
		return Math.abs(x - y) / avgMag < relEps;
	}
}
