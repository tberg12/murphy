package opt;

import java.util.ArrayList;
import java.util.List;

import lbfgsb.Bound;
import arrays.a;
import tuple.Pair;

public class LBFGSFortranWrapperMinimizer implements Minimizer {
	
	private static double MACHINE_EPS;
	static {
		MACHINE_EPS = 1.0;
		do
			MACHINE_EPS /= 2.0;
		while (1.0 + (MACHINE_EPS / 2.0) != 1.0);
	}
	
	double tolerance;
	int maxIters;
	int rank;
	double lowerBound;
	double upperBound;
	
	public LBFGSFortranWrapperMinimizer(double tolerance, int maxIters) {
		this.tolerance = tolerance;
		this.maxIters = maxIters;
		this.lowerBound = Double.NEGATIVE_INFINITY;
		this.upperBound = Double.POSITIVE_INFINITY;
		this.rank = -1;
	}
	
	public LBFGSFortranWrapperMinimizer(double tolerance, int maxIters, double lowerBound, double upperBound) {
		this.tolerance = tolerance;
		this.maxIters = maxIters;
		this.lowerBound = lowerBound;
		this.upperBound = upperBound;
		this.rank = -1;
	}
	
	public LBFGSFortranWrapperMinimizer(double tolerance, int maxIters, double lowerBound, double upperBound, int rank) {
		this.tolerance = tolerance;
		this.maxIters = maxIters;
		this.lowerBound = lowerBound;
		this.upperBound = upperBound;
		this.rank = rank;
	}

	public double[] minimize(final DifferentiableFunction function, double[] initial, boolean verbose, final Callback iterCallbackFunction) {
		lbfgsb.Minimizer alg = new lbfgsb.Minimizer();
		if (lowerBound != Double.NEGATIVE_INFINITY || upperBound != Double.POSITIVE_INFINITY) {
			List<Bound> bounds = new ArrayList<Bound>();
			for (int d=0; d<initial.length; ++d) {
				bounds.add(new Bound((lowerBound == Double.NEGATIVE_INFINITY ? null : lowerBound), (upperBound == Double.POSITIVE_INFINITY ? null : upperBound)));
			}
			alg.setBounds(bounds);
		}
		if (rank > 0) {
			alg.setCorrectionsNo(rank);
		}
		alg.getStopConditions().setMaxIterations(maxIters);
		alg.getStopConditions().setFunctionReductionFactor(tolerance / MACHINE_EPS);
		alg.setDebugLevel((verbose ? 1 : 0));
		alg.setIterationFinishedListener(new lbfgsb.IterationFinishedListener() {
			public boolean iterationFinished(double[] point, double val, double[] grad) {
				if (iterCallbackFunction != null) iterCallbackFunction.callback(point, -1, val, grad);
				return true;
			}
		});
		lbfgsb.Result result = null;
		try {
			result = alg.run(new lbfgsb.DifferentiableFunction() {
				public lbfgsb.FunctionValues getValues(double[] x) {
					Pair<Double,double[]> valAndGrad = function.calculate(x);
					return new lbfgsb.FunctionValues(valAndGrad.getFirst(), valAndGrad.getSecond()); 
				}
			}, initial);
		} catch (lbfgsb.LBFGSBException e) {
			e.printStackTrace();
		}
		return result.point;
	}
	
	public static void main(String[] args) {
		DifferentiableFunction function = new DifferentiableFunction() {
			public Pair<Double, double[]> calculate(double[] x) {
				return Pair.makePair(-a.sum(x) + 2.0 * a.innerProd(x, x), a.comb(a.scale(a.onesDouble(x.length), -1.0), 1.0, a.scale(x, 2.0 * 2.0), 1.0));
			}
		};
		
//		Minimizer minimizer = new LBFGSFortranWrapperMinimizer(1e-5, 1000);
		Minimizer minimizer = new LBFGSFortranWrapperMinimizer(1e-5, 1000, 0.33, 0.59, 10);
		System.out.println(a.toString(minimizer.minimize(function, a.zerosDouble(10), true, null)));
	}

}
