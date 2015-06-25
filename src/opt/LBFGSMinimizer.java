package opt;

import java.util.Arrays;
import java.util.LinkedList;

import tuple.Pair;
import arrays.a;

public class LBFGSMinimizer implements Minimizer {
	
	private static class CachingFunctionWrapper implements DifferentiableFunction {
		DifferentiableFunction func;
		double[] x;
		Pair<Double,double[]> valAndGrad;
		public CachingFunctionWrapper(DifferentiableFunction func) {
			this.func = func;
		}
		private void ensureCache(double[] x) {
			if (this.x == null || !Arrays.equals(this.x, x)) {
				valAndGrad = func.calculate(x);
			}
		}
		public Pair<Double, double[]> calculate(double[] x) {
			ensureCache(x);
			return valAndGrad;
		}
	}
	
	private static final double EPS = 1e-12;
	private static final double LINE_SEARCH_SUFF_DECR = 1e-4;
	
	int maxHistorySize = 5;
	double initialStepSizeMultiplier = 0.5;
	double stepSizeMultiplier = 0.5;
	double stepSizeGrowMultiplier = 1.5;
	boolean finishOnFirstConverge = true;
	double tolerance;
	int maxIters;
	
	LinkedList<double[]> inputDifferenceVectorList;
	LinkedList<double[]> derivativeDifferenceVectorList;

	public LBFGSMinimizer(double tolerance, int maxIters, boolean finishOnFirstConverge, double initialStepSizeMultiplier, double stepSizeMultiplier, double stepSizeGrowMultiplier, int maxHistorySize) {
		this.tolerance = tolerance;
		this.maxIters = maxIters;
		this.finishOnFirstConverge = finishOnFirstConverge;
		this.initialStepSizeMultiplier = initialStepSizeMultiplier;
		this.stepSizeMultiplier = stepSizeMultiplier;
		this.stepSizeGrowMultiplier = stepSizeGrowMultiplier;
		this.maxHistorySize = maxHistorySize;
	}
	
	public LBFGSMinimizer(double tolerance, int maxIters, boolean finishOnFirstConverge, double initialStepSizeMultiplier, double stepSizeMultiplier, double stepSizeGrowMultiplier) {
		this.tolerance = tolerance;
		this.maxIters = maxIters;
		this.finishOnFirstConverge = finishOnFirstConverge;
		this.initialStepSizeMultiplier = initialStepSizeMultiplier;
		this.stepSizeMultiplier = stepSizeMultiplier;
		this.stepSizeGrowMultiplier = stepSizeGrowMultiplier;
	}
	
	public LBFGSMinimizer(double tolerance, int maxIters, boolean finishOnFirstConverge) {
		this.tolerance = tolerance;
		this.maxIters = maxIters;
		this.finishOnFirstConverge = finishOnFirstConverge;
	}
	
	public LBFGSMinimizer(double tolerance, int maxIters) {
		this.tolerance = tolerance;
		this.maxIters = maxIters;
	}

	public double[] minimize(DifferentiableFunction function, double[] initial, boolean verbose, Callback iterCallbackFunction) {
		inputDifferenceVectorList = new LinkedList<double[]>();
		derivativeDifferenceVectorList = new LinkedList<double[]>();
		
		function = new CachingFunctionWrapper(function);
		double[] guess = a.copy(initial);
		
		Pair<Double,double[]> valAndGrad = function.calculate(guess);
		double val = valAndGrad.getFirst();
		double[] grad = valAndGrad.getSecond();
		boolean lastIterConverged = false;
		double stepSize = 1.0;
		for (int iteration = 0; iteration < maxIters; iteration++) {
			double[] initialInverseHessianDiagonal = getInitialInverseHessianDiagonal(function, initial.length);
			double[] direction = implicitMultiply(initialInverseHessianDiagonal, grad);
			a.scalei(direction, -1.0);

			Pair<double[],Double> nextGuessAndStepSize = lineSearch(function, guess, direction, stepSize, (iteration == 0 ? initialStepSizeMultiplier : stepSizeMultiplier));
			if (nextGuessAndStepSize == null) {
				clearHistories();
				if (verbose) System.out.println("[LBFGSMinimizer.minimize] Cleared history.");
				stepSize = 1.0;
				nextGuessAndStepSize = lineSearch(function, guess, direction, stepSize, stepSizeMultiplier);
				if (nextGuessAndStepSize == null) {
					throw new Error("[LBFGSMinimizer.minimize] Cannot find step that will decrease function value.");
				}
			}
			stepSize = nextGuessAndStepSize.getSecond() * stepSizeGrowMultiplier;
			double[] nextGuess = nextGuessAndStepSize.getFirst();
			
			Pair<Double,double[]> nextValAndGrad = function.calculate(nextGuess);
			double nextVal = nextValAndGrad.getFirst();
			double[] nextGrad = nextValAndGrad.getSecond();

			if (verbose) System.out.println(String.format("[LBFGSMinimizer.minimize] Iteration %d ended with value %.6f", iteration, nextVal));

			if (converged(val, nextVal, tolerance)) {
				if (finishOnFirstConverge || lastIterConverged) {
					return nextGuess;
				} else {
					clearHistories();
					if (verbose) System.out.println("[LBFGSMinimizer.minimize] Cleared history.");
					stepSize = 1.0;
					lastIterConverged = true;
				}
			} else {
				lastIterConverged = false;
			}

			updateHistories(guess, nextGuess, grad, nextGrad);
			guess = nextGuess;
			val = nextVal;
			grad = nextGrad;
			
			if (iterCallbackFunction != null) iterCallbackFunction.callback(guess, iteration, val, grad);
		}
		if (verbose) System.out.println("[LBFGSMinimizer.minimize] Exceeded max iterations without converging.");
		return guess;
	}
	
	private static Pair<double[],Double> lineSearch(DifferentiableFunction function, double[] initial, double[] direction, double initialStepSize, double stepSizeMultiplier) {
		double stepSize = initialStepSize;
		Pair<Double,double[]> initialValAndGrad = function.calculate(initial);
		double val = initialValAndGrad.getFirst();
		final double[] grad = initialValAndGrad.getSecond();
		double initialDirectionalDerivative = a.innerProd(grad, direction);
		double gradMax = a.max(a.abs(grad));
		double[] guess = null;
		double guessValue = 0.0;
		boolean sufficientDecreaseObtained = false;
		while (!sufficientDecreaseObtained) {
			guess = a.comb(initial, 1.0, direction, stepSize);
			guessValue = function.calculate(guess).getFirst();
			double sufficientDecreaseValue = val + LINE_SEARCH_SUFF_DECR * initialDirectionalDerivative * stepSize;
			sufficientDecreaseObtained = (guessValue <= sufficientDecreaseValue + EPS);
			if (!sufficientDecreaseObtained) {
				if (stepSize < EPS && stepSize * gradMax < EPS) {
					System.out.printf("[LBFGSMinimizer.minimize]: Line search step size underflow: %.15f, %.15f, %.15f, %.15f, %.15f, %.15f\n", stepSize, initialDirectionalDerivative, gradMax, guessValue, sufficientDecreaseValue, val);
					return null;
				}
				stepSize *= stepSizeMultiplier;
			}
		}
		return Pair.makePair(guess, stepSize);
	}
	
	private boolean converged(double value, double nextValue, double tolerance) {
		double valueChange = Math.abs(nextValue - value);
		if (valueChange <= EPS) return true;
		double valueAverageMag = (Math.abs(nextValue) + Math.abs(value) + EPS) / 2.0;
		if (valueChange / valueAverageMag < tolerance) return true;
		return false;
	}

	private void updateHistories(double[] guess, double[] nextGuess, double[] derivative, double[] nextDerivative) {
		double[] guessChange = a.comb(nextGuess, 1.0, guess, -1.0);
		double[] derivativeChange = a.comb(nextDerivative, 1.0, derivative, -1.0);
		pushOntoList(guessChange, inputDifferenceVectorList);
		pushOntoList(derivativeChange, derivativeDifferenceVectorList);
	}

	private void pushOntoList(double[] vector, LinkedList<double[]> vectorList) {
		vectorList.addFirst(vector);
		if (vectorList.size() > maxHistorySize) vectorList.removeLast();
	}

	private void clearHistories() {
		inputDifferenceVectorList.clear();
		derivativeDifferenceVectorList.clear();
	}

	private int historySize() {
		return inputDifferenceVectorList.size();
	}

	private double[] getInputDifference(int num) {
		// 0 is previous, 1 is the one before that
		return inputDifferenceVectorList.get(num);
	}

	private double[] getDerivativeDifference(int num) {
		return derivativeDifferenceVectorList.get(num);
	}

	private double[] getLastDerivativeDifference() {
		return derivativeDifferenceVectorList.getFirst();
	}

	private double[] getLastInputDifference() {
		return inputDifferenceVectorList.getFirst();
	}

	private double[] implicitMultiply(double[] initialInverseHessianDiagonal, double[] derivative) {
		double[] rho = new double[historySize()];
		double[] alpha = new double[historySize()];
		double[] right = a.copy(derivative);
		// loop last backward
		for (int i = historySize() - 1; i >= 0; i--) {
			double[] inputDifference = getInputDifference(i);
			double[] derivativeDifference = getDerivativeDifference(i);
			rho[i] = a.innerProd(inputDifference, derivativeDifference);
			if (rho[i] == 0.0) throw new RuntimeException("LBFGSMinimizer.implicitMultiply: Curvature problem.");
			alpha[i] = a.innerProd(inputDifference, right) / rho[i];
			right = a.comb(right, 1.0, derivativeDifference, -1.0 * alpha[i]);
		}
		double[] left = a.pointwiseMult(initialInverseHessianDiagonal, right);
		for (int i = 0; i < historySize(); i++) {
			double[] inputDifference = getInputDifference(i);
			double[] derivativeDifference = getDerivativeDifference(i);
			double beta = a.innerProd(derivativeDifference, left) / rho[i];
			left = a.comb(left, 1.0, inputDifference, alpha[i] - beta);
		}
		return left;
	}

	private double[] getInitialInverseHessianDiagonal(DifferentiableFunction function, int dim) {
		double scale = 1.0;
		if (derivativeDifferenceVectorList.size() >= 1) {
			double[] lastDerivativeDifference = getLastDerivativeDifference();
			double[] lastInputDifference = getLastInputDifference();
			double num = a.innerProd(lastDerivativeDifference, lastInputDifference);
			double den = a.innerProd(lastDerivativeDifference, lastDerivativeDifference);
			scale = num / den;
		}
		double[] result = new double[dim];
		Arrays.fill(result, scale);
		return result;	
	}
	
	public static void main(String[] args) {
		DifferentiableFunction function = new DifferentiableFunction() {
			public Pair<Double, double[]> calculate(double[] x) {
				return Pair.makePair(-a.sum(x) + 2.0 * a.innerProd(x, x), a.comb(a.scale(a.onesDouble(x.length), -1.0), 1.0, a.scale(x, 2.0 * 2.0), 1.0));
			}
		};
		
		Minimizer minimizer = new LBFGSMinimizer(1e-5, 1000);
		minimizer.minimize(function, a.zerosDouble(10), true, null);
	}

}
