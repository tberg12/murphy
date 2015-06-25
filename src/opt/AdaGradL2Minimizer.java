package opt;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import arrays.a;
import opt.Minimizer.Callback;
import tuple.Pair;

public class AdaGradL2Minimizer implements OnlineMinimizer {
	
	double eta;
	double delta;
	double regConstant;
	int epochs;

	public AdaGradL2Minimizer(double eta, double delta, double regConstant, int epochs) {
		this.eta = eta;
		this.delta = delta;
		this.regConstant = regConstant;
		this.epochs = epochs;
	}
	
	public double[] minimize(List<DifferentiableFunction> functions, double[] initial, boolean verbose, Callback iterCallbackFunction) {
		Random rand = new Random(0);
		double[] guess = a.copy(initial);
		double[] sqrGradSum = new double[guess.length];
		for (int epoch=0; epoch<epochs; ++epoch) {
			double epochValSum = 0.0;
			double[] epochGradSum = new double[guess.length];
			for (int funcIndex : a.shuffle(a.enumerate(0, functions.size()), rand)) {
				DifferentiableFunction func = functions.get(funcIndex);
				Pair<Double,double[]> valAndGrad = func.calculate(guess);
				epochValSum += valAndGrad.getFirst();
				double[] grad = valAndGrad.getSecond();
				a.combi(epochGradSum, 1.0, grad, 1.0);
			
				double[] sqrGrad = a.sqr(grad);
				a.combi(sqrGradSum, 1.0, sqrGrad, 1.0);
				
				for (int i=0; i<guess.length; ++i) {
					double s = Math.sqrt(sqrGradSum[i]);
					guess[i] = (s * guess[i] - eta * grad[i]) / (eta * regConstant + delta + s);
				}
			}
			if (verbose) System.out.println(String.format("[AdaGradMinimizer.minimize] Epoch %d ended with value %.6f", epoch, epochValSum + regConstant * a.innerProd(guess, guess)));
			if (iterCallbackFunction != null) iterCallbackFunction.callback(guess, epoch, epochValSum, epochGradSum);
		}
		return guess;
	}
	
	public static void main(String[] args) {
		List<DifferentiableFunction> functions = new ArrayList<DifferentiableFunction>();
		functions.add(new DifferentiableFunction() {
			public Pair<Double, double[]> calculate(double[] x) {
				return Pair.makePair(-a.sum(x) + 2.0 * a.innerProd(x, x), a.comb(a.scale(a.onesDouble(x.length), -1.0), 1.0, a.scale(x, 2.0 * 2.0), 1.0));
			}
		});
		functions.add(new DifferentiableFunction() {
			public Pair<Double, double[]> calculate(double[] x) {
				return Pair.makePair(2.0 * a.sum(x) + 2.0 * a.innerProd(x, x), a.comb(a.scale(a.onesDouble(x.length), 2.0), 1.0, a.scale(x, 2.0 * 2.0), 1.0));
			}
		});
		
		final double regConstant = 10.0;
		DifferentiableFunction sumAndReg = new DifferentiableFunction() {
			public Pair<Double, double[]> calculate(double[] x) {
				return Pair.makePair(-a.sum(x) + 2.0 * a.innerProd(x, x) + 2.0 * a.sum(x) + 2.0 * a.innerProd(x, x) + regConstant * a.innerProd(x, x), a.comb(a.scale(x, 2.0 * regConstant), 1.0, a.comb(a.comb(a.scale(a.onesDouble(x.length), -1.0), 1.0, a.scale(x, 2.0 * 2.0), 1.0), 1.0, a.comb(a.scale(a.onesDouble(x.length), 2.0), 1.0, a.scale(x, 2.0 * 2.0), 1.0), 1.0), 1.0));
			}
		};
		EmpiricalGradientTester.test(sumAndReg, a.onesDouble(3), 1e-5, 1.0, 1e-10);
		
		{
			OnlineMinimizer minimizer = new AdaGradL2Minimizer(1e-1, 1e-2, regConstant, 1000);
			minimizer.minimize(functions, a.onesDouble(10), true, null);
		}
		
		{
			Minimizer minimizer = new LBFGSFortranWrapperMinimizer(1e-10, 1000);
			minimizer.minimize(sumAndReg, a.onesDouble(10), true, null);
		}
	}

}
