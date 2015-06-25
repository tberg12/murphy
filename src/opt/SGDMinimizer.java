package opt;

import java.util.List;
import java.util.Random;

import opt.Minimizer.Callback;
import tuple.Pair;
import arrays.a;

public class SGDMinimizer implements OnlineMinimizer {
	
	double startLearningRate;
	double endLearningRate;
	int epochs;
	
	public SGDMinimizer(double startLearningRate, double endLearningRate, int epochs) {
		this.startLearningRate = startLearningRate;
		this.endLearningRate = endLearningRate;
		this.epochs = epochs;
	}
	
	public double[] minimize(List<DifferentiableFunction> functions, double[] initial, boolean verbose, Callback iterCallbackFunction) {
		Random rand = new Random(0);
		double[] guess = a.copy(initial);
		double update = 0;
		for (int epoch=0; epoch<epochs; ++epoch) {
			double valSum = 0.0;
			double[] gradSum = new double[guess.length];
			int[] indices = a.shuffle(a.enumerate(0, functions.size()), rand);
			for (int funcIndex : indices) {
				DifferentiableFunction func = functions.get(funcIndex);
				Pair<Double,double[]> valAndGrad = func.calculate(guess);
				valSum += valAndGrad.getFirst();
				double[] grad = valAndGrad.getSecond();
				double learningRate = startLearningRate + update/(epochs*functions.size()) * (endLearningRate - startLearningRate);
				a.combi(guess, 1.0, grad, -learningRate);
				a.combi(gradSum, 1.0, grad, 1.0);
				update++;
			}
			if (verbose) System.out.println(String.format("[SGDMinimizer.minimize] Epoch %d ended with value %.6f", epoch, valSum));
			if (iterCallbackFunction != null) iterCallbackFunction.callback(guess, epoch, valSum, gradSum);
		}
		return guess;
	}
	
}
