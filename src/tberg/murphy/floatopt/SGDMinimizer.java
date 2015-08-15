package tberg.murphy.floatopt;

import java.util.List;
import java.util.Map;
import java.util.Random;

import tberg.murphy.arrays.a;
import tberg.murphy.counter.CounterInterface;
import tberg.murphy.tuple.Pair;

public class SGDMinimizer implements OnlineMinimizer {
	
	double startLearningRate;
	double endLearningRate;
	int epochs;
	
	public SGDMinimizer(double startLearningRate, double endLearningRate, int epochs) {
		this.startLearningRate = startLearningRate;
		this.endLearningRate = endLearningRate;
		this.epochs = epochs;
	}
	
  public float[] minimize(List<DifferentiableFunction> functions, float[] initial, boolean verbose, Callback iterCallbackFunction) {
    Random rand = new Random(0);
    float[] guess = a.copy(initial);
    double update = 0;
    for (int epoch=0; epoch<epochs; ++epoch) {
      double valSum = 0.0;
      int[] indices = a.shuffle(a.enumerate(0, functions.size()), rand);
      for (int funcIndex : indices) {
        DifferentiableFunction func = functions.get(funcIndex);
        Pair<Double,CounterInterface<Integer>> valAndGrad = func.calculate(guess);
        valSum += valAndGrad.getFirst();
        CounterInterface<Integer> grad = valAndGrad.getSecond();
        double learningRate = startLearningRate + update/(epochs*functions.size()) * (endLearningRate - startLearningRate);
        for (Map.Entry<Integer,Double> entry : grad.entries()) {
          guess[entry.getKey()] += -learningRate * entry.getValue();
        }
        update++;
      }
      if (verbose) System.out.println(String.format("[SGDMinimizer.minimize] Epoch %d ended with value %.6f", epoch, valSum));
      if (iterCallbackFunction != null) iterCallbackFunction.callback(guess, epoch, valSum);
    }
    return guess;
  }

}
