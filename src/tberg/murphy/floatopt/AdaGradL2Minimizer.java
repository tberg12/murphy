package tberg.murphy.floatopt;

import java.util.List;
import java.util.Map;
import java.util.Random;

import tberg.murphy.arrays.a;
import tberg.murphy.counter.CounterInterface;
import tberg.murphy.tuple.Pair;

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

  public float[] minimize(List<DifferentiableFunction> functions, float[] initial, boolean verbose, Callback iterCallbackFunction) {
    Random rand = new Random(0);
    float[] guess = a.copy(initial);
    float[] sqrGradSum = new float[guess.length];
    a.addi(sqrGradSum, (float) delta);
    final float r = (float) (eta * regConstant);
    for (int epoch=0; epoch<epochs; ++epoch) {
      float epochValSum = 0.0f;
      for (int funcIndex : a.shuffle(a.enumerate(0, functions.size()), rand)) {
        DifferentiableFunction func = functions.get(funcIndex);
        Pair<Double,CounterInterface<Integer>> valAndGrad = func.calculate(guess);
        epochValSum += valAndGrad.getFirst();
        CounterInterface<Integer> grad = valAndGrad.getSecond();

        for (Map.Entry<Integer,Double> entry : grad.entries()) {
          final int key = entry.getKey();
          final double val = entry.getValue();
          sqrGradSum[key] += val * val;
        }

        for (int i = 0; i < guess.length; ++i) {
          float s = (float) Math.sqrt(sqrGradSum[i]);
          guess[i] = (s * guess[i]) / (r + s);
        }

        for (Map.Entry<Integer,Double> entry : grad.entries()) {
          final int key = entry.getKey();
          final float val = entry.getValue().floatValue();
          float s = (float) Math.sqrt(sqrGradSum[key]);
          guess[key] += -eta * val / (r + s);
        }
      }
      if (verbose) System.out.println(String.format("[AdaGradMinimizer.minimize] Epoch %d ended with value %.6f", epoch, epochValSum + regConstant * a.innerProd(guess, guess)));
      if (iterCallbackFunction != null) iterCallbackFunction.callback(guess, epoch, epochValSum + regConstant * a.innerProd(guess, guess));
    }
    return guess;
  }

}
