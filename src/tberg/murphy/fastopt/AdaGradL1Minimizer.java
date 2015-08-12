package tberg.murphy.fastopt;

import java.util.List;
import java.util.Map;
import java.util.Random;

import tberg.murphy.arrays.a;
import tberg.murphy.counter.CounterInterface;
import tberg.murphy.tuple.Pair;

public class AdaGradL1Minimizer implements OnlineMinimizer {
	
	float eta;
	float delta;
	float regConstant;
	int epochs;

	public AdaGradL1Minimizer(double eta, double delta, double regConstant, int epochs) {
		this.eta = (float) eta;
		this.delta = (float) delta;
		this.regConstant = (float) regConstant;
		this.epochs = epochs;
	}
	
	 public float[] minimize(List<DifferentiableFunction> functions, float[] initial, boolean verbose, Callback iterCallbackFunction) {
	    Random rand = new Random(0);
	    float[] guess = a.copy(initial);
	    float[] sqrGradSum = new float[guess.length];
	    final float r = (float) (regConstant * eta);
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
	        
	        for (Map.Entry<Integer,Double> entry : grad.entries()) {
	          final int key = entry.getKey();
	          final float val = entry.getValue().floatValue();
	          float s = (float) Math.sqrt(sqrGradSum[key]) + delta;
	          guess[key] += -(eta/s) * val;
	        }
	        
	        for (int i=0; i<guess.length; ++i) {
	          final float s = (float) Math.sqrt(sqrGradSum[i]) + delta;
	          final float xHalf = guess[i];
	          final float x = (float) Math.abs(xHalf) - (r/s);
	          if (x > 0) {
	            guess[i] = (xHalf > 0 ? 1.0f : -1.0f) * x;
	          } else {
	            guess[i] = 0.0f;
	          }
	        }
	      }
	      if (verbose) System.out.println(String.format("[AdaGradMinimizer.minimize] Epoch %d ended with value %.6f", epoch, epochValSum + regConstant * a.sum(a.abs(guess))));
	      if (iterCallbackFunction != null) iterCallbackFunction.callback(guess, epoch, epochValSum);
	    }
	    return guess;
	  }

}
