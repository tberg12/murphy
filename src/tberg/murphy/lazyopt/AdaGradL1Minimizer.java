package tberg.murphy.lazyopt;

import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;

import tberg.murphy.arrays.a;
import tberg.murphy.counter.CounterInterface;
import tberg.murphy.counter.IntCounter;
import tberg.murphy.tuple.Pair;

public class AdaGradL1Minimizer implements OnlineMinimizer {
  
  float eta;
  float delta;
  float regConstant;
  int epochs;
  float r;

  private class LazyAdaGradResult implements CounterInterface<Integer> {
    
    int[] current = null;
    int[] lastUpdate = null;
    float[] x = null;
    float[] sqrGradSum = null;
    
    public LazyAdaGradResult(int[] current, int[] lastUpdate, float[] x, float[] sqrGradSum) {
      this.current = current;
      this.lastUpdate = lastUpdate;
      this.x = x;
      this.sqrGradSum = sqrGradSum;
    }
    
    public double dotProduct(CounterInterface<Integer> c) {
      throw new Error("Method not implemented.");
    }

    public Iterable<Entry<Integer, Double>> entries() {
      throw new Error("Method not implemented.");
    }

    public double incrementCount(Integer key, double d) {
      throw new Error("Method not implemented.");
    }

    public <T extends Integer> void incrementAll(CounterInterface<T> c, double d) {
      throw new Error("Method not implemented.");
    }

    public <T extends Integer> void incrementAll(CounterInterface<T> newWeights) {
      throw new Error("Method not implemented.");
    }

    public void scale(double d) {
      throw new Error("Method not implemented.");
    }

    public final double getCount(final Integer index) {
      flushShrinkageUpdates(index, current, lastUpdate, x, sqrGradSum, r);
      return (double) x[index];
    }

    public void setCount(Integer k, double d) {
      throw new Error("Method not implemented.");
    }

    public double totalCount() {
      throw new Error("Method not implemented.");
    }

    public final int size() {
      return x.length;
    }

    public Iterable<Integer> keySet() {
      throw new Error("Method not implemented.");
    }
    
  }

  public AdaGradL1Minimizer(double eta, double delta, double regConstant, int epochs) {
    this.eta = (float) eta;
    this.delta = (float) delta;
    this.regConstant = (float) regConstant;
    this.epochs = epochs;
    this.r = (float) (regConstant * eta);
  }

  private static final void flushShrinkageUpdates(final int feat, final int[] current, final int[] lastUpdate, final float[] x, final float[] sqrGradSum, final float r) {
    final int dt = current[0] - lastUpdate[feat];
    if (dt > 0) {
      final float s = (float) Math.sqrt(sqrGradSum[feat]);
      final float change = dt * r / s;
      final float w = x[feat];

      if (w < 0.0 && w < -change) x[feat] = w + change;
      else if (w > 0.0 && w > change) x[feat] = w - change;
      else x[feat] = 0.0f;

      lastUpdate[feat] = current[0];
    }
  }

  public CounterInterface<Integer> minimize(List<DifferentiableFunction> functions, float[] initial, boolean verbose, Callback iterCallbackFunction) {
    Random rand = new Random(0);
    
    int[] current = new int[] {0};
    float[] x = a.copy(initial);
    int[] lastUpdate = new int[x.length];
    float[] sqrGradSum = new float[x.length];
    a.addi(sqrGradSum, (float) delta);
    
    LazyAdaGradResult lazyResult = new LazyAdaGradResult(current, lastUpdate, x, sqrGradSum);
    
    for (int epoch=0; epoch<epochs; ++epoch) {
      float epochValSum = 0.0f;
      for (int funcIndex : a.shuffle(a.enumerate(0, functions.size()), rand)) {
        DifferentiableFunction func = functions.get(funcIndex);
        Pair<Double,CounterInterface<Integer>> valAndGrad = func.calculate(lazyResult);
        epochValSum += valAndGrad.getFirst();
        CounterInterface<Integer> grad = valAndGrad.getSecond();
      
        for (Map.Entry<Integer,Double> entry : grad.entries()) {
          final int index = entry.getKey();
          final double gradVal = entry.getValue();
          sqrGradSum[index] += gradVal * gradVal;
        }
        
        for (Map.Entry<Integer,Double> entry : grad.entries()) {
          final int index = entry.getKey();
          flushShrinkageUpdates(index, current, lastUpdate, x, sqrGradSum, r);
          final float gradVal = entry.getValue().floatValue();
          float s = (float) Math.sqrt(sqrGradSum[index]);
          x[index] += -(eta/s) * gradVal;
        }
        
        current[0] += 1;
      }
      
      if (verbose || iterCallbackFunction != null) {
        float[] result = new float[initial.length];
        for (int i=0; i<result.length; ++i) {
          result[i] = (float) lazyResult.getCount(i);
        }
        double funcVal = epochValSum + regConstant * a.sum(a.abs(result));
        if (verbose) System.out.println(String.format("[AdaGradMinimizer.minimize] Epoch %d ended with value %.6f", epoch, funcVal));
        if (iterCallbackFunction != null) iterCallbackFunction.callback(x, epoch, funcVal);
      }
    }
    
    double[] result = new double[initial.length];
    for (int i=0; i<result.length; ++i) {
      result[i] = lazyResult.getCount(i);
    }
    return IntCounter.wrapArray(result, result.length);
  }
}
