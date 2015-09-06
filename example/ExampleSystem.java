package example;

import java.util.ArrayList;
import java.util.List;

import tberg.murphy.counter.CounterInterface;
import tberg.murphy.counter.IntCounter;
import tberg.murphy.structpred.PrimalSubgradientSVMLearner;
import tberg.murphy.structpred.LossAugmentedLearner;

public class ExampleSystem {
  public static void main(String[] args) {
    double C = 0.0001;
    double delta = 1e-6;
    double stepSize = 0.01;
    boolean l1 = true;
    int batchSize = 1;
    int iters = 2;

    List<DataSample> data = new ArrayList<DataSample>();
    data.add(new DataSample("0 0 0"));
    data.add(new DataSample("0 0 1"));
    data.add(new DataSample("1 1 0"));
    data.add(new DataSample("1 1 1"));
    int numFeatures = data.get(0).features.length * 2;

    CounterInterface<Integer> initWeights = new IntCounter();

    StructpredInterface model = new StructpredInterface(initWeights);

		LossAugmentedLearner<DataSample> trainer = new PrimalSubgradientSVMLearner<DataSample>(C, delta, stepSize, numFeatures, l1, batchSize);

		CounterInterface<Integer> finalWeights = trainer.train(initWeights, model, data, iters);

    System.out.printf("Final weights:\n%s", finalWeights.toString());
  }
}
