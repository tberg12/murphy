package structpred;

import java.util.ArrayList;
import java.util.List;

import util.Maxer;
import counter.CounterInterface;
import counter.IntCounter;

public class OneSlackSVMLearner<T> extends NSlackSVMLearner<T> {

	private List<List<UpdateBundle>> cache;

	public OneSlackSVMLearner(double C, double epsilon) {
		this(C, epsilon, new SvmOpts());
	}

	public OneSlackSVMLearner(double C, double epsilon, SvmOpts svmOpts) {
		super(C, epsilon, svmOpts);
		cache = new ArrayList<List<UpdateBundle>>();
	}

	@Override
	public int reapConstraints(boolean initial, LossAugmentedLinearModel<T> model, List<T> data, CounterInterface<Integer> weights, int currStart,
		int numMiniBatches, int currMiniBatchIndex, int totalDataSize) {
		opts.minDecodeToSmoTimeRatio = 0.0;
		opts.smoMiniBatch = false;
		if (initial) {
			clearConstraints(numMiniBatches);
			for (int i = 0; i < numMiniBatches; ++i) {
				addConstraint(i, new IntCounter(), 0.0);
			}

		}
		int numAddedFromCache = tryCache(data, weights, currStart, currMiniBatchIndex);

		if (numAddedFromCache > 0) {
			System.out.printf("Using cache\n");
			return numAddedFromCache;
		}

		List<UpdateBundle> ubs = batchLossAugmentedDecode(model, data, weights, 1.0);

		IntCounter avgDelta = new IntCounter();
		double avgLoss = 0.0;
		for (int i = currStart; i < currStart + data.size(); ++i) {
			UpdateBundle ub = ubs.get(i - currStart);
			if (opts.oneSlackCacheSize > 0) {
				final List<UpdateBundle> currCache = cache.get(i);
				currCache.add(ub);
				while (currCache.size() > opts.oneSlackCacheSize) {
					currCache.remove(0);
				}
			}
			CounterInterface<Integer> delta = new IntCounter();
			delta.incrementAll(ub.gold);
			delta.incrementAll(ub.guess, -1.0);
			double loss = ub.loss;
			avgDelta.incrementAll(delta);
			avgLoss += loss;
		}
		avgDelta.scale(1.0 / data.size());
		avgLoss /= data.size();

		int numAdded = addConstraintIfNecessary(weights, currMiniBatchIndex, avgDelta, avgLoss);

		return numAdded;
	}

	/**
	 * @param data
	 * @param weights
	 * @param currStart
	 * @param currMiniBatchIndex
	 * @return
	 */
	private int tryCache(List<T> data, CounterInterface<Integer> weights, int currStart, int currMiniBatchIndex) {
		if (opts.oneSlackCacheSize == 0) return 0;
		int numAdded_ = 0;
		boolean added = false;
		while (currStart + data.size() > cache.size()) {
			cache.add(new ArrayList<UpdateBundle>());
			added = true;
		}

		if (!added) {
			IntCounter avgDelta = new IntCounter();
			double avgLoss = 0.0;
			for (int i = currStart; i < currStart + data.size(); ++i) {
				List<UpdateBundle> cached = cache.get(i);
				if (cached.size() < opts.cacheWarmup) continue;
				Maxer<UpdateBundle> maxer = new Maxer<UpdateBundle>();
				for (UpdateBundle ub : cached) {
					CounterInterface<Integer> delta = getDelta(ub.guess, ub.gold);
					maxer.observe(ub, delta.dotProduct(weights) + ub.loss);

				}
				UpdateBundle max = maxer.argMax();
				avgDelta.incrementAll(max.gold);
				avgDelta.incrementAll(max.guess, -1.0);
				avgLoss += max.loss;

			}
			avgDelta.scale(1.0 / data.size());
			avgLoss /= data.size();
			numAdded_ = addConstraintIfNecessary(weights, currMiniBatchIndex, avgDelta, avgLoss);
		}
		return numAdded_;
	}

}
