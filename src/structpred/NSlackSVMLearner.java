package structpred;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import tuple.Pair;
import arrays.a;
import counter.CounterInterface;
import counter.IntCounter;

public class NSlackSVMLearner<T> implements LossAugmentedLearner<T> {

	public static class SvmOpts
	{
		public static double precision = 1e-20;

		public static double smoTol = 1e-4;

		public static int expGradIters = 0;

		public static int smoIters = 10000;

		public static int innerSmoIters = 10;

		public static boolean refreshAlphas = false;

		public static double lossAugCheckTol = 0.1;

		public static double newAlphaMag = 0.0;

		public static boolean smoCheckPrimal = true;

		public static int miniBatchSize = 8;

		public static int oneSlackCacheSize = 0;

		public static boolean primalSmo = true;

		public static int cacheWarmup = 5;

		public static double minDecodeToSmoTimeRatio = 1.0;

		public static boolean smoMiniBatch = true;

		public static boolean expGradient = false;

		public static int maxInactiveAlphaCount = Integer.MAX_VALUE;

		static public boolean projGradient = false;

		public static int innerInnerSmoIters = 3;

		public static double minCountThresh = 1e-6;
		
		public static boolean svmVerbose = true;

	}

	private double totalSMOTime;

	private double totalDecodeTime;

	int maxLength;

	double C;

	int N;

	double epsilon;

	List<Integer> activeAlphasPriorityQueue[];

	List<IntCounter>[] indexToDelta;

	List<Double>[] indexToDeltaNormSquared;

	List<Double>[] indexToAlpha;

	List<Double>[] indexToLoss;

	double[][] dotProdCache;

	int[][] alphaRelIndicesCache;

	int[] alphaAbsIndicesCacheInstance;

	int[] alphaAbsIndicesCacheConstraint;

	double[] weights;

	int numDecodes;

	int numFeaturesSoFar;

	int maxFeatureIndexSoFar = 0;;

	Random rand = new Random();

	protected SvmOpts opts;

	private LossAugmentedLinearModel<T> model;

	public NSlackSVMLearner(double C, double epsilon) {
		this(C, epsilon, new SvmOpts());
	}

	public NSlackSVMLearner(double C, double epsilon, SvmOpts opts) {
		this.C = C;
		this.epsilon = epsilon;
		this.opts = opts;
		numFeaturesSoFar = 0;
	}

	public CounterInterface<Integer> train(CounterInterface<Integer> initWeights, LossAugmentedLinearModel<T> model, List<T> data, int maxIters) {
		clearDotProductCache();
		clearIndicesCache();
		numDecodes = 0;
		totalDecodeTime = 0;
		totalSMOTime = 0;

		this.model = model;
		int miniBatchSize = Math.min(data.size(), opts.miniBatchSize);
		weights = toArray(initWeights);
		model.setWeights(wrapCounter(weights, maxFeatureIndexSoFar + 1));
		model.startIteration(0);
		for (int t = 0; t < maxIters; ++t) {
			System.out.println("Iteration " + t);
			int numAdded = 0;
			int currStart = 0;
			do {

				final int currEnd = Math.min(data.size(), currStart + miniBatchSize);
				List<T> currData = data.subList(currStart, currEnd);
				if (opts.svmVerbose) System.out.println("Decoding batch from " + currStart + " to " + currEnd);
				final int numMiniBatches = roundUp(data, miniBatchSize);
				final int currMiniBatchIndex = currStart / miniBatchSize;
				final boolean isFirst = t == 0 && currStart == 0;
				long startDecodeTime = System.currentTimeMillis();
				numAdded += reapConstraints(isFirst, model, currData, wrapCounter(weights, maxFeatureIndexSoFar + 1), currStart,
					numMiniBatches, currMiniBatchIndex, data.size());
				long endDecodeTime = System.currentTimeMillis();

				clearIndicesCache();

				if (opts.refreshAlphas || isFirst) {
					if (opts.expGradient) {
						uniformInitializeAlphas();
					} else {
						zeroInitializeAlphas();
					}
					weights = computeWeights();
					model.setWeights(wrapCounter(weights, maxFeatureIndexSoFar + 1));
				}
				long startSmoTime = System.currentTimeMillis();
				final int numSmoIters = opts.innerSmoIters;
				final int smoStart = opts.smoMiniBatch ? currStart : 0;
				final int smoEnd = opts.smoMiniBatch ? currEnd : indexToAlpha.length;
				final boolean checkConvergence = !opts.smoMiniBatch;
				optimizeDual(numSmoIters, smoStart, smoEnd, checkConvergence);
				long endSmoTime = System.currentTimeMillis();
				currStart += miniBatchSize;

				miniBatchSize = updateMiniBatchSize(miniBatchSize, currData, startDecodeTime, endDecodeTime, startSmoTime, endSmoTime);

				//				if (currentWeights != null) {
				//					if (!opts.primalSmo) {
				//						CounterInterface<Integer> weightsDelta = new IntCounter();
				//						weightsDelta.incrementAll(newWeights);
				//						weightsDelta.incrementAll(currentWeights, -1.0);
				//						System.out.printf("Mag of weights delta: %.8f\n", Math.sqrt(weightsDelta.dotProduct(weightsDelta)));
				//					}
				//					currentWeights = newWeights;
				//				} else {
				//					currentWeights = getWeights();
				//				}

			} while (currStart < data.size());

			System.out.println("Iteration ");

			if (numAdded > 0) {
				optimizeDual(opts.smoIters, 0, indexToAlpha.length, true);
				pruneInactiveAlphas();
				//				Logger.logss("True primal is " + computeTruePrimal(data));
			}

			model.setWeights(wrapCounter(weights, maxFeatureIndexSoFar + 1));
			assert checkWeights();
			model.startIteration(t + 1);
			if (numAdded == 0) break;

			if (opts.svmVerbose) {
				System.out.printf("Num constraints: %d\n", numConstraints());
				System.out.printf("Num decodes so far: %d\n", numDecodes);
				System.out.printf("Total SMO time so far: %f\n", totalSMOTime);
				System.out.printf("Total decode time so far: %f\n", totalDecodeTime);
			}
		}

		System.out.printf("Num decodes: %d\n", numDecodes);
		System.out.printf("Total SMO time: %f\n", totalSMOTime);
		System.out.printf("Total decode time: %f\n", totalDecodeTime);

		final IntCounter wrapCounter = wrapCounter(weights, maxFeatureIndexSoFar + 1);
		model.setWeights(wrapCounter);
		return wrapCounter;
	}

	private boolean checkWeights() {
		double[] computeWeights = computeWeights();
		for (int i = 0; i < maxFeatureIndexSoFar + 1; ++i) {
			assert Math.abs(computeWeights[i] - weights[i]) < 1e-3;
		}
		return true;
	}

	/**
	 * @param numSmoIters
	 * @param checkConvergence
	 * @return
	 */
	private void optimizeDual(final int numSmoIters, int start, int end, boolean checkConvergence) {
		if (numSmoIters == 0) return;
		if (opts.svmVerbose) if (checkConvergence) System.out.println("Optimizing dual");

		CounterInterface<Integer> newWeights = null;
		assert opts.primalSmo : "Dual SMO is broken";
		//		if (opts.primalSmo) {
		if (checkConvergence) {
			if (opts.projGradient) {
				optimizeDualObjectiveProjectedGradientPrimal(numSmoIters, start, end, checkConvergence);
			} else if (opts.expGradient) {
				optimizeDualObjectiveExponentiatedGradientPrimal(numSmoIters, start, end, checkConvergence);
			} else {
				//			optimizeDualObjectiveAnalytic(numSmoIters, start, end, checkConvergence);
				optimizeDualObjectiveSMOPrimalDual(numSmoIters, start, end, checkConvergence);
			}
		} else {
			optimizeDualObjectiveSMOPrimalDual(numSmoIters, start, end, checkConvergence);
		}
		//		} else {
		//			assert false : "Code is now broken";
		//			buildDotProdCache();
		//			optimizeDualObjectiveSMO();
		////			newWeights = getWeights();
		//
		//			System.out.printf("Primal objective: %.8f\n", getPrimalObjective());
		//			System.out.printf("Dual objective: %.8f\n", getDualObjective());
		//		}
		if (opts.svmVerbose) if (checkConvergence) System.out.println();
		//		return newWeights;
	}

	/**
	 * @param miniBatchSize
	 * @param currData
	 * @param startDecodeTime
	 * @param endDecodeTime
	 * @param startSmoTime
	 * @param endSmoTime
	 * @return
	 */
	private int updateMiniBatchSize(int miniBatchSize, List<T> currData, long startDecodeTime, long endDecodeTime, long startSmoTime, long endSmoTime) {
		final double currSmoTime = (endSmoTime - startSmoTime);
		final double currDecodeTime = endDecodeTime - startDecodeTime;
		totalSMOTime += currSmoTime;
		totalDecodeTime += currDecodeTime;
		double currDecodeToSmoRatio = currDecodeTime / currSmoTime;
		if (currDecodeToSmoRatio < opts.minDecodeToSmoTimeRatio) {
			double avgDecodeTime = currDecodeTime / currData.size();
			miniBatchSize = (int) Math.round(opts.minDecodeToSmoTimeRatio * currSmoTime / avgDecodeTime);
		}
		return miniBatchSize;
	}

	public double getDualObjective(double wNormSquared) {
		double obj = 0.0;
		obj += -0.5 * wNormSquared;
		for (int i = 0; i < indexToAlpha.length; ++i) {
			final int numConstraintsI = numConstraints(i);
			for (int yi = 0; yi < numConstraintsI; ++yi) {
				obj += (C / indexToAlpha.length) * indexToAlpha[i].get(yi) * indexToLoss[i].get(yi);
			}
		}
		return obj;
	}

	public double getDualObjectiveChange(int i, double[] alphas, double[] newAlphas, double[][] dotProdCaches) {
		double obj = 0.0;

		for (int yi = 0; yi < alphas.length; ++yi) {
			double ti = C / indexToAlpha.length * (newAlphas[yi] - alphas[yi]);
			obj -= 0.5 * indexToDeltaNormSquared[i].get(yi) * ti * ti;
			obj -= 0.5 * 2.0 * indexToDelta[i].get(yi).dotProduct(weights) * ti;
			for (int yj = yi + 1; yj < alphas.length; ++yj) {
				double tj = C / indexToAlpha.length * (newAlphas[yj] - alphas[yj]);
				obj -= 0.5 * 2.0 * dotProdCaches[yi][yj] * ti * tj;
			}
		}

		final int numConstraintsI = numConstraints(i);
		for (int yi = 0; yi < numConstraintsI; ++yi) {
			obj += (C / indexToAlpha.length) * (newAlphas[yi] - alphas[yi]) * indexToLoss[i].get(yi);
		}
		return obj;
	}

	private void clearIndicesCache() {
		alphaRelIndicesCache = null;
		alphaAbsIndicesCacheInstance = null;
		alphaAbsIndicesCacheConstraint = null;
	}

	private void clearDotProductCache() {
		dotProdCache = null;
	}

	protected static CounterInterface<Integer> getDelta(CounterInterface<Integer> gold, CounterInterface<Integer> guess) {
		CounterInterface<Integer> delta = new IntCounter();
		delta.incrementAll(gold);
		delta.incrementAll(guess, -1.0);
		return delta;
	}

	/**
	 * @param data
	 * @param miniBatchSize
	 * @return
	 */
	private int roundUp(List<T> data, int miniBatchSize) {
		return data.size() / miniBatchSize + (data.size() % miniBatchSize == 0 ? 0 : 1);
	}

	Pair<Integer, Integer> getAlphaRelativeIndicesFromAbsoluteIndex(final int absIndex_) {
		if (alphaAbsIndicesCacheInstance == null) {
			alphaAbsIndicesCacheInstance = new int[numConstraints()];
			alphaAbsIndicesCacheConstraint = new int[numConstraints()];
			Arrays.fill(alphaAbsIndicesCacheInstance, -1);
		}
		if (alphaAbsIndicesCacheInstance[absIndex_] < 0) {
			int i = 0;
			int absIndex = absIndex_;
			while (absIndex >= numConstraints(i)) {
				absIndex -= numConstraints(i);
				i++;
			}
			alphaAbsIndicesCacheInstance[absIndex_] = i;
			alphaAbsIndicesCacheConstraint[absIndex_] = absIndex;
		}
		return Pair.makePair(alphaAbsIndicesCacheInstance[absIndex_], alphaAbsIndicesCacheConstraint[absIndex_]);
	}

	int getAlphaAbsoluteIndexFromRelativeIndices(int i, int yi) {
		if (alphaRelIndicesCache == null) {
			alphaRelIndicesCache = new int[indexToAlpha.length][];
			for (int j = 0; j < indexToAlpha.length; ++j) {
				final int[] js = new int[numConstraints(j)];
				Arrays.fill(js, -1);
				alphaRelIndicesCache[j] = js;
			}

		}
		if (alphaRelIndicesCache[i][yi] < 0) {
			int absoluteIndex = 0;
			for (int j = 0; j < i; ++j) {
				absoluteIndex += numConstraints(j);
			}
			absoluteIndex += yi;
			alphaRelIndicesCache[i][yi] = absoluteIndex;
		}
		return alphaRelIndicesCache[i][yi];

	}

	public double getPrimalObjective() {
		//		double[] weights = toArray(getWeights());
		//		double[] weights = getWeights();

		return getPrimalObjective(weights, a.innerProd(weights, weights));
	}

	private double[] toArray(CounterInterface<Integer> weights) {
		double[] array = new double[maxFeatureIndexSoFar + 1];
		//		IntCounter.incrementDenseArray(array, weights, 1.0);
		int maxKey = weights.size();
		for (Entry<Integer, Double> entry : weights.entries()) {
			if (entry.getKey() >= array.length) {
				final int currKey = entry.getKey() + 1;
				maxKey = Math.max(currKey, maxKey);
				array = Arrays.copyOf(array, Math.max(currKey, array.length * 3 / 2));
			}
			array[entry.getKey()] = entry.getValue();
		}
		return array;//Arrays.copyOf(array, maxKey);
	}

	/**
	 * @param weights
	 * @return
	 */
	private double getPrimalObjective(double[] weights, double weightNormSquared) {
		double obj = 0.0;
		obj += 0.5 * weightNormSquared;
		double w2 = 0.5 * weightNormSquared;

		for (int i = 0; i < indexToAlpha.length; ++i) {
			double slack = 0.0;
			for (int yi = 0; yi < numConstraints(i); ++yi) {
				slack = Math.max(slack, getContraintSlack(i, yi, weights));
			}
			obj += (C / indexToAlpha.length) * slack;
		}
		return obj;
	}

	public double[] computeWeights() {
		IntCounter w = new IntCounter(numFeaturesSoFar);
		for (int i = 0; i < indexToAlpha.length; ++i) {
			final double currC = C / indexToAlpha.length;
			for (int yi = 0; yi < numConstraints(i); ++yi) {
				final double currAlpha = indexToAlpha[i].get(yi);
				assert currAlpha >= -1e-4;
				assert currAlpha <= 1 + 1e-4;
				for (Map.Entry<Integer, Double> entry : indexToDelta[i].get(yi).entries()) {
					maxFeatureIndexSoFar = Math.max(entry.getKey(), maxFeatureIndexSoFar);
					if (entry.getValue() != 0.0) w.incrementCount(entry.getKey(), currC * currAlpha * entry.getValue());
				}
			}
		}
		numFeaturesSoFar = Math.max(numFeaturesSoFar, w.size());
		return toArray(w);
	}

	double getContraintSlack(int i, int yi, CounterInterface<Integer> weights) {
		return Math.max(0.0, indexToLoss[i].get(yi) - indexToDelta[i].get(yi).dotProduct(weights));
	}

	double getContraintSlack(int i, int yi, double[] weights) {
		return Math.max(0.0, indexToLoss[i].get(yi) - indexToDelta[i].get(yi).dotProduct(weights));
	}

	double getContraintSlack(CounterInterface<Integer> weights, CounterInterface<Integer> delta, double loss) {
		return Math.max(loss - delta.dotProduct(weights), 0.0);
	}

	public double getDualObjective() {
		double obj = 0.0;
		final double C_norm = C / indexToAlpha.length;
		for (int i = 0; i < indexToAlpha.length; ++i) {
			final int numConstraintsI = numConstraints(i);
			for (int yi = 0; yi < numConstraintsI; ++yi) {
				final double[] dotProdCacheI = dotProdCache[getAlphaAbsoluteIndexFromRelativeIndices(i, yi)];
				final double alphaI = indexToAlpha[i].get(yi);
				obj += C_norm * alphaI * indexToLoss[i].get(yi);
				final double mult = 0.5 * C_norm * C_norm * alphaI;
				for (int j = 0; j < indexToAlpha.length; ++j) {
					final int numConstraintsJ = numConstraints(j);
					final List<Double> currAlphaJs = indexToAlpha[j];
					for (int yj = 0; yj < numConstraintsJ; ++yj) {
						final double alphaJ = currAlphaJs.get(yj);
						obj -= mult * alphaJ * dotProdCacheI[getAlphaAbsoluteIndexFromRelativeIndices(j, yj)];
					}
				}
			}
		}
		return obj;
	}

	double[] getDualGradient(int i) {
		double[] grad = new double[numConstraints(i)];
		for (int yi = 0; yi < numConstraints(i); ++yi) {
			grad[yi] = (C / indexToAlpha.length) * indexToLoss[i].get(yi);
			for (int j = 0; j < indexToAlpha.length; ++j) {
				for (int yj = 0; yj < numConstraints(j); ++yj) {
					grad[yi] -= (C / indexToAlpha.length) * (C / indexToAlpha.length) * indexToAlpha[j].get(yj)
						* dotProdCache[getAlphaAbsoluteIndexFromRelativeIndices(i, yi)][getAlphaAbsoluteIndexFromRelativeIndices(j, yj)];
				}
			}
		}
		return grad;
	}

	double[] getDualGradient(int i, double[] weights) {
		double[] grad = new double[numConstraints(i)];
		for (int yi = 0; yi < numConstraints(i); ++yi) {
			grad[yi] = (C / indexToAlpha.length) * indexToLoss[i].get(yi);
			grad[yi] -= (C / indexToAlpha.length) * indexToDelta[i].get(yi).dotProduct(weights);
			//			for (int j = 0; j < indexToAlpha.length; ++j) {
			//				for (int yj = 0; yj < numConstraints(j); ++yj) {
			//					grad[yi] -= (C / indexToAlpha.length) * (C / indexToAlpha.length) * indexToAlpha[j].get(yj)
			//						* dotProdCache[getAlphaAbsoluteIndexFromRelativeIndices(i, yi)][getAlphaAbsoluteIndexFromRelativeIndices(j, yj)];
			//				}
			//			}
		}
		return grad;
	}

	public void optimizeDualObjectiveExponentiatedGradient() {
		double[] stepSizes = new double[indexToAlpha.length];
		Arrays.fill(stepSizes, 0.1);
		for (int iter = 1; iter <= opts.expGradIters; iter++) {
			double objective = getDualObjective();
			for (int i = 0; i < indexToAlpha.length; ++i) {
				double[] alphas = getAlphas(i);
				double[] grad = getDualGradient(i);
				double newObjective = Double.NEGATIVE_INFINITY;
				while (true) {
					double[] direction = a.scale(grad, stepSizes[i]);
					double[] scale = a.exp(direction);
					double[] newAlphas = a.pointwiseMult(alphas, scale);
					normalize(newAlphas);
					setAlphas(i, newAlphas);
					newObjective = getDualObjective();
					boolean isFinite = !Double.isInfinite(newObjective) && !Double.isNaN(newObjective);
					if (isFinite && newObjective >= objective - opts.precision) {
						stepSizes[i] *= 1.1;
					} else {
						stepSizes[i] *= 0.5;
					}
					if (isFinite) break;
				}
				objective = newObjective;
			}
			if (iter == 1 || iter % 500 == 0 || iter == opts.expGradIters) System.out.printf(String.format("[ExpGrad] Iter %d: %.8f\n", iter, objective));
		}
	}

	public void optimizeDualObjectiveExponentiatedGradientPrimal(int maxIters, int start, int end, boolean checkConvergence) {

		double[] stepSizes = new double[indexToAlpha.length];
		Arrays.fill(stepSizes, 1.0);
		IntCounter deltaScratch = new IntCounter();
		for (int iter = 1; iter <= maxIters; iter++) {
			double weightsNormSquared = a.innerProd(weights, weights);

			double objective = getDualObjective(weightsNormSquared);
			for (int i = start; i < end; ++i) {
				double[] alphas = getAlphas(i);
				double[] grad = getDualGradient(i, weights);
				double newObjective = Double.NEGATIVE_INFINITY;
				double newWeightsNormSquared = Double.NaN;
				while (true) {
					double[] direction = a.scale(grad, stepSizes[i]);
					double maxDirection = a.max(direction);
					a.addi(direction, -maxDirection);

					double[] scale = a.exp(direction);

					double[] newAlphas = a.pointwiseMult(alphas, scale);
					normalize(newAlphas);
					for (int yi = 0; yi < newAlphas.length; ++yi) {
						newAlphas[yi] = Math.max(newAlphas[yi], Double.MIN_VALUE);
					}
					if (a.hasnan(newAlphas) || a.hasinf(newAlphas)) {
						stepSizes[i] *= 0.2;
						continue;
					}
					deltaScratch.clear();
					IntCounter weightsDelta = setAlphas(i, alphas, newAlphas, deltaScratch);
					if (checkConvergence) newWeightsNormSquared = weightsNormSquared + 2.0 * weightsDelta.dotProduct(weights) + weightsDelta.normSquared();
					if (checkConvergence) newObjective = getDualObjective(newWeightsNormSquared);

					boolean isFinite = !checkConvergence || !Double.isInfinite(newObjective) && !Double.isNaN(newObjective);
					if (isFinite && newObjective >= objective) {
						if (stepSizes[i] < 1e5) stepSizes[i] *= 2.0;
					} else {
						stepSizes[i] *= 0.2;
						if (stepSizes[i] == 0) break;
						continue;
					}
					incrementAll(weights, weightsDelta, 1.0);
					break;
				}
				objective = newObjective;
				weightsNormSquared = newWeightsNormSquared;
			}
			if (checkConvergence) {
				double primalObjective = getPrimalObjective(weights, weightsNormSquared);
				if (converged(primalObjective, objective, opts.smoTol, Double.NaN)) {
					break;
				}
				System.out.printf(String.format("[ExpGrad]  Iter %d: %.8f, %.8f\n", iter, objective, primalObjective));
			}
		}

	}

	public void optimizeDualObjectiveProjectedGradientPrimal(int maxIters, int start, int end, boolean checkConvergence) {

		if (checkConvergence) {
			System.out.println("Building delta cache");
		}
		double[][][] dotProdCaches = new double[end - start][][];
		for (int i = start; i < end; ++i) {
			final int numConstraints = numConstraints(i);
			dotProdCaches[i - start] = new double[numConstraints][numConstraints];

			for (int yi = 0; yi < numConstraints; ++yi) {
				IntCounter delta_yi = indexToDelta[i].get(yi);
				for (int yj = yi + 1; yj < numConstraints; ++yj) {

					IntCounter delta_yj = indexToDelta[i].get(yj);
					dotProdCaches[i - start][yi][yj] = delta_yi.dotProduct(delta_yj);

				}
			}
		}

		double[] stepSizes = new double[indexToAlpha.length];
		Arrays.fill(stepSizes, 1.0);
		for (int iter = 1; iter <= maxIters; iter++) {
			double weightsNormSquared = a.innerProd(weights, weights);

			double objective = getDualObjective(weightsNormSquared);
			for (int i = start; i < end; ++i) {
				double[] alphas = getAlphas(i);
				double[] grad = getDualGradient(i, weights);
				double objectiveChange = 0.0;
				while (true) {
					double[] newAlphas = projectToSimplex(toList(a.comb(alphas, 1.0, grad, stepSizes[i])));
					if (Arrays.equals(alphas, newAlphas)) break;
					objectiveChange = getDualObjectiveChange(i, alphas, newAlphas, dotProdCaches[i]);

					//					deltaScratch.clear();

					//					double weightsNormChange = 2.0 * weightsDelta.dotProduct(weights) + weightsDelta.normSquared();

					if (objectiveChange == 0.0)
						break;
					else if (objectiveChange > 0.0) {
						if (stepSizes[i] < 1e10) stepSizes[i] *= 1.2;
					} else {
						stepSizes[i] *= 0.5;
						if (stepSizes[i] == 0) break;
						continue;
					}
					for (int yi = 0; yi < newAlphas.length; ++yi) {
						indexToAlpha[i].set(yi, newAlphas[yi]);
						updateWeights(indexToDelta[i].get(yi), C / indexToAlpha.length * (newAlphas[yi] - alphas[yi]));
						//						incrementAll(weights, weightsDelta, 1.0);
					}
					break;
				}
				objective += objectiveChange;
				//				assert NumUtils.approxEquals(objective, getDualObjective(ArrayUtil.normSquared(weights)), 1e-4);
			}
			if (checkConvergence) {
				double primalObjective = getPrimalObjective(weights, a.innerProd(weights, weights));
				if (converged(primalObjective, objective, opts.smoTol, Double.NaN)) {
					System.out.printf(String.format("[ProjGrad]  Final Iter %d: %.8f, %.8f\n", iter, objective, primalObjective));
					break;
				}
				System.out.printf(String.format("[ProjGrad]  Iter %d: %.8f, %.8f\n", iter, objective, primalObjective));
			}
		}

	}

	private static List<Double> toList(double[] a) {
		List<Double> list = new ArrayList<Double>(a.length);
		for (double d : a)
			list.add(d);
		return list;
	}

	static void normalize(double[] vect) {
		double norm = 0.0;
		for (double val : vect)
			norm += val;
		for (int i = 0; i < vect.length; ++i) {
			if (norm > 0) vect[i] /= norm;
		}
	}

	void setAlphas(int i, double[] alphas) {
		for (int yi = 0; yi < numConstraints(i); ++yi) {
			indexToAlpha[i].set(yi, alphas[yi]);
		}
	}

	IntCounter setAlphas(int i, double[] oldAlphas, double[] alphas, IntCounter weightsDelta) {
		for (int yi = 0; yi < numConstraints(i); ++yi) {
			double step = alphas[yi] - oldAlphas[yi];
			if (step == 0.0) continue;
			indexToAlpha[i].set(yi, alphas[yi]);
			final double d = step * C / indexToAlpha.length;

			weightsDelta.ensureCapacity(weightsDelta.size() + indexToDelta[i].get(yi).size());
			weightsDelta.incrementAll(indexToDelta[i].get(yi), d);
		}
		return weightsDelta;
	}

	double[] getAlphas(int i) {
		double[] alphas = new double[numConstraints(i)];
		for (int yi = 0; yi < numConstraints(i); ++yi) {
			alphas[yi] = indexToAlpha[i].get(yi);
		}
		return alphas;
	}

	public void optimizeDualObjectiveSMO() {
		double lastDual = Double.NEGATIVE_INFINITY;
		for (int iter = 1; iter <= opts.smoIters; ++iter) {
			for (int i = 0; i < indexToAlpha.length; ++i) {
				for (int yi = 0; yi < numConstraints(i); ++yi) {
					for (int yj = 0; yj < numConstraints(i); ++yj) {
						if (yi != yj) updateAlphas(i, yi, yj);
					}
				}
			}
			double dual = getDualObjective();
			double primal = opts.smoCheckPrimal ? getPrimalObjective() : Double.NaN;
			if (iter == 1 || iter % 200 == 0 || converged(primal, dual, opts.smoTol, lastDual) || iter == opts.smoIters)
				System.out.printf("[SMO] Round %d: %.8f\n", iter, dual);
			if (converged(primal, dual, opts.smoTol, lastDual)) break;
			lastDual = dual;
		}
	}

	private double computeTruePrimal(List<T> data) {
		double w2 = 0.5 * a.innerProd(weights, weights);
		double obj = w2;
		List<UpdateBundle> ubs = model.getLossAugmentedUpdateBundleBatch(data, 1.0);
		for (UpdateBundle b : ubs) {
			IntCounter delta = new IntCounter();
			delta.incrementAll(b.gold);
			delta.incrementAll(b.guess, -1.0);
			obj += getContraintSlack(IntCounter.wrapArray(weights, maxFeatureIndexSoFar + 1), delta, b.loss);
		}
		return obj;
	}

	public void optimizeDualObjectiveSMOPrimalDual(int maxIters, int start, int end, boolean checkConvergence) {
		if (checkConvergence) {

			if (opts.svmVerbose) System.out.println("Building delta cache");
		}
		double[][][] deltaDeltaNormCache = new double[end - start][][];
		double[][][] deltaDeltaDotProductCache = new double[end - start][][];
		for (int i = start; i < end; ++i) {
			final int numConstraints = numConstraints(i);
			deltaDeltaNormCache[i - start] = new double[numConstraints][numConstraints];
			deltaDeltaDotProductCache[i - start] = new double[numConstraints][numConstraints];

			for (int yi = 0; yi < numConstraints; ++yi) {
				final double delta_yi_sq = indexToDeltaNormSquared[i].get(yi);
				IntCounter delta_yi = indexToDelta[i].get(yi);
				for (int yj = yi + 1; yj < numConstraints; ++yj) {

					final double delta_yj_sq = indexToDeltaNormSquared[i].get(yj);
					IntCounter delta_yj = indexToDelta[i].get(yj);
					//							IntCounter delta = new IntCounter();
					//							delta.incrementAll(indexToDelta[i].get(yi));
					//							delta.incrementAll(indexToDelta[i].get(yj), -1.0);
					//							IntCounter delta_ = new IntCounter();
					//							delta_.incrementAll(delta);
					final double dotProd = delta_yi.dotProduct(delta_yj);
					deltaDeltaDotProductCache[i - start][yi][yj] = dotProd;
					deltaDeltaNormCache[i - start][yi][yj] = delta_yi_sq - 2.0 * dotProd + delta_yj_sq;

				}
			}
		}

		double lastDual = Double.NEGATIVE_INFINITY;

		for (int iter = 1; iter <= maxIters; ++iter) {
			boolean weightsChanged = false;

			for (int i : shuffle(start, end)) {
				final int currNumConstraints = numConstraints(i);
				final double[][] deltaDeltaNormCacheHere = deltaDeltaNormCache[i - start];
				final double[][] deltaDeltaDotProductCacheHere = deltaDeltaDotProductCache[i - start];
				//				INNER: for (int yi = 0; yi < 1/* currNumConstraints */; ++yi) {
				//					for (int yj = 1/* 0 */; yj < currNumConstraints; ++yj) {
				final List<Integer> shuffleOuter = shuffle(0, currNumConstraints);
				double[] deltWeightsDotProductCache = new double[currNumConstraints];
				for (int yi = 0; yi < currNumConstraints; ++yi) {
					deltWeightsDotProductCache[yi] = indexToDelta[i].get(yi).dotProduct(weights);
				}

				for (int innerIter = 0; innerIter < opts.innerInnerSmoIters; ++innerIter) {
					boolean innerWeightsChanged = false;
					final List<Integer> shuffleInner = shuffle(0, currNumConstraints);
					for (int yi : shuffleOuter) {
						for (int yj : shuffleInner) {
							if (yi < yj) {
								innerWeightsChanged |= updateAlphasPrimal(i, yi, yj, deltWeightsDotProductCache, deltaDeltaNormCacheHere,
									deltaDeltaDotProductCacheHere);
								//							if (weightsChanged) continue INNER;

							}
						}
					}
					if (!innerWeightsChanged)
						break;
					else
						weightsChanged = true;
				}
			}
			if (checkConvergence) {
				double wNormSquared = a.innerProd(weights, weights);//weights.dotProduct(weights);
				double dual = getDualObjective(wNormSquared);

				double primal = opts.smoCheckPrimal ? getPrimalObjective(weights, wNormSquared) : Double.NaN;
				if (opts.svmVerbose) System.out.printf("[SMO] Round %d: %.8f, %.8f\n", iter, dual, primal);

				if (converged(primal, dual, opts.smoTol, lastDual) || iter == maxIters) {
					if (opts.svmVerbose) System.out.printf("[SMO] Final Round %d: %.8f, %.8f\n", iter, dual, primal);
					break;
				}
				lastDual = dual;
			} else {
				if (!weightsChanged) {//
					break;
				}
			}
		}
	}

	//	public void optimizeDualObjectiveAnalytic(int maxIters, int start, int end, boolean checkConvergence) {
	//		//		if (checkConvergence) {
	//		//
	//		//			System.out.println("Building delta cache");
	//		//		}
	//		//		double[][][] deltaDeltaNormCache = new double[end - start][][];
	//		//		for (int i = start; i < end; ++i) {
	//		//			final int numConstraints = numConstraints(i);
	//		//			deltaDeltaNormCache[i - start] = new double[numConstraints][numConstraints];
	//		//
	//		//			for (int yi = 0; yi < numConstraints; ++yi) {
	//		//				final double delta_yi_sq = indexToDeltaNormSquared[i].get(yi);
	//		//				IntCounter delta_yi = indexToDelta[i].get(yi);
	//		//				for (int yj = yi + 1; yj < numConstraints; ++yj) {
	//		//
	//		//					final double delta_yj_sq = indexToDeltaNormSquared[i].get(yj);
	//		//					IntCounter delta_yj = indexToDelta[i].get(yj);
	//		//					//							IntCounter delta = new IntCounter();
	//		//					//							delta.incrementAll(indexToDelta[i].get(yi));
	//		//					//							delta.incrementAll(indexToDelta[i].get(yj), -1.0);
	//		//					//							IntCounter delta_ = new IntCounter();
	//		//					//							delta_.incrementAll(delta);
	//		//					deltaDeltaNormCache[i - start][yi][yj] = delta_yi_sq - 2.0 * delta_yi.dotProduct(delta_yj) + delta_yj_sq;
	//		//
	//		//				}
	//		//			}
	//		//		}
	//		//
	//		//		if (checkConvergence) Logger.endTrack();
	//		double lastDual = Double.NEGATIVE_INFINITY;
	//
	//		for (int iter = 1; iter <= maxIters; ++iter) {
	//			boolean weightsChanged = false;
	//			for (int i : shuffle(start, end)) {
	//				final int currNumConstraints = numConstraints(i);
	//				//				final double[][] deltaDeltaNormCacheHere = deltaDeltaNormCache[i - start];
	//				//				double[] cache = new double[currNumConstraints];
	//				//				Arrays.fill(cache, Double.NaN);
	//				//				INNER: for (int yi = 0; yi < 1/* currNumConstraints */; ++yi) {
	//				//					for (int yj = 1/* 0 */; yj < currNumConstraints; ++yj) {
	//				//				final List<Integer> shuffleOuter = shuffle(0, currNumConstraints);
	//				//				for (int yi : shuffleOuter) {
	//				//					final List<Integer> shuffleInner = shuffle(yi + 1, currNumConstraints);
	//				//					for (int yj : shuffleInner) {
	//				//						if (yi < yj) {
	//				weightsChanged |= updateAlphasAnalytic(i);
	//				//							if (weightsChanged) continue INNER;
	//
	//				//						}
	//				//					}
	//				//				}
	//			}
	//			if (checkConvergence) {
	//				double wNormSquared = ArrayUtil.normSquared(weights);//weights.dotProduct(weights);
	//				double dual = getDualObjective(wNormSquared);
	//
	//				double primal = opts.smoCheckPrimal ? getPrimalObjective(weights, wNormSquared) : Double.NaN;
	//				System.out.printf("[SMO] Round %d: %.8f, %.8f\n", iter, dual, primal);
	//
	//				if (converged(primal, dual, opts.smoTol, lastDual) || iter == maxIters) {
	//					System.out.printf("[SMO] Final Round %d: %.8f, %.8f\n", iter, dual, primal);
	//					break;
	//				}
	//				lastDual = dual;
	//			} else {
	//				if (!weightsChanged) {//
	//					break;
	//				}
	//			}
	//		}
	//	}

	//	private boolean updateAlphasAnalytic(int i) {
	//		double[] oldAlphas = getAlphas(i);
	//		final int currNumConstraints = numConstraints(i);
	//		List<Pair<Integer, Double>> Ds = new ArrayList<Pair<Integer, Double>>(currNumConstraints);
	//		for (int yi = 0; yi < currNumConstraints; ++yi) {
	//			final double deltaNormSquared = indexToDeltaNormSquared[i].get(yi);
	//			Ds.add(Pair.newPair(yi, deltaNormSquared == 0.0 ? 0.0 : (indexToLoss[i].get(yi) - indexToDelta[i].get(yi).dotProduct(weights)) / deltaNormSquared));
	//		}
	//		Collections.sort(Ds, new Comparator<Pair<Integer, Double>>()
	//		{
	//
	//			@Override
	//			public int compare(Pair<Integer, Double> arg0, Pair<Integer, Double> arg1) {
	//				return Double.compare(arg1.getSecond(), arg0.getSecond());
	//			}
	//		});
	//		int r = -1;
	//		double phi = 1.0;
	//		double lastPhi = phi;
	//		while (phi > 0.0) {
	//			r++;
	//			lastPhi = phi;
	//			phi = (r == currNumConstraints - 1) ? 0.0 : (phi - r * (Ds.get(r).getSecond() - Ds.get(r + 1).getSecond()));
	//		}
	//		double theta = Ds.get(r).getSecond() - lastPhi / r;
	//		double[] newAlphas = new double[currNumConstraints];
	//		for (int q = 0; q < currNumConstraints; ++q) {
	//			final int yi = Ds.get(q).getFirst();
	//			final double v = q < r ? theta : Ds.get(q).getSecond();
	//			final double alpha = -v;
	//			newAlphas[yi] = alpha;
	//		}
	//
	//		IntCounter weightsDelta = setAlphas(i, oldAlphas, newAlphas);
	//		incrementAll(weights, weightsDelta, 1.0);
	//
	//		return true;
	//
	//	}

	/**
	 * @param start
	 * @param end
	 * @return
	 */
	private List<Integer> shuffle(int start, int end) {
		List<Integer> result = a.toList(a.enumerate(start, end));
		Collections.shuffle(result);
		return result;
	}

	public static IntCounter toCounter(double[] weights) {
		IntCounter counter = new IntCounter(weights.length);
		for (int i = 0; i < weights.length; ++i) {
			counter.setCount(i, weights[i]);
		}
		return counter;
	}

	public static IntCounter wrapCounter(double[] weights, int size) {
		IntCounter counter = IntCounter.wrapArray(weights, size);
		return counter;
	}

	boolean converged(double primal, double dual, double tol, double lastDual) {
		if (opts.smoCheckPrimal) {
			double valueAverage = (Math.abs(dual) + Math.abs(primal)) / 2.0;
			if (Math.abs(primal - dual) < opts.precision || Math.abs(primal - dual) / valueAverage < tol) return true;
		} else {
			double diff = dual - lastDual;
			if (diff < opts.precision || diff / dual < opts.smoTol) return true;
		}
		return false;
	}

	public void updateAlphas(int i, int yi, int yj) {

		int yiAbs = getAlphaAbsoluteIndexFromRelativeIndices(i, yi);
		int yjAbs = getAlphaAbsoluteIndexFromRelativeIndices(i, yj);
		if (dotProdCache[yiAbs][yiAbs] == 0 && dotProdCache[yjAbs][yjAbs] == 0) return;
		double numerator = indexToLoss[i].get(yi) - indexToLoss[i].get(yj);
		double x = 0.0;
		double y = 0.0;
		for (int k = 0; k < indexToAlpha.length; ++k) {
			for (int yk = 0; yk < numConstraints(k); ++yk) {
				int ykAbs = getAlphaAbsoluteIndexFromRelativeIndices(k, yk);
				x -= (C / indexToAlpha.length) * indexToAlpha[k].get(yk) * dotProdCache[yiAbs][ykAbs];
				y += (C / indexToAlpha.length) * indexToAlpha[k].get(yk) * dotProdCache[yjAbs][ykAbs];
			}
		}
		double denomenator = 0.0;
		numerator += x + y;
		denomenator += (C / indexToAlpha.length) * dotProdCache[yiAbs][yiAbs];
		denomenator -= 2.0 * (C / indexToAlpha.length) * dotProdCache[yiAbs][yjAbs];
		denomenator += (C / indexToAlpha.length) * dotProdCache[yjAbs][yjAbs];
		if (denomenator == 0) return;

		double delta = Math.max(-indexToAlpha[i].get(yi), Math.min(indexToAlpha[i].get(yj), numerator / denomenator));
		indexToAlpha[i].set(yi, indexToAlpha[i].get(yi) + delta);
		indexToAlpha[i].set(yj, indexToAlpha[i].get(yj) - delta);
	}

	public boolean updateAlphasPrimal(int i, int yi, int yj, double[] deltaWeightsDotProdCache, double[][] deltaDeltaNormCache,
		double[][] deltaDeltaDotProductCache) {

		final double alpha_yi = indexToAlpha[i].get(yi);
		final double alpha_yj = indexToAlpha[i].get(yj);
		if (alpha_yi == 0.0 && alpha_yj == 0.0) return false;
		final double C_norm = C / indexToAlpha.length;
		double b = C_norm * deltaDeltaNormCache[yi][yj];
		if (b == 0.0) return false;

		final IntCounter delta_yi = indexToDelta[i].get(yi);
		final IntCounter delta_yj = indexToDelta[i].get(yj);
		final double delta_T_w_yi = deltaWeightsDotProdCache[yi];
		final double delta_T_w_yj = deltaWeightsDotProdCache[yj];
		final double deltaWeightsDotProd = delta_T_w_yi - delta_T_w_yj;
		double t = computeSmoStep(i, yi, yj, b, alpha_yi, alpha_yj, deltaWeightsDotProd);
		if (t == 0) return false;

		final double newAlphaYi = alpha_yi + t;
		final double newAlphaYj = alpha_yj - t;
		assert newAlphaYi >= -1e-7;
		assert newAlphaYi < 1 + 1e-7;
		assert newAlphaYj >= -1e-7;
		assert newAlphaYj < 1 + 1e-7;
		indexToAlpha[i].set(yi, newAlphaYi);
		indexToAlpha[i].set(yj, newAlphaYj);
		updateWeightsPair(delta_yi, delta_yj, t * C_norm);
		for (int yk = 0; yk < deltaWeightsDotProdCache.length; ++yk) {

			deltaWeightsDotProdCache[yk] += t * C_norm * getDotProd(i, yi, yk, deltaDeltaDotProductCache);
			deltaWeightsDotProdCache[yk] -= t * C_norm * getDotProd(i, yj, yk, deltaDeltaDotProductCache);
		}
		return true;
	}

	private double getDotProd(int i, int yj, int yk, double[][] deltaDeltaDotProductCache) {
		if (yj == yk) return indexToDeltaNormSquared[i].get(yj);
		if (yj < yk)
			return deltaDeltaDotProductCache[yj][yk];
		else
			return deltaDeltaDotProductCache[yk][yj];
	}

	/**
	 * @param C_norm
	 * @param delta_yi
	 * @param delta_yj
	 * @param t
	 */
	private void updateWeightsPair(final IntCounter delta_yi, final IntCounter delta_yj, double t) {
		updateWeights(delta_yi, t);
		updateWeights(delta_yj, -1.0 * t);
	}

	/**
	 * @param delta_yi
	 * @param t
	 */
	private void updateWeights(final IntCounter delta_yi, double t) {
		incrementAll(weights, delta_yi, t);
		model.updateWeights(delta_yi, t);
	}

	/**
	 * @param i
	 * @param yi
	 * @param yj
	 * @param b
	 * @param alpha_yi
	 * @param alpha_yj
	 * @param deltaWeightsDotProd
	 * @return
	 */
	private double computeSmoStep(int i, int yi, int yj, double b, final double alpha_yi, final double alpha_yj, final double deltaWeightsDotProd) {
		double a = -1.0 * deltaWeightsDotProd + indexToLoss[i].get(yi) - indexToLoss[i].get(yj);
		double c = -1.0 * alpha_yi;
		double d = alpha_yj;
		double t = Math.max(c, Math.min(d, a / b));
		return t;
	}

	private void incrementAll(final double[] weights, IntCounter currDelta, double d) {
		IntCounter.incrementDenseArray(weights, currDelta, d);
	}

	void clearConstraints(int numConstraintSets) {
		this.indexToDelta = new List[numConstraintSets];
		this.indexToDeltaNormSquared = new List[numConstraintSets];
		this.indexToAlpha = new List[numConstraintSets];
		this.indexToLoss = new List[numConstraintSets];
		this.activeAlphasPriorityQueue = new List[numConstraintSets];
		for (int i = 0; i < numConstraintSets; ++i) {
			this.indexToDelta[i] = new ArrayList<IntCounter>();
			this.activeAlphasPriorityQueue[i] = new ArrayList<Integer>();
			this.indexToDeltaNormSquared[i] = new ArrayList<Double>();
			this.indexToAlpha[i] = new ArrayList<Double>();
			this.indexToLoss[i] = new ArrayList<Double>();
		}
	}

	private void pruneInactiveAlphas() {
		if (opts.maxInactiveAlphaCount == Integer.MAX_VALUE) return;
		for (int i = 0; i < indexToAlpha.length; ++i) {
			for (int yi = 0; yi < numConstraints(i); ++yi) {
				if (indexToAlpha[i].get(yi) < 1e-30) {
					int curr = activeAlphasPriorityQueue[i].get(yi);
					if (curr > opts.maxInactiveAlphaCount) {
						deleteConstraint(i, yi);
					} else {
						activeAlphasPriorityQueue[i].set(yi, curr + 1);
					}
				} else {
					activeAlphasPriorityQueue[i].set(yi, 0);
				}

			}

		}
	}

	private void deleteConstraint(int i, int yi) {
		double alpha = indexToAlpha[i].remove(yi);
		indexToDeltaNormSquared[i].remove(yi);
		IntCounter delta_yi = indexToDelta[i].remove(yi);
		indexToLoss[i].remove(yi);
		updateWeights(delta_yi, -alpha);

	}

	public List<UpdateBundle> batchLossAugmentedDecode(LossAugmentedLinearModel<T> model, List<T> data, CounterInterface<Integer> weights, double lossWeight) {
		//				model.setWeights(weights);
		List<UpdateBundle> ubs = model.getLossAugmentedUpdateBundleBatch(data, lossWeight);
		numDecodes += data.size();
		return ubs;
	}

	public int reapConstraints(boolean initial, LossAugmentedLinearModel<T> model, List<T> data, CounterInterface<Integer> weights, int currStart,
		int numMiniBatches, int currMiniBatchIndex, int totalDataSize) {
		if (initial) {
			clearConstraints(totalDataSize);
			for (int i = 0; i < totalDataSize; ++i) {
				addConstraint(i, new IntCounter(), 0.0);
			}
		}

		int numAdded = 0;

		List<UpdateBundle> ubs = batchLossAugmentedDecode(model, data, weights, 1.0);

		for (int i = currStart; i < currStart + data.size(); ++i) {
			UpdateBundle ub = ubs.get(i - currStart);
			if (ub.loss == Double.POSITIVE_INFINITY) {
				System.out.println("Hmmm, infinite loss, ignoring");
				continue;
			}
			IntCounter delta = new IntCounter();
			delta.incrementAll(ub.gold);
			delta.incrementAll(ub.guess, -1.0);
			double loss = ub.loss;

			numAdded += addConstraintIfNecessary(weights, i, delta, loss);
		}

		return numAdded;
	}

	/**
	 * @param weights
	 * @param numAdded
	 * @param i
	 * @param delta
	 * @param loss
	 * @return
	 */
	protected int addConstraintIfNecessary(CounterInterface<Integer> weights, int i, IntCounter delta, double loss) {
		int numAdded = 0;
		double currentSlack = Double.NEGATIVE_INFINITY;
		for (int yi = 0; yi < numConstraints(i); ++yi) {
			currentSlack = Math.max(currentSlack, getContraintSlack(i, yi, weights));
		}
		final double newSlack = getContraintSlack(weights, delta, loss);
		if (shouldCheckLossAugmentedDecoding() && newSlack < currentSlack - opts.lossAugCheckTol * currentSlack) {
			//throw new RuntimeException("Something probably wrong with loss-augmented decoding"); 
			System.out.println("Something wrong with loss augmented decoding, new slack is " + newSlack + " and current slack is " + currentSlack);
		}
		if (newSlack > currentSlack + epsilon) {
			addConstraint(i, delta, loss);
			numAdded += 1;
		}
		return numAdded;
	}

	protected boolean shouldCheckLossAugmentedDecoding() {
		return true;
	}

	public void buildDotProdCache() {
		if (!opts.primalSmo) {

			System.out.println("Building dotprod cache");
			final int numConstraints = numConstraints();
			this.dotProdCache = new double[numConstraints][numConstraints];
			for (int i = 0; i < numConstraints; ++i) {
				Pair<Integer, Integer> relIndicesI = getAlphaRelativeIndicesFromAbsoluteIndex(i);
				final CounterInterface<Integer> deltaI = indexToDelta[relIndicesI.getFirst()].get(relIndicesI.getSecond());
				for (int j = 0; j < numConstraints; ++j) {
					Pair<Integer, Integer> relIndicesJ = getAlphaRelativeIndicesFromAbsoluteIndex(j);
					dotProdCache[i][j] = deltaI.dotProduct(indexToDelta[relIndicesJ.getFirst()].get(relIndicesJ.getSecond()));
				}
			}
		}
	}

	public void zeroInitializeAlphas() {
		for (int i = 0; i < indexToAlpha.length; ++i) {
			for (int yi = 0; yi < numConstraints(i); ++yi) {
				if (yi == 0) {
					indexToAlpha[i].set(yi, 1.0);
				} else {
					indexToAlpha[i].set(yi, 0.0);
				}
			}
		}
	}

	public void uniformInitializeAlphas() {
		for (int i = 0; i < indexToAlpha.length; ++i) {
			for (int yi = 0; yi < numConstraints(i); ++yi) {
				if (yi == 0) {
					indexToAlpha[i].set(yi, 0.9);
				} else {
					indexToAlpha[i].set(yi, 0.1 / (numConstraints(i) - 1.0));
				}
			}
		}
	}

	public void addConstraint(int i, IntCounter delta, double loss) {
		indexToAlpha[i].add(0.0);
		activeAlphasPriorityQueue[i].add(0);
		normalizeAlphas(i);

		double maxFeatureCount = 0.0;
		for (Map.Entry<Integer, Double> entry : delta.entries()) {
			maxFeatureIndexSoFar = Math.max(entry.getKey(), maxFeatureIndexSoFar);
			maxFeatureCount = Math.abs(Math.max(maxFeatureCount, entry.getValue()));
		}
		maxFeatureCount = Math.min(1.0, maxFeatureCount);
		if (maxFeatureIndexSoFar >= weights.length) {
			weights = Arrays.copyOf(weights, Math.max(maxFeatureIndexSoFar + 1, weights.length * 3 / 2));
		}

		// prune low count features. Really helps for small expected counts
		List<Integer> toClear = new ArrayList<Integer>();
		for (Entry<Integer,Double> entry : delta.entries()) {
			if (Math.abs(entry.getValue()) < maxFeatureCount * opts.minCountThresh) {
				toClear.add(entry.getKey());
			}

		}
		for (int key : toClear)
			delta.setCount(key, 0.0);
		IntCounter fresh = new IntCounter();
		fresh.incrementAll(delta);
		delta = null;
		fresh.toSorted();
		indexToDelta[i].add(fresh);
		indexToDeltaNormSquared[i].add(fresh.normSquared());
		indexToLoss[i].add(loss);
		double[] alphas = getAlphas(i);
		double[] newAlphas = Arrays.copyOf(alphas, alphas.length);
		newAlphas[newAlphas.length - 1] = opts.newAlphaMag;

		IntCounter weightsDelta = setAlphas(i, alphas, newAlphas, new IntCounter());
		incrementAll(weights, weightsDelta, 1.0);
	}

	void normalizeAlphas(int i) {
		double norm = 0.0;
		for (int yi = 0; yi < numConstraints(i); ++yi) {
			norm += indexToAlpha[i].get(yi);
		}
		for (int yi = 0; yi < numConstraints(i); ++yi) {
			if (norm > 0) indexToAlpha[i].set(yi, indexToAlpha[i].get(yi) / norm);
		}
	}

	public int numConstraints(int i) {
		if (indexToAlpha == null) return 0;
		return indexToAlpha[i].size();
	}

	public int numConstraints() {
		if (indexToAlpha == null) return 0;
		int result = 0;
		for (int i = 0; i < indexToAlpha.length; ++i) {
			result += numConstraints(i);
		}
		return result;
	}

	static <D> int index(D thing, List<D> indexToThing, Map<D, Integer> thingToIndex) {
		Integer index = thingToIndex.get(thing);
		if (index == null) {
			index = indexToThing.size();
			thingToIndex.put(thing, index);
			indexToThing.add(thing);
		}
		return index;
	}

	public static double[] projectToSimplex(List<Double> v) {
		List<Double> u = new ArrayList<Double>(v);
		Collections.sort(u);
		Collections.reverse(u);
		int p = 0;
		for (int j = 0; j < u.size(); ++j) {
			double sum = 0.0;
			for (int r = 0; r <= j; ++r) {
				sum += u.get(r);
			}
			if (u.get(j) - (1.0 / (j + 1.0)) * (sum - 1.0) > 0) {
				p = Math.max(p, j);
			}
		}
		double sum = 0.0;
		for (int i = 0; i <= p; ++i) {
			sum += u.get(i);
		}
		double theta = (1.0 / (p + 1.0)) * (sum - 1.0);
		for (int i = 0; i < v.size(); ++i) {
			v.set(i, Math.max(v.get(i) - theta, 0));
		}

		double sumv = 0.0;
		for (double val : v) {
			sumv += val;
		}
		assert sumv > 1.0 - 1e-6 && sumv < 1.0 + 1e-6;
		double[] ret = new double[v.size()];
		for (int i = 0; i < v.size(); ++i)
			ret[i] = v.get(i);
		return ret;
	}

}
