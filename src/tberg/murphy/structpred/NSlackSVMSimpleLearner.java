package tberg.murphy.structpred;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import tberg.murphy.counter.CounterInterface;
import tberg.murphy.counter.IntCounter;
import tberg.murphy.tuple.Pair;

public class NSlackSVMSimpleLearner<T> implements LossAugmentedLearner<T> {

	public static class SvmOpts {
		public double EPSILON = 1e-20;

		public double SMO_TOL = 1e-4;

		public int SMO_ITERS = 10000;

		public boolean REFRESH_ALPHAS = false;

		public double NEW_ALPHA_MAG = 0.0;

		public boolean smoCheckPrimal = true;
	}

	int maxLength;

	double C;

	int N;

	double epsilon;

	List<IntCounter>[] indexToDelta;

	List<Double>[] indexToAlpha;

	List<Double>[] indexToLoss;

	double[][] dotProdCache;

	private SvmOpts opts;

	public NSlackSVMSimpleLearner(double C, double epsilon) {
		this(C, epsilon, new SvmOpts());
	}

	public NSlackSVMSimpleLearner(double C, double epsilon, SvmOpts opts) {
		this.C = C;
		this.epsilon = epsilon;
		this.opts = opts;
	}
	

	public CounterInterface<Integer> train(CounterInterface<Integer> initWeights, LossAugmentedLinearModel<T> model, List<T> data, int maxIters) {
		CounterInterface<Integer> currentWeights = initWeights;
		for (int t = 0; t < maxIters; ++t) {
			model.startIteration(t);

			int numAdded = 0;
			if (t == 0) {
				numAdded = reapConstraints(true, model, data, currentWeights);
			} else {
				numAdded = reapConstraints(false, model, data, currentWeights);
			}
			System.out.printf("Added %d contraints.\n", numAdded);
			System.out.printf("Iteration %d...\n", t);
			System.out.printf("Num constraints: %d\n", numConstraints());

			if (opts.REFRESH_ALPHAS || t == 0) uniformInitializeAlphas();
			buildDotProdCache();
			optimizeDualObjectiveSMO();

			System.out.printf("Primal objective: %.8f\n", getPrimalObjective());
			System.out.printf("Dual objective: %.8f\n", getDualObjective());
			if (currentWeights != null) {
				CounterInterface<Integer> newWeights = getWeights();
				CounterInterface<Integer> weightsDelta = new IntCounter();
				weightsDelta.incrementAll(newWeights);
				weightsDelta.incrementAll(currentWeights, -1.0);
				System.out.printf("Mag of weights delta: %.8f\n", Math.sqrt(weightsDelta.dotProduct(weightsDelta)));
				currentWeights = newWeights;
			} else {
				currentWeights = getWeights();
			}

			if (numAdded == 0) break;
		}

		model.setWeights(getWeights());
		return getWeights();
	}

	Pair<Integer, Integer> getAlphaRelativeIndicesFromAbsoluteIndex(int absIndex) {
		int i = 0;
		while (absIndex >= numConstraints(i)) {
			absIndex -= numConstraints(i);
			i++;
		}
		return Pair.makePair(i, absIndex);
	}

	int getAlphaAbsoluteIndexFromRelativeIndices(int i, int yi) {
		int absoluteIndex = 0;
		for (int j = 0; j < i; ++j) {
			absoluteIndex += numConstraints(j);
		}
		absoluteIndex += yi;
		return absoluteIndex;
	}

	public double getPrimalObjective() {
		double obj = 0.0;
		CounterInterface<Integer> weights = getWeights();
		obj += 0.5 * weights.dotProduct(weights);
		for (int i = 0; i < indexToAlpha.length; ++i) {
			double slack = 0.0;
			for (int yi = 0; yi < numConstraints(i); ++yi) {
				slack = Math.max(slack, getContraintSlack(i, yi, weights));
			}
			obj += (C / indexToAlpha.length) * slack;
		}
		return obj;
	}

	double getContraintSlack(int i, int yi, CounterInterface<Integer> weights) {
		return indexToLoss[i].get(yi) - indexToDelta[i].get(yi).dotProduct(weights);
	}

	double getContraintSlack(CounterInterface<Integer> weights, CounterInterface<Integer> delta, double loss) {
		return loss - weights.dotProduct(delta);
	}

	public double getDualObjective() {
		double obj = 0.0;
		for (int i = 0; i < indexToAlpha.length; ++i) {
			for (int yi = 0; yi < numConstraints(i); ++yi) {
				obj += (C / indexToAlpha.length) * indexToAlpha[i].get(yi) * indexToLoss[i].get(yi);
				for (int j = 0; j < indexToAlpha.length; ++j) {
					for (int yj = 0; yj < numConstraints(j); ++yj) {
						obj -= 0.5 * (C / indexToAlpha.length) * (C / indexToAlpha.length) * indexToAlpha[i].get(yi) * indexToAlpha[j].get(yj)
							* dotProdCache[getAlphaAbsoluteIndexFromRelativeIndices(i, yi)][getAlphaAbsoluteIndexFromRelativeIndices(j, yj)];
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

	double[] getAlphas(int i) {
		double[] alphas = new double[numConstraints(i)];
		for (int yi = 0; yi < numConstraints(i); ++yi) {
			alphas[yi] = indexToAlpha[i].get(yi);
		}
		return alphas;
	}

	public void optimizeDualObjectiveSMO() {
		double lastDual = Double.NEGATIVE_INFINITY;
		for (int iter = 1; iter <= opts.SMO_ITERS; ++iter) {
			for (int i = 0; i < indexToAlpha.length; ++i) {
				for (int yi = 0; yi < numConstraints(i); ++yi) {
					for (int yj = 0; yj < numConstraints(i); ++yj) {
						if (yi != yj) updateAlphas(i, yi, yj);
					}
				}
			}
			double dual = getDualObjective();
			double primal = opts.smoCheckPrimal ? getPrimalObjective() : Double.NaN;
			if (iter == 1 || converged(primal, dual, opts.SMO_TOL, lastDual) || iter == opts.SMO_ITERS)
//			if (iter == 1 || iter % 200 == 0 || converged(primal, dual, opts.SMO_TOL, lastDual) || iter == opts.SMO_ITERS)
				System.out.printf("[SMO] Round %d: %.8f\n", iter, dual);
			if (converged(primal, dual, opts.SMO_TOL, lastDual)) break;
			lastDual = dual;
		}
	}

	boolean converged(double primal, double dual, double tol, double lastDual) {
		if (opts.smoCheckPrimal) {
			double valueAverage = (Math.abs(dual) + Math.abs(primal)) / 2.0;
			if (Math.abs(primal - dual) < opts.EPSILON || Math.abs(primal - dual) / valueAverage < tol) return true;
		} else {
			double diff = dual - lastDual;
			if (diff / dual < opts.SMO_TOL) return true;
		}
		return false;
	}

	public void updateAlphas(int i, int yi, int yj) {
		int yiAbs = getAlphaAbsoluteIndexFromRelativeIndices(i, yi);
		int yjAbs = getAlphaAbsoluteIndexFromRelativeIndices(i, yj);
		if (dotProdCache[yiAbs][yiAbs] == 0 && dotProdCache[yjAbs][yjAbs] == 0) return;
		double numerator = indexToLoss[i].get(yi) - indexToLoss[i].get(yj);
		for (int k = 0; k < indexToAlpha.length; ++k) {
			for (int yk = 0; yk < numConstraints(k); ++yk) {
				int ykAbs = getAlphaAbsoluteIndexFromRelativeIndices(k, yk);
				numerator -= (C / indexToAlpha.length) * indexToAlpha[k].get(yk) * dotProdCache[yiAbs][ykAbs];
				numerator += (C / indexToAlpha.length) * indexToAlpha[k].get(yk) * dotProdCache[yjAbs][ykAbs];
			}
		}
		double denomenator = 0.0;
		denomenator += (C / indexToAlpha.length) * dotProdCache[yiAbs][yiAbs];
		denomenator -= 2.0 * (C / indexToAlpha.length) * dotProdCache[yiAbs][yjAbs];
		denomenator += (C / indexToAlpha.length) * dotProdCache[yjAbs][yjAbs];
		if (denomenator == 0) return;

		double delta = Math.max(-indexToAlpha[i].get(yi), Math.min(indexToAlpha[i].get(yj), numerator / denomenator));

		indexToAlpha[i].set(yi, indexToAlpha[i].get(yi) + delta);
		indexToAlpha[i].set(yj, indexToAlpha[i].get(yj) - delta);
	}

	void clearConstraints(int numConstraintSets) {
		this.indexToDelta = new List[numConstraintSets];
		this.indexToAlpha = new List[numConstraintSets];
		this.indexToLoss = new List[numConstraintSets];
		for (int i = 0; i < numConstraintSets; ++i) {
			this.indexToDelta[i] = new ArrayList<IntCounter>();
			this.indexToAlpha[i] = new ArrayList<Double>();
			this.indexToLoss[i] = new ArrayList<Double>();
		}
	}

	public int reapConstraints(boolean initial, LossAugmentedLinearModel<T> model, List<T> data, CounterInterface<Integer> weights) {
		if (initial) {
			clearConstraints(data.size());
			for (int i = 0; i < data.size(); ++i) {
				addConstraint(i, new IntCounter(), 0.0);
			}
		}

		model.setWeights(weights);
		List<UpdateBundle> ubs = model.getLossAugmentedUpdateBundleBatch(data, 1.0);

		int numAdded = 0;
		for (int i = 0; i < data.size(); ++i) {
			UpdateBundle ub = ubs.get(i);
			CounterInterface<Integer> delta = new IntCounter();
			delta.incrementAll(ub.gold);
			delta.incrementAll(ub.guess, -1.0);
			double loss = ub.loss;

			double currentSlack = Double.NEGATIVE_INFINITY;
			for (int yi = 0; yi < numConstraints(i); ++yi) {
				currentSlack = Math.max(currentSlack, getContraintSlack(i, yi, weights));
			}
			if (getContraintSlack(weights, delta, loss) > currentSlack + epsilon) {
				addConstraint(i, delta, loss);
				numAdded += 1;
			}
		}

		return numAdded;
	}

	public void buildDotProdCache() {
		this.dotProdCache = new double[numConstraints()][numConstraints()];
		for (int i = 0; i < numConstraints(); ++i) {
			for (int j = 0; j < numConstraints(); ++j) {
				Pair<Integer, Integer> relIndicesI = getAlphaRelativeIndicesFromAbsoluteIndex(i);
				Pair<Integer, Integer> relIndicesJ = getAlphaRelativeIndicesFromAbsoluteIndex(j);
				dotProdCache[i][j] = indexToDelta[relIndicesI.getFirst()].get(relIndicesI.getSecond()).dotProduct(
					indexToDelta[relIndicesJ.getFirst()].get(relIndicesJ.getSecond()));
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

	public void addConstraint(int i, CounterInterface<Integer> delta, double loss) {
		indexToAlpha[i].add(opts.NEW_ALPHA_MAG);
		normalizeAlphas(i);
		indexToDelta[i].add(new IntCounter(delta));
		indexToLoss[i].add(loss);
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

	public CounterInterface<Integer> getWeights() {
		CounterInterface<Integer> weights = new IntCounter();
		for (int i = 0; i < indexToAlpha.length; ++i) {
			for (int yi = 0; yi < numConstraints(i); ++yi) {
				for (Map.Entry<Integer,Double> entry : indexToDelta[i].get(yi).entries()) {
					weights.incrementCount(entry.getKey(), (C / indexToAlpha.length) * indexToAlpha[i].get(yi) * entry.getValue());
				}
			}
		}
		return weights;
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

}
