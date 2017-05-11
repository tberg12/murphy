package example;

import java.util.ArrayList;
import java.util.List;

import tberg.murphy.arrays.a;
import tberg.murphy.counter.CounterInterface;
import tberg.murphy.counter.IntCounter;
import tberg.murphy.fileio.f;
import tberg.murphy.sequence.ForwardBackward;
import tberg.murphy.structpred.LossAugmentedLearner;
import tberg.murphy.structpred.LossAugmentedLinearModel;
import tberg.murphy.structpred.UpdateBundle;
import tberg.murphy.indexer.HashMapIndexer;
import tberg.murphy.indexer.Indexer;
import tberg.murphy.lazystructpred.PrimalSubgradientSVMLearner;
import tberg.murphy.sequence.ForwardBackward.Lattice;

public class NERExample {
	
	public static void main(String[] args) {
		
		int numTest = 200;
		double regConstant = 0.0001;
		double stepSize = 2e-1;
		int trainIters = 30;
		boolean L1Regularizer = false;
		
		Indexer<String> wordIndexer = new HashMapIndexer<String>();
		Indexer<String> tagIndexer = new HashMapIndexer<String>();
		// make sure null tag has index 0
		tagIndexer.getIndex("O");
		
		List<Datum> allData = readAndIndexDataset(args[0], wordIndexer, tagIndexer);
		List<Datum> train = allData.subList(0, allData.size()-numTest);
		List<Datum> test = allData.subList(allData.size()-numTest, allData.size());
		
		System.out.println("num train: "+train.size());
		System.out.println("num test: "+test.size());
		
		List<FeatureFunction> featFuncs = new ArrayList<FeatureFunction>();
		featFuncs.add(new SimpleFeatureFunction());
		
		Indexer<String> featIndexer = new HashMapIndexer<String>();
		
		List<FeaturizedDatum> featTrain = new ArrayList<FeaturizedDatum>();
		for (Datum datum : train) featTrain.add(featurize(datum, featFuncs, featIndexer, wordIndexer, tagIndexer));
		featIndexer.lock();
		List<FeaturizedDatum> featTest = new ArrayList<FeaturizedDatum>();
		for (Datum datum : test) featTest.add(featurize(datum, featFuncs, featIndexer, wordIndexer, tagIndexer));
		
		System.out.println("num features: "+featIndexer.size());
		
		NERModel model = new NERModel();
		LossAugmentedLearner<FeaturizedDatum> learner = new PrimalSubgradientSVMLearner<FeaturizedDatum>(regConstant, 1e-5, stepSize, featIndexer.size(), L1Regularizer);
		CounterInterface<Integer> learnedWeights = learner.train(new IntCounter(), model, featTrain, trainIters);
		
		model.setWeights(learnedWeights);
		
		System.out.println();
		System.out.println("train error: "+computeErrorRate(featTrain, model));
		System.out.println();
		double[] trainBreakdown = computeErrorBreakdown(featTrain, model, tagIndexer.size());
		for (int t=0; t<trainBreakdown.length; ++t) System.out.println(tagIndexer.getObject(t) + " : " + trainBreakdown[t]);
		System.out.println();
		System.out.println("test error: "+computeErrorRate(featTest, model));
		System.out.println();
		double[] testBreakdown = computeErrorBreakdown(featTest, model, tagIndexer.size());
		for (int t=0; t<testBreakdown.length; ++t) System.out.println(tagIndexer.getObject(t) + " : " + testBreakdown[t]);
		
	}
	
	public static class SimpleFeatureFunction implements FeatureFunction {

		public List<String> getFeatures(int[] words, int position, int prevTag, int curTag, Indexer<String> wordIndexer, Indexer<String> tagIndexer) {
			List<String> features = new ArrayList<String>();
			
			String word = wordIndexer.getObject(words[position]);
			boolean firstIsUpper = word.substring(0,1).toUpperCase().equals(word.substring(0,1));
			boolean allIsUpper = word.toUpperCase().equals(word);
			int length = Math.min(word.length(), 6);
			boolean containsPeriod = word.contains(".");

			features.add("BIAS");
			features.add("BIAS_TAG_"+tagIndexer.getObject(curTag));
			features.add("TRANS_"+tagIndexer.getObject(prevTag)+"_"+tagIndexer.getObject(curTag));
			
			features.add("FIRSTISUPPER_"+tagIndexer.getObject(curTag)+"_"+firstIsUpper);
			features.add("ALLISUPPER_"+tagIndexer.getObject(curTag)+"_"+allIsUpper);
			features.add("PERIOD_"+tagIndexer.getObject(curTag)+"_"+containsPeriod);
			features.add("LENGTH_"+tagIndexer.getObject(curTag)+"_"+length);
			features.add("LENGTH_ALLISUPPER_"+tagIndexer.getObject(curTag)+"_"+length+"_"+allIsUpper);
			features.add("LENGTH_FIRSTISUPPER_"+tagIndexer.getObject(curTag)+"_"+length+"_"+firstIsUpper);
			features.add("LENGTH_PERIOD_"+tagIndexer.getObject(curTag)+"_"+length+"_"+containsPeriod);
			
			return features;
		}
		
	}
	
	public static interface FeatureFunction {
		public List<String> getFeatures(int[] words, int position, int prevTag, int curTag, Indexer<String> wordIndexer, Indexer<String> tagIndexer);
	}
	
	public static double[] computeErrorBreakdown(List<FeaturizedDatum> data, NERModel model, int numTags) {
		double total[] = new double[numTags];
		double incorrect[] = new double[numTags];
		for (FeaturizedDatum datum : data) {
			int[] decode = model.decode(datum);
			for (int pos=0; pos<decode.length; ++pos) {
				total[datum.datum.tags[pos]] += 1.0;
				incorrect[datum.datum.tags[pos]] += (datum.datum.tags[pos] == decode[pos] ? 0.0 : 1.0);
			}
		}
		return a.pointwiseDiv(incorrect, total);
	}
	
	public static double computeErrorRate(List<FeaturizedDatum> data, NERModel model) {
		double total = 0.0;
		double incorrect = 0.0;
		for (FeaturizedDatum datum : data) {
			int[] decode = model.decode(datum);
			for (int pos=0; pos<decode.length; ++pos) {
				total += 1.0;
				incorrect += (datum.datum.tags[pos] == decode[pos] ? 0.0 : 1.0);
			}
		}
		return incorrect / total;
	}
	
	public static double computeHammingLoss(FeaturizedDatum datum, int[] decode) {
		double loss = 0.0;
		for (int pos=0; pos<datum.datum.tags.length; ++pos) if (datum.datum.tags[pos] != decode[pos]) loss += (datum.datum.tags[pos] == 0 ? 0.01 : 1.0);
		return loss;
	}
	
	public static class NERModel implements LossAugmentedLinearModel<FeaturizedDatum> {
		
		CounterInterface<Integer> weights;
		
		public NERModel() {
			this.weights = new IntCounter();
		}
		
		public int[] decode(FeaturizedDatum datum) {
			return lossAugmentedDecode(datum, 0.0);
		}
		
		public int[] lossAugmentedDecode(FeaturizedDatum datum, double lossWeight) {
			int numTags = datum.feats[0].length;
			double[][][] forwardPotentials = new double[datum.datum.words.length][numTags][numTags];
			for (int pos=0; pos<datum.datum.words.length; ++pos) {
				for (int prevTag=0; prevTag<numTags; ++prevTag) {
					for (int curTag=0; curTag<numTags; ++curTag) {
						double score = 0.0;
						for (int feat : datum.feats[pos][prevTag][curTag]) score += weights.getCount(feat);
						forwardPotentials[pos][prevTag][curTag] = score + (datum.datum.tags[pos] != curTag ? lossWeight * (datum.datum.tags[pos] == 0 ? 0.01 : 1.0) : 0.0);
					}
				}
			}
			double[][][] backwardPotentials = new double[datum.datum.words.length][][];
			for (int pos=0; pos<datum.datum.words.length; ++pos) backwardPotentials[pos] = a.transpose(forwardPotentials[pos]);
			
			Lattice lattice = new Lattice() {
				public int numSequences() {
					return 1;
				}
				public int sequenceLength(int d) {
					return datum.datum.words.length;
				}
				public int numStates(int d, int t) {
					return numTags;
				}
				public double nodeLogPotential(int d, int t, int s) {
					if (t == 0) {
						return forwardPotentials[0][0][s];
					} else {
						return 0.0;
					}
				}
				public double[] allowedEdgesLogPotentials(int d, int t, int s, boolean backward) {
					return (backward ? backwardPotentials[t][s] : forwardPotentials[t][s]);
				}
				public double nodePotential(int d, int t, int s) {
					return Math.exp(nodeLogPotential(d, t, s));
				}
				public double[] allowedEdgesPotentials(int d, int t, int s, boolean backward) {
					return a.exp(allowedEdgesLogPotentials(d, t, s, backward));
				}
				public int[] allowedEdges(int d, int t, int s, boolean backward) {
					return a.enumerate(0, numTags);
				}
			};
			
			return ForwardBackward.computeViterbiPathsLogSpace(lattice, 1)[0];
		}

		public UpdateBundle getLossAugmentedUpdateBundle(FeaturizedDatum datum, double lossWeight) {
			int[] decode = lossAugmentedDecode(datum, lossWeight);
			UpdateBundle ub = new UpdateBundle();
			ub.guess = getFeaturesFullDecode(datum, decode);
			ub.gold = getFeaturesFullDecode(datum, datum.datum.tags);
			ub.loss = computeHammingLoss(datum, decode);
			return ub;
		}
		
		public static CounterInterface<Integer> getFeaturesFullDecode(FeaturizedDatum datum, int[] decode) {
			CounterInterface<Integer> result = new IntCounter();
			for (int pos=0; pos<decode.length; ++pos) {
				if (pos == 0) {
					for (int feat : datum.feats[pos][0][decode[pos]]) result.incrementCount(feat, 1.0);
				} else {
					for (int feat : datum.feats[pos][decode[pos-1]][decode[pos]]) result.incrementCount(feat, 1.0);
				}
			}
			return result;
		}
		
		public List<UpdateBundle> getLossAugmentedUpdateBundleBatch(List<FeaturizedDatum> data, double lossWeight) {
			List<UpdateBundle> result = new ArrayList<UpdateBundle>();
			for (FeaturizedDatum datum : data) result.add(getLossAugmentedUpdateBundle(datum, lossWeight));
			return result;
		}
		
		public UpdateBundle getUpdateBundle(FeaturizedDatum datum) {
			return getLossAugmentedUpdateBundle(datum, 0.0);
		}

		public List<UpdateBundle> getUpdateBundleBatch(List<FeaturizedDatum> data) {
			return getLossAugmentedUpdateBundleBatch(data, 0.0);
		}

		public void setWeights(CounterInterface<Integer> weights) {
			this.weights = weights;
		}

		public CounterInterface<Integer> getWeights() {
			return weights;
		}

		public void updateWeights(IntCounter weightsDelta, double scale) {
			weights.incrementAll(weightsDelta, scale);
		}

		public void startIteration(int t) {
		}

	}
	
	public static FeaturizedDatum featurize(Datum datum, List<FeatureFunction> featFuncs, Indexer<String> featIndexer, Indexer<String> wordIndexer, Indexer<String> tagIndexer) {
		int[][][][] feats = new int[datum.words.length][tagIndexer.size()][tagIndexer.size()][];
		for (int pos=0; pos<datum.words.length; ++pos) {
			for (int prevTag=0; prevTag<tagIndexer.size(); ++prevTag) {
				for (int curTag=0; curTag<tagIndexer.size(); ++curTag) {
					List<String> strLocalFeats = new ArrayList<String>();
					for (FeatureFunction featFunc : featFuncs) {
						strLocalFeats.addAll(featFunc.getFeatures(datum.words, pos, prevTag, curTag, wordIndexer, tagIndexer));
					}
					ArrayList<Integer> localFeats = new ArrayList<Integer>();
					for (int i=0; i<strLocalFeats.size(); ++i) {
						int feat = featIndexer.getIndex(strLocalFeats.get(i));
						if (feat >= 0) localFeats.add(feat);
					}
					feats[pos][prevTag][curTag] = a.toIntArray(localFeats);
				}
			}
		}
		return new FeaturizedDatum(datum, feats);
	}
	
	public static class FeaturizedDatum {
		public FeaturizedDatum(Datum datum, int[][][][] feats) {
			this.datum = datum;
			this.feats = feats;
		}
		Datum datum;
		int[][][][] feats;
	}
	
	public static class Datum {
		public Datum(int[] words, int[] tags) {
			this.words = words;
			this.tags = tags;
		}
		int[] words;
		int[] tags;
	}
	
	public static List<Datum> readAndIndexDataset(String path, Indexer<String> wordIndexer, Indexer<String> tagIndexer) {
		List<String> lines = f.readLines(path);
		List<Datum> data = new ArrayList<Datum>();
		List<Integer> words = new ArrayList<Integer>();
		List<Integer> tags = new ArrayList<Integer>();
		String line = lines.remove(0);
		while (lines.size() > 0) {
			line = lines.remove(0);
			if (line.trim().equals("") || line.contains("-DOCSTART-")) {
				if (tags.size() > 0) {
					data.add(new Datum(a.toIntArray(words), a.toIntArray(tags)));
					words = new ArrayList<Integer>();
					tags = new ArrayList<Integer>();
				}
			} else {
				String[] split = line.trim().split("\\s+");
				words.add(wordIndexer.getIndex(split[0]));
				String tag = (split[3].length() == 1 ? split[3] : split[3].substring(2, split[3].length()));
				tags.add(tagIndexer.getIndex(tag));
			}
		}
		if (tags.size() > 0) {
			data.add(new Datum(a.toIntArray(words), a.toIntArray(tags)));
		}
		return data;
	}

}
