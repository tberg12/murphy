package tberg.murphy.sequence;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import tberg.murphy.tuple.Pair;

public class DenseSemiMarkovDP2 {

	public static interface Model {
		public int length();
		public int numStates();
		public int maxAllowedWidth(int s);
		public double logStartPotential(int w, int s);
		public double logEndPotential(int s);
		public double logPotential(int t, int w, int prevS, int s);
	}
	
	public static List<Pair<Integer,Pair<Integer,Integer>>> viterbiDecode(Model model) {

		int[][] prevTimes = new int[model.length()+1][model.numStates()];
		int[][] prevStates = new int[model.length()+1][model.numStates()];
		double[][] alphas = new double[model.length()+1][model.numStates()];
		for (int t=0; t<model.length()+1; ++t) {
			Arrays.fill(alphas[t], Double.NEGATIVE_INFINITY);
		}
		
		for (int t=0; t<model.length(); ++t) {
			if (t == 0) {
				for (int s=0; s<model.numStates(); ++s) {
					int maxAllowedWidth = model.maxAllowedWidth(s);
					for (int w=1; w<=maxAllowedWidth; ++w) {
						int nextT = t+w;
						if (nextT <= model.length()) {
							double score = model.logStartPotential(w, s); 
							double alpha = alphas[nextT][s];
							if (score > alpha) {
								alphas[nextT][s] = score;
								prevTimes[nextT][s] = 0;
								prevStates[nextT][s] = -1;
							}
						}
					}
				}
			} else {
				for (int s=0; s<model.numStates(); ++s) {
				  int maxAllowedWidth = model.maxAllowedWidth(s);
					for (int prevS=0; prevS<model.numStates(); ++prevS) {
					  for (int w=1; w<=maxAllowedWidth; ++w) {
					    int nextT = t+w;
					    if (nextT <= model.length()) {
					      double alpha = alphas[nextT][s];
					      double score = alphas[t][prevS] + model.logPotential(t, w, prevS, s);
					      if (score > alpha) {
					        alphas[nextT][s] = score;
					        prevTimes[nextT][s] = t;
					        prevStates[nextT][s] = prevS;
					      }
					    }
					  }
					}
				}
			}
		}
		
		List<Pair<Integer,Pair<Integer,Integer>>> decode = new ArrayList<Pair<Integer,Pair<Integer,Integer>>>();
		int currentS = -1;
		int currentT = model.length();
		double bestScore = Double.NEGATIVE_INFINITY;
		for (int s=0; s<model.numStates(); ++s) {
			double score = alphas[model.length()][s] + model.logEndPotential(s);
			if (score > bestScore) {
				bestScore = score;
				currentS = s;
			}
		}
		while (currentS != -1) {
			int prevS = prevStates[currentT][currentS]; 
			int prevT = prevTimes[currentT][currentS];
			decode.add(Pair.makePair(currentS, Pair.makePair(prevT, currentT)));
			currentS = prevS;
			currentT = prevT;
		}
		Collections.reverse(decode);
		
		return decode;
	}
	
}