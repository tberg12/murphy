package floatsequence;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import tuple.Pair;

public class DenseSemiMarkovDP {

	public static interface Model {
		public int length();
		public int numStates();
		public int[] allowedWidths(int s);
		public float logEdgePotential(int t, int prevS, int s);
		public float logNodePotential(int t, int wi, int s);
	}
	
	public static List<Pair<Integer,Pair<Integer,Integer>>> viterbiDecode(Model model) {

		int[][] prevTimes = new int[model.length()+1][model.numStates()];
		int[][] prevStates = new int[model.length()+1][model.numStates()];
		float[][] alphas = new float[model.length()+1][model.numStates()];
		for (int t=0; t<model.length()+1; ++t) {
			Arrays.fill(alphas[t], Float.NEGATIVE_INFINITY);
		}
		
		for (int t=0; t<model.length(); ++t) {
			if (t == 0) {
				for (int s=0; s<model.numStates(); ++s) {
					int[] allowedWidths = model.allowedWidths(s);
					for (int wi=0; wi<allowedWidths.length; ++wi) {
						int w = allowedWidths[wi];
						int nextT = t+w;
						if (nextT <= model.length()) {
							float nodesScore = model.logNodePotential(t, wi, s); 
							float alpha = alphas[nextT][s];
							if (nodesScore > alpha) {
								alphas[nextT][s] = nodesScore;
								prevTimes[nextT][s] = 0;
								prevStates[nextT][s] = -1;
							}
						}
					}
				}
			} else {
				for (int s=0; s<model.numStates(); ++s) {
					float bestPrevAlphaAndTransScore = Float.NEGATIVE_INFINITY;
					int bestPrevS = -1;
					for (int prevS=0; prevS<model.numStates(); ++prevS) {
						float score = alphas[t][prevS] + model.logEdgePotential(t, prevS, s);
						if (score > bestPrevAlphaAndTransScore) {
							bestPrevAlphaAndTransScore = score;
							bestPrevS = prevS;
						}
					}
					int[] allowedWidths = model.allowedWidths(s);
					for (int wi=0; wi<allowedWidths.length; ++wi) {
						int w = allowedWidths[wi];
						int nextT = t+w;
						if (nextT <= model.length()) {
							float alpha = alphas[nextT][s];
							float nodesScore = model.logNodePotential(t, wi, s);
							float score = bestPrevAlphaAndTransScore+nodesScore;
							if (score > alpha) {
								alphas[nextT][s] = score;
								prevTimes[nextT][s] = t;
								prevStates[nextT][s] = bestPrevS;
							}
						}
					}
				}
			}
		}
		
		List<Pair<Integer,Pair<Integer,Integer>>> decode = new ArrayList<Pair<Integer,Pair<Integer,Integer>>>();
		int currentS = -1;
		int currentT = model.length();
		float bestScore = Float.NEGATIVE_INFINITY;
		for (int s=0; s<model.numStates(); ++s) {
			float score = alphas[model.length()][s];
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