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
		public float logStartPotential(int w, int s);
		public float logEndPotential(int s);
		public float logPotential(int t, int w, int prevS, int s);
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
					int maxAllowedWidth = model.maxAllowedWidth(s);
					for (int w=1; w<=maxAllowedWidth; ++w) {
						int nextT = t+w;
						if (nextT <= model.length()) {
						  float score = model.logStartPotential(w, s); 
						  float alpha = alphas[nextT][s];
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
					      float alpha = alphas[nextT][s];
					      float score = alphas[t][prevS] + model.logPotential(t, w, prevS, s);
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
		float bestScore = Float.NEGATIVE_INFINITY;
		for (int s=0; s<model.numStates(); ++s) {
		  float score = alphas[model.length()][s] + model.logEndPotential(s);
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