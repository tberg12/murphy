package floatsequence;

import java.util.Arrays;
import java.util.Iterator;

import threading.BetterThreader;
import tuple.Pair;

import arrays.a;
import math.m;

public class ForwardBackward {	

	public static final float SCALE = (float) Math.exp(20);
	public static final float INVSCALE = 1.0f / SCALE;
	public static final float LOG_SCALE = (float) Math.log(SCALE);
	
	private static float getScaleFactor(float logScale) {
		if (logScale == 0.0) return 1.0f;
		if (logScale == 1.0) return SCALE;
		if (logScale == 2.0) return SCALE * SCALE;
		if (logScale == 3.0) return SCALE * SCALE * SCALE;		
		if (logScale == -1.0) return 1.0f * INVSCALE;
		if (logScale == -2.0) return 1.0f * INVSCALE * INVSCALE;
		if (logScale == -3.0) return 1.0f * INVSCALE * INVSCALE * INVSCALE;		
		return (float) Math.pow(SCALE, logScale);
	}
	
	public static interface StationaryLattice {
		public int numSequences();
		public int sequenceLength(int d);
		public abstract int numStates(int d);
		public float nodeLogPotential(int d, int t, int s);
		public abstract float[] allowedEdgesLogPotentials(int d, int s, boolean backward);
		public float nodePotential(int d, int t, int s);
		public abstract float[] allowedEdgesPotentials(int d, int s, boolean backward);
		public abstract int[] allowedEdges(int d, int s, boolean backward);
	}
	
	public static class StationaryLatticeWrapper implements Lattice {
		StationaryLattice lattice;
		public StationaryLatticeWrapper(StationaryLattice lattice) {
			this.lattice = lattice;
		}
		public int numSequences() {
			return lattice.numSequences();
		}
		public int sequenceLength(int d) {
			return lattice.sequenceLength(d);
		}
		public int numStates(int d, int t) {
			return lattice.numStates(d);
		}
		public float nodeLogPotential(int d, int t, int s) {
			return lattice.nodeLogPotential(d, t, s);
		}
		public float[] allowedEdgesLogPotentials(int d, int t, int s, boolean backward) {
			return lattice.allowedEdgesLogPotentials(d, s, backward);
		}
		public float nodePotential(int d, int t, int s) {
			return lattice.nodePotential(d, t, s);
		}
		public float[] allowedEdgesPotentials(int d, int t, int s, boolean backward) {
			return lattice.allowedEdgesPotentials(d, s, backward);
		}
		public int[] allowedEdges(int d, int t, int s, boolean backward) {
			return lattice.allowedEdges(d, s, backward);
		}
	}
	
	public static interface Lattice {
		public int numSequences();
		public int sequenceLength(int d);
		public int numStates(int d, int t);
		public float nodeLogPotential(int d, int t, int s);
		public float[] allowedEdgesLogPotentials(int d, int t, int s, boolean backward);
		public float nodePotential(int d, int t, int s);
		public float[] allowedEdgesPotentials(int d, int t, int s, boolean backward);
		public int[] allowedEdges(int d, int t, int s, boolean backward);
	}

	public static interface StationaryEdgeMarginals {
		public float[] startNodeCondProbs(int d);
		public float[] endNodeCondProbs(int d);
		public int[] allowedForwardEdges(int d, int s);
		public float[] allowedForwardEdgesExpectedCounts(int d, int s);
		public float sequenceLogMarginalProb(int d);
		public float logMarginalProb();
		public int numSequences();
		public int numStates(int d);
		public Iterator<Pair<Pair<Integer,Pair<Integer,Integer>>,Float>> getEdgeMarginalsIterator();
		public Iterator<Pair<Pair<Integer,Integer>,Float>> getStartMarginalsIterator();
		public Iterator<Pair<Pair<Integer,Integer>,Float>> getEndMarginalsIterator();
		public double estimateMemoryUsage();
	}
	
	public static interface NonStationaryEdgeMarginals {
		public float[] startNodeCondProbs(int d);
		public float[] endNodeCondProbs(int d);
		public int[] allowedForwardEdges(int d, int t, int s);
		public float[] allowedForwardEdgesExpectedCounts(int d, int t, int s);
		public float sequenceLogMarginalProb(int d);
		public float logMarginalProb();
		public int numSequences();
		public int sequenceLength(int d);
		public int numStates(int d, int t);
		public Iterator<Pair<Pair<Pair<Integer,Integer>,Pair<Integer,Integer>>,Float>> getEdgeMarginalsIterator();
		public Iterator<Pair<Pair<Integer,Integer>,Float>> getStartMarginalsIterator();
		public Iterator<Pair<Pair<Integer,Integer>,Float>> getEndMarginalsIterator();
		public double estimateMemoryUsage();
	}

	public static interface NodeMarginals {
		public float[] nodeCondProbs(int d, int t);
		public float sequenceLogMarginalProb(int d);
		public float logMarginalProb();
		public int numSequences();
		public int sequenceLength(int d);
		public int numStates(int d);
		public Iterator<Pair<Pair<Pair<Integer,Integer>,Integer>,Float>> getNodeMarginalsIterator();
		public double estimateMemoryUsage();
	}
	
	public static interface StationaryStateProjector {
		public int domainSize(int d, int t);
		public int rangeSize(int d);
		public int project(int d, int t, int s);
	}
	
	private static class NodeMarginalsLogSpace implements NodeMarginals {
		Lattice lattice;
		float[][][] nodeCondProbs;
		float[] sequenceLogMarginalProbs;
		StationaryStateProjector stateProjector;

		public NodeMarginalsLogSpace(Lattice lattice, StationaryStateProjector stateProjector) {
			this.lattice = lattice;
			this.stateProjector = stateProjector;
			this.sequenceLogMarginalProbs = new float[lattice.numSequences()];
			this.nodeCondProbs = new float[lattice.numSequences()][][];
		}
		
		public void incrementExpectedCounts(float[][] alphas, float[][] betas, int d, boolean viterbi) {
			this.sequenceLogMarginalProbs[d] = Float.NEGATIVE_INFINITY;
			for (int s=0; s<lattice.numStates(d, 0); ++s) {
				if (viterbi) {
					this.sequenceLogMarginalProbs[d] = Math.max(sequenceLogMarginalProbs[d], betas[0][s]);
				} else {
					this.sequenceLogMarginalProbs[d] = m.logAdd(sequenceLogMarginalProbs[d], betas[0][s]);
				}
			}
			
			this.nodeCondProbs[d] = new float[lattice.sequenceLength(d)][];
			for (int t=0; t<lattice.sequenceLength(d); ++t) {
				this.nodeCondProbs[d][t] = new float[stateProjector.rangeSize(d)];
				if (viterbi) {
					Arrays.fill(this.nodeCondProbs[d][t], Float.NEGATIVE_INFINITY);
				}
				int numStates = lattice.numStates(d, t);
				for (int s=0; s<numStates; ++s) {
					int projectedS = stateProjector.project(d, t, s);
					if (alphas[t][s] != Float.NEGATIVE_INFINITY && betas[t][s] != Float.NEGATIVE_INFINITY) {
						if (viterbi) {
							this.nodeCondProbs[d][t][projectedS] = Math.max(this.nodeCondProbs[d][t][projectedS], (float) Math.exp(alphas[t][s] - lattice.nodeLogPotential(d, t, s) + betas[t][s] - sequenceLogMarginalProbs[d]));
						} else {
							this.nodeCondProbs[d][t][projectedS] += Math.exp(alphas[t][s] - lattice.nodeLogPotential(d, t, s) + betas[t][s] - sequenceLogMarginalProbs[d]);
						}
					}
				}
			}
		}
		
		public double estimateMemoryUsage() {
			double numElements = 0.0;
			for (int d=0; d<numSequences(); ++d) {
				if (nodeCondProbs[d] != null) {
					for (int t=0; t<sequenceLength(d); ++t) {
						numElements += nodeCondProbs[d][t].length;
					}
				}
			}
			return 8 * numElements / 1e9;
		}

		public float sequenceLogMarginalProb(int d) {
			return sequenceLogMarginalProbs[d];
		}

		public float logMarginalProb() {
			return a.sum(sequenceLogMarginalProbs);
		}

		public float[] nodeCondProbs(int d, int t) {
			return nodeCondProbs[d][t];
		}
		
		public int numSequences() {
			return lattice.numSequences();
		}

		public int sequenceLength(int d) {
			return lattice.sequenceLength(d);
		}

		public int numStates(int d) {
			return stateProjector.rangeSize(d);
		}

		private class NodeMarginalsIterator implements Iterator<Pair<Pair<Pair<Integer, Integer>, Integer>, Float>> {
			int d;
			int t;
			int s;
			float[] nodeCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && t == sequenceLength(d)-1 && s == numStates(d)-1);
			}
			public Pair<Pair<Pair<Integer, Integer>, Integer>, Float> next() {
				if (nodeCondProbs == null) {
					this.d = 0;
					this.t = 0;
					this.s = 0;
					this.nodeCondProbs = nodeCondProbs(0,0);
				} else {
					if (s == numStates(d)-1) {
						this.s = 0;
						if (t == sequenceLength(d)-1) {
							this.t=0;
							this.d++;
						} else {
							this.t++;
						}
						this.nodeCondProbs = nodeCondProbs(d,t);
					} else {
						this.s++;
					}
				}
				return Pair.makePair(Pair.makePair(Pair.makePair(d, t), s), nodeCondProbs[s]);
			}
			public void remove() {
			}
		}
		
		public Iterator<Pair<Pair<Pair<Integer, Integer>, Integer>, Float>> getNodeMarginalsIterator() {
			return new NodeMarginalsIterator();
		}
		
	}
	
	private static class NodeMarginalsScaling implements NodeMarginals {
		Lattice lattice;
		float[][][] nodeCondProbs;
		float[] sequenceLogMarginalProbs;
		StationaryStateProjector stateProjector;

		public NodeMarginalsScaling(Lattice lattice, StationaryStateProjector stateProjector) {
			this.lattice = lattice;
			this.stateProjector = stateProjector;
			this.sequenceLogMarginalProbs = new float[lattice.numSequences()];
			this.nodeCondProbs = new float[lattice.numSequences()][][];
		}
		
		public void incrementExpectedCounts(float[][] alphas, float[] alphaLogScales, float[][] betas, float[] betaLogScales, int d, boolean viterbi) {
			float sequenceMarginalProb = (viterbi ? Float.NEGATIVE_INFINITY : 0.0f);
			float sequenceMarginalProbLogScale = betaLogScales[0];
			for (int s=0; s<lattice.numStates(d, 0); ++s) {
				float nodePotential = lattice.nodePotential(d, 0, s);
				if (nodePotential > 0.0) {
					if (viterbi) {
						sequenceMarginalProb = Math.max(sequenceMarginalProb, betas[0][s]);
					} else {
						sequenceMarginalProb += betas[0][s];
					}
				}
			}
			this.sequenceLogMarginalProbs[d] = sequenceMarginalProbLogScale * LOG_SCALE + (float) Math.log(sequenceMarginalProb);
			
			this.nodeCondProbs[d] = new float[lattice.sequenceLength(d)][];
			for (int t=0; t<lattice.sequenceLength(d); ++t) {
				this.nodeCondProbs[d][t] = new float[stateProjector.rangeSize(d)];
				if (viterbi) {
					Arrays.fill(nodeCondProbs[d][t], Float.NEGATIVE_INFINITY);
				}
				int numStates = lattice.numStates(d, t);
				for (int s=0; s<numStates; ++s) {
					float scale = getScaleFactor(alphaLogScales[t] + betaLogScales[t] - sequenceMarginalProbLogScale);
					int projectedS = stateProjector.project(d, t, s);
					if (alphas[t][s] != 0.0 && betas[t][s] != 0.0) {
						if (viterbi) {
							this.nodeCondProbs[d][t][projectedS] = Math.max(this.nodeCondProbs[d][t][projectedS], (alphas[t][s] / lattice.nodePotential(d, t, s)) * (betas[t][s] / sequenceMarginalProb) * scale);
						} else {
							this.nodeCondProbs[d][t][projectedS] += (alphas[t][s] / lattice.nodePotential(d, t, s)) * (betas[t][s] / sequenceMarginalProb) * scale;
						}
					}
				}
			}
		}
		
		public double estimateMemoryUsage() {
			double numElements = 0.0;
			for (int d=0; d<numSequences(); ++d) {
				if (nodeCondProbs[d] != null) {
					for (int t=0; t<sequenceLength(d); ++t) {
						numElements += nodeCondProbs[d][t].length;
					}
				}
			}
			return 8 * numElements / 1e9;
		}
		
		public float sequenceLogMarginalProb(int d) {
			return sequenceLogMarginalProbs[d];
		}
		
		public float logMarginalProb() {
			return a.sum(sequenceLogMarginalProbs);
		}
		
		public float[] nodeCondProbs(int d, int t) {
			return nodeCondProbs[d][t];
		}
		
		public int numSequences() {
			return lattice.numSequences();
		}

		public int sequenceLength(int d) {
			return lattice.sequenceLength(d);
		}

		public int numStates(int d) {
			return stateProjector.rangeSize(d);
		}
		
		private class NodeMarginalsIterator implements Iterator<Pair<Pair<Pair<Integer, Integer>, Integer>, Float>> {
			int d;
			int t;
			int s;
			float[] nodeCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && t == sequenceLength(d)-1 && s == numStates(d)-1);
			}
			public Pair<Pair<Pair<Integer, Integer>, Integer>, Float> next() {
				if (nodeCondProbs == null) {
					this.d = 0;
					this.t = 0;
					this.s = 0;
					this.nodeCondProbs = nodeCondProbs(0,0);
				} else {
					if (s == numStates(d)-1) {
						this.s = 0;
						if (t == sequenceLength(d)-1) {
							this.t=0;
							this.d++;
						} else {
							this.t++;
						}
						this.nodeCondProbs = nodeCondProbs(d,t);
					} else {
						this.s++;
					}
				}
				return Pair.makePair(Pair.makePair(Pair.makePair(d, t), s), nodeCondProbs[s]);
			}
			public void remove() {
			}
		}

		public Iterator<Pair<Pair<Pair<Integer, Integer>, Integer>, Float>> getNodeMarginalsIterator() {
			return new NodeMarginalsIterator();
		}
		
	}
	
	private static class NonStationaryEdgeMarginalsLogSpace implements NonStationaryEdgeMarginals {
		Lattice lattice;
		float[] sequenceLogMarginalProbs;
		float[][] startNodeCondProbs;
		float[][] endNodeCondProbs;
		float[][][] allAlphas;
		float[][][] allBetas;

		public NonStationaryEdgeMarginalsLogSpace(Lattice lattice) {
			this.lattice = lattice;
			this.sequenceLogMarginalProbs = new float[lattice.numSequences()];
			this.startNodeCondProbs = new float[lattice.numSequences()][];
			this.endNodeCondProbs = new float[lattice.numSequences()][];
			this.allAlphas = new float[lattice.numSequences()][][];
			this.allBetas = new float[lattice.numSequences()][][];
		}

		public int[] allowedForwardEdges(int d, int t, int s) {
			return lattice.allowedEdges(d, t, s, false);
		}
		
		public void incrementExpectedCounts(float[][] alphas, float[][] betas, int d) {
			this.allAlphas[d] = alphas;
			this.allBetas[d] = betas;
			
			this.sequenceLogMarginalProbs[d] = Float.NEGATIVE_INFINITY;
			for (int s=0; s<lattice.numStates(d, 0); ++s) {
				this.sequenceLogMarginalProbs[d] = m.logAdd(sequenceLogMarginalProbs[d], betas[0][s]);
			}
			
			this.startNodeCondProbs[d] = new float[lattice.numStates(d, 0)];
			for (int s=0; s<lattice.numStates(d, 0); ++s) {
				this.startNodeCondProbs[d][s] = (float) Math.exp(betas[0][s] - sequenceLogMarginalProbs[d]);
			}
			
			this.endNodeCondProbs[d] = new float[lattice.numStates(d, lattice.sequenceLength(d)-1)];
			for (int s=0; s<lattice.numStates(d, lattice.sequenceLength(d)-1); ++s) {
				this.endNodeCondProbs[d][s] = (float) Math.exp(alphas[lattice.sequenceLength(d)-1][s] - sequenceLogMarginalProbs[d]);
			}
		}
		
		public double estimateMemoryUsage() {
			double numElements = 0.0;
			for (int d=0; d<numSequences(); ++d) {
				if (allAlphas[d] != null) {
					for (int t=0; t<sequenceLength(d); ++t) {
						numElements += allAlphas[d][t].length;
					}
				}
			}
			for (int d=0; d<numSequences(); ++d) {
				if (allBetas[d] != null) {
					for (int t=0; t<sequenceLength(d); ++t) {
						numElements += allBetas[d][t].length;
					}
				}
			}
			return 8 * numElements / 1e9;
		}

		public float[] allowedForwardEdgesExpectedCounts(int d, int t, int s) {
			int[] allowedEdges = lattice.allowedEdges(d, t, s, false);
			float[] allowedForwardEdgesExpectedCounts = new float[allowedEdges.length];
			float[] alowedEdgesLogPotentials = lattice.allowedEdgesLogPotentials(d, t, s, false);
			for (int i=0; i<allowedEdges.length; ++i) {
				int nextS = allowedEdges[i];
				float edgeLogPotential = alowedEdgesLogPotentials[i];
				allowedForwardEdgesExpectedCounts[i] += Math.exp(allAlphas[d][t][s] + edgeLogPotential + allBetas[d][t+1][nextS] - sequenceLogMarginalProbs[d]);
			}
			return allowedForwardEdgesExpectedCounts;
		}

		public float[] startNodeCondProbs(int d) {
			return startNodeCondProbs[d];
		}

		public float[] endNodeCondProbs(int d) {
			return endNodeCondProbs[d];
		}
		
		public float sequenceLogMarginalProb(int d) {
			return sequenceLogMarginalProbs[d];
		}

		public float logMarginalProb() {
			return a.sum(sequenceLogMarginalProbs);
		}
		
		public int numSequences() {
			return lattice.numSequences();
		}

		public int sequenceLength(int d) {
			return lattice.sequenceLength(d);
		}

		public int numStates(int d, int t) {
			return lattice.numStates(d, t);
		}
		
		private class EdgeMarginalsIterator implements Iterator<Pair<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Float>> {
			int d;
			int t;
			int s1;
			int s2i;
			float[] edgeCondProbs = null;
			public EdgeMarginalsIterator() {
				this.d = 0;
				this.t = 0;
				this.s1 = 0;
				this.s2i = 0;
				this.edgeCondProbs = allowedForwardEdgesExpectedCounts(d, t, s1);
				while (d < numSequences() && edgeCondProbs.length == 0) advance();
			}
			public boolean hasNext() {
				return d < numSequences();
			}
			public Pair<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Float> next() {
				Pair<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Float> result = Pair.makePair(Pair.makePair(Pair.makePair(d, t), Pair.makePair(s1, allowedForwardEdges(d, t, s1)[s2i])), edgeCondProbs[s2i]);
				advance();
				while (d < numSequences() && edgeCondProbs.length == 0) advance();
				return result;
			}
			private void advance() {
				if (s2i >= allowedForwardEdges(d, t, s1).length-1) {
					this.s2i = 0;
					if (s1 >= numStates(d, t)-1) {
						this.s1 = 0;
						if (t >= sequenceLength(d)-2) {
							this.t=0;
							this.d++;
						} else {
							this.t++;
						}
					} else {
						this.s1++;
					}
					if (d < numSequences()) this.edgeCondProbs = allowedForwardEdgesExpectedCounts(d, t, s1);
				} else {
					this.s2i++;
				}
			}
			public void remove() {
			}
		}

		public Iterator<Pair<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Float>> getEdgeMarginalsIterator() {
			return new EdgeMarginalsIterator();
		}
		
		private class StartMarginalsIterator implements Iterator<Pair<Pair<Integer, Integer>, Float>> {
			int d;
			int s;
			float[] startCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && s == numStates(d, 0)-1);
			}
			public Pair<Pair<Integer, Integer>, Float> next() {
				if (startCondProbs == null) {
					this.d = 0;
					this.s = 0;
					this.startCondProbs = startNodeCondProbs(d);
				} else {
					if (s == numStates(d,0)-1) {
						this.s = 0;
						this.d++;
						this.startCondProbs = startNodeCondProbs(d);
					} else {
						this.s++;
					}
				}
				return Pair.makePair(Pair.makePair(d, s), startCondProbs[s]);
			}
			public void remove() {
			}
		}

		public Iterator<Pair<Pair<Integer, Integer>, Float>> getStartMarginalsIterator() {
			return new StartMarginalsIterator();
		}
		
		private class EndMarginalsIterator implements Iterator<Pair<Pair<Integer, Integer>, Float>> {
			int d;
			int s;
			float[] endCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && s == numStates(d, sequenceLength(d)-1)-1);
			}
			public Pair<Pair<Integer, Integer>, Float> next() {
				if (endCondProbs == null) {
					this.d = 0;
					this.s = 0;
					this.endCondProbs = endNodeCondProbs(d);
				} else {
					if (s == numStates(d,sequenceLength(d)-1)-1) {
						this.s = 0;
						this.d++;
						this.endCondProbs = endNodeCondProbs(d);
					} else {
						this.s++;
					}
				}
				return Pair.makePair(Pair.makePair(d, s), endCondProbs[s]);
			}
			public void remove() {
			}
		}

		public Iterator<Pair<Pair<Integer, Integer>, Float>> getEndMarginalsIterator() {
			return new EndMarginalsIterator();
		}
		
	}
	
	private static class NonStationaryEdgeMarginalsScaling implements NonStationaryEdgeMarginals {
		Lattice lattice;
		float[] sequenceLogMarginalProbs;
		float[] sequenceMarginalProbs;
		float[] sequenceMarginalProbLogScales;
		float[][] startNodeCondProbs;
		float[][] endNodeCondProbs;
		float[][][] allAlphas;
		float[][][] allBetas;
		float[][] allAlphaLogScales;
		float[][] allBetaLogScales;
		
		public NonStationaryEdgeMarginalsScaling(Lattice lattice) {
			this.lattice = lattice;
			this.sequenceLogMarginalProbs = new float[lattice.numSequences()];
			this.sequenceMarginalProbs = new float[lattice.numSequences()];
			this.sequenceMarginalProbLogScales = new float[lattice.numSequences()];
			this.startNodeCondProbs = new float[lattice.numSequences()][];
			this.endNodeCondProbs = new float[lattice.numSequences()][];
			this.allAlphas = new float[lattice.numSequences()][][];
			this.allAlphaLogScales = new float[lattice.numSequences()][];
			this.allBetaLogScales = new float[lattice.numSequences()][];
		}

		public int[] allowedForwardEdges(int d, int t, int s) {
			return lattice.allowedEdges(d, t, s, false);
		}
		
		public void incrementExpectedCounts(float[][] alphas, float[] alphaLogScales, float[][] betas, float betaLogScales[], int d) {
			this.allAlphas[d] = alphas;
			this.allAlphaLogScales[d] = alphaLogScales;
			this.allBetas[d] = betas;
			this.allBetaLogScales[d] = betaLogScales;
			
			sequenceMarginalProbs[d] = 0.0f;
			sequenceMarginalProbLogScales[d] = betaLogScales[0];
			for (int s=0; s<lattice.numStates(d, 0); ++s) {
				float nodePotential = lattice.nodePotential(d, 0, s);
				if (nodePotential > 0.0) {
					sequenceMarginalProbs[d] += betas[0][s];
				}
			}
			this.sequenceLogMarginalProbs[d] = sequenceMarginalProbLogScales[d] * LOG_SCALE + (float) Math.log(sequenceMarginalProbs[d]);
			
			this.startNodeCondProbs[d] = new float[lattice.numStates(d, 0)];
			{
				float scale = getScaleFactor(betaLogScales[0] - sequenceMarginalProbLogScales[d]);
				for (int s=0; s<lattice.numStates(d, 0); ++s) {
					this.startNodeCondProbs[d][s] = (betas[0][s] / sequenceMarginalProbs[d]) * scale;
				}
			}
			
			this.endNodeCondProbs[d] = new float[lattice.numStates(d, lattice.sequenceLength(d)-1)];
			{
				float scale = getScaleFactor(alphaLogScales[lattice.sequenceLength(d)-1] - sequenceMarginalProbLogScales[d]);
				for (int s=0; s<lattice.numStates(d, lattice.sequenceLength(d)-1); ++s) {
					this.endNodeCondProbs[d][s] = (alphas[lattice.sequenceLength(d)-1][s] / sequenceMarginalProbs[d]) * scale;
				}
			}
		}
		
		public double estimateMemoryUsage() {
			double numElements = 0.0;
			for (int d=0; d<numSequences(); ++d) {
				if (allAlphas[d] != null) {
					for (int t=0; t<sequenceLength(d); ++t) {
						numElements += allAlphas[d][t].length;
					}
				}
			}
			for (int d=0; d<numSequences(); ++d) {
				if (allBetas[d] != null) {
					for (int t=0; t<sequenceLength(d); ++t) {
						numElements += allBetas[d][t].length;
					}
				}
			}
			return 8 * numElements / 1e9;
		}
		
		public float[] allowedForwardEdgesExpectedCounts(int d, int t, int s) {
			int[] allowedEdges = lattice.allowedEdges(d, t, s, false);
			float[] allowedForwardEdgesExpectedCounts = new float[allowedEdges.length];
			float[] alowedEdgesPotentials = lattice.allowedEdgesPotentials(d, t, s, false);
			float scale = getScaleFactor(allAlphaLogScales[d][t] + allBetaLogScales[d][t+1] - sequenceMarginalProbLogScales[d]);
			for (int i=0; i<allowedEdges.length; ++i) {
				int nextS = allowedEdges[i];
				float edgePotential = alowedEdgesPotentials[i];
				allowedForwardEdgesExpectedCounts[i] += (allAlphas[d][t][s] * edgePotential *  allBetas[d][t+1][nextS] / sequenceMarginalProbs[d]) * scale;
			}
			return allowedForwardEdgesExpectedCounts;
		}
		
		public float[] startNodeCondProbs(int d) {
			return startNodeCondProbs[d];
		}
		
		public float[] endNodeCondProbs(int d) {
			return endNodeCondProbs[d];
		}
		
		public float sequenceLogMarginalProb(int d) {
			return sequenceLogMarginalProbs[d];
		}
		
		public float logMarginalProb() {
			return a.sum(sequenceLogMarginalProbs);
		}
		
		public int numSequences() {
			return lattice.numSequences();
		}

		public int sequenceLength(int d) {
			return lattice.sequenceLength(d);
		}

		public int numStates(int d, int t) {
			return lattice.numStates(d, t);
		}
		
		private class EdgeMarginalsIterator implements Iterator<Pair<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Float>> {
			int d;
			int t;
			int s1;
			int s2i;
			float[] edgeCondProbs = null;
			public EdgeMarginalsIterator() {
				this.d = 0;
				this.t = 0;
				this.s1 = 0;
				this.s2i = 0;
				this.edgeCondProbs = allowedForwardEdgesExpectedCounts(d, t, s1);
				while (d < numSequences() && edgeCondProbs.length == 0) advance();
			}
			public boolean hasNext() {
				return d < numSequences();
			}
			public Pair<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Float> next() {
				Pair<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Float> result = Pair.makePair(Pair.makePair(Pair.makePair(d, t), Pair.makePair(s1, allowedForwardEdges(d, t, s1)[s2i])), edgeCondProbs[s2i]);
				advance();
				while (d < numSequences() && edgeCondProbs.length == 0) advance();
				return result;
			}
			private void advance() {
				if (s2i >= allowedForwardEdges(d, t, s1).length-1) {
					this.s2i = 0;
					if (s1 >= numStates(d, t)-1) {
						this.s1 = 0;
						if (t >= sequenceLength(d)-2) {
							this.t=0;
							this.d++;
						} else {
							this.t++;
						}
					} else {
						this.s1++;
					}
					if (d < numSequences()) this.edgeCondProbs = allowedForwardEdgesExpectedCounts(d, t, s1);
				} else {
					this.s2i++;
				}
			}
			public void remove() {
			}
		}

		public Iterator<Pair<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Float>> getEdgeMarginalsIterator() {
			return new EdgeMarginalsIterator();
		}
		
		private class StartMarginalsIterator implements Iterator<Pair<Pair<Integer, Integer>, Float>> {
			int d;
			int s;
			float[] startCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && s == numStates(d, 0)-1);
			}
			public Pair<Pair<Integer, Integer>, Float> next() {
				if (startCondProbs == null) {
					this.d = 0;
					this.s = 0;
					this.startCondProbs = startNodeCondProbs(d);
				} else {
					if (s == numStates(d,0)-1) {
						this.s = 0;
						this.d++;
						this.startCondProbs = startNodeCondProbs(d);
					} else {
						this.s++;
					}
				}
				return Pair.makePair(Pair.makePair(d, s), startCondProbs[s]);
			}
			public void remove() {
			}
		}

		public Iterator<Pair<Pair<Integer, Integer>, Float>> getStartMarginalsIterator() {
			return new StartMarginalsIterator();
		}
		
		private class EndMarginalsIterator implements Iterator<Pair<Pair<Integer, Integer>, Float>> {
			int d;
			int s;
			float[] endCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && s == numStates(d, sequenceLength(d)-1)-1);
			}
			public Pair<Pair<Integer, Integer>, Float> next() {
				if (endCondProbs == null) {
					this.d = 0;
					this.s = 0;
					this.endCondProbs = endNodeCondProbs(d);
				} else {
					if (s == numStates(d,sequenceLength(d)-1)-1) {
						this.s = 0;
						this.d++;
						this.endCondProbs = endNodeCondProbs(d);
					} else {
						this.s++;
					}
				}
				return Pair.makePair(Pair.makePair(d, s), endCondProbs[s]);
			}
			public void remove() {
			}
		}

		public Iterator<Pair<Pair<Integer, Integer>, Float>> getEndMarginalsIterator() {
			return new EndMarginalsIterator();
		}
		
	}

	private static class StationaryEdgeMarginalsLogSpace implements StationaryEdgeMarginals {
		StationaryLattice lattice;
		float[] sequenceLogMarginalProbs;
		float[][][] allowedForwardEdgesExpectedCounts;
		float[][] startNodeCondProbs;
		float[][] endNodeCondProbs;

		public StationaryEdgeMarginalsLogSpace(StationaryLattice lattice) {
			this.lattice = lattice;
			this.sequenceLogMarginalProbs = new float[lattice.numSequences()];
			this.allowedForwardEdgesExpectedCounts = new float[lattice.numSequences()][][];
			this.startNodeCondProbs = new float[lattice.numSequences()][];
			this.endNodeCondProbs = new float[lattice.numSequences()][];
		}
		
		public int[] allowedForwardEdges(int d, int s) {
			return lattice.allowedEdges(d, s, false);
		}
		
		public void incrementExpectedCounts(float[][] alphas, float[][] betas, int d) {
			this.sequenceLogMarginalProbs[d] = Float.NEGATIVE_INFINITY;
			for (int s=0; s<lattice.numStates(d); ++s) {
				this.sequenceLogMarginalProbs[d] = m.logAdd(sequenceLogMarginalProbs[d], betas[0][s]);
			}
			
			this.allowedForwardEdgesExpectedCounts[d] = new float[lattice.numStates(d)][];
			for (int s=0; s<lattice.numStates(d); ++s) {
				this.allowedForwardEdgesExpectedCounts[d][s] = new float[lattice.allowedEdges(d, s, false).length];
			}
			for (int t=0; t<lattice.sequenceLength(d)-1; ++t) {
				int numStates = lattice.numStates(d);
				for (int s=0; s<numStates; ++s) {
					int[] allowedEdges = lattice.allowedEdges(d, s, false);
					float[] alowedEdgesLogPotentials = lattice.allowedEdgesLogPotentials(d, s, false);
					for (int i=0; i<allowedEdges.length; ++i) {
						int nextS = allowedEdges[i];
						float edgeLogPotential = alowedEdgesLogPotentials[i];
						this.allowedForwardEdgesExpectedCounts[d][s][i] += Math.exp(alphas[t][s] + edgeLogPotential + betas[t+1][nextS] - sequenceLogMarginalProbs[d]);
					}
				}
			}
			
			this.startNodeCondProbs[d] = new float[lattice.numStates(d)];
			for (int s=0; s<lattice.numStates(d); ++s) {
				this.startNodeCondProbs[d][s] = (float) Math.exp(betas[0][s] - sequenceLogMarginalProbs[d]);
			}
			
			this.endNodeCondProbs[d] = new float[lattice.numStates(d)];
			for (int s=0; s<lattice.numStates(d); ++s) {
				this.endNodeCondProbs[d][s] = (float) Math.exp(alphas[lattice.sequenceLength(d)-1][s] - sequenceLogMarginalProbs[d]);
			}
		}
		
		public double estimateMemoryUsage() {
			double numElements = 0.0;
			for (int d=0; d<numSequences(); ++d) {
				if (allowedForwardEdgesExpectedCounts[d] != null) {
					for (int s=0; s<numStates(d); ++s) {
						numElements += allowedForwardEdgesExpectedCounts[d][s].length;
					}
				}
			}
			return 8 * numElements / 1e9;
		}

		public float[] allowedForwardEdgesExpectedCounts(int d, int s) {
			return allowedForwardEdgesExpectedCounts[d][s];
		}

		public float[] startNodeCondProbs(int d) {
			return startNodeCondProbs[d];
		}

		public float[] endNodeCondProbs(int d) {
			return endNodeCondProbs[d];
		}
		
		public float sequenceLogMarginalProb(int d) {
			return sequenceLogMarginalProbs[d];
		}

		public float logMarginalProb() {
			return a.sum(sequenceLogMarginalProbs);
		}
		
		public int numSequences() {
			return lattice.numSequences();
		}

		public int numStates(int d) {
			return lattice.numStates(d);
		}
		
		private class EdgeMarginalsIterator implements Iterator<Pair<Pair<Integer, Pair<Integer, Integer>>, Float>> {
			int d;
			int s1;
			int s2i;
			float[] edgeCondProbs = null;
			public EdgeMarginalsIterator() {
				this.d = 0;
				this.s1 = 0;
				this.s2i = 0;
				this.edgeCondProbs = allowedForwardEdgesExpectedCounts(d, s1);
				while (d < numSequences() && edgeCondProbs.length == 0) advance();
			}
			public boolean hasNext() {
				return d < numSequences();
			}
			public Pair<Pair<Integer, Pair<Integer, Integer>>, Float> next() {
				Pair<Pair<Integer, Pair<Integer, Integer>>, Float> result = Pair.makePair(Pair.makePair(d, Pair.makePair(s1, allowedForwardEdges(d, s1)[s2i])), edgeCondProbs[s2i]);
				advance();
				while (d < numSequences() && edgeCondProbs.length == 0) advance();
				return result;
			}
			private void advance() {
				if (s2i >= allowedForwardEdges(d, s1).length-1) {
					this.s2i = 0;
					if (s1 >= numStates(d)-1) {
						this.s1 = 0;
						this.d++;
					} else {
						this.s1++;
					}
					if (d < numSequences()) this.edgeCondProbs = allowedForwardEdgesExpectedCounts(d, s1);
				} else {
					this.s2i++;
				}
			}
			public void remove() {
			}
		}

		public Iterator<Pair<Pair<Integer, Pair<Integer, Integer>>, Float>> getEdgeMarginalsIterator() {
			return new EdgeMarginalsIterator();
		}
		
		private class StartMarginalsIterator implements Iterator<Pair<Pair<Integer, Integer>, Float>> {
			int d;
			int s;
			float[] startCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && s == numStates(d)-1);
			}
			public Pair<Pair<Integer, Integer>, Float> next() {
				if (startCondProbs == null) {
					this.d = 0;
					this.s = 0;
					this.startCondProbs = startNodeCondProbs(d);
				} else {
					if (s == numStates(d)-1) {
						this.s = 0;
						this.d++;
						this.startCondProbs = startNodeCondProbs(d);
					} else {
						this.s++;
					}
				}
				return Pair.makePair(Pair.makePair(d, s), startCondProbs[s]);
			}
			public void remove() {
			}
		}
		
		public Iterator<Pair<Pair<Integer, Integer>, Float>> getStartMarginalsIterator() {
			return new StartMarginalsIterator();
		}
		
		private class EndMarginalsIterator implements Iterator<Pair<Pair<Integer, Integer>, Float>> {
			int d;
			int s;
			float[] endCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && s == numStates(d)-1);
			}
			public Pair<Pair<Integer, Integer>, Float> next() {
				if (endCondProbs == null) {
					this.d = 0;
					this.s = 0;
					this.endCondProbs = endNodeCondProbs(d);
				} else {
					if (s == numStates(d)-1) {
						this.s = 0;
						this.d++;
						this.endCondProbs = endNodeCondProbs(d);
					} else {
						this.s++;
					}
				}
				return Pair.makePair(Pair.makePair(d, s), endCondProbs[s]);
			}
			public void remove() {
			}
		}
		
		public Iterator<Pair<Pair<Integer, Integer>, Float>> getEndMarginalsIterator() {
			return new EndMarginalsIterator();
		}
		
	}
	
	private static class StationaryEdgeMarginalsScaling implements StationaryEdgeMarginals {
		StationaryLattice lattice;
		float[] sequenceLogMarginalProbs;
		float[][][] allowedForwardEdgesExpectedCounts;
		float[][] startNodeCondProbs;
		float[][] endNodeCondProbs;
		
		public StationaryEdgeMarginalsScaling(StationaryLattice lattice) {
			this.lattice = lattice;
			this.sequenceLogMarginalProbs = new float[lattice.numSequences()];
			this.allowedForwardEdgesExpectedCounts = new float[lattice.numSequences()][][];
			this.startNodeCondProbs = new float[lattice.numSequences()][];
			this.endNodeCondProbs = new float[lattice.numSequences()][];
		}
		
		public int[] allowedForwardEdges(int d, int s) {
			return lattice.allowedEdges(d, s, false);
		}
		
		public void incrementExpectedCounts(float[][] alphas, float[] alphaLogScales, float[][] betas, float betaLogScales[], int d) {
			float sequenceMarginalProb = 0.0f;
			float sequenceMarginalProbLogScale = betaLogScales[0];
			for (int s=0; s<lattice.numStates(d); ++s) {
				float nodePotential = lattice.nodePotential(d, 0, s);
				if (nodePotential > 0.0) {
					sequenceMarginalProb += betas[0][s];
				}
			}
			this.sequenceLogMarginalProbs[d] = sequenceMarginalProbLogScale * LOG_SCALE + (float) Math.log(sequenceMarginalProb);
			
			this.allowedForwardEdgesExpectedCounts[d] = new float[lattice.numStates(d)][];
			for (int s=0; s<lattice.numStates(d); ++s) {
				this.allowedForwardEdgesExpectedCounts[d][s] = new float[lattice.allowedEdges(d, s, false).length];
			}
			for (int t=0; t<lattice.sequenceLength(d)-1; ++t) {
				float scale = getScaleFactor(alphaLogScales[t] + betaLogScales[t+1] - sequenceMarginalProbLogScale);
				int numStates = lattice.numStates(d);
				for (int s=0; s<numStates; ++s) {
					int[] allowedEdges = lattice.allowedEdges(d, s, false);
					float[] alowedEdgesPotentials = lattice.allowedEdgesPotentials(d, s, false);
					for (int i=0; i<allowedEdges.length; ++i) {
						int nextS = allowedEdges[i];
						float edgePotential = alowedEdgesPotentials[i];
						this.allowedForwardEdgesExpectedCounts[d][s][i] += (alphas[t][s] * edgePotential *  betas[t+1][nextS] / sequenceMarginalProb) * scale;
					}
				}
			}
			
			this.startNodeCondProbs[d] = new float[lattice.numStates(d)];
			{
				float scale = getScaleFactor(betaLogScales[0] - sequenceMarginalProbLogScale);
				for (int s=0; s<lattice.numStates(d); ++s) {
					this.startNodeCondProbs[d][s] = (betas[0][s] / sequenceMarginalProb) * scale;
				}
			}
			
			this.endNodeCondProbs[d] = new float[lattice.numStates(d)];
			{
				float scale = getScaleFactor(alphaLogScales[lattice.sequenceLength(d)-1] - sequenceMarginalProbLogScale);
				for (int s=0; s<lattice.numStates(d); ++s) {
					this.endNodeCondProbs[d][s] = (alphas[lattice.sequenceLength(d)-1][s] / sequenceMarginalProb) * scale;
				}
			}
		}
		
		public double estimateMemoryUsage() {
			double numElements = 0.0;
			for (int d=0; d<numSequences(); ++d) {
				if (allowedForwardEdgesExpectedCounts[d] != null) {
					for (int s=0; s<numStates(d); ++s) {
						numElements += allowedForwardEdgesExpectedCounts[d][s].length;
					}
				}
			}
			return 8 * numElements / 1e9;
		}
		
		public float[] allowedForwardEdgesExpectedCounts(int d, int s) {
			return allowedForwardEdgesExpectedCounts[d][s];
		}
		
		public float[] startNodeCondProbs(int d) {
			return startNodeCondProbs[d];
		}
		
		public float[] endNodeCondProbs(int d) {
			return endNodeCondProbs[d];
		}
		
		public float sequenceLogMarginalProb(int d) {
			return sequenceLogMarginalProbs[d];
		}
		
		public float logMarginalProb() {
			return a.sum(sequenceLogMarginalProbs);
		}
		
		public int numSequences() {
			return lattice.numSequences();
		}

		public int numStates(int d) {
			return lattice.numStates(d);
		}

		private class EdgeMarginalsIterator implements Iterator<Pair<Pair<Integer, Pair<Integer, Integer>>, Float>> {
			int d;
			int s1;
			int s2i;
			float[] edgeCondProbs = null;
			public EdgeMarginalsIterator() {
				this.d = 0;
				this.s1 = 0;
				this.s2i = 0;
				this.edgeCondProbs = allowedForwardEdgesExpectedCounts(d, s1);
				while (d < numSequences() && edgeCondProbs.length == 0) advance();
			}
			public boolean hasNext() {
				return d < numSequences();
			}
			public Pair<Pair<Integer, Pair<Integer, Integer>>, Float> next() {
				Pair<Pair<Integer, Pair<Integer, Integer>>, Float> result = Pair.makePair(Pair.makePair(d, Pair.makePair(s1, allowedForwardEdges(d, s1)[s2i])), edgeCondProbs[s2i]);
				advance();
				while (d < numSequences() && edgeCondProbs.length == 0) advance();
				return result;
			}
			private void advance() {
				if (s2i >= allowedForwardEdges(d, s1).length-1) {
					this.s2i = 0;
					if (s1 >= numStates(d)-1) {
						this.s1 = 0;
						this.d++;
					} else {
						this.s1++;
					}
					if (d < numSequences()) this.edgeCondProbs = allowedForwardEdgesExpectedCounts(d, s1);
				} else {
					this.s2i++;
				}
			}
			public void remove() {
			}
		}
		
		public Iterator<Pair<Pair<Integer, Pair<Integer, Integer>>, Float>> getEdgeMarginalsIterator() {
			return new EdgeMarginalsIterator();
		}
		
		private class StartMarginalsIterator implements Iterator<Pair<Pair<Integer, Integer>, Float>> {
			int d;
			int s;
			float[] startCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && s == numStates(d)-1);
			}
			public Pair<Pair<Integer, Integer>, Float> next() {
				if (startCondProbs == null) {
					this.d = 0;
					this.s = 0;
					this.startCondProbs = startNodeCondProbs(d);
				} else {
					if (s == numStates(d)-1) {
						this.s = 0;
						this.d++;
						this.startCondProbs = startNodeCondProbs(d);
					} else {
						this.s++;
					}
				}
				return Pair.makePair(Pair.makePair(d, s), startCondProbs[s]);
			}
			public void remove() {
			}
		}

		public Iterator<Pair<Pair<Integer, Integer>, Float>> getStartMarginalsIterator() {
			return new StartMarginalsIterator();
		}
		
		private class EndMarginalsIterator implements Iterator<Pair<Pair<Integer, Integer>, Float>> {
			int d;
			int s;
			float[] endCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && s == numStates(d)-1);
			}
			public Pair<Pair<Integer, Integer>, Float> next() {
				if (endCondProbs == null) {
					this.d = 0;
					this.s = 0;
					this.endCondProbs = endNodeCondProbs(d);
				} else {
					if (s == numStates(d)-1) {
						this.s = 0;
						this.d++;
						this.endCondProbs = endNodeCondProbs(d);
					} else {
						this.s++;
					}
				}
				return Pair.makePair(Pair.makePair(d, s), endCondProbs[s]);
			}
			public void remove() {
			}
		}
		

		public Iterator<Pair<Pair<Integer, Integer>, Float>> getEndMarginalsIterator() {
			return new EndMarginalsIterator();
		}
		
	}

	public static Pair<NodeMarginals,StationaryEdgeMarginals> computeMarginalsLogSpace(final StationaryLattice lattice, final StationaryStateProjector nodeMarginalsStateProjector, final boolean viterbiEmissionOnly, int numThreads) {
		final NodeMarginalsLogSpace projectedNodeMarginals = new NodeMarginalsLogSpace(new StationaryLatticeWrapper(lattice), nodeMarginalsStateProjector);
		final StationaryEdgeMarginalsLogSpace edgeMarginals = (viterbiEmissionOnly ? null : new StationaryEdgeMarginalsLogSpace(lattice));
		BetterThreader.Function<Integer,Object> func = new BetterThreader.Function<Integer,Object>(){public void call(Integer d, Object ignore){
			float[][] alphas = doPassLogSpace(new StationaryLatticeWrapper(lattice), false, viterbiEmissionOnly, d);
			float[][] betas = doPassLogSpace(new StationaryLatticeWrapper(lattice), true, viterbiEmissionOnly, d);
			projectedNodeMarginals.incrementExpectedCounts(alphas, betas, d, viterbiEmissionOnly);
			if (!viterbiEmissionOnly) edgeMarginals.incrementExpectedCounts(alphas, betas, d);
		}};
		BetterThreader<Integer,Object> threader = new BetterThreader<Integer,Object>(func, numThreads);
		for (int d=0; d<lattice.numSequences(); ++d) threader.addFunctionArgument(d);
		threader.run();
//		System.out.printf("Estimated node marginals size: %.3fgb\n", projectedNodeMarginals.estimateMemoryUsage());
//		if (!viterbiEmissionOnly) System.out.printf("Estimated edge marginals size: %.3fgb\n", edgeMarginals.estimateMemoryUsage());
		return Pair.makePair((NodeMarginals) projectedNodeMarginals, (StationaryEdgeMarginals) edgeMarginals);
	}
	
	public static Pair<NodeMarginals,NonStationaryEdgeMarginals> computeMarginalsLogSpace(final Lattice lattice, final StationaryStateProjector nodeMarginalsStateProjector, final boolean viterbiEmissionOnly, int numThreads) {
		final NodeMarginalsLogSpace projectedNodeMarginals = new NodeMarginalsLogSpace(lattice, nodeMarginalsStateProjector);
		final NonStationaryEdgeMarginalsLogSpace edgeMarginals = (viterbiEmissionOnly ? null : new NonStationaryEdgeMarginalsLogSpace(lattice));
		BetterThreader.Function<Integer,Object> func = new BetterThreader.Function<Integer,Object>(){public void call(Integer d, Object ignore){
			float[][] alphas = doPassLogSpace(lattice, false, viterbiEmissionOnly, d);
			float[][] betas = doPassLogSpace(lattice, true, viterbiEmissionOnly, d);
			projectedNodeMarginals.incrementExpectedCounts(alphas, betas, d, viterbiEmissionOnly);
			if (!viterbiEmissionOnly) edgeMarginals.incrementExpectedCounts(alphas, betas, d);
		}};
		BetterThreader<Integer,Object> threader = new BetterThreader<Integer,Object>(func, numThreads);
		for (int d=0; d<lattice.numSequences(); ++d) threader.addFunctionArgument(d);
		threader.run();
//		System.out.printf("Estimated node marginals size: %.3fgb\n", projectedNodeMarginals.estimateMemoryUsage());
//		if (!viterbiEmissionOnly) System.out.printf("Estimated edge marginals size: %.3fgb\n", edgeMarginals.estimateMemoryUsage());
		return Pair.makePair((NodeMarginals) projectedNodeMarginals, (NonStationaryEdgeMarginals) edgeMarginals);
	}
	
	public static Pair<NodeMarginals,StationaryEdgeMarginals> computeMarginalsScaling(final StationaryLattice lattice, final StationaryStateProjector nodeMarginalsStateProjector, final boolean viterbiEmissionOnly, int numThreads) {
		final NodeMarginalsScaling projectedNodeMarginals = new NodeMarginalsScaling(new StationaryLatticeWrapper(lattice), nodeMarginalsStateProjector);
		final StationaryEdgeMarginalsScaling edgeMarginals = (viterbiEmissionOnly ? null : new StationaryEdgeMarginalsScaling(lattice));
		BetterThreader.Function<Integer,Object> func = new BetterThreader.Function<Integer,Object>(){public void call(Integer d, Object ignore){
			Pair<float[][],float[]> alphasAndScales = doPassScaling(new StationaryLatticeWrapper(lattice), false, viterbiEmissionOnly, d);
			float[][] alphas = alphasAndScales.getFirst();
			float [] alphaLogScales = alphasAndScales.getSecond();
			Pair<float[][],float[]> betasAndScales = doPassScaling(new StationaryLatticeWrapper(lattice), true, viterbiEmissionOnly, d);
			float[][] betas = betasAndScales.getFirst();
			float [] betaLogScales = betasAndScales.getSecond();
			projectedNodeMarginals.incrementExpectedCounts(alphas, alphaLogScales, betas, betaLogScales, d, viterbiEmissionOnly);
			if (!viterbiEmissionOnly) edgeMarginals.incrementExpectedCounts(alphas, alphaLogScales, betas, betaLogScales, d);
		}};
		BetterThreader<Integer,Object> threader = new BetterThreader<Integer,Object>(func, numThreads);
		for (int d=0; d<lattice.numSequences(); ++d) threader.addFunctionArgument(d);
		threader.run();
//		System.out.printf("Estimated node marginals size: %.3fgb\n", projectedNodeMarginals.estimateMemoryUsage());
//		if (!viterbiEmissionOnly) System.out.printf("Estimated edge marginals size: %.3fgb\n", edgeMarginals.estimateMemoryUsage());
		return Pair.makePair((NodeMarginals) projectedNodeMarginals, (StationaryEdgeMarginals) edgeMarginals);
	}
	
	public static Pair<NodeMarginals,NonStationaryEdgeMarginals> computeMarginalsScaling(final Lattice lattice, final StationaryStateProjector nodeMarginalsStateProjector, final boolean viterbiEmissionOnly, int numThreads) {
		final NodeMarginalsScaling projectedNodeMarginals = new NodeMarginalsScaling(lattice, nodeMarginalsStateProjector);
		final NonStationaryEdgeMarginalsScaling edgeMarginals = (viterbiEmissionOnly ? null : new NonStationaryEdgeMarginalsScaling(lattice));
		BetterThreader.Function<Integer,Object> func = new BetterThreader.Function<Integer,Object>(){public void call(Integer d, Object ignore){
			Pair<float[][],float[]> alphasAndScales = doPassScaling(lattice, false, viterbiEmissionOnly, d);
			float[][] alphas = alphasAndScales.getFirst();
			float [] alphaLogScales = alphasAndScales.getSecond();
			Pair<float[][],float[]> betasAndScales = doPassScaling(lattice, true, viterbiEmissionOnly, d);
			float[][] betas = betasAndScales.getFirst();
			float [] betaLogScales = betasAndScales.getSecond();
			projectedNodeMarginals.incrementExpectedCounts(alphas, alphaLogScales, betas, betaLogScales, d, viterbiEmissionOnly);
			if (!viterbiEmissionOnly) edgeMarginals.incrementExpectedCounts(alphas, alphaLogScales, betas, betaLogScales, d);
		}};
		BetterThreader<Integer,Object> threader = new BetterThreader<Integer,Object>(func, numThreads);
		for (int d=0; d<lattice.numSequences(); ++d) threader.addFunctionArgument(d);
		threader.run();
//		System.out.printf("Estimated node marginals size: %.3fgb\n", projectedNodeMarginals.estimateMemoryUsage());
//		if (!viterbiEmissionOnly) System.out.printf("Estimated edge marginals size: %.3fgb\n", edgeMarginals.estimateMemoryUsage());
		return Pair.makePair((NodeMarginals) projectedNodeMarginals, (NonStationaryEdgeMarginals) edgeMarginals);
	}
	
	public static int[][] computeViterbiPathsScaling(final Lattice lattice, final int numThreads) {
		final int[][] viterbiSequences = new int[lattice.numSequences()][];
		BetterThreader.Function<Integer,Object> func = new BetterThreader.Function<Integer,Object>(){public void call(Integer d, Object ignore){
			float[][] alphas = doPassScaling(lattice, false, true, d).getFirst();
			viterbiSequences[d] = extractViterbiPath(lattice, null, alphas, true, d);
		}};
		BetterThreader<Integer,Object> threader = new BetterThreader<Integer,Object>(func, numThreads);
		for (int d=0; d<lattice.numSequences(); ++d) threader.addFunctionArgument(d);
		threader.run();
		return viterbiSequences;
	}
	
	public static int[][] computeViterbiPathsScaling(final Lattice lattice, final StationaryStateProjector stateProjector, final int numThreads) {
		final int[][] viterbiSequences = new int[lattice.numSequences()][];
		BetterThreader.Function<Integer,Object> func = new BetterThreader.Function<Integer,Object>(){public void call(Integer d, Object ignore){
			float[][] alphas = doPassScaling(lattice, false, true, d).getFirst();
			viterbiSequences[d] = extractViterbiPath(lattice, stateProjector, alphas, true, d);
		}};
		BetterThreader<Integer,Object> threader = new BetterThreader<Integer,Object>(func, numThreads);
		for (int d=0; d<lattice.numSequences(); ++d) threader.addFunctionArgument(d);
		threader.run();
		return viterbiSequences;
	}
	
	public static int[][] computeViterbiPathsLogSpace(final Lattice lattice, final int numThreads) {
		final int[][] viterbiSequences = new int[lattice.numSequences()][];
		BetterThreader.Function<Integer,Object> func = new BetterThreader.Function<Integer,Object>(){public void call(Integer d, Object ignore){
			float[][] alphas = doPassLogSpace(lattice, false, true, d);
			viterbiSequences[d] = extractViterbiPath(lattice, null, alphas, false, d);
		}};
		BetterThreader<Integer,Object> threader = new BetterThreader<Integer,Object>(func, numThreads);
		for (int d=0; d<lattice.numSequences(); ++d) threader.addFunctionArgument(d);
		threader.run();
		return viterbiSequences;
	}
	
	public static int[][] computeViterbiPathsLogSpace(final Lattice lattice, final StationaryStateProjector stateProjector, final int numThreads) {
		final int[][] viterbiSequences = new int[lattice.numSequences()][];
		BetterThreader.Function<Integer,Object> func = new BetterThreader.Function<Integer,Object>(){public void call(Integer d, Object ignore){
			float[][] alphas = doPassLogSpace(lattice, false, true, d);
			viterbiSequences[d] = extractViterbiPath(lattice, stateProjector, alphas, false, d);
		}};
		BetterThreader<Integer,Object> threader = new BetterThreader<Integer,Object>(func, numThreads);
		for (int d=0; d<lattice.numSequences(); ++d) threader.addFunctionArgument(d);
		threader.run();
		return viterbiSequences;
	}

	private static int[] extractViterbiPath(final Lattice lattice, final StationaryStateProjector stateProjector, final float[][] alphas, boolean scaling, int d) {
		int[] viterbiSequence = new int[lattice.sequenceLength(d)];
		viterbiSequence[lattice.sequenceLength(d)-1] = a.argmax(alphas[lattice.sequenceLength(d)-1]);
		for (int t=lattice.sequenceLength(d)-2; t>=0; --t) {
			int s = viterbiSequence[t+1];
			int[] prevStates = lattice.allowedEdges(d, t+1, s, true);
			int bestPrevState = -1;
			float bestScore = Float.NEGATIVE_INFINITY;
			if (scaling) {
				float[] prevStatesEdgePotentials = lattice.allowedEdgesPotentials(d, t+1, s, true);
				for (int i=0; i<prevStates.length; ++i) {
					int prevState = prevStates[i];
					float score = alphas[t][prevState]*prevStatesEdgePotentials[i];
					if (score > bestScore) {
						bestScore = score;
						bestPrevState = prevState;
					}
				}
			} else {
				float[] prevNodesEdgeLogPotentials = lattice.allowedEdgesLogPotentials(d, t+1, s, true);
				for (int i=0; i<prevStates.length; ++i) {
					int prevState = prevStates[i];
					float score = alphas[t][prevState] + prevNodesEdgeLogPotentials[i];
					if (score > bestScore) {
						bestScore = score;
						bestPrevState = prevState;
					}
				}
			}
			viterbiSequence[t] = bestPrevState;
		}
		if (stateProjector != null) {
			for (int t=0; t<viterbiSequence.length; ++t) {
				viterbiSequence[t] = stateProjector.project(d, t, viterbiSequence[t]);
			}
		}
		return viterbiSequence;
	}

	private static float[][] doPassLogSpace(final Lattice lattice, final boolean backward, final boolean viterbi, int d) {
		float[][] alphas = new float[lattice.sequenceLength(d)][];
		for (int t=0; t<lattice.sequenceLength(d); ++t) {
			alphas[t] = new float[lattice.numStates(d, t)];
		}
		
		int[] timeOrder = (backward ? a.enumerate(lattice.sequenceLength(d),0) : a.enumerate(0,lattice.sequenceLength(d)));
		for (int ti=0; ti<timeOrder.length; ++ti) {
			int t = timeOrder[ti];
			Arrays.fill(alphas[t], Float.NEGATIVE_INFINITY);
			if (ti == 0) {
				int numStates = lattice.numStates(d, t);
				for (int s=0; s<numStates; ++s) {
					alphas[t][s] = lattice.nodeLogPotential(d, t, s);
				}
			} else {
				int prevT = timeOrder[ti-1];
				int numStates = lattice.numStates(d, prevT);
				for (int prevS=0; prevS<numStates; ++prevS) {
					float prevAlpha = alphas[prevT][prevS];
					int[] nextStates = lattice.allowedEdges(d, prevT, prevS, backward);
					float[] nextStatesEdgeLogPotentials = lattice.allowedEdgesLogPotentials(d, prevT, prevS, backward);
					float[] currentAlphas = alphas[t];
					for (int i=0; i<nextStates.length; ++i) {
						int nextState = nextStates[i];
						float edgeLogPotential = nextStatesEdgeLogPotentials[i];
						if (viterbi) {
							currentAlphas[nextState] = Math.max(currentAlphas[nextState], prevAlpha + edgeLogPotential);
						} else {
							currentAlphas[nextState] = m.logAdd(currentAlphas[nextState], prevAlpha + edgeLogPotential);
						}
					}
				}
				for (int s=0; s<lattice.numStates(d, t); ++s) {
					alphas[t][s] += lattice.nodeLogPotential(d, t, s);
				}
			}
		}
		return alphas;
	}
	
	private static Pair<float[][],float[]> doPassScaling(final Lattice lattice, final boolean backward, final boolean viterbi, int d) {
		float[] logScales = new float[lattice.sequenceLength(d)];
		float[][] alphas = new float[lattice.sequenceLength(d)][];
		for (int t=0; t<lattice.sequenceLength(d); ++t) {
			alphas[t] = new float[lattice.numStates(d, t)];
		}
		
		int[] timeOrder = (backward ? a.enumerate(lattice.sequenceLength(d),0) : a.enumerate(0,lattice.sequenceLength(d)));
		for (int ti=0; ti<timeOrder.length; ++ti) {
			int t = timeOrder[ti];
			Arrays.fill(alphas[t], 0.0f);
			float max = Float.NEGATIVE_INFINITY;
			if (ti == 0) {
				int numStates = lattice.numStates(d, t);
				for (int s=0; s<numStates; ++s) {
					float alpha = lattice.nodePotential(d, t, s);
					alphas[t][s] = alpha;
					if (alpha > max) max = alpha;
				}
			} else {
				int prevT = timeOrder[ti-1];
				int numStates = lattice.numStates(d, prevT);
				for (int prevS=0; prevS<numStates; ++prevS) {
					float prevAlpha = alphas[prevT][prevS];
					int[] nextStates = lattice.allowedEdges(d, prevT, prevS, backward);
					float[] nextStatesEdgePotentials = lattice.allowedEdgesPotentials(d, prevT, prevS, backward);
					float[] currentAlphas = alphas[t];
					for (int i=0; i<nextStates.length; ++i) {
						int nextState = nextStates[i];
						float edgePotential = nextStatesEdgePotentials[i];
						if (viterbi) {
							currentAlphas[nextState] = Math.max(currentAlphas[nextState], prevAlpha * edgePotential);
						} else {
							currentAlphas[nextState] += prevAlpha * edgePotential;
						}
					}
				}
				for (int s=0; s<lattice.numStates(d, t); ++s) {
					alphas[t][s] *= lattice.nodePotential(d, t, s);
					float alpha = alphas[t][s];
					if (alpha > max) max = alpha;
				}
			}
			
			int logScale = 0;
			float scale = 1.0f;
			while (max > SCALE) {
				max /= SCALE;
				scale *= SCALE;
				logScale += 1;
			}
			while (max > 0.0 && max < 1.0 / SCALE) {
				max *= SCALE;
				scale /= SCALE;
				logScale -= 1;
			}
			if (logScale != 0) {
				for (int s=0; s<lattice.numStates(d, t); ++s) {
					alphas[t][s] /= scale;
				}
			}
			if (ti == 0) {
				logScales[t] = logScale;
			} else {
				int prevT = timeOrder[ti-1];
				logScales[t] = logScales[prevT] + logScale;
			}
		}
		return Pair.makePair(alphas, logScales);
	}

}
