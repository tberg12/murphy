package sequence;

import java.util.Arrays;
import java.util.Iterator;

import threading.BetterThreader;
import tuple.Pair;

import arrays.a;
import math.m;

public class ForwardBackward {	

	public static final double SCALE = Math.exp(100);
	public static final double INVSCALE = 1.0 / SCALE;
	public static final double LOG_SCALE = Math.log(SCALE);
	
	private static double getScaleFactor(double logScale) {
		if (logScale == 0.0) return 1.0;
		if (logScale == 1.0) return SCALE;
		if (logScale == 2.0) return SCALE * SCALE;
		if (logScale == 3.0) return SCALE * SCALE * SCALE;		
		if (logScale == -1.0) return 1.0 * INVSCALE;
		if (logScale == -2.0) return 1.0 * INVSCALE * INVSCALE;
		if (logScale == -3.0) return 1.0 * INVSCALE * INVSCALE * INVSCALE;		
		return Math.pow(SCALE, logScale);
	}
	
	public static interface StationaryLattice {
		public int numSequences();
		public int sequenceLength(int d);
		public abstract int numStates(int d);
		public double nodeLogPotential(int d, int t, int s);
		public abstract double[] allowedEdgesLogPotentials(int d, int s, boolean backward);
		public double nodePotential(int d, int t, int s);
		public abstract double[] allowedEdgesPotentials(int d, int s, boolean backward);
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
		public double nodeLogPotential(int d, int t, int s) {
			return lattice.nodeLogPotential(d, t, s);
		}
		public double[] allowedEdgesLogPotentials(int d, int t, int s, boolean backward) {
			return lattice.allowedEdgesLogPotentials(d, s, backward);
		}
		public double nodePotential(int d, int t, int s) {
			return lattice.nodePotential(d, t, s);
		}
		public double[] allowedEdgesPotentials(int d, int t, int s, boolean backward) {
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
		public double nodeLogPotential(int d, int t, int s);
		public double[] allowedEdgesLogPotentials(int d, int t, int s, boolean backward);
		public double nodePotential(int d, int t, int s);
		public double[] allowedEdgesPotentials(int d, int t, int s, boolean backward);
		public int[] allowedEdges(int d, int t, int s, boolean backward);
	}

	public static interface StationaryEdgeMarginals {
		public double[] startNodeCondProbs(int d);
		public double[] endNodeCondProbs(int d);
		public int[] allowedForwardEdges(int d, int s);
		public double[] allowedForwardEdgesExpectedCounts(int d, int s);
		public double sequenceLogMarginalProb(int d);
		public double logMarginalProb();
		public int numSequences();
		public int numStates(int d);
		public Iterator<Pair<Pair<Integer,Pair<Integer,Integer>>,Double>> getEdgeMarginalsIterator();
		public Iterator<Pair<Pair<Integer,Integer>,Double>> getStartMarginalsIterator();
		public Iterator<Pair<Pair<Integer,Integer>,Double>> getEndMarginalsIterator();
		public double estimateMemoryUsage();
	}
	
	public static interface NonStationaryEdgeMarginals {
		public double[] startNodeCondProbs(int d);
		public double[] endNodeCondProbs(int d);
		public int[] allowedForwardEdges(int d, int t, int s);
		public double[] allowedForwardEdgesExpectedCounts(int d, int t, int s);
		public double sequenceLogMarginalProb(int d);
		public double logMarginalProb();
		public int numSequences();
		public int sequenceLength(int d);
		public int numStates(int d, int t);
		public Iterator<Pair<Pair<Pair<Integer,Integer>,Pair<Integer,Integer>>,Double>> getEdgeMarginalsIterator();
		public Iterator<Pair<Pair<Integer,Integer>,Double>> getStartMarginalsIterator();
		public Iterator<Pair<Pair<Integer,Integer>,Double>> getEndMarginalsIterator();
		public double estimateMemoryUsage();
	}

	public static interface NodeMarginals {
		public double[] nodeCondProbs(int d, int t);
		public double sequenceLogMarginalProb(int d);
		public double logMarginalProb();
		public int numSequences();
		public int sequenceLength(int d);
		public int numStates(int d);
		public Iterator<Pair<Pair<Pair<Integer,Integer>,Integer>,Double>> getNodeMarginalsIterator();
		public double estimateMemoryUsage();
	}
	
	public static interface StationaryStateProjector {
		public int domainSize(int d, int t);
		public int rangeSize(int d);
		public int project(int d, int t, int s);
	}
	
	private static class NodeMarginalsLogSpace implements NodeMarginals {
		Lattice lattice;
		double[][][] nodeCondProbs;
		double[] sequenceLogMarginalProbs;
		StationaryStateProjector stateProjector;

		public NodeMarginalsLogSpace(Lattice lattice, StationaryStateProjector stateProjector) {
			this.lattice = lattice;
			this.stateProjector = stateProjector;
			this.sequenceLogMarginalProbs = new double[lattice.numSequences()];
			this.nodeCondProbs = new double[lattice.numSequences()][][];
		}
		
		public void incrementExpectedCounts(double[][] alphas, double[][] betas, int d, boolean viterbi) {
			this.sequenceLogMarginalProbs[d] = Double.NEGATIVE_INFINITY;
			for (int s=0; s<lattice.numStates(d, 0); ++s) {
				if (viterbi) {
					this.sequenceLogMarginalProbs[d] = Math.max(sequenceLogMarginalProbs[d], betas[0][s]);
				} else {
					this.sequenceLogMarginalProbs[d] = m.logAdd(sequenceLogMarginalProbs[d], betas[0][s]);
				}
			}
			
			this.nodeCondProbs[d] = new double[lattice.sequenceLength(d)][];
			for (int t=0; t<lattice.sequenceLength(d); ++t) {
				this.nodeCondProbs[d][t] = new double[stateProjector.rangeSize(d)];
				if (viterbi) {
					Arrays.fill(this.nodeCondProbs[d][t], Double.NEGATIVE_INFINITY);
				}
				int numStates = lattice.numStates(d, t);
				for (int s=0; s<numStates; ++s) {
					int projectedS = stateProjector.project(d, t, s);
					if (alphas[t][s] != Double.NEGATIVE_INFINITY && betas[t][s] != Double.NEGATIVE_INFINITY) {
						if (viterbi) {
							this.nodeCondProbs[d][t][projectedS] = Math.max(this.nodeCondProbs[d][t][projectedS], Math.exp(alphas[t][s] - lattice.nodeLogPotential(d, t, s) + betas[t][s] - sequenceLogMarginalProbs[d]));
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

		public double sequenceLogMarginalProb(int d) {
			return sequenceLogMarginalProbs[d];
		}

		public double logMarginalProb() {
			return a.sum(sequenceLogMarginalProbs);
		}

		public double[] nodeCondProbs(int d, int t) {
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

		private class NodeMarginalsIterator implements Iterator<Pair<Pair<Pair<Integer, Integer>, Integer>, Double>> {
			int d;
			int t;
			int s;
			double[] nodeCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && t == sequenceLength(d)-1 && s == numStates(d)-1);
			}
			public Pair<Pair<Pair<Integer, Integer>, Integer>, Double> next() {
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
		
		public Iterator<Pair<Pair<Pair<Integer, Integer>, Integer>, Double>> getNodeMarginalsIterator() {
			return new NodeMarginalsIterator();
		}
		
	}
	
	private static class NodeMarginalsScaling implements NodeMarginals {
		Lattice lattice;
		double[][][] nodeCondProbs;
		double[] sequenceLogMarginalProbs;
		StationaryStateProjector stateProjector;

		public NodeMarginalsScaling(Lattice lattice, StationaryStateProjector stateProjector) {
			this.lattice = lattice;
			this.stateProjector = stateProjector;
			this.sequenceLogMarginalProbs = new double[lattice.numSequences()];
			this.nodeCondProbs = new double[lattice.numSequences()][][];
		}
		
		public void incrementExpectedCounts(double[][] alphas, double[] alphaLogScales, double[][] betas, double[] betaLogScales, int d, boolean viterbi) {
			double sequenceMarginalProb = (viterbi ? Double.NEGATIVE_INFINITY : 0.0);
			double sequenceMarginalProbLogScale = betaLogScales[0];
			for (int s=0; s<lattice.numStates(d, 0); ++s) {
				double nodePotential = lattice.nodePotential(d, 0, s);
				if (nodePotential > 0.0) {
					if (viterbi) {
						sequenceMarginalProb = Math.max(sequenceMarginalProb, betas[0][s]);
					} else {
						sequenceMarginalProb += betas[0][s];
					}
				}
			}
			this.sequenceLogMarginalProbs[d] = sequenceMarginalProbLogScale * LOG_SCALE + Math.log(sequenceMarginalProb);
			
			this.nodeCondProbs[d] = new double[lattice.sequenceLength(d)][];
			for (int t=0; t<lattice.sequenceLength(d); ++t) {
				this.nodeCondProbs[d][t] = new double[stateProjector.rangeSize(d)];
				if (viterbi) {
					Arrays.fill(nodeCondProbs[d][t], Double.NEGATIVE_INFINITY);
				}
				int numStates = lattice.numStates(d, t);
				for (int s=0; s<numStates; ++s) {
					double scale = getScaleFactor(alphaLogScales[t] + betaLogScales[t] - sequenceMarginalProbLogScale);
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
		
		public double sequenceLogMarginalProb(int d) {
			return sequenceLogMarginalProbs[d];
		}
		
		public double logMarginalProb() {
			return a.sum(sequenceLogMarginalProbs);
		}
		
		public double[] nodeCondProbs(int d, int t) {
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
		
		private class NodeMarginalsIterator implements Iterator<Pair<Pair<Pair<Integer, Integer>, Integer>, Double>> {
			int d;
			int t;
			int s;
			double[] nodeCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && t == sequenceLength(d)-1 && s == numStates(d)-1);
			}
			public Pair<Pair<Pair<Integer, Integer>, Integer>, Double> next() {
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

		public Iterator<Pair<Pair<Pair<Integer, Integer>, Integer>, Double>> getNodeMarginalsIterator() {
			return new NodeMarginalsIterator();
		}
		
	}
	
	private static class NonStationaryEdgeMarginalsLogSpace implements NonStationaryEdgeMarginals {
		Lattice lattice;
		double[] sequenceLogMarginalProbs;
		double[][] startNodeCondProbs;
		double[][] endNodeCondProbs;
		double[][][] allAlphas;
		double[][][] allBetas;

		public NonStationaryEdgeMarginalsLogSpace(Lattice lattice) {
			this.lattice = lattice;
			this.sequenceLogMarginalProbs = new double[lattice.numSequences()];
			this.startNodeCondProbs = new double[lattice.numSequences()][];
			this.endNodeCondProbs = new double[lattice.numSequences()][];
			this.allAlphas = new double[lattice.numSequences()][][];
			this.allBetas = new double[lattice.numSequences()][][];
		}

		public int[] allowedForwardEdges(int d, int t, int s) {
			return lattice.allowedEdges(d, t, s, false);
		}
		
		public void incrementExpectedCounts(double[][] alphas, double[][] betas, int d) {
			this.allAlphas[d] = alphas;
			this.allBetas[d] = betas;
			
			this.sequenceLogMarginalProbs[d] = Double.NEGATIVE_INFINITY;
			for (int s=0; s<lattice.numStates(d, 0); ++s) {
				this.sequenceLogMarginalProbs[d] = m.logAdd(sequenceLogMarginalProbs[d], betas[0][s]);
			}
			
			this.startNodeCondProbs[d] = new double[lattice.numStates(d, 0)];
			for (int s=0; s<lattice.numStates(d, 0); ++s) {
				this.startNodeCondProbs[d][s] = Math.exp(betas[0][s] - sequenceLogMarginalProbs[d]);
			}
			
			this.endNodeCondProbs[d] = new double[lattice.numStates(d, lattice.sequenceLength(d)-1)];
			for (int s=0; s<lattice.numStates(d, lattice.sequenceLength(d)-1); ++s) {
				this.endNodeCondProbs[d][s] = Math.exp(alphas[lattice.sequenceLength(d)-1][s] - sequenceLogMarginalProbs[d]);
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

		public double[] allowedForwardEdgesExpectedCounts(int d, int t, int s) {
			int[] allowedEdges = lattice.allowedEdges(d, t, s, false);
			double[] allowedForwardEdgesExpectedCounts = new double[allowedEdges.length];
			double[] alowedEdgesLogPotentials = lattice.allowedEdgesLogPotentials(d, t, s, false);
			for (int i=0; i<allowedEdges.length; ++i) {
				int nextS = allowedEdges[i];
				double edgeLogPotential = alowedEdgesLogPotentials[i];
				allowedForwardEdgesExpectedCounts[i] += Math.exp(allAlphas[d][t][s] + edgeLogPotential + allBetas[d][t+1][nextS] - sequenceLogMarginalProbs[d]);
			}
			return allowedForwardEdgesExpectedCounts;
		}

		public double[] startNodeCondProbs(int d) {
			return startNodeCondProbs[d];
		}

		public double[] endNodeCondProbs(int d) {
			return endNodeCondProbs[d];
		}
		
		public double sequenceLogMarginalProb(int d) {
			return sequenceLogMarginalProbs[d];
		}

		public double logMarginalProb() {
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
		
		private class EdgeMarginalsIterator implements Iterator<Pair<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Double>> {
			int d;
			int t;
			int s1;
			int s2i;
			double[] edgeCondProbs = null;
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
			public Pair<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Double> next() {
				Pair<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Double> result = Pair.makePair(Pair.makePair(Pair.makePair(d, t), Pair.makePair(s1, allowedForwardEdges(d, t, s1)[s2i])), edgeCondProbs[s2i]);
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

		public Iterator<Pair<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Double>> getEdgeMarginalsIterator() {
			return new EdgeMarginalsIterator();
		}
		
		private class StartMarginalsIterator implements Iterator<Pair<Pair<Integer, Integer>, Double>> {
			int d;
			int s;
			double[] startCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && s == numStates(d, 0)-1);
			}
			public Pair<Pair<Integer, Integer>, Double> next() {
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

		public Iterator<Pair<Pair<Integer, Integer>, Double>> getStartMarginalsIterator() {
			return new StartMarginalsIterator();
		}
		
		private class EndMarginalsIterator implements Iterator<Pair<Pair<Integer, Integer>, Double>> {
			int d;
			int s;
			double[] endCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && s == numStates(d, sequenceLength(d)-1)-1);
			}
			public Pair<Pair<Integer, Integer>, Double> next() {
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

		public Iterator<Pair<Pair<Integer, Integer>, Double>> getEndMarginalsIterator() {
			return new EndMarginalsIterator();
		}
		
	}
	
	private static class NonStationaryEdgeMarginalsScaling implements NonStationaryEdgeMarginals {
		Lattice lattice;
		double[] sequenceLogMarginalProbs;
		double[] sequenceMarginalProbs;
		double[] sequenceMarginalProbLogScales;
		double[][] startNodeCondProbs;
		double[][] endNodeCondProbs;
		double[][][] allAlphas;
		double[][][] allBetas;
		double[][] allAlphaLogScales;
		double[][] allBetaLogScales;
		
		public NonStationaryEdgeMarginalsScaling(Lattice lattice) {
			this.lattice = lattice;
			this.sequenceLogMarginalProbs = new double[lattice.numSequences()];
			this.sequenceMarginalProbs = new double[lattice.numSequences()];
			this.sequenceMarginalProbLogScales = new double[lattice.numSequences()];
			this.startNodeCondProbs = new double[lattice.numSequences()][];
			this.endNodeCondProbs = new double[lattice.numSequences()][];
			this.allAlphas = new double[lattice.numSequences()][][];
			this.allAlphaLogScales = new double[lattice.numSequences()][];
			this.allBetaLogScales = new double[lattice.numSequences()][];
		}

		public int[] allowedForwardEdges(int d, int t, int s) {
			return lattice.allowedEdges(d, t, s, false);
		}
		
		public void incrementExpectedCounts(double[][] alphas, double[] alphaLogScales, double[][] betas, double betaLogScales[], int d) {
			this.allAlphas[d] = alphas;
			this.allAlphaLogScales[d] = alphaLogScales;
			this.allBetas[d] = betas;
			this.allBetaLogScales[d] = betaLogScales;
			
			sequenceMarginalProbs[d] = 0.0;
			sequenceMarginalProbLogScales[d] = betaLogScales[0];
			for (int s=0; s<lattice.numStates(d, 0); ++s) {
				double nodePotential = lattice.nodePotential(d, 0, s);
				if (nodePotential > 0.0) {
					sequenceMarginalProbs[d] += betas[0][s];
				}
			}
			this.sequenceLogMarginalProbs[d] = sequenceMarginalProbLogScales[d] * LOG_SCALE + Math.log(sequenceMarginalProbs[d]);
			
			this.startNodeCondProbs[d] = new double[lattice.numStates(d, 0)];
			{
				double scale = getScaleFactor(betaLogScales[0] - sequenceMarginalProbLogScales[d]);
				for (int s=0; s<lattice.numStates(d, 0); ++s) {
					this.startNodeCondProbs[d][s] = (betas[0][s] / sequenceMarginalProbs[d]) * scale;
				}
			}
			
			this.endNodeCondProbs[d] = new double[lattice.numStates(d, lattice.sequenceLength(d)-1)];
			{
				double scale = getScaleFactor(alphaLogScales[lattice.sequenceLength(d)-1] - sequenceMarginalProbLogScales[d]);
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
		
		public double[] allowedForwardEdgesExpectedCounts(int d, int t, int s) {
			int[] allowedEdges = lattice.allowedEdges(d, t, s, false);
			double[] allowedForwardEdgesExpectedCounts = new double[allowedEdges.length];
			double[] alowedEdgesPotentials = lattice.allowedEdgesPotentials(d, t, s, false);
			double scale = getScaleFactor(allAlphaLogScales[d][t] + allBetaLogScales[d][t+1] - sequenceMarginalProbLogScales[d]);
			for (int i=0; i<allowedEdges.length; ++i) {
				int nextS = allowedEdges[i];
				double edgePotential = alowedEdgesPotentials[i];
				allowedForwardEdgesExpectedCounts[i] += (allAlphas[d][t][s] * edgePotential *  allBetas[d][t+1][nextS] / sequenceMarginalProbs[d]) * scale;
			}
			return allowedForwardEdgesExpectedCounts;
		}
		
		public double[] startNodeCondProbs(int d) {
			return startNodeCondProbs[d];
		}
		
		public double[] endNodeCondProbs(int d) {
			return endNodeCondProbs[d];
		}
		
		public double sequenceLogMarginalProb(int d) {
			return sequenceLogMarginalProbs[d];
		}
		
		public double logMarginalProb() {
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
		
		private class EdgeMarginalsIterator implements Iterator<Pair<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Double>> {
			int d;
			int t;
			int s1;
			int s2i;
			double[] edgeCondProbs = null;
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
			public Pair<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Double> next() {
				Pair<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Double> result = Pair.makePair(Pair.makePair(Pair.makePair(d, t), Pair.makePair(s1, allowedForwardEdges(d, t, s1)[s2i])), edgeCondProbs[s2i]);
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

		public Iterator<Pair<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Double>> getEdgeMarginalsIterator() {
			return new EdgeMarginalsIterator();
		}
		
		private class StartMarginalsIterator implements Iterator<Pair<Pair<Integer, Integer>, Double>> {
			int d;
			int s;
			double[] startCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && s == numStates(d, 0)-1);
			}
			public Pair<Pair<Integer, Integer>, Double> next() {
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

		public Iterator<Pair<Pair<Integer, Integer>, Double>> getStartMarginalsIterator() {
			return new StartMarginalsIterator();
		}
		
		private class EndMarginalsIterator implements Iterator<Pair<Pair<Integer, Integer>, Double>> {
			int d;
			int s;
			double[] endCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && s == numStates(d, sequenceLength(d)-1)-1);
			}
			public Pair<Pair<Integer, Integer>, Double> next() {
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

		public Iterator<Pair<Pair<Integer, Integer>, Double>> getEndMarginalsIterator() {
			return new EndMarginalsIterator();
		}
		
	}

	private static class StationaryEdgeMarginalsLogSpace implements StationaryEdgeMarginals {
		StationaryLattice lattice;
		double[] sequenceLogMarginalProbs;
		double[][][] allowedForwardEdgesExpectedCounts;
		double[][] startNodeCondProbs;
		double[][] endNodeCondProbs;

		public StationaryEdgeMarginalsLogSpace(StationaryLattice lattice) {
			this.lattice = lattice;
			this.sequenceLogMarginalProbs = new double[lattice.numSequences()];
			this.allowedForwardEdgesExpectedCounts = new double[lattice.numSequences()][][];
			this.startNodeCondProbs = new double[lattice.numSequences()][];
			this.endNodeCondProbs = new double[lattice.numSequences()][];
		}
		
		public int[] allowedForwardEdges(int d, int s) {
			return lattice.allowedEdges(d, s, false);
		}
		
		public void incrementExpectedCounts(double[][] alphas, double[][] betas, int d) {
			this.sequenceLogMarginalProbs[d] = Double.NEGATIVE_INFINITY;
			for (int s=0; s<lattice.numStates(d); ++s) {
				this.sequenceLogMarginalProbs[d] = m.logAdd(sequenceLogMarginalProbs[d], betas[0][s]);
			}
			
			this.allowedForwardEdgesExpectedCounts[d] = new double[lattice.numStates(d)][];
			for (int s=0; s<lattice.numStates(d); ++s) {
				this.allowedForwardEdgesExpectedCounts[d][s] = new double[lattice.allowedEdges(d, s, false).length];
			}
			for (int t=0; t<lattice.sequenceLength(d)-1; ++t) {
				int numStates = lattice.numStates(d);
				for (int s=0; s<numStates; ++s) {
					int[] allowedEdges = lattice.allowedEdges(d, s, false);
					double[] alowedEdgesLogPotentials = lattice.allowedEdgesLogPotentials(d, s, false);
					for (int i=0; i<allowedEdges.length; ++i) {
						int nextS = allowedEdges[i];
						double edgeLogPotential = alowedEdgesLogPotentials[i];
						this.allowedForwardEdgesExpectedCounts[d][s][i] += Math.exp(alphas[t][s] + edgeLogPotential + betas[t+1][nextS] - sequenceLogMarginalProbs[d]);
					}
				}
			}
			
			this.startNodeCondProbs[d] = new double[lattice.numStates(d)];
			for (int s=0; s<lattice.numStates(d); ++s) {
				this.startNodeCondProbs[d][s] = Math.exp(betas[0][s] - sequenceLogMarginalProbs[d]);
			}
			
			this.endNodeCondProbs[d] = new double[lattice.numStates(d)];
			for (int s=0; s<lattice.numStates(d); ++s) {
				this.endNodeCondProbs[d][s] = Math.exp(alphas[lattice.sequenceLength(d)-1][s] - sequenceLogMarginalProbs[d]);
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

		public double[] allowedForwardEdgesExpectedCounts(int d, int s) {
			return allowedForwardEdgesExpectedCounts[d][s];
		}

		public double[] startNodeCondProbs(int d) {
			return startNodeCondProbs[d];
		}

		public double[] endNodeCondProbs(int d) {
			return endNodeCondProbs[d];
		}
		
		public double sequenceLogMarginalProb(int d) {
			return sequenceLogMarginalProbs[d];
		}

		public double logMarginalProb() {
			return a.sum(sequenceLogMarginalProbs);
		}
		
		public int numSequences() {
			return lattice.numSequences();
		}

		public int numStates(int d) {
			return lattice.numStates(d);
		}
		
		private class EdgeMarginalsIterator implements Iterator<Pair<Pair<Integer, Pair<Integer, Integer>>, Double>> {
			int d;
			int s1;
			int s2i;
			double[] edgeCondProbs = null;
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
			public Pair<Pair<Integer, Pair<Integer, Integer>>, Double> next() {
				Pair<Pair<Integer, Pair<Integer, Integer>>, Double> result = Pair.makePair(Pair.makePair(d, Pair.makePair(s1, allowedForwardEdges(d, s1)[s2i])), edgeCondProbs[s2i]);
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

		public Iterator<Pair<Pair<Integer, Pair<Integer, Integer>>, Double>> getEdgeMarginalsIterator() {
			return new EdgeMarginalsIterator();
		}
		
		private class StartMarginalsIterator implements Iterator<Pair<Pair<Integer, Integer>, Double>> {
			int d;
			int s;
			double[] startCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && s == numStates(d)-1);
			}
			public Pair<Pair<Integer, Integer>, Double> next() {
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
		
		public Iterator<Pair<Pair<Integer, Integer>, Double>> getStartMarginalsIterator() {
			return new StartMarginalsIterator();
		}
		
		private class EndMarginalsIterator implements Iterator<Pair<Pair<Integer, Integer>, Double>> {
			int d;
			int s;
			double[] endCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && s == numStates(d)-1);
			}
			public Pair<Pair<Integer, Integer>, Double> next() {
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
		
		public Iterator<Pair<Pair<Integer, Integer>, Double>> getEndMarginalsIterator() {
			return new EndMarginalsIterator();
		}
		
	}
	
	private static class StationaryEdgeMarginalsScaling implements StationaryEdgeMarginals {
		StationaryLattice lattice;
		double[] sequenceLogMarginalProbs;
		double[][][] allowedForwardEdgesExpectedCounts;
		double[][] startNodeCondProbs;
		double[][] endNodeCondProbs;
		
		public StationaryEdgeMarginalsScaling(StationaryLattice lattice) {
			this.lattice = lattice;
			this.sequenceLogMarginalProbs = new double[lattice.numSequences()];
			this.allowedForwardEdgesExpectedCounts = new double[lattice.numSequences()][][];
			this.startNodeCondProbs = new double[lattice.numSequences()][];
			this.endNodeCondProbs = new double[lattice.numSequences()][];
		}
		
		public int[] allowedForwardEdges(int d, int s) {
			return lattice.allowedEdges(d, s, false);
		}
		
		public void incrementExpectedCounts(double[][] alphas, double[] alphaLogScales, double[][] betas, double betaLogScales[], int d) {
			double sequenceMarginalProb = 0.0;
			double sequenceMarginalProbLogScale = betaLogScales[0];
			for (int s=0; s<lattice.numStates(d); ++s) {
				double nodePotential = lattice.nodePotential(d, 0, s);
				if (nodePotential > 0.0) {
					sequenceMarginalProb += betas[0][s];
				}
			}
			this.sequenceLogMarginalProbs[d] = sequenceMarginalProbLogScale * LOG_SCALE + Math.log(sequenceMarginalProb);
			
			this.allowedForwardEdgesExpectedCounts[d] = new double[lattice.numStates(d)][];
			for (int s=0; s<lattice.numStates(d); ++s) {
				this.allowedForwardEdgesExpectedCounts[d][s] = new double[lattice.allowedEdges(d, s, false).length];
			}
			for (int t=0; t<lattice.sequenceLength(d)-1; ++t) {
				double scale = getScaleFactor(alphaLogScales[t] + betaLogScales[t+1] - sequenceMarginalProbLogScale);
				int numStates = lattice.numStates(d);
				for (int s=0; s<numStates; ++s) {
					int[] allowedEdges = lattice.allowedEdges(d, s, false);
					double[] alowedEdgesPotentials = lattice.allowedEdgesPotentials(d, s, false);
					for (int i=0; i<allowedEdges.length; ++i) {
						int nextS = allowedEdges[i];
						double edgePotential = alowedEdgesPotentials[i];
						this.allowedForwardEdgesExpectedCounts[d][s][i] += (alphas[t][s] * edgePotential *  betas[t+1][nextS] / sequenceMarginalProb) * scale;
					}
				}
			}
			
			this.startNodeCondProbs[d] = new double[lattice.numStates(d)];
			{
				double scale = getScaleFactor(betaLogScales[0] - sequenceMarginalProbLogScale);
				for (int s=0; s<lattice.numStates(d); ++s) {
					this.startNodeCondProbs[d][s] = (betas[0][s] / sequenceMarginalProb) * scale;
				}
			}
			
			this.endNodeCondProbs[d] = new double[lattice.numStates(d)];
			{
				double scale = getScaleFactor(alphaLogScales[lattice.sequenceLength(d)-1] - sequenceMarginalProbLogScale);
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
		
		public double[] allowedForwardEdgesExpectedCounts(int d, int s) {
			return allowedForwardEdgesExpectedCounts[d][s];
		}
		
		public double[] startNodeCondProbs(int d) {
			return startNodeCondProbs[d];
		}
		
		public double[] endNodeCondProbs(int d) {
			return endNodeCondProbs[d];
		}
		
		public double sequenceLogMarginalProb(int d) {
			return sequenceLogMarginalProbs[d];
		}
		
		public double logMarginalProb() {
			return a.sum(sequenceLogMarginalProbs);
		}
		
		public int numSequences() {
			return lattice.numSequences();
		}

		public int numStates(int d) {
			return lattice.numStates(d);
		}

		private class EdgeMarginalsIterator implements Iterator<Pair<Pair<Integer, Pair<Integer, Integer>>, Double>> {
			int d;
			int s1;
			int s2i;
			double[] edgeCondProbs = null;
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
			public Pair<Pair<Integer, Pair<Integer, Integer>>, Double> next() {
				Pair<Pair<Integer, Pair<Integer, Integer>>, Double> result = Pair.makePair(Pair.makePair(d, Pair.makePair(s1, allowedForwardEdges(d, s1)[s2i])), edgeCondProbs[s2i]);
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
		
		public Iterator<Pair<Pair<Integer, Pair<Integer, Integer>>, Double>> getEdgeMarginalsIterator() {
			return new EdgeMarginalsIterator();
		}
		
		private class StartMarginalsIterator implements Iterator<Pair<Pair<Integer, Integer>, Double>> {
			int d;
			int s;
			double[] startCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && s == numStates(d)-1);
			}
			public Pair<Pair<Integer, Integer>, Double> next() {
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

		public Iterator<Pair<Pair<Integer, Integer>, Double>> getStartMarginalsIterator() {
			return new StartMarginalsIterator();
		}
		
		private class EndMarginalsIterator implements Iterator<Pair<Pair<Integer, Integer>, Double>> {
			int d;
			int s;
			double[] endCondProbs = null;
			
			public boolean hasNext() {
				return !(d == numSequences()-1 && s == numStates(d)-1);
			}
			public Pair<Pair<Integer, Integer>, Double> next() {
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
		

		public Iterator<Pair<Pair<Integer, Integer>, Double>> getEndMarginalsIterator() {
			return new EndMarginalsIterator();
		}
		
	}

	public static Pair<NodeMarginals,StationaryEdgeMarginals> computeMarginalsLogSpace(final StationaryLattice lattice, final StationaryStateProjector nodeMarginalsStateProjector, final boolean viterbiEmissionOnly, int numThreads) {
		final NodeMarginalsLogSpace projectedNodeMarginals = new NodeMarginalsLogSpace(new StationaryLatticeWrapper(lattice), nodeMarginalsStateProjector);
		final StationaryEdgeMarginalsLogSpace edgeMarginals = (viterbiEmissionOnly ? null : new StationaryEdgeMarginalsLogSpace(lattice));
		BetterThreader.Function<Integer,Object> func = new BetterThreader.Function<Integer,Object>(){public void call(Integer d, Object ignore){
			double[][] alphas = doPassLogSpace(new StationaryLatticeWrapper(lattice), false, viterbiEmissionOnly, d);
			double[][] betas = doPassLogSpace(new StationaryLatticeWrapper(lattice), true, viterbiEmissionOnly, d);
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
			double[][] alphas = doPassLogSpace(lattice, false, viterbiEmissionOnly, d);
			double[][] betas = doPassLogSpace(lattice, true, viterbiEmissionOnly, d);
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
			Pair<double[][],double[]> alphasAndScales = doPassScaling(new StationaryLatticeWrapper(lattice), false, viterbiEmissionOnly, d);
			double[][] alphas = alphasAndScales.getFirst();
			double [] alphaLogScales = alphasAndScales.getSecond();
			Pair<double[][],double[]> betasAndScales = doPassScaling(new StationaryLatticeWrapper(lattice), true, viterbiEmissionOnly, d);
			double[][] betas = betasAndScales.getFirst();
			double [] betaLogScales = betasAndScales.getSecond();
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
			Pair<double[][],double[]> alphasAndScales = doPassScaling(lattice, false, viterbiEmissionOnly, d);
			double[][] alphas = alphasAndScales.getFirst();
			double [] alphaLogScales = alphasAndScales.getSecond();
			Pair<double[][],double[]> betasAndScales = doPassScaling(lattice, true, viterbiEmissionOnly, d);
			double[][] betas = betasAndScales.getFirst();
			double [] betaLogScales = betasAndScales.getSecond();
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
			double[][] alphas = doPassScaling(lattice, false, true, d).getFirst();
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
			double[][] alphas = doPassScaling(lattice, false, true, d).getFirst();
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
			double[][] alphas = doPassLogSpace(lattice, false, true, d);
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
			double[][] alphas = doPassLogSpace(lattice, false, true, d);
			viterbiSequences[d] = extractViterbiPath(lattice, stateProjector, alphas, false, d);
		}};
		BetterThreader<Integer,Object> threader = new BetterThreader<Integer,Object>(func, numThreads);
		for (int d=0; d<lattice.numSequences(); ++d) threader.addFunctionArgument(d);
		threader.run();
		return viterbiSequences;
	}

	private static int[] extractViterbiPath(final Lattice lattice, final StationaryStateProjector stateProjector, final double[][] alphas, boolean scaling, int d) {
		int[] viterbiSequence = new int[lattice.sequenceLength(d)];
		viterbiSequence[lattice.sequenceLength(d)-1] = a.argmax(alphas[lattice.sequenceLength(d)-1]);
		for (int t=lattice.sequenceLength(d)-2; t>=0; --t) {
			int s = viterbiSequence[t+1];
			int[] prevStates = lattice.allowedEdges(d, t+1, s, true);
			int bestPrevState = -1;
			double bestScore = Double.NEGATIVE_INFINITY;
			if (scaling) {
				double[] prevStatesEdgePotentials = lattice.allowedEdgesPotentials(d, t+1, s, true);
				for (int i=0; i<prevStates.length; ++i) {
					int prevState = prevStates[i];
					double score = alphas[t][prevState]*prevStatesEdgePotentials[i];
					if (score > bestScore) {
						bestScore = score;
						bestPrevState = prevState;
					}
				}
			} else {
				double[] prevNodesEdgeLogPotentials = lattice.allowedEdgesLogPotentials(d, t+1, s, true);
				for (int i=0; i<prevStates.length; ++i) {
					int prevState = prevStates[i];
					double score = alphas[t][prevState] + prevNodesEdgeLogPotentials[i];
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

	private static double[][] doPassLogSpace(final Lattice lattice, final boolean backward, final boolean viterbi, int d) {
		double[][] alphas = new double[lattice.sequenceLength(d)][];
		for (int t=0; t<lattice.sequenceLength(d); ++t) {
			alphas[t] = new double[lattice.numStates(d, t)];
		}
		
		int[] timeOrder = (backward ? a.enumerate(lattice.sequenceLength(d),0) : a.enumerate(0,lattice.sequenceLength(d)));
		for (int ti=0; ti<timeOrder.length; ++ti) {
			int t = timeOrder[ti];
			Arrays.fill(alphas[t], Double.NEGATIVE_INFINITY);
			if (ti == 0) {
				int numStates = lattice.numStates(d, t);
				for (int s=0; s<numStates; ++s) {
					alphas[t][s] = lattice.nodeLogPotential(d, t, s);
				}
			} else {
				int prevT = timeOrder[ti-1];
				int numStates = lattice.numStates(d, prevT);
				for (int prevS=0; prevS<numStates; ++prevS) {
					double prevAlpha = alphas[prevT][prevS];
					int[] nextStates = lattice.allowedEdges(d, prevT, prevS, backward);
					double[] nextStatesEdgeLogPotentials = lattice.allowedEdgesLogPotentials(d, prevT, prevS, backward);
					double[] currentAlphas = alphas[t];
					for (int i=0; i<nextStates.length; ++i) {
						int nextState = nextStates[i];
						double edgeLogPotential = nextStatesEdgeLogPotentials[i];
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
	
	private static Pair<double[][],double[]> doPassScaling(final Lattice lattice, final boolean backward, final boolean viterbi, int d) {
		double[] logScales = new double[lattice.sequenceLength(d)];
		double[][] alphas = new double[lattice.sequenceLength(d)][];
		for (int t=0; t<lattice.sequenceLength(d); ++t) {
			alphas[t] = new double[lattice.numStates(d, t)];
		}
		
		int[] timeOrder = (backward ? a.enumerate(lattice.sequenceLength(d),0) : a.enumerate(0,lattice.sequenceLength(d)));
		for (int ti=0; ti<timeOrder.length; ++ti) {
			int t = timeOrder[ti];
			Arrays.fill(alphas[t], 0.0);
			double max = Double.NEGATIVE_INFINITY;
			if (ti == 0) {
				int numStates = lattice.numStates(d, t);
				for (int s=0; s<numStates; ++s) {
					double alpha = lattice.nodePotential(d, t, s);
					alphas[t][s] = alpha;
					if (alpha > max) max = alpha;
				}
			} else {
				int prevT = timeOrder[ti-1];
				int numStates = lattice.numStates(d, prevT);
				for (int prevS=0; prevS<numStates; ++prevS) {
					double prevAlpha = alphas[prevT][prevS];
					int[] nextStates = lattice.allowedEdges(d, prevT, prevS, backward);
					double[] nextStatesEdgePotentials = lattice.allowedEdgesPotentials(d, prevT, prevS, backward);
					double[] currentAlphas = alphas[t];
					for (int i=0; i<nextStates.length; ++i) {
						int nextState = nextStates[i];
						double edgePotential = nextStatesEdgePotentials[i];
						if (viterbi) {
							currentAlphas[nextState] = Math.max(currentAlphas[nextState], prevAlpha * edgePotential);
						} else {
							currentAlphas[nextState] += prevAlpha * edgePotential;
						}
					}
				}
				for (int s=0; s<lattice.numStates(d, t); ++s) {
					alphas[t][s] *= lattice.nodePotential(d, t, s);
					double alpha = alphas[t][s];
					if (alpha > max) max = alpha;
				}
			}
			
			int logScale = 0;
			double scale = 1.0;
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
