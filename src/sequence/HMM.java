package sequence;

import sequence.ForwardBackward.StationaryLattice;
import sequence.ForwardBackward.StationaryStateProjector;

public class HMM {
	
	public interface StationaryTransitionModel {
		public int numSequences();
		public int sequenceLength(int d);
		public int numStates(int d);
		public double startLogProb(int d, int ts);
		public double endLogProb(int d, int ts);
		public double startProb(int d, int ts);
		public double endProb(int d, int ts);
		public int[] allowedForwardTransitions(int d, int ts);
		public double[] allowedForwardTransitionsLogProbs(int d, int ts);
		public double[] allowedForwardTransitionsProbs(int d, int ts);
		public int[] allowedBackwardTransitions(int d, int ts);
		public double[] allowedBackwardTransitionsLogProbs(int d, int ts);
		public double[] allowedBackwardTransitionsProbs(int d, int ts);
	}
	
	public interface TransitionModel {
		public int numSequences();
		public int sequenceLength(int d);
		public int numStates(int d, int t);
		public double startLogProb(int d, int ts);
		public double endLogProb(int d, int ts);
		public double startProb(int d, int ts);
		public double endProb(int d, int ts);
		public int[] allowedForwardTransitions(int d, int t, int ts);
		public double[] allowedForwardTransitionsLogProbs(int d, int t, int ts);
		public double[] allowedForwardTransitionsProbs(int d, int t, int ts);
		public int[] allowedBackwardTransitions(int d, int t, int ts);
		public double[] allowedBackwardTransitionsLogProbs(int d, int t, int ts);
		public double[] allowedBackwardTransitionsProbs(int d, int t, int ts);
	}
	
	public interface StationaryEmissionModel<A> {
		public int numSequences();
		public int sequenceLength(int d);
		public int numStates();
		public double emissionLogProb(int d, int t, int es);
		public double emissionProb(int d, int t, int es);
	}
	
	public static class StationaryHMMLattice<A> implements StationaryLattice {
		private StationaryTransitionModel transitionModel;
		private StationaryEmissionModel<A> emissionModel;
		private StationaryStateProjector transitionToEmissionState;
		private A[][] observations;
		
		public StationaryHMMLattice(StationaryTransitionModel transitionModel, StationaryEmissionModel<A> emissionModel, StationaryStateProjector transitionToEmissionState, A[][] observations) {
			this.transitionModel = transitionModel;
			this.emissionModel = emissionModel;
			this.transitionToEmissionState = transitionToEmissionState;
			this.observations = observations;
		}
		
		public int numSequences() {
			return observations.length;
		}
		
		public int sequenceLength(int d) {
			return observations[d].length;
		}

		public int numStates(int d) {
			return transitionModel.numStates(d);
		}

		public double nodeLogPotential(int d, int t, int ts) {
			int es = transitionToEmissionState.project(d, t, ts);
			if (t == 0) {
				return transitionModel.startLogProb(d, ts) + emissionModel.emissionLogProb(d, t, es);
			} else if (t == observations[d].length-1) {
				return transitionModel.endLogProb(d, ts) + emissionModel.emissionLogProb(d, t, es);
			} else {
				return emissionModel.emissionLogProb(d, t, es);
			}
		}
		
		public double nodePotential(int d, int t, int ts) {
			int es = transitionToEmissionState.project(d, t, ts);
			if (t == 0) {
				return transitionModel.startProb(d, ts) * emissionModel.emissionProb(d, t, es);
			} else if (t == observations[d].length-1) {
				return transitionModel.endProb(d, ts) * emissionModel.emissionProb(d, t, es);
			} else {
				return emissionModel.emissionProb(d, t, es);
			}
		}

		public double[] allowedEdgesLogPotentials(int d, int ts, boolean backward) {
			if (backward) {
				return transitionModel.allowedBackwardTransitionsLogProbs(d, ts);
			} else {
				return transitionModel.allowedForwardTransitionsLogProbs(d, ts);
			}
		}
		
		public double[] allowedEdgesPotentials(int d, int ts, boolean backward) {
			if (backward) {
				return transitionModel.allowedBackwardTransitionsProbs(d, ts);
			} else {
				return transitionModel.allowedForwardTransitionsProbs(d, ts);
			}
		}

		public int[] allowedEdges(int d, int ts, boolean backward) {
			if (backward) {
				return transitionModel.allowedBackwardTransitions(d, ts);
			} else {
				return transitionModel.allowedForwardTransitions(d, ts);
			}
		}
		
	}
	
	public static class NonStationaryHMMLattice<A> implements ForwardBackward.Lattice {
		private TransitionModel transitionModel;
		private StationaryEmissionModel<A> emissionModel;
		private StationaryStateProjector transitionToEmissionState;
		private A[][] observations;
		
		public NonStationaryHMMLattice(TransitionModel transitionModel, StationaryEmissionModel<A> emissionModel, StationaryStateProjector transitionToEmissionState, A[][] observations) {
			this.transitionModel = transitionModel;
			this.emissionModel = emissionModel;
			this.transitionToEmissionState = transitionToEmissionState;
			this.observations = observations;
		}
		
		public int numSequences() {
			return observations.length;
		}
		
		public int sequenceLength(int d) {
			return observations[d].length;
		}

		public int numStates(int d, int t) {
			return transitionModel.numStates(d, t);
		}

		public double nodeLogPotential(int d, int t, int ts) {
			int es = transitionToEmissionState.project(d, t, ts);
			if (t == 0) {
				return transitionModel.startLogProb(d, ts) + emissionModel.emissionLogProb(d, t, es);
			} else if (t == observations[d].length-1) {
				return transitionModel.endLogProb(d, ts) + emissionModel.emissionLogProb(d, t, es);
			} else {
				return emissionModel.emissionLogProb(d, t, es);
			}
		}
		
		public double nodePotential(int d, int t, int ts) {
			int es = transitionToEmissionState.project(d, t, ts);
			if (t == 0) {
				return transitionModel.startProb(d, ts) * emissionModel.emissionProb(d, t, es);
			} else if (t == observations[d].length-1) {
				return transitionModel.endProb(d, ts) * emissionModel.emissionProb(d, t, es);
			} else {
				return emissionModel.emissionProb(d, t, es);
			}
		}

		public double[] allowedEdgesLogPotentials(int d, int t, int ts, boolean backward) {
			if (backward) {
				return transitionModel.allowedBackwardTransitionsLogProbs(d, t, ts);
			} else {
				return transitionModel.allowedForwardTransitionsLogProbs(d, t, ts);
			}
		}
		
		public double[] allowedEdgesPotentials(int d, int t, int ts, boolean backward) {
			if (backward) {
				return transitionModel.allowedBackwardTransitionsProbs(d, t, ts);
			} else {
				return transitionModel.allowedForwardTransitionsProbs(d, t, ts);
			}
		}

		public int[] allowedEdges(int d, int t, int ts, boolean backward) {
			if (backward) {
				return transitionModel.allowedBackwardTransitions(d, t, ts);
			} else {
				return transitionModel.allowedForwardTransitions(d, t, ts);
			}
		}
		
	}
	
}
