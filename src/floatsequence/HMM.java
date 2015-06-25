package floatsequence;

import floatsequence.ForwardBackward.StationaryLattice;
import floatsequence.ForwardBackward.StationaryStateProjector;

public class HMM {
	
	public interface StationaryTransitionModel {
		public int numSequences();
		public int sequenceLength(int d);
		public int numStates(int d);
		public float startLogProb(int d, int ts);
		public float endLogProb(int d, int ts);
		public float startProb(int d, int ts);
		public float endProb(int d, int ts);
		public int[] allowedForwardTransitions(int d, int ts);
		public float[] allowedForwardTransitionsLogProbs(int d, int ts);
		public float[] allowedForwardTransitionsProbs(int d, int ts);
		public int[] allowedBackwardTransitions(int d, int ts);
		public float[] allowedBackwardTransitionsLogProbs(int d, int ts);
		public float[] allowedBackwardTransitionsProbs(int d, int ts);
	}
	
	public interface TransitionModel {
		public int numSequences();
		public int sequenceLength(int d);
		public int numStates(int d, int t);
		public float startLogProb(int d, int ts);
		public float endLogProb(int d, int ts);
		public float startProb(int d, int ts);
		public float endProb(int d, int ts);
		public int[] allowedForwardTransitions(int d, int t, int ts);
		public float[] allowedForwardTransitionsLogProbs(int d, int t, int ts);
		public float[] allowedForwardTransitionsProbs(int d, int t, int ts);
		public int[] allowedBackwardTransitions(int d, int t, int ts);
		public float[] allowedBackwardTransitionsLogProbs(int d, int t, int ts);
		public float[] allowedBackwardTransitionsProbs(int d, int t, int ts);
	}
	
	public interface StationaryEmissionModel<A> {
		public int numSequences();
		public int sequenceLength(int d);
		public int numStates();
		public float emissionLogProb(int d, int t, int es);
		public float emissionProb(int d, int t, int es);
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

		public float nodeLogPotential(int d, int t, int ts) {
			int es = transitionToEmissionState.project(d, t, ts);
			if (t == 0) {
				return transitionModel.startLogProb(d, ts) + emissionModel.emissionLogProb(d, t, es);
			} else if (t == observations[d].length-1) {
				return transitionModel.endLogProb(d, ts) + emissionModel.emissionLogProb(d, t, es);
			} else {
				return emissionModel.emissionLogProb(d, t, es);
			}
		}
		
		public float nodePotential(int d, int t, int ts) {
			int es = transitionToEmissionState.project(d, t, ts);
			if (t == 0) {
				return transitionModel.startProb(d, ts) * emissionModel.emissionProb(d, t, es);
			} else if (t == observations[d].length-1) {
				return transitionModel.endProb(d, ts) * emissionModel.emissionProb(d, t, es);
			} else {
				return emissionModel.emissionProb(d, t, es);
			}
		}

		public float[] allowedEdgesLogPotentials(int d, int ts, boolean backward) {
			if (backward) {
				return transitionModel.allowedBackwardTransitionsLogProbs(d, ts);
			} else {
				return transitionModel.allowedForwardTransitionsLogProbs(d, ts);
			}
		}
		
		public float[] allowedEdgesPotentials(int d, int ts, boolean backward) {
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

		public float nodeLogPotential(int d, int t, int ts) {
			int es = transitionToEmissionState.project(d, t, ts);
			if (t == 0) {
				return transitionModel.startLogProb(d, ts) + emissionModel.emissionLogProb(d, t, es);
			} else if (t == observations[d].length-1) {
				return transitionModel.endLogProb(d, ts) + emissionModel.emissionLogProb(d, t, es);
			} else {
				return emissionModel.emissionLogProb(d, t, es);
			}
		}
		
		public float nodePotential(int d, int t, int ts) {
			int es = transitionToEmissionState.project(d, t, ts);
			if (t == 0) {
				return transitionModel.startProb(d, ts) * emissionModel.emissionProb(d, t, es);
			} else if (t == observations[d].length-1) {
				return transitionModel.endProb(d, ts) * emissionModel.emissionProb(d, t, es);
			} else {
				return emissionModel.emissionProb(d, t, es);
			}
		}

		public float[] allowedEdgesLogPotentials(int d, int t, int ts, boolean backward) {
			if (backward) {
				return transitionModel.allowedBackwardTransitionsLogProbs(d, t, ts);
			} else {
				return transitionModel.allowedForwardTransitionsLogProbs(d, t, ts);
			}
		}
		
		public float[] allowedEdgesPotentials(int d, int t, int ts, boolean backward) {
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
