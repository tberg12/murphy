package tberg.murphy.sequence;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import tberg.murphy.arrays.a;
import tberg.murphy.tuple.Pair;
import tberg.murphy.util.GeneralPriorityQueue;

public class SparseSemiMarkovDP {
  
  public static interface EmissionModel {
    public int sequenceLength();
    public int[] allowedWidths(int emissionStateIndex);
    public double score(int startPos, int emissionStateIndex, int width);
  }
  
  public static interface TransitionModel {
    public Collection<Pair<TransitionState,Double>> startStates();
  }
  
  public static interface TransitionState {
    public int getEmissionStateIndex();
    public Collection<Pair<TransitionState,Double>> forwardTransitions();
    public double endScore();
  }
  
  private static class BeamState {
    private final TransitionState transState;
    public double score = Double.NEGATIVE_INFINITY;
    public Pair<Integer,TransitionState> backPointer = null;
    public BeamState(TransitionState transState) {
      this.transState = transState;
    }
    public int hashCode() {
      return transState.hashCode();
    }
    public boolean equals(Object obj) {
      if (obj instanceof BeamState) {
        return transState.equals(((BeamState) obj).transState);
      } else {
        return false;
      }
    }
  }
  
  public static Pair<TransitionState[],int[]> decode(EmissionModel emissionModel, TransitionModel transitionModel, int beamSize) {
    GeneralPriorityQueue<SparseSemiMarkovDP.BeamState>[] alphas = doForwardPass(emissionModel, transitionModel, beamSize);
    Pair<TransitionState[],int[]> statesAndWidths = followBackpointers(alphas, emissionModel);
    TransitionState[] decodeStates = statesAndWidths.getFirst();
    int[] decodeWidths = statesAndWidths.getSecond();
    return Pair.makePair(decodeStates, decodeWidths);
  }
  
  private static GeneralPriorityQueue<SparseSemiMarkovDP.BeamState>[] doForwardPass(EmissionModel emissionModel, TransitionModel transitionModel, int beamSize) {
    GeneralPriorityQueue<SparseSemiMarkovDP.BeamState>[] alphas = new GeneralPriorityQueue[emissionModel.sequenceLength()+1];
    for (int t=0; t<emissionModel.sequenceLength()+1; ++t) {
      alphas[t] = new GeneralPriorityQueue<SparseSemiMarkovDP.BeamState>();
    }
    
    for (int t=0; t<emissionModel.sequenceLength()+1; ++t) {
      if (t == 0) {
        for (SparseSemiMarkovDP.BeamState startBeamState : addNullBackpointers(transitionModel.startStates())) {
          TransitionState nextTs = startBeamState.transState;
          double startScore = startBeamState.score;
          if (startScore != Double.NEGATIVE_INFINITY) {
            int e = nextTs.getEmissionStateIndex(); 
            for (int w : emissionModel.allowedWidths(e)) {
              if (t + w < emissionModel.sequenceLength()+1) {
                int nextT = t + w;
                double emissionScore = emissionModel.score(t, e, nextT-t);
                double score = startScore + emissionScore;
                if (score != Double.NEGATIVE_INFINITY) {
                  addToBeam(alphas[nextT], nextTs, score, new Pair<Integer,TransitionState>(0, startBeamState.backPointer.getSecond()), beamSize);
                }
              }
            }
          }
        }
      } else {
        for (SparseSemiMarkovDP.BeamState beamState : alphas[t].getObjects()) {
          Collection<Pair<TransitionState,Double>> allowedTrans = beamState.transState.forwardTransitions();
          for (Pair<TransitionState,Double> trans : allowedTrans) {
            TransitionState nextTs = trans.getFirst();
            double transScore = trans.getSecond();
            int e = nextTs.getEmissionStateIndex();
            for (int w : emissionModel.allowedWidths(e)) {
              if (t + w < emissionModel.sequenceLength()+1) {
                int nextT = t + w;
                double emissionScore = emissionModel.score(t, e, nextT-t);
                double score = beamState.score + transScore + emissionScore;
                if (score != Double.NEGATIVE_INFINITY) {
                  addToBeam(alphas[nextT], nextTs, score, Pair.makePair(t, beamState.transState), beamSize);
                }
              }
            }
          }
        }
      }
    }
    
    return alphas;
  }
  
  private static void addToBeam(GeneralPriorityQueue<BeamState> queue, TransitionState nextTs, double score, Pair<Integer,TransitionState> backPointer, int beamSize) {
    double priority = -score;
    if (queue.isEmpty() || priority < queue.getPriority()) {
      BeamState key = new BeamState(nextTs);
      if (queue.containsKey(key)) {
        queue.decreasePriority(key, priority);
      } else {
        queue.setPriority(key, priority);
      }
      SparseSemiMarkovDP.BeamState object = queue.getObject(key);
      if (object.score < score) {
        object.score = score;
        object.backPointer = backPointer;
      }
      while (queue.size() > beamSize) {
        queue.removeFirst(); 
      }
    }
  }
  
  private static Collection<BeamState> addNullBackpointers(Collection<Pair<TransitionState,Double>> without) {
    List<BeamState> with = new ArrayList<BeamState>();
    for (Pair<TransitionState,Double> startPair : without) {
      BeamState beamState = new BeamState(startPair.getFirst());
      beamState.score = startPair.getSecond();
      beamState.backPointer = Pair.makePair(-1, null);
      with.add(beamState);
    }
    return with;
  }

  private static Pair<TransitionState[],int[]> followBackpointers(GeneralPriorityQueue<SparseSemiMarkovDP.BeamState>[] alphas, EmissionModel emissionModel) {
    List<TransitionState> transStateDecodeList = new ArrayList<TransitionState>();
    List<Integer> widthsDecodeList = new ArrayList<Integer>();
    TransitionState bestFinalTs = null;
    double bestFinalScore = Double.NEGATIVE_INFINITY;
    for (BeamState beamState : alphas[emissionModel.sequenceLength()].getObjects()) {
      double score = beamState.score + beamState.transState.endScore();
      if (score > bestFinalScore) {
        bestFinalScore = score;
        bestFinalTs = beamState.transState;
      }
    }

    int currentT = emissionModel.sequenceLength();
    TransitionState currentTs = bestFinalTs;
    while (true) {
      Pair<Integer,TransitionState> backpointer = alphas[currentT].getObject(new BeamState(currentTs)).backPointer;
      int width =  currentT - backpointer.getFirst();
      transStateDecodeList.add(currentTs);
      widthsDecodeList.add(width);
      currentT = backpointer.getFirst();
      currentTs = backpointer.getSecond();
      if (currentT == 0) {
        break;
      }
    }

    Collections.reverse(transStateDecodeList);
    Collections.reverse(widthsDecodeList);
    int[] widthsDecode = a.toIntArray(widthsDecodeList);
    return Pair.makePair(transStateDecodeList.toArray(new TransitionState[0]), widthsDecode);
  }

}
