package tberg.murphy.gpu;

import tberg.murphy.gpu.JOCLBlasUtil.Matrix;

public class AutoDiff {
	
	public static interface Func {
		public Matrix val(Matrix[] x);
		public Matrix[] grad(Matrix[] x);
	}
	
	public static class Node {
		
		public Node(Func f, Node[] inNodes, Node[] outNodes, int[] outBackPointers) {
			this.f = f;
			this.inNodes = inNodes;
			this.outNodes = outNodes;
			this.outBackPointers = outBackPointers;
		}
		
		private Func f;
		private Node[] inNodes;
		private Node[] outNodes;
		private int[] outBackPointers;
		
		private Matrix val = null;
		private Matrix[] grad = null;
		
		public Matrix getVal() {return val;}
		public Matrix[] getGrad() {return grad;}
		
		public void clear() {
			if (val != null || grad != null) {
				this.val = null;
				this.grad = null;
				for (Node node : inNodes) node.clear();
				for (Node node : outNodes) node.clear();
			}
		}
		
		private void computeVal() {
			if (val == null) {
				Matrix[] inVals = new Matrix[inNodes.length];
				for (int i=0; i<inNodes.length; ++i) {
					inNodes[i].computeVal();
					inVals[i] = inNodes[i].val;
				}
				val = f.val(inVals);
			}
		}
		
		private void computeGrad() {
			if (grad == null) {
				Matrix[] inVals = new Matrix[inNodes.length];
				for (int i=0; i<inNodes.length; ++i) {
					inVals[i] = inNodes[i].val;
				}
				grad = new Matrix[inNodes.length];
				Matrix[] localGrad = f.grad(inVals);
				for (int i=0; i<outNodes.length; ++i) {
					outNodes[i].computeGrad();
					Matrix outNodeGrad = outNodes[i].grad[outBackPointers[i]];
					for (int j=0; j<inNodes.length; ++j) {
						if (grad[j] == null) {
							grad[j] = outNodeGrad.mmul(localGrad[j]);
						} else {
							grad[j].addi(outNodeGrad.mmul(localGrad[j]));
						}
					}
				}
			}
		}
		

	}
	
	public static void main(String[] args) {
		
	}

}
