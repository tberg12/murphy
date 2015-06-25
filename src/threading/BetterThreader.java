package threading;

import java.util.ArrayList;
import java.util.List;

import arrays.a;

public class BetterThreader<A,B> {

	public static interface Function<A, B> {
		public void call(A a, B b);
	}
	
	boolean locked;
	double doneCount;
	Function<A,B> func;
	List<A> funcArguments;
	Thread[] pool;
	
	public BetterThreader(Function<A,B> func, int numThreads) {
		funcArguments = new ArrayList<A>();
		this.func = func;
		this.pool = new Thread[numThreads];
		for (int t=0; t<numThreads; ++t) {
			this.pool[t] = new ThreaderThread();
		}
		locked = false;
	}
	
	public void run() {
		locked = true;
		doneCount = 0;
		if (pool.length == 1) {
			A a = null;
			while ((a = popWork()) != null) {
				func.call(a, ((ThreaderThread) pool[0]).getArgument());
				doneCount++;
			}
		} else {
			for (Thread t : pool) t.start();
			for (Thread t : pool) {
				try {
					t.join();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
	}
	
	public void run(int ms) {
		locked = true;
		doneCount = 0;
		long startTime = System.currentTimeMillis();
		for (Thread t : pool) t.start();
		for (Thread t : pool) {
			try {
				long soFar = System.currentTimeMillis() - startTime;
				if (soFar >= ms) return;
				t.join(ms - soFar);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	public void run(int ms, double percentage) {
		locked = true;
		doneCount = 0;
		double initialFuncArgsSize = 0;
		synchronized (funcArguments) {
			initialFuncArgsSize = funcArguments.size();
		}
		long startTime = System.currentTimeMillis();
		for (Thread t : pool) t.start();
		long soFar = System.currentTimeMillis() - startTime;
		while (soFar < ms) {
			try {
				synchronized (funcArguments) {
					if (doneCount / initialFuncArgsSize >= percentage) {
						return;
					}
				}
				Thread.sleep(200);
			} catch (Exception e) {
				e.printStackTrace();
			}
			soFar = System.currentTimeMillis() - startTime;
		}
	}
	
	public int numThreads() {
		return pool.length;
	}

	public void addFunctionArgument(A a) {
		synchronized (funcArguments) {
			if (!locked) funcArguments.add(a);
			else throw new RuntimeException("Better threader is locked.");
		}
	}

	public void setThreadArgument(int t, B b) {
		synchronized (funcArguments) {
			if (!locked) ((ThreaderThread) pool[t]).setArgument(b);
			else throw new RuntimeException("Better threader is locked.");
		}
	}
	
	private A popWork() {
		synchronized (funcArguments) {
			if (funcArguments.isEmpty()) return null;
			else return funcArguments.remove(0);
		}
	}
	
	private class ThreaderThread extends Thread {
		B b = null;
		
		public B getArgument() {
			return b;
		}
		
		public void setArgument(B b) {
			this.b = b;
		}
		
		public void run() {
			A a = null;
			while ((a = popWork()) != null) {
				func.call(a, b);
				synchronized (funcArguments) {
					doneCount++;
				}
			}
		}
	}
	
	public static void main(String[] args) {
		int numThreads = 8;
		final double[] data = new double[] {8, 2, 3, 4};
		BetterThreader.Function<Integer,Integer> func = new BetterThreader.Function<Integer,Integer>(){public void call(Integer funcArg, Integer threadArg){
			data[funcArg] = threadArg;
		}};
		BetterThreader<Integer,Integer> threader = new BetterThreader<Integer,Integer>(func, numThreads);
		for (int i=0; i<data.length; ++i) threader.addFunctionArgument(i);
		for (int t=0; t<threader.numThreads(); ++t) threader.setThreadArgument(t, t);
		threader.run();
		System.out.println(a.toString(data));
	}
	
}
