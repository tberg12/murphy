package util;

import java.text.NumberFormat;
import java.util.Arrays;
import java.util.NoSuchElementException;

/**
 * Max-heap
 * 
 * @author Dan Klein
 */
public class IntPriorityQueue
{
	private static final long serialVersionUID = 1L;

	private int size;

	private int capacity;

	private int[] elements;

	private final int[] elementPositions;

	private double[] priorities;

	private final int maxElem;

	protected void grow(int newCapacity) {
		elements = elements == null ? new int[newCapacity] : Arrays.copyOf(elements, newCapacity);
		priorities = priorities == null ? new double[newCapacity] : Arrays.copyOf(priorities, newCapacity);

		capacity = newCapacity;
	}

	protected int parent(int loc) {
		return (loc - 1) / 2;
	}

	protected int leftChild(int loc) {
		return 2 * loc + 1;
	}

	protected int rightChild(int loc) {
		return 2 * loc + 2;
	}

	protected void heapifyUp(int loc) {
		if (loc == 0) return;
		int parent = parent(loc);
		if (priorities[loc] > priorities[parent]) {
			swap(loc, parent);
			heapifyUp(parent);
		}
	}

	protected void heapifyDown(int loc) {
		int max = loc;
		int leftChild = leftChild(loc);
		if (leftChild < size()) {
			double priority = priorities[loc];
			double leftChildPriority = priorities[leftChild];
			if (leftChildPriority > priority) max = leftChild;
			int rightChild = rightChild(loc);
			if (rightChild < size()) {
				double rightChildPriority = priorities[rightChild(loc)];
				if (rightChildPriority > priority && rightChildPriority > leftChildPriority) max = rightChild;
			}
		}
		if (max == loc) return;
		swap(loc, max);
		heapifyDown(max);
	}

	protected void swap(int loc1, int loc2) {
		double tempPriority = priorities[loc1];
		int tempElement = (elements[loc1]);
		priorities[loc1] = priorities[loc2];
		final int element2 = elements[loc2];
		elements[loc1] = element2;
		if (elementPositions != null) elementPositions[element2] = loc1;
		priorities[loc2] = tempPriority;
		elements[loc2] = (tempElement);
		if (elementPositions != null) elementPositions[tempElement] = loc2;
	}

	protected void removeFirst() {
		if (size < 1) return;
		swap(0, size - 1);
		if (elementPositions != null) elementPositions[elements[size]] = -1;
		size--;
		heapifyDown(0);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.util.PriorityQueueInterface#hasNext()
	 */
	public boolean hasNext() {
		return !isEmpty();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.util.PriorityQueueInterface#next()
	 */
	public int next() {
		int first = peek();
		removeFirst();
		return first;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.util.PriorityQueueInterface#remove()
	 */
	public void remove() {
		throw new UnsupportedOperationException();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.util.PriorityQueueInterface#peek()
	 */
	public int peek() {
		if (size() > 0) return elements[0];
		throw new NoSuchElementException();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.util.PriorityQueueInterface#getPriority()
	 */
	public double getPriorityOfBest() {
		if (size() > 0) return priorities[0];
		return Double.NaN;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.util.PriorityQueueInterface#size()
	 */
	public int size() {
		return size;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.util.PriorityQueueInterface#isEmpty()
	 */
	public boolean isEmpty() {
		return size == 0;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.util.PriorityQueueInterface#add(E, double)
	 */
	public boolean add(int key, double priority) {
		if (size == capacity) {
			grow(2 * capacity + 1);
		}
		elements[size] = key;
		if (elementPositions != null) elementPositions[key] = size;
		priorities[size] = priority;
		heapifyUp(size);
		size++;
		return true;
	}

	public void put(int key, double priority) {
		add(key, priority);
	}

	/**
	 * Returns a representation of the queue in decreasing priority order.
	 */
	@Override
	public String toString() {
		return toString(size(), false);
	}

	/**
	 * Returns a representation of the queue in decreasing priority order,
	 * displaying at most maxKeysToPrint elements and optionally printing one
	 * element per line.
	 * 
	 * @param maxKeysToPrint
	 * @param multiline
	 *            TODO
	 */
	public String toString(int maxKeysToPrint, boolean multiline) {
		IntPriorityQueue pq = clone();
		StringBuilder sb = new StringBuilder(multiline ? "" : "[");
		int numKeysPrinted = 0;
		NumberFormat f = NumberFormat.getInstance();
		f.setMaximumFractionDigits(5);
		while (numKeysPrinted < maxKeysToPrint && pq.hasNext()) {
			double priority = pq.getPriorityOfBest();
			int element = pq.next();
			sb.append("" + element);
			sb.append(" : ");
			sb.append(f.format(priority));
			if (numKeysPrinted < size() - 1) sb.append(multiline ? "\n" : ", ");
			numKeysPrinted++;
		}
		if (numKeysPrinted < size()) sb.append("...");
		if (!multiline) sb.append("]");
		return sb.toString();
	}

	/**
	 * Returns a clone of this priority queue. Modifications to one will not
	 * affect modifications to the other.
	 */
	@Override
	public IntPriorityQueue clone() {
		IntPriorityQueue clonePQ = new IntPriorityQueue(maxElem);
		clonePQ.size = size;
		clonePQ.capacity = capacity;
		clonePQ.elements = Arrays.copyOf(elements, elements.length);
		clonePQ.priorities = Arrays.copyOf(priorities, priorities.length);
		if (elementPositions != null) System.arraycopy(elementPositions, 0, clonePQ.elementPositions, 0, elementPositions.length);
		return clonePQ;
	}

	public IntPriorityQueue(int maxElem) {
		this(maxElem, 15);
	}

	public IntPriorityQueue(int maxElem, int capacity) {
		this.maxElem = maxElem;
		elementPositions = maxElem < 0 ? null : new int[maxElem + 1];
		if (elementPositions != null) Arrays.fill(elementPositions, -1);
		int legalCapacity = 0;
		while (legalCapacity < capacity) {
			legalCapacity = 2 * legalCapacity + 1;
		}
		grow(legalCapacity);
	}
	
	public void clear() {
		size = 0;
		if (elementPositions != null) Arrays.fill(elementPositions, -1);
		Arrays.fill(elements, 0);
		Arrays.fill(priorities, 0);
	}

	public static void main(String[] args) {
		IntPriorityQueue pq = new IntPriorityQueue(3);
		System.out.println(pq);
		pq.put(1, 1);
		System.out.println(pq);
		pq.put(3, 3);
		System.out.println(pq);
		pq.put(1, 1.1);
		System.out.println(pq);
		pq.put(2, 2);
		System.out.println(pq);
		System.out.println(pq.toString(2, false));
		while (pq.hasNext()) {
			System.out.println(pq.next());
		}
	}

	public int[] toSortedList() {
		int[] l = new int[size()];
		IntPriorityQueue pq = clone();
		int k = 0;
		while (pq.hasNext()) {
			l[k++] = pq.next();
		}
		return l;
	}

	public double getPriorityOfElement(int element) {
		assert (elementPositions == null);
		int loc = elementPositions[element];
		if (loc < 0 || loc >= size) return Double.NaN;
		return priorities[loc];
	}

	public void decreaseKey(int element, double cost) {
		assert (elementPositions == null);
		int loc = elementPositions[element];
		assert loc >= 0;
		assert cost < priorities[loc];
		priorities[loc] = cost;
		heapifyDown(loc);
	}
}
