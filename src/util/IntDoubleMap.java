package util;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;

public final class IntDoubleMap {

	private int[] keys;

	private double[] values;

	private int size = 0;

	private static final int EMPTY_KEY = -1;

	private double maxLoadFactor = 0.5;

	private boolean sorted = false;

	private double deflt = Double.NEGATIVE_INFINITY;

	public IntDoubleMap() {
		this(100);
	}

	public double estimateMemoryUsage() {
		return (8.0*keys.length + 8.0*values.length) / 1e9;
	}
	
	public void setLoadFactor(double loadFactor) {
		this.maxLoadFactor = loadFactor;
		ensureCapacity(values.length);
	}

	public IntDoubleMap(int initCapacity_) {
		int initCapacity = toSize(initCapacity_);
		keys = new int[initCapacity];
		values = new double[initCapacity];
		Arrays.fill(keys, EMPTY_KEY);
	}

	private int toSize(int initCapacity_) {
		return Math.max(5, (int) (initCapacity_ / maxLoadFactor) + 1);
	}

	public boolean put(int k, double v) {
		checkNotImmutable();
		if (size / (double) keys.length > maxLoadFactor) {
			rehash();
		}
		return putHelp(k, v, keys, values);

	}

	private void checkNotImmutable() {
		if (keys == null) throw new RuntimeException("Cannot change wrapped IntCounter");
		if (sorted) throw new RuntimeException("Cannot change sorted IntCounter");
	}

	private void rehash() {
		final int length = keys.length * 2 + 1;
		rehash(length);
	}

	private void rehash(final int length) {
		checkNotImmutable();
		int[] newKeys = new int[length];
		double[] newValues = new double[length];
		Arrays.fill(newKeys, EMPTY_KEY);
		size = 0;
		for (int i = 0; i < keys.length; ++i) {
			int curr = keys[i];
			if (curr != EMPTY_KEY) {
				double val = values[i];
				putHelp(curr, val, newKeys, newValues);
			}
		}
		keys = newKeys;
		values = newValues;
	}

	private boolean putHelp(int k, double v, int[] keyArray, double[] valueArray) {
		checkNotImmutable();
		assert k >= 0;
		int pos = find(k, true, keyArray, valueArray);
		//		int pos = getInitialPos(k, keyArray);
		int currKey = keyArray[pos];
		//		while (currKey != EMPTY_KEY && currKey != k) {
		//			pos++;
		//			if (pos == keyArray.length) pos = 0;
		//			currKey = keyArray[pos];
		//		}
		//
		valueArray[pos] = v;
		if (currKey == EMPTY_KEY) {
			size++;
			keyArray[pos] = k;
			return true;
		}
		return false;
	}

	private static int getInitialPos(final int k, final int[] keyArray) {
		if (keyArray == null) return k;
		int hash = k;
		if (hash < 0) hash = -hash;
		int pos = hash % keyArray.length;
		return pos;
	}

	public double get(int k) {
		int pos = find(k, false);
		if (pos == EMPTY_KEY) return deflt;
		return values[pos];
	}

	private int find(int k, boolean returnLastEmpty) {
		return find(k, returnLastEmpty, keys, values);
	}

	private int find(int k, boolean returnLastEmpty, int[] keyArray, double[] valueArray) {
		if (keyArray == null) {
			return (k < valueArray.length ? k : EMPTY_KEY);
		} else if (sorted) {
			final int pos = Arrays.binarySearch(keyArray, k);
			return pos < 0 ? EMPTY_KEY : pos;

		} else {
			final int[] localKeys = keyArray;
			final int length = localKeys.length;
			int pos = getInitialPos(k, localKeys);
			long curr = localKeys[pos];
			while (curr != EMPTY_KEY && curr != k) {
				pos++;
				if (pos == length) pos = 0;
				curr = localKeys[pos];
			}
			return returnLastEmpty ? pos : (curr == EMPTY_KEY ? EMPTY_KEY : pos);
		}
	}

	public void setDefault(double d) {
		this.deflt = d;
	}

	public boolean isEmpty() {
		return size == 0;
	}

	public class Entry implements Map.Entry<Integer, Double>
	{
		private int index;

		public int key;

		public double value;

		public Entry(int key, double value, int index) {
			super();
			this.key = key;
			assert key >= 0;
			this.value = value;
			this.index = index;
		}

		public Integer getKey() {
			return key;
		}

		public Double getValue() {
			return value;
		}

		public Double setValue(Double value) {
			this.value = value;
			values[index] = value;
			return this.value;
		}
	}

	private class KeyIterator extends MapIterator<Integer>
	{
		public Integer next() {
			final int nextIndex = nextIndex();
			return keys == null ? nextIndex : keys[nextIndex];
		}
	}

	private abstract class MapIterator<E> implements Iterator<E>
	{
		public MapIterator() {
			end = keys == null ? size : values.length;
			next = -1;
			nextIndex();
		}

		public boolean hasNext() {
			return end > 0 && next < end;
		}

		int nextIndex() {
			int curr = next;
			do {
				next++;
			} while (next < end && keys != null && keys[next] == EMPTY_KEY);
			return curr;
		}

		public void remove() {
			throw new UnsupportedOperationException();
		}

		private int next, end;
	}

	public void ensureCapacity(int capacity) {
		checkNotImmutable();
		int newSize = toSize(capacity);
		if (newSize > keys.length) {
			rehash(newSize);
		}
	}

	public int size() {
		return size;
	}

	public Iterable<Integer> keys() {
		return new Iterable<Integer>()
		{
			public Iterator<Integer> iterator() {
				return (new KeyIterator());
			}
		};
	}

	public void clear() {
		Arrays.fill(keys, EMPTY_KEY);
		Arrays.fill(values, deflt);
		size = 0;
	}
	
}
