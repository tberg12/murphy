package indexer;

import java.io.Serializable;
import java.util.Collection;

public interface Indexer<A> extends Serializable {
	
	public boolean locked();

	public void lock();

	public int size();

	public boolean contains(A object);

	public int getIndex(A object);

	public A getObject(int index);

	public void index(A[] vect);
	
	public void forgetIndexLookup();
	
	public Collection<A> getObjects();

}