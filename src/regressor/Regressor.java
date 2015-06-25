package regressor;

public interface Regressor {
	
	public void train(float[][] x, float[][] y);
	
	public float[][] predict(float[][] x);

}