package example;

public class DataSample {
  public int gold = -1;
  public int[] features = null;

  public DataSample(String line) {
    String[] parts = line.split(" ");
    gold = Integer.parseInt(parts[0]);
    features = new int[parts.length - 1];
    for (int i = 1; i < parts.length; i++)
      features[i - 1] = Integer.parseInt(parts[i]);
  }
}
