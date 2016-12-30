package tberg.murphy.fig;

/**
 * Some predefined constants for marking options. Options allow arbitrary
 * strings for tags, but these are some predefined ones.
 * 
 * @author adampauls
 * 
 */
public class OptionTag
{
	/**
	 * Everybody should know about this option
	 */
	public static final String IMPORTANT_TAG = "important";

	/**
	 * Option only exists for debugging and is only for power users
	 */
	public static final String DEBUGGING_TAG = "debugging";

	/**
	 * A path that varies from machine to machine.
	 */
	public static final String MACHINE_PATH_TAG = "machine_path";

	/**
	 * A user-defined parameter in some algorithm
	 */
	public static final String PARAM_TAG = "param";

	/**
	 * Changes the behavior of the
	 */
	public static final String BEHAVIOR_TAG = "behavior";

	/**
	 * Changes the performance (speed, memory usage, etc.) of the code
	 */
	public static final String PERFORMANCE_TAG = "performance";

	/**
	 * Indicates a non-standard option (for power users only)
	 */
	public static final String NON_STANDARD_TAG = "non_standard";

	/**
	 * Options dealing with input and output, initialization, saving, etc.
	 */
	public static final String IO_TAG = "io";

}
