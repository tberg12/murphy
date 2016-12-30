package tberg.murphy.fig;

import static tberg.murphy.fig.LogInfo.error;
import static tberg.murphy.fig.LogInfo.logs;
import static tberg.murphy.fig.LogInfo.stderr;
import static tberg.murphy.fig.LogInfo.stdout;

import java.io.File;
import java.util.HashSet;
import java.util.Set;

import tberg.murphy.util.SysInfoUtils;

/**
 * Represents all the settings and output of an execution of a program.
 * An execution is defined by all the options registered with OptionsParser.
 * Creates a directory for the execution in the execution pool dir.
 */
public class Execution {

  @Option(gloss = "Whether to create a directory for this run; if not, don't generate output files")
  public static boolean create = false;

  // How to create the execution directory
  @Option(gloss = "Directory to put all output files; if blank, use execPoolDir.")
  public static String execDir;
  @Option(gloss = "Directory which contains all the executions (or symlinks).")
  public static String execPoolDir;
  @Option(gloss = "Directory which actually holds the executions.")
  public static String actualExecPoolDir;
  @Option(gloss = "Overwrite the contents of the execDir if it doesn't exist (e.g., when running a thunk).")
  public static boolean overwriteExecDir;
  @Option(gloss = "Assume in the run directory, automatically set execPoolDir and actualExecPoolDir")
  public static boolean useStandardExecPoolDirStrategy = false;

  @Option(gloss = "Simply print options and exit.")
  public static boolean printOptionsAndExit = false;

  @Option(gloss = "Character encoding")
  public static String charEncoding;

  // Execution directory that we write to (execDir is just a suggestion)
  // Could be a symlink to a directory in actualExecPoolDir
  private static String virtualExecDir;

  // Passed to the options parser
  public static boolean ignoreUnknownOpts = false;

  static OrderedStringMap inputMap = new OrderedStringMap(); // Accessed by monitor thread
  private static OrderedStringMap outputMap = new OrderedStringMap();
  private static OptionsParser parser;
  static int exitCode = 0;

  static boolean shouldBail = false; // Set by monitor thread

  public static boolean shouldBail() {
    return shouldBail;
  }

  private static void mkdirHard(File f) {
    if (!f.mkdir()) {
      stderr.println("Cannot create directory: " + f);
      System.exit(1);
    }
  }

  public static String getVirtualExecDir() {
    return virtualExecDir;
  }

  public static void setVirtualExecDir(String dir) {
    virtualExecDir = dir;
  }

  /**
   * Return an unused directory in the execution pool directory.
   * Set virtualExecDir
   */
  public static String createVirtualExecDir() {
    if (useStandardExecPoolDirStrategy) {
      // Assume we are in the run directory, so set the standard paths
      execPoolDir = new File(SysInfoUtils.getcwd(), "state/execs").toString();
      actualExecPoolDir = new File(SysInfoUtils.getcwd(), "state/hosts/" + SysInfoUtils.getShortHostName()).toString();
      if (!new File(actualExecPoolDir).isDirectory())
        actualExecPoolDir = null;
    }
    if (!StrUtils.isEmpty(execPoolDir) && !new File(execPoolDir).isDirectory())
      throw Exceptions.bad("Execution pool directory '" + execPoolDir + "' doesn't exist");
    if (!StrUtils.isEmpty(actualExecPoolDir) && !new File(actualExecPoolDir).isDirectory())
      throw Exceptions.bad("Actual execution pool directory '" + actualExecPoolDir + "' doesn't exist");

    if (!StrUtils.isEmpty(execDir)) { // Use specified execDir
      boolean exists = new File(execDir).isDirectory();
      if (exists && !overwriteExecDir)
        throw Exceptions.bad("Directory already exists and overwrite flag is false");
      if (!exists)
        mkdirHard(new File(execDir));
      else {
        // This part looks at actualExecPoolDir
        // This case is overwriting an existing execution directory, which
        // happens when we are executing a thunk.  We have to be careful here
        // because the actual symlinked directory that was created when thunking
        // might be using a different actualPoolDir.  If this happens, we need
        // to move the actual thunked symlinked directory into the actual
        // execution pool directory requested.  In fact, we always do this for simplicity.
        String oldActualExecDir = Utils.systemGetStringOutputEasy("readlink " + execDir);
        if (oldActualExecDir == null) { // Not symlink
          if (!StrUtils.isEmpty(actualExecPoolDir))
            throw Exceptions.bad("The old execution directory was not created with actualExecPoolDir but now we want an actualExecPoolDir");
          // Do nothing, just use the directory as is
        } else { // Symlink
          oldActualExecDir = oldActualExecDir.trim();
          if (StrUtils.isEmpty(actualExecPoolDir))
            throw Exceptions.bad("The old execution directory was created with actualExecPoolDir but now we don't want an actualExecPoolDir");
          // Note that now the execution numbers might not correspond between the
          // actual and virtual execution pool directories.
          File newActualExecDir = null;
          for (int i = 0;; i++) {
            newActualExecDir = new File(actualExecPoolDir, i + "a.exec");
            if (!newActualExecDir.exists())
              break;
          }
          // Move the old directory to the new directory
          Utils.systemHard(String.format("mv %s %s", oldActualExecDir, newActualExecDir));
          // Update the symlink (execDir -> newActualExecDir)
          Utils.systemHard(String.format("ln -sf %s %s", newActualExecDir.getAbsolutePath(), execDir));
        }
      }
      return virtualExecDir = execDir;
    }

    // execDir hasn't been specified, so we need to pick one from a pool directory
    // execPoolDir must exist; actualExecPoolDir is optional

    // Get a list of files that already exists
    Set<String> files = new HashSet<String>();
    for (String f : new File(execPoolDir).list())
      files.add(f);

    // Go through and pick out a file that doesn't exist
    int numFailures = 0;
    for (int i = 0; numFailures < 3; i++) {
      // Either the virtual file (a link) or the actual file
      final String execDirSuffix = ".exec";
      File f = new File(execPoolDir, i + execDirSuffix);
      // Actual file
      File g = StrUtils.isEmpty(actualExecPoolDir) ? null : new File(actualExecPoolDir, i + execDirSuffix);

      if (!files.contains(i + execDirSuffix) && (g == null || !g.exists())) {
        if (g == null || g.equals(f)) {
          mkdirHard(f);
          return virtualExecDir = f.toString();
        }
        // Create symlink before mkdir to try to reserve the name and avoid race conditions
        if (Utils.createSymLink(g.getAbsolutePath(), f.getAbsolutePath())) {
          mkdirHard(g);
          return virtualExecDir = f.toString();
        }

        // Probably because someone else already linked to it
        // in the race condition: so try again
        stderr.println("Cannot create symlink from " + f + " to " + g);
        numFailures++;
      }
    }
    throw Exceptions.bad("Failed many times to create execution directory");
  }

  public static boolean isVirtualExecDirSet() {
    return !StrUtils.isEmpty(virtualExecDir);
  }

  // Get the path of the file (in the execution directory)
  public static String getFile(String file) {
    if (!isVirtualExecDirSet())
      return file;
    //    if(StrUtils.isEmpty(file)) return null;
    return new File(virtualExecDir, file).toString();
  }

  public static void linkFileToExec(String realFileName, String file) {
    if (StrUtils.isEmpty(realFileName) || StrUtils.isEmpty(file))
      return;
    File f = new File(realFileName);
    Utils.createSymLink(f.getAbsolutePath(), getFile(file));
  }

  public static void linkFileFromExec(String file, String realFileName) {
    if (StrUtils.isEmpty(realFileName) || StrUtils.isEmpty(file))
      return;
    File f = new File(realFileName);
    Utils.createSymLink(getFile(file), f.getAbsolutePath());
  }

  // Getting input and writing output
  public static boolean getBooleanInput(String s) {
    String t = inputMap.get(s, "0");
    return t.equals("true") || t.equals("1");
  }

  public static String getInput(String s) {
    return inputMap.get(s);
  }

  public synchronized static void putOutput(String s, Object t) {
    outputMap.put(s, StrUtils.toString(t));
  }

  public synchronized static void printOutputMapToStderr() {
    outputMap.print(stderr);
  }

  public synchronized static void printOutputMap(String path) {
    if (StrUtils.isEmpty(path))
      return;
    // First write to a temporary directory and then rename the file
    String tmpPath = path + ".tmp";
    if (outputMap.printEasy(tmpPath))
      new File(tmpPath).renameTo(new File(path));
  }

  public static void setExecStatus(String newStatus, boolean override) {
    String oldStatus = outputMap.get("exec.status");
    if (oldStatus == null || oldStatus.equals("running"))
      override = true;
    if (override)
      putOutput("exec.status", newStatus);
  }

  static OrderedStringMap getInfo() {
    OrderedStringMap map = new OrderedStringMap();
    map.put("Date", SysInfoUtils.getCurrentDateStr());
    map.put("Host", SysInfoUtils.getHostName());
    map.put("CPU speed", SysInfoUtils.getCPUSpeedStr());
    map.put("Max memory", SysInfoUtils.getMaxMemoryStr());
    map.put("Num CPUs", SysInfoUtils.getNumCPUs());
    return map;
  }

  public static void init(String[] args, Object... objects) {
    //// Parse options
    // If one of the objects is an options parser, use that; otherwise, create a new one
    for (int i = 0; i < objects.length; i++) {
      if (objects[i] instanceof OptionsParser) {
        parser = (OptionsParser) objects[i];
        objects[i] = null;
      }
    }
    if (parser == null)
      parser = new OptionsParser();
    parser.doRegister("log", LogInfo.class);
    parser.doRegister("exec", Execution.class);
    parser.doRegisterAll(objects);
    // These options are specific to the execution, so we don't want to overwrite them
    // with a previous execution's.
    parser.setDefaultDirFileName("options.map");
    parser.setIgnoreOptsFromFileName("options.map", ListUtils.newList("log.file", "exec.execDir", "exec.execPoolDir",
                                                                      "exec.actualPoolDir", "exec.makeThunk"));
    if (ignoreUnknownOpts)
      parser.ignoreUnknownOpts();
    if (!parser.doParse(args))
      System.exit(1);

    // Set character encoding
    if (charEncoding != null)
      CharEncUtils.setCharEncoding(charEncoding);

    if (printOptionsAndExit) { // Just print options and exit
      parser.doGetOptionPairs().print(stdout);
      System.exit(0);
    }

    // Create a new directory
    if (create) {
      createVirtualExecDir();
      //stderr.println(virtualExecDir);
      LogInfo.file = getFile("log");
    } else {
      LogInfo.file = "";
    }

    LogInfo.init();
    // Output options
    logs("Execution directory: " + virtualExecDir);
    getInfo().printEasy(getFile("info.map"));
    printOptions();
  }

  // Might want to call this again after some command-line options were changed.
  public static void printOptions() {
    parser.doGetOptionPairs().printEasy(getFile("options.map"));
    parser.doGetOptionStrings().printEasy(getFile("options.help"));
  }

  public static void raiseException(Throwable t) {
    error(t + ":\n" + StrUtils.join(t.getStackTrace(), "\n"));
    t = t.getCause();
    if (t != null)
      error("Caused by " + t + ":\n" + StrUtils.join(t.getStackTrace(), "\n"));
    putOutput("exec.status", "exception");
    exitCode = 1;
  }

  // This should be all we need to put in a main function.
  // args are the commandline arguments
  // First object is the Runnable object to call run on.
  // All of them are objects whose options args is to supposed to populate.
  public static void run(String[] args, Object... objects) {
    runWithObjArray(args, objects);
  }

  public static void runWithObjArray(String[] args, Object[] objects) {
    init(args, objects);
    Object mainObj;
    if (objects[0] instanceof String)
      mainObj = objects[1];
    else
      mainObj = objects[0];
    try {
      ((Runnable) mainObj).run();
    } catch (Throwable t) {
      raiseException(t);
    }
    System.exit(exitCode);
  }
}
