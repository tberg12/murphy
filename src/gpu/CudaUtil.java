package gpu;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;

public class CudaUtil {
	
	public static CUdevice device;
	public static CUcontext context;
	
	public static void startup(int deviceId) {
        JCudaDriver.setExceptionsEnabled(true);
        JCudaDriver.cuInit(0);
        device = new CUdevice();
        cuDeviceGet(device, deviceId);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);
	}
	
	public static void shutdown() {
		JCuda.cudaDeviceReset();
	}
	
	public static CUmodule compileAndLoad(String kernelName, String kernelSrc, boolean forceCompile) {
		return loadModule(preparePtxFile(kernelName, kernelSrc, forceCompile));
	}
	
	public static String preparePtxFile(String kernelName, String kernelSrc, boolean forceCompile) {
		String ptxFileName = kernelName+".ptx";
		try {
			File ptxFile = new File(ptxFileName);
			if (!forceCompile && ptxFile.exists()) {
				return ptxFileName;
			}

			long start = System.nanoTime();
			File cuFile = new File(kernelName+".cu");
			BufferedWriter out = new BufferedWriter(new FileWriter(cuFile));
			out.append(kernelSrc);
			out.flush();
			out.close();

			String modelString = "-m"+System.getProperty("sun.arch.data.model");
			int[] major = new int[1];
			int[] minor = new int[1];
			JCudaDriver.cuDeviceComputeCapability(major, minor, device);
			String command = "nvcc -use_fast_math -arch=sm_"+major[0]+""+minor[0]+" " + modelString + " -ptx "+ cuFile.getPath()+" -o "+ptxFileName;
			System.out.println("Executing\n"+command);
			Process process = Runtime.getRuntime().exec(command);

			String errorMessage =
					new String(toByteArray(process.getErrorStream()));
			String outputMessage =
					new String(toByteArray(process.getInputStream()));
			int exitValue = 0;
			try
			{
				exitValue = process.waitFor();
			}
			catch (InterruptedException e)
			{
				Thread.currentThread().interrupt();
				throw new IOException(
						"Interrupted while waiting for nvcc output", e);
			}

			if (exitValue != 0)
			{
				System.out.println("nvcc process exitValue "+exitValue);
				System.out.println("errorMessage:\n"+errorMessage);
				System.out.println("outputMessage:\n"+outputMessage);
				throw new IOException(
						"Could not create .ptx file: "+errorMessage);
			}
			System.out.println("Finished creating PTX file");

										long end = System.nanoTime();
			System.out.println("Compile time: "+(end - start) / 1e6 + "ms");
		} catch (IOException e) {
			e.printStackTrace();
		}
		return ptxFileName;
    }
	
	public static CUmodule loadModule(String name) {
        CUmodule module = new CUmodule();
        cuModuleLoad(module, name);
        return module;
	}

    private static byte[] toByteArray(InputStream inputStream) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }
    
	// ignores the higher 16 bits
	public static float toFloat( int hbits )
	{
	    int mant = hbits & 0x03ff;            // 10 bits mantissa
	    int exp =  hbits & 0x7c00;            // 5 bits exponent
	    if( exp == 0x7c00 )                   // NaN/Inf
	        exp = 0x3fc00;                    // -> NaN/Inf
	    else if( exp != 0 )                   // normalized value
	    {
	        exp += 0x1c000;                   // exp - 15 + 127
	        if( mant == 0 && exp > 0x1c400 )  // smooth transition
	            return Float.intBitsToFloat( ( hbits & 0x8000 ) << 16
	                                            | exp << 13 | 0x3ff );
	    }
	    else if( mant != 0 )                  // && exp==0 -> subnormal
	    {
	        exp = 0x1c400;                    // make it normal
	        do {
	            mant <<= 1;                   // mantissa * 2
	            exp -= 0x400;                 // decrease exp by 1
	        } while( ( mant & 0x400 ) == 0 ); // while not normal
	        mant &= 0x3ff;                    // discard subnormal bit
	    }                                     // else +/-0 -> +/-0
	    return Float.intBitsToFloat(          // combine all parts
	        ( hbits & 0x8000 ) << 16          // sign  << ( 31 - 15 )
	        | ( exp | mant ) << 13 );         // value << ( 23 - 10 )
	}
	
	// returns all higher 16 bits as 0 for all results
	public static int fromFloat( float fval )
	{
	    int fbits = Float.floatToIntBits( fval );
	    int sign = fbits >>> 16 & 0x8000;          // sign only
	    int val = ( fbits & 0x7fffffff ) + 0x1000; // rounded value

	    if( val >= 0x47800000 )               // might be or become NaN/Inf
	    {                                     // avoid Inf due to rounding
	        if( ( fbits & 0x7fffffff ) >= 0x47800000 )
	        {                                 // is or must become NaN/Inf
	            if( val < 0x7f800000 )        // was value but too large
	                return sign | 0x7c00;     // make it +/-Inf
	            return sign | 0x7c00 |        // remains +/-Inf or NaN
	                ( fbits & 0x007fffff ) >>> 13; // keep NaN (and Inf) bits
	        }
	        return sign | 0x7bff;             // unrounded not quite Inf
	    }
	    if( val >= 0x38800000 )               // remains normalized value
	        return sign | val - 0x38000000 >>> 13; // exp - 127 + 15
	    if( val < 0x33000000 )                // too small for subnormal
	        return sign;                      // becomes +/-0
	    val = ( fbits & 0x7fffffff ) >>> 23;  // tmp exp for subnormal calc
	    return sign | ( ( fbits & 0x7fffff | 0x800000 ) // add subnormal bit
	         + ( 0x800000 >>> val - 102 )     // round depending on cut off
	      >>> 126 - val );   // div by 2^(1-(exp-127+15)) and >> 13 | exp=0
	}
	
	public static char[] convertToHalfFloat(float[] vect) {
		char[] result = new char[vect.length];
		for (int i=0; i<vect.length; ++i) {
			result[i] = (char) fromFloat(vect[i]);
		}
		return result;
	}
	
	public static float[] convertFromHalfFloat(char[] vect) {
		float[] result = new float[vect.length];
		for (int i=0; i<vect.length; ++i) {
			result[i] = toFloat(vect[i]);
		}
		return result;
	}
	
	public static float[] flatten(float[][] mat) {
		float[] result = new float[mat.length*mat[0].length];
		for (int i=0; i<mat.length; ++i) {
			System.arraycopy(mat[i], 0, result, i*mat[0].length, mat[i].length);
		}
		return result;
	}
	
	public static double[] flatten(double[][] mat) {
		double[] result = new double[mat.length*mat[0].length];
		for (int i=0; i<mat.length; ++i) {
			System.arraycopy(mat[i], 0, result, i*mat[0].length, mat[i].length);
		}
		return result;
	}
	
	public static float[] flatten(float[][][] tens) {
		float[] result = new float[tens.length*tens[0].length*tens[0][0].length];
		for (int i=0; i<tens.length; ++i) {
			for (int j=0; j<tens[0].length; ++j) {
				System.arraycopy(tens[i][j], 0, result, i*tens[0].length*tens[0][0].length + j*tens[0][0].length, tens[i][j].length);
			}
		}
		return result;
	}
	
	public static float[] flatten(List<float[]> mat) {
		float[] result = new float[mat.size()*mat.get(0).length];
		for (int i=0; i<mat.size(); ++i) {
			System.arraycopy(mat.get(i), 0, result, i*mat.get(0).length, mat.get(i).length);
		}
		return result;
	}
	
	public static int flatten(int I, int J, int i, int j) {
		return (i*J)+j;
	}
	
	public static int unflattenFirst(int I, int J, int f) {
		return f / J;
	}
	
	public static int unflattenSecond(int I, int J, int f) {
		return f % J;
	}
	
	public static String flatten(int I, int J, int i, String j) {
		return "("+(i*J)+" + "+j+")";
	}
	
	public static String flatten(int I, int J, String i, int j) {
		return "("+i+" * "+J+" + "+j+")";
	}
	
	public static String flatten(int I, int J, String i, String j) {
		return "("+i+" * "+J+" + "+j+")";
	}

	public static int flatten(int I, int J, int K, int i, int j, int k) {
		return i*J*K+j*K+k;
	}	
	
	public static String flatten(int I, int J, int K, int i, int j, String k) {
		return "("+(i*J*K+j*K)+" + "+k+")";
	}	
	
	public static String flatten(int I, int J, int K, String i, int j, int k) {
		return "("+i+" * "+(J*K) + " + "+(j*K)+" + "+k+")";
	}
	
	public static String flatten(int I, int J, int K, int i, String j, int k) {
		return "("+(i*J*K) + " + "+j+"  * "+K+" + "+k+")";
	}	
	
	public static String flatten(int I, int J, int K, String i, String j, int k) {
		return "("+i+" * "+(J*K) + " + "+j+"  * "+K+" + "+k+")";
	}	
	
	public static String flatten(int I, int J, int K, String i, int j, String k) {
		return "("+i+" * "+(J*K) + " + "+(j*K)+" + "+k+")";
	}

	public static String flatten(int I, int J, int K, int i, String j, String k) {
		return "("+(i*J*K) + " + "+j+"  * "+K+" + "+k+")";
	}	
	
	public static String flatten(int I, int J, int K, String i, String j, String k) {
		return "("+i+" * "+(J*K) + " + "+j+"  * "+K+" + "+k+")";
	}	
	
	public static float[] extendWithZeros(float[] x, int l) {
		float[] result = new float[l];
		for (int i=0; i<l; ++i) {
			if (i < x.length) {
				result[i] = x[i];
			} else {
				result[i] = 0.0f;
			}
		}
		return result;
	}

}
