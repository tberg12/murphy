package tberg.murphy.gpu;

import static org.jocl.CL.*;
import static org.jocl.blas.CLBLAS.clblasSetup;
import static org.jocl.blas.CLBLAS.clblasTeardown;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Random;

import org.jblas.FloatMatrix;
import org.jocl.*;
import org.jocl.blas.*;

import tberg.murphy.arrays.a;

public class JOCLBlasUtil {
	
	public static final boolean DEBUG_SYNC = false;
	
	private static cl_context context;
	private static cl_command_queue commandQueue;
	private static cl_program program;
	private static cl_event lastEvent = null;
	private static Map<String,cl_kernel> builtKernels = null;
	public static LinkedList<Matrix> allocated;
	
	private static String getString(cl_device_id device, int paramName) {

		// Obtain the length of the string that will be queried
		long size[] = new long[1];
		clGetDeviceInfo(device, paramName, 0, null, size);

		// Create a buffer of the appropriate size and fill it with the info
		byte buffer[] = new byte[(int)size[0]];
		clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

		// Create a string from the buffer (excluding the trailing \0 byte)
		return new String(buffer, 0, buffer.length-1);

	}
	
	public static void startup() {
		final int platformIndex = 0;
		final long deviceType = CL_DEVICE_TYPE_ALL;
		CL.setExceptionsEnabled(true);
		int numPlatformsArray[] = new int[1];
		clGetPlatformIDs(0, null, numPlatformsArray);
		int numPlatforms = numPlatformsArray[0];
		cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
		clGetPlatformIDs(platforms.length, platforms, null);
		cl_platform_id platform = platforms[platformIndex];
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
		int numDevicesArray[] = new int[1];
		clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
		int numDevices = numDevicesArray[0];
		startup(numDevices-1);
	}
	
	public static void startup(int deviceIndex) {
		// The platform, device type and device number
		// that will be used

		final int platformIndex = 0;
		final long deviceType = CL_DEVICE_TYPE_ALL;

		// Enable exceptions and subsequently omit error checks in this sample
		CL.setExceptionsEnabled(true);

		// Obtain the number of platforms
		int numPlatformsArray[] = new int[1];
		clGetPlatformIDs(0, null, numPlatformsArray);
		int numPlatforms = numPlatformsArray[0];

		// Obtain a platform ID
		cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
		clGetPlatformIDs(platforms.length, platforms, null);
		cl_platform_id platform = platforms[platformIndex];

		// Initialize the context properties
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

		// Obtain the number of devices for the platform
		int numDevicesArray[] = new int[1];
		clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
		int numDevices = numDevicesArray[0];

		// Obtain a device ID
		cl_device_id devices[] = new cl_device_id[numDevices];
		clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
		cl_device_id device = devices[deviceIndex];

		// Create a context for the selected device
		context = clCreateContext(contextProperties, 1, new cl_device_id[]{device}, null, null, null);
		String deviceName = getString(devices[deviceIndex], CL_DEVICE_NAME);
		System.out.printf("CL_DEVICE_NAME: %s\n", deviceName);

		// Create a command-queue
		commandQueue = clCreateCommandQueue(context, devices[deviceIndex], 0, null);
		
        // Create the program from the source code
        program = clCreateProgramWithSource(context, 1, new String[]{ kernels }, null, null);
        
        // Build the program
        clBuildProgram(program, 0, null, "-cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math -cl-no-signed-zeros", null, null);
 
		allocated = new LinkedList<Matrix>();
		lastEvent = new cl_event();
		clEnqueueBarrierWithWaitList(commandQueue, 0, null, lastEvent);
		
		// Setup CLBLAS
		CLBLAS.setExceptionsEnabled(DEBUG_SYNC);
		clblasSetup( );
		
		// Build kernels
		builtKernels = new HashMap<String,cl_kernel>();
		for (String kernelName : kernelNames) {
			cl_kernel kernel = clCreateKernel(program, kernelName, null);
			builtKernels.put(kernelName, kernel);
		}
	}
	
	public static void shutdown() {
		clFinish(commandQueue);
		freeAll(true);
		for (cl_kernel kernel : builtKernels.values()) {
			clReleaseKernel(kernel);
		}
		clblasTeardown();
		clReleaseCommandQueue(commandQueue);
		clReleaseContext(context);
	}
	
	public static void freeAll() {
		freeAll(false);
	}
	
	public static void freeAll(boolean freeDontFree) {
		clFinish(commandQueue);
		LinkedList<Matrix> remainingAllocated = new LinkedList<Matrix>();
		while (!allocated.isEmpty()) {
			Matrix mat = allocated.poll();
			if (freeDontFree || !mat.dontFree) {
				mat.free();
			} else {
				remainingAllocated.add(mat);
			}
		}
		allocated = remainingAllocated;
	}
	
	public static void freeAllBut(Matrix... args) {
		Collection<Matrix> keep = new HashSet<Matrix>();
		for (Matrix mat : args) keep.add(mat);
		freeAllBut(keep);
	}
	
	public static void freeAllBut(Collection<Matrix> keep) {
		clFinish(commandQueue);
		LinkedList<Matrix> remainingAllocated = new LinkedList<Matrix>();
		while (!allocated.isEmpty()) {
			Matrix mat = allocated.poll();
			if (!keep.contains(mat) && !mat.dontFree) {
				mat.free();
			} else {
				remainingAllocated.add(mat);
			}
		}
		allocated = remainingAllocated;
	}
	
	public static class Matrix {
		
		private boolean dontFree;
		private boolean alreadyFreed;
		private int rows;
		private int cols;
		private cl_mem data_d;
		
		public Matrix(int rows, int cols) {
			this.dontFree = false;
			this.alreadyFreed = false;
			this.rows = rows;
			this.cols = cols;
			this.data_d = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * cols * Sizeof.cl_float, null, null);
			if (DEBUG_SYNC) clFinish(commandQueue);
			JOCLBlasUtil.allocated.add(this);
		}
		
		public void setDontFree(boolean dontFree) {
			this.dontFree = dontFree;
		}
		
		public void setAlreadyFreed(boolean alreadyFreed) {
			this.alreadyFreed = alreadyFreed;
		}
		
		public boolean dontFree() {
			return dontFree;
		}
		
		public boolean alreadyFreed() {
			return alreadyFreed;
		}
		
		public boolean equals(Object other) {
		    if (other instanceof Matrix) {
		    	Matrix that = (Matrix) other;
		    	if (!this.data_d.equals(that.data_d)) {
		    		return false;
		    	} else {
		    		return true;
		    	}
		    } else {
		    	return false;
		    }
		}
		public int hashCode() {
			return this.data_d.hashCode();
		}
		
		public static Matrix build(float[][] mat) {
			Matrix result = new Matrix(mat.length, mat[0].length);
			float[] data_h = toColMajor(mat);
			clEnqueueWriteBuffer(commandQueue, result.data_d, CL_TRUE, 0, result.rows * result.cols * Sizeof.cl_float, Pointer.to(data_h), 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
			return result;
		}
		
		public static Matrix build(int rows, int cols, float[] data_h) {
			Matrix result = new Matrix(rows, cols);
			clEnqueueWriteBuffer(commandQueue, result.data_d, CL_TRUE, 0, result.rows * result.cols * Sizeof.cl_float, Pointer.to(data_h), 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
			return result;
		}
		
		public static Matrix rand(int rows, int cols, Random rand) {
			return Matrix.build(a.randFloat(rows, cols, rand));
		}
		
		public static Matrix ones(int rows, int cols) {
			Matrix result = new Matrix(rows, cols);
			result.set(1.0f);
			return result;
		}
		
		public static Matrix zeros(int rows, int cols) {
			Matrix result = new Matrix(rows, cols);
			result.zeroi();
			return result;
		}
		
		public static Matrix eye(int n) {
			Matrix result = zeros(n, n);
			result.diagAddi(1.0f);
			return result;
		}
		
		public boolean isVector() {
			return rows == 1 || cols == 1;
		}
		
		public boolean isScalar() {
			return rows == 1 && cols == 1;
		}
		
		public int rows() {
			return rows;
		}

		public int cols() {
			return cols;
		}
		
		public Matrix copy() {
			Matrix result = new Matrix(rows, cols);
			clEnqueueCopyBuffer(commandQueue, data_d, result.data_d, 0, 0, rows * cols * Sizeof.cl_float, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
			return result;
		}
		
		public Matrix copySubmatrix(int r0, int r1, int c0, int c1) {
			Matrix result = new Matrix(r1-r0, c1-c0);
			CLBLAS.clblasCopySubMatrixAsync(clblasOrder.clblasColumnMajor, Sizeof.cl_float, this.data_d, 0, this.rows, this.rows, this.cols, r0, c0, result.data_d, 0, result.rows, result.rows, result.cols, 0, 0, (r1-r0), (c1-c0), commandQueue, 1, new cl_event[] {lastEvent}, new cl_event[] {lastEvent});
			if (DEBUG_SYNC) clFinish(commandQueue);
			return result;
		}
		
		public Matrix setSubmatrix(int r, int c, Matrix that, int r0, int r1, int c0, int c1) {
			CLBLAS.clblasCopySubMatrixAsync(clblasOrder.clblasColumnMajor, Sizeof.cl_float, that.data_d, 0, that.rows, that.rows, that.cols, r0, c0, this.data_d, 0, this.rows, this.rows, this.cols, r, c, (r1-r0), (c1-c0), commandQueue, 1, new cl_event[] {lastEvent}, new cl_event[] {lastEvent});
			if (DEBUG_SYNC) clFinish(commandQueue);
			return this;
		}
		
		public Matrix setSubmatrix(Matrix that, int r, int c) {
			CLBLAS.clblasCopySubMatrixAsync(clblasOrder.clblasColumnMajor, Sizeof.cl_float, that.data_d, 0, that.rows, that.rows, that.cols, 0, 0, this.data_d, 0, this.rows, this.rows, this.cols, r, c, that.rows, that.cols, commandQueue, 1, new cl_event[] {lastEvent}, new cl_event[] {lastEvent});
			if (DEBUG_SYNC) clFinish(commandQueue);
			return this;
		}
		
		public Matrix setSubmatrix(float[][] mat, int r, int c) {
			setSubmatrix(Matrix.build(mat), r, c);
			return this;
		}
		
		public Matrix set(int r, int c, float alpha) {
			return setSubmatrix(new float[][] {{alpha}}, r, c);
		}
		
		public Matrix copyRow(int r) {
			return copySubmatrix(r, r+1, 0, cols);
		}
		
		public Matrix copyCol(int c) {
			return copySubmatrix(0, rows, c, c+1);
		}
		
		public Matrix setRow(int r, Matrix row) {
			return setSubmatrix(row, r, 0);
		}
		
		public Matrix setCol(int c, Matrix col) {
			return setSubmatrix(col, 0, c);
		}
		
		public Matrix set(float alpha) {
			scalarSet(this, alpha);
			if (DEBUG_SYNC) clFinish(commandQueue);
			return this;
		}
		
		public float[] toArray() {
			float[] data_h = new float[rows*cols];
			clEnqueueReadBuffer(commandQueue, data_d, CL_TRUE, 0, rows * cols * Sizeof.cl_float, Pointer.to(data_h), 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
			return data_h;
		}
		
		public float[][] toArray2() {
			float[] data_h = toArray();
			return fromColMajor(data_h, rows);
		}
		
		public void free() {
			setDontFree(false);
			if (data_d != null && !alreadyFreed()) clReleaseMemObject(data_d);
			setAlreadyFreed(true);
		}
		
		//////////////////////////////////////////
		
		public Matrix diagAdd(float alpha) {
			Matrix diag = new Matrix(1, this.cols);
			diag.set(alpha);
			return diagAdd(diag);
		}
		
		public Matrix diagAddi(float alpha) {
			Matrix diag = new Matrix(1, this.cols);
			diag.set(alpha);
			return diagAddi(diag);
		}
		
		public Matrix diagAdd(Matrix diag) {
			Matrix result = this.copy();
			return result.diagAddi(diag);
		}
		
		public Matrix diagAddi(Matrix diag) {
			if (this.rows != this.cols) throw new Error("diagAddi: matrix not square");
			if (this.rows != diag.rows*diag.cols) throw new Error("diagAddi: sizes do not match");
			CLBLAS.clblasSaxpy(diag.rows*diag.cols, 1.0f, diag.data_d, 0, 1, this.data_d, 0, this.rows+1, 1, new cl_command_queue[] {commandQueue}, 1, new cl_event[] {lastEvent}, new cl_event[] {lastEvent});
			if (DEBUG_SYNC) clFinish(commandQueue);
			return this;
		}

		public Matrix rowMul(Matrix row) {
			if (this.cols != row.rows*row.cols) throw new Error("rowMul: sizes do not match");
			Matrix result = new Matrix(this.rows, this.cols);
			rowMul(this, row, result);
			return result;
		}
		
		public Matrix colMul(Matrix col) {
			if (this.rows != col.rows*col.cols) throw new Error("colMul: sizes do not match");
			Matrix result = new Matrix(this.rows, this.cols);
			colMul(this, col, result);
			return result;
		}
		
		public Matrix rowMuli(Matrix row) {
			if (this.cols != row.rows*row.cols) throw new Error("rowMuli: sizes do not match");
			rowMuli(this, row);
			return this;
		}
		
		public Matrix colMuli(Matrix col) {
			if (this.rows != col.rows*col.cols) throw new Error("colMuli: sizes do not match");
			colMuli(this, col);
			return this;
		}
		
		public Matrix rowDiv(Matrix row) {
			if (this.cols != row.rows*row.cols) throw new Error("rowDiv: sizes do not match");
			Matrix result = new Matrix(this.rows, this.cols);
			rowDiv(this, row, result);
			return result;
		}
		
		public Matrix colDiv(Matrix col) {
			if (this.rows != col.rows*col.cols) throw new Error("colDiv: sizes do not match");
			Matrix result = new Matrix(this.rows, this.cols);
			colDiv(this, col, result);
			return result;
		}
		
		public Matrix rowDivi(Matrix row) {
			if (this.cols != row.rows*row.cols) throw new Error("rowDivi: sizes do not match");
			rowDivi(this, row);
			return this;
		}
		
		public Matrix colDivi(Matrix col) {
			if (this.rows != col.rows*col.cols) throw new Error("colDivi: sizes do not match");
			colDivi(this, col);
			return this;
		}
		
		public Matrix rowAdd(Matrix row) {
			return this.rowComb(1.0f, row);
		}
		
		public Matrix rowAddi(Matrix row) {
			return this.rowCombi(1.0f, row);
		}

		public Matrix rowSub(Matrix row) {
			return this.rowComb(-1.0f, row);
		}
		
		public Matrix rowSubi(Matrix row) {
			return this.rowCombi(-1.0f, row);
		}
		
		public Matrix colAdd(Matrix col) {
			return this.colComb(1.0f, col);
		}
		
		public Matrix colAddi(Matrix col) {
			return this.colCombi(1.0f, col);
		}
		
		public Matrix colSub(Matrix col) {
			return this.colComb(-1.0f, col);
		}
		
		public Matrix colSubi(Matrix col) {
			return this.colCombi(-1.0f, col);
		}
		
		public Matrix rowSum() {
			Matrix ones = Matrix.ones(1, this.rows);
			return ones.mmul(this);
		}
		
		public Matrix colSum() {
			Matrix ones = Matrix.ones(this.cols, 1);
			return this.mmul(ones);
		}
		
		public Matrix sub(Matrix that) {
			return comb(1.0f, -1.0f, that);
		}

		public Matrix subi(Matrix that) {
			return combi(1.0f, -1.0f, that);
		}
		
		public Matrix add(Matrix that) {
			return comb(1.0f, 1.0f, that);
		}

		public Matrix addi(Matrix that) {
			return combi(1.0f, 1.0f, that);
		}
		
		public Matrix rowComb(float alpha, Matrix row) {
			Matrix result = this.copy();
			return result.rowCombi(alpha, row);
		}
		
		public Matrix rowCombi(float alpha, Matrix row) {
			if (this.cols != row.rows*row.cols) throw new Error("rowCombi: sizes do not match");
			Matrix weights = Matrix.ones(this.rows, 1);
			ger(alpha, weights, row, this);
			return this;
		}
		
		public Matrix colComb(float alpha, Matrix row) {
			Matrix result = this.copy();
			return result.colCombi(alpha, row);
		}
		
		public Matrix colCombi(float alpha, Matrix col) {
			if (this.rows != col.rows*col.cols) throw new Error("colCombi: sizes do not match");
			Matrix weights = Matrix.ones(this.cols, 1);
			ger(alpha, col, weights, this);
			return this;
		}
		
		// result = alpha * this + beta * that
		public Matrix comb(float alpha, float beta, Matrix that) {
			if (this.rows*this.cols != that.rows*that.cols) throw new Error("comb: sizes do not match");
			Matrix result = new Matrix(rows, cols);
			vectComb(this, that, result, alpha, beta);
			return result;
		}

		public Matrix combi(float alpha, float beta, Matrix that) {
			if (this.rows*this.cols != that.rows*that.cols) throw new Error("combi: sizes do not match");
			vectCombi(this, that, alpha, beta);
			return this;
		}
		
		public Matrix dot(Matrix that) {
			if (this.rows*this.cols != that.rows*that.cols) throw new Error("dot: sizes do not match");
			Matrix result = new Matrix(1, 1);
			dot(this, that, result);
			return result;
		}
		
		public Matrix mmul(Matrix that) {
			if (this.cols != that.rows) throw new Error("mmul: sizes do not match");
			Matrix result = new Matrix(this.rows, that.cols);
			if (this.rows == 1 && that.cols == 1) {
				dot(this, that, result);
			} else if (this.cols == 1 && that.rows == 1) {
				result.set(0.0f);
				ger(1.0f, this, that, result);
			} else if (this.rows == 1) {
				gemv(1.0f, that, this, 0.0f, result, true);
			} else if (that.cols == 1) {
				gemv(1.0f, this, that, 0.0f, result, false);
			} else {
				gemm(1.0f, this, that, 0.0f, result);
			}
			return result;
		}
		
		public Matrix mmuli(Matrix that) {
			replaceRef(mmul(that), this);
			return this;
		}
		
		public Matrix add(float alpha) {
			Matrix result = new Matrix(rows, cols);
			scalarAdd(this, alpha, result);
			return result;
		}
		
		public Matrix addi(float alpha) {
			scalarAddi(this, alpha);
			return this;
		}
		
		public Matrix log() {
			Matrix result = new Matrix(rows, cols);
			log(this, result);
			return result;
		}
		
		public Matrix logi() {
			logi(this);
			return this;
		}
		
		public Matrix exp() {
			Matrix result = new Matrix(rows, cols);
			exp(this, result);
			return result;
		}
		
		public Matrix expi() {
			expi(this);
			return this;
		}
		
		public Matrix sign() {
			Matrix result = new Matrix(rows, cols);
			sign(this, result);
			return result;
		}
		
		public Matrix signi() {
			signi(this);
			return this;
		}
		
		public Matrix abs() {
			Matrix result = new Matrix(rows, cols);
			abs(this, result);
			return result;
		}
		
		public Matrix absi() {
			absi(this);
			return this;
		}
		
		public Matrix mul(Matrix that) {
			if (this.rows*this.cols != that.rows*that.cols) throw new Error("mul: sizes do not match");
			Matrix result = new Matrix(rows, cols);
			mul(this, that, result);
			return result;
		}
		
		public Matrix muli(Matrix that) {
			if (this.rows*this.cols != that.rows*that.cols) throw new Error("muli: sizes do not match");
			muli(this, that);
			return this;
		}
		
		public Matrix mul(float alpha) {
			Matrix result = new Matrix(rows, cols);
			scalarMul(this, alpha, result);
			return result;
		}
		
		public Matrix muli(float alpha) {
			scalarMuli(this, alpha);
			return this;
		}
		
		public Matrix div(Matrix that) {
			if (this.rows*this.cols != that.rows*that.cols) throw new Error("div: sizes do not match");
			Matrix result = new Matrix(rows, cols);
			div(this, that, result);
			return result;
		}
		
		public Matrix divi(Matrix that) {
			if (this.rows*this.cols != that.rows*that.cols) throw new Error("divi: sizes do not match");
			divi(this, that);
			return this;
		}
		
		public Matrix max(float alpha) {
			Matrix result = new Matrix(rows, cols);
			max(this, result, alpha);
			return result;
		}
		
		public Matrix maxi(float alpha) {
			maxi(this, alpha);
			return this;
		}
		
		public Matrix min(float alpha) {
			Matrix result = new Matrix(rows, cols);
			min(this, result, alpha);
			return result;
		}
		
		public Matrix mini(float alpha) {
			mini(this, alpha);
			return this;
		}
		
		public Matrix pow(float alpha) {
			Matrix result = new Matrix(rows, cols);
			pow(this, result, alpha);
			return result;
		}
		
		public Matrix powi(float alpha) {
			powi(this, alpha);
			return this;
		}
		
		public Matrix sqr() {
			Matrix result = new Matrix(rows, cols);
			sqr(this, result);
			return result;
		}
		
		public Matrix sqri() {
			sqri(this);
			return this;
		}
		
		public Matrix sqrt() {
			Matrix result = new Matrix(rows, cols);
			sqrt(this, result);
			return result;
		}
		
		public Matrix sqrti() {
			sqrti(this);
			return this;
		}
		
		public Matrix transpose() {
			if (isScalar()) {
				return this.copy();
			} else if (isVector()) {
				Matrix result = this.copy();
				int rowsTmp = result.rows;
				result.rows = result.cols;
				result.cols = rowsTmp;
				return result;
			} else {
				Matrix result = new Matrix(cols, rows);
				transpose(this, result);
				return result;
			}
		}
		
		public Matrix transposei() {
			if (isScalar()) {
			} else if (isVector()) {
				int rowsTmp = this.rows;
				this.rows = this.cols;
				this.cols = rowsTmp;
			} else {
				replaceRef(transpose(), this);
			}
			return this;
		}
		
		public Matrix zeroi() {
			scalarSet(this, 0.0f);
			if (DEBUG_SYNC) clFinish(commandQueue);
			return this;
		}
		
		public float norm1() {
			Matrix result = Matrix.zeros(1, 1);
			Matrix scratch = new Matrix(rows*cols*2, 1);
//			CLBLAS.clblasSasum(N, asum, offAsum, X, offx, incx, scratchBuff, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events)
			CLBLAS.clblasSasum(rows*cols, result.data_d, 0, data_d, 0, 1, scratch.data_d, 1, new cl_command_queue[] {commandQueue}, 1, new cl_event[] {lastEvent}, new cl_event[] {lastEvent});
//			JCublas2.cublasSasum(cublasHandle, rows*cols, data_d, 1, Pointer.to(result));
			if (DEBUG_SYNC) clFinish(commandQueue);
			return result.toArray()[0];
		}
		
		public float norm2() {
			Matrix result = Matrix.zeros(1, 1);
			Matrix scratch = new Matrix(rows*cols*2, 1);
//			CLBLAS.clblasSnrm2(N, NRM2, offNRM2, X, offx, incx, scratchBuff, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events)
			CLBLAS.clblasSnrm2(rows*cols, result.data_d, 0, data_d, 0, 1, scratch.data_d, 1, new cl_command_queue[] {commandQueue}, 1, new cl_event[] {lastEvent}, new cl_event[] {lastEvent});
//			JCublas2.cublasSnrm2(cublasHandle, rows*cols, data_d, 1, Pointer.to(result));
			if (DEBUG_SYNC) clFinish(commandQueue);
			return result.toArray()[0];
		}
		
		public float distance1(Matrix that) {
			return comb(1.0f, -1.0f, that).norm1();
		}
		
		public float distance2(Matrix that) {
			return comb(1.0f, -1.0f, that).norm2();
		}
		
		//////////////////////////////////////////

		private static float[] toColMajor(float[][] mat) {
			int rows = mat.length; 
			int cols = mat[0].length;
			float[] data = new float[rows * cols];
			int i=0;
			for (int c=0; c<cols; ++c) {
				for (int r=0; r<rows; ++r) {
					data[i] = mat[r][c];
					i++;
				}
			}
			return data;
		}
		
		private static float[][] fromColMajor(float[] data, int rows) {
			int cols = data.length / rows;
			float[][] mat = new float[rows][cols];
			int i=0;
			for (int c=0; c<cols; ++c) {
				for (int r=0; r<rows; ++r) {
					mat[r][c] = data[i];
					i++;
				}
			}
			return mat;
		}
		
		private static void replaceRef(Matrix A, Matrix B) {
			B.free();
			B.rows = A.rows;
			B.cols = A.cols;
			B.data_d = A.data_d;
		}
		
		// C = alpha * A * B + beta * C
		private static void gemm(float alpha, Matrix A, Matrix B, float beta, Matrix C) {
//			CLBLAS.clblasSgemm(order, transA, transB, M, N, K, alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events)
			CLBLAS.clblasSgemm(clblasOrder.clblasColumnMajor, clblasTranspose.clblasNoTrans, clblasTranspose.clblasNoTrans, C.rows, C.cols, B.rows, alpha, A.data_d, 0, A.rows, B.data_d, 0, B.rows, beta, C.data_d, 0, C.rows, 1, new cl_command_queue[] {commandQueue}, 1, new cl_event[] {lastEvent}, new cl_event[] {lastEvent});
//			JCublas2.cublasSgemm(cublasHandle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, C.rows, C.cols, B.rows, Pointer.to(new float[] {alpha}), A.data_d, A.rows, B.data_d, B.rows, Pointer.to(new float[] {beta}), C.data_d, C.rows);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// y = alpha * A * x + eta * y or y = alpha * A^T * x + eta * y
		private static void gemv(float alpha, Matrix A, Matrix x, float beta, Matrix y, boolean left) {
//			CLBLAS.clblasSgemv(order, transA, M, N, alpha, A, offA, lda, x, offx, incx, beta, y, offy, incy, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events)
			CLBLAS.clblasSgemv(clblasOrder.clblasColumnMajor, (left ? clblasTranspose.clblasTrans : clblasTranspose.clblasNoTrans), A.rows, A.cols, alpha, A.data_d, 0, A.rows, x.data_d, 0, 1, beta, y.data_d, 0, 1, 1, new cl_command_queue[] {commandQueue}, 1, new cl_event[] {lastEvent}, new cl_event[] {lastEvent});
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// d = x^T * y
		private static void dot(Matrix x, Matrix y, Matrix d) {
			Matrix scratch = new Matrix(x.rows*x.cols, 1);
//			CLBLAS.clblasSdot(N, dotProduct, offDP, X, offx, incx, Y, offy, incy, scratchBuff, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events)
			CLBLAS.clblasSdot(x.rows*x.cols, d.data_d, 0, x.data_d, 0, 1, y.data_d, 0, 1, scratch.data_d, 1, new cl_command_queue[] {commandQueue}, 1, new cl_event[] {lastEvent}, new cl_event[] {lastEvent});
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// A = alpha * x * y^T + A
		private static void ger(float alpha, Matrix x, Matrix y, Matrix A) {
//			CLBLAS.clblasSger(order, M, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events)
			CLBLAS.clblasSger(clblasOrder.clblasColumnMajor, A.rows, A.cols, alpha, x.data_d, 0, 1, y.data_d, 0, 1, A.data_d, 0, A.rows, 1, new cl_command_queue[] {commandQueue}, 1, new cl_event[] {lastEvent}, new cl_event[] {lastEvent});
//			JCublas2.cublasSger(cublasHandle, A.rows, A.cols, Pointer.to(new float[] {alpha}), x.data_d, 1, y.data_d, 1, A.data_d, A.rows);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// A = alpha
		private static void scalarSet(Matrix A, float alpha) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorScalarSet");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_float, Pointer.to(new float[] {alpha}));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// B = A + alpha
		private static void scalarAdd(Matrix A, float alpha, Matrix B) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorScalarAdd");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_float, Pointer.to(new float[] {alpha}));
			clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// A = A + alpha
		private static void scalarAddi(Matrix A, float alpha) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorScalarAddi");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_float, Pointer.to(new float[] {alpha}));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// B = A + alpha
		private static void scalarMul(Matrix A, float alpha, Matrix B) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorScalarMul");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_float, Pointer.to(new float[] {alpha}));
			clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// A = A + alpha
		private static void scalarMuli(Matrix A, float alpha) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorScalarMuli");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_float, Pointer.to(new float[] {alpha}));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}

		// B = log(A)
		private static void log(Matrix A, Matrix B) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorLog");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// A = log(A)
		private static void logi(Matrix A) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorLogi");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// B = exp(A)
		private static void exp(Matrix A, Matrix B) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorExp");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// A = exp(A)
		private static void expi(Matrix A) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorExpi");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// B = sign(A)
		private static void sign(Matrix A, Matrix B) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorSign");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// A = sign(A)
		private static void signi(Matrix A) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorSigni");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// B = abs(A)
		private static void abs(Matrix A, Matrix B) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorAbs");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// A = abs(A)
		private static void absi(Matrix A) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorAbsi");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// C = A ./ B
		private static void div(Matrix A, Matrix B, Matrix C) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorDiv");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(C.data_d));
			clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}

		// A = A ./ B
		private static void divi(Matrix A, Matrix B) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorDivi");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// C = A .* B
		private static void mul(Matrix A, Matrix B, Matrix C) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorMul");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(C.data_d));
			clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// A = A .* B
		private static void muli(Matrix A, Matrix B) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorMuli");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// C =  a * A + b * B
		private static void vectComb(Matrix A, Matrix B, Matrix C, float a, float b) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorComb");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(C.data_d));
			clSetKernelArg(kernel, 3, Sizeof.cl_float, Pointer.to(new float[] {a}));
			clSetKernelArg(kernel, 4, Sizeof.cl_float, Pointer.to(new float[] {b}));
			clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// A =  a * A + b * B
		private static void vectCombi(Matrix A, Matrix B, float a, float b) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorCombi");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_float, Pointer.to(new float[] {a}));
			clSetKernelArg(kernel, 3, Sizeof.cl_float, Pointer.to(new float[] {b}));
			clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// B = max(A, val)
		private static void max(Matrix A, Matrix B, float val) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorMax");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_float, Pointer.to(new float[] {val}));
			clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// A = max(A, val)
		private static void maxi(Matrix A, float val) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorMaxi");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_float, Pointer.to(new float[] {val}));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}

		// B = min(A, val)
		private static void min(Matrix A, Matrix B, float val) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorMin");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_float, Pointer.to(new float[] {val}));
			clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// A = min(A, val)
		private static void mini(Matrix A, float val) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorMini");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_float, Pointer.to(new float[] {val}));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// B = pow(A, val)
		private static void pow(Matrix A, Matrix B, float val) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorPow");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_float, Pointer.to(new float[] {val}));
			clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// A = pow(A, val)
		private static void powi(Matrix A, float val) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorPowi");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_float, Pointer.to(new float[] {val}));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// B = sqr(A)
		private static void sqr(Matrix A, Matrix B) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorSqr");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// A = sqr(A)
		private static void sqri(Matrix A) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorSqri");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// B = sqrt(A)
		private static void sqrt(Matrix A, Matrix B) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorSqrt");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// A = sqrt(A)
		private static void sqrti(Matrix A) {
			int n = A.rows*A.cols;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("vectorSqrti");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		// B = tr(A)
		private static void transpose(Matrix A, Matrix B) {
			int gridSizeX = (int) Math.ceil((double) A.rows / (double) TR_BLOCK_SIZE);
			int gridSizeY = (int) Math.ceil((double) A.cols / (double) TR_BLOCK_SIZE);
			cl_kernel kernel = builtKernels.get("transpose");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {0}));
			clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {A.rows}));
			clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[] {A.cols}));
			clSetKernelArg(kernel, 5, (TR_BLOCK_SIZE + 1) * TR_BLOCK_SIZE * Sizeof.cl_float, null);
			clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[] {gridSizeX*TR_BLOCK_SIZE, gridSizeY*TR_BLOCK_SIZE}, new long[] {TR_BLOCK_SIZE, TR_BLOCK_SIZE}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		private static void rowMul(Matrix A, Matrix x, Matrix B) {
			int n = A.rows*A.cols;
			int rows = A.rows;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("rowMul");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(x.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {rows}));
			clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		private static void rowMuli(Matrix A, Matrix x) {
			int n = A.rows*A.cols;
			int rows = A.rows;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("rowMuli");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(x.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {rows}));
			clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		private static void colMul(Matrix A, Matrix x, Matrix B) {
			int n = A.rows*A.cols;
			int rows = A.rows;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("colMul");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(x.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {rows}));
			clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		private static void colMuli(Matrix A, Matrix x) {
			int n = A.rows*A.cols;
			int rows = A.rows;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("colMuli");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(x.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {rows}));
			clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		private static void rowDiv(Matrix A, Matrix x, Matrix B) {
			int n = A.rows*A.cols;
			int rows = A.rows;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("rowDiv");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(x.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {rows}));
			clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		private static void rowDivi(Matrix A, Matrix x) {
			int n = A.rows*A.cols;
			int rows = A.rows;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("rowDivi");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(x.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {rows}));
			clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		private static void colDiv(Matrix A, Matrix x, Matrix B) {
			int n = A.rows*A.cols;
			int rows = A.rows;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("colDiv");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(x.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(B.data_d));
			clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {rows}));
			clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
		private static void colDivi(Matrix A, Matrix x) {
			int n = A.rows*A.cols;
			int rows = A.rows;
			int blockSize = Math.min(n, BLOCK_SIZE);
			int gridSizeX = (int) Math.ceil((double) n / blockSize);
			cl_kernel kernel = builtKernels.get("colDivi");
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(A.data_d));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(x.data_d));
			clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {rows}));
			clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {n}));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[] {gridSizeX*blockSize}, new long[] {blockSize}, 1, new cl_event[] {lastEvent}, lastEvent);
			if (DEBUG_SYNC) clFinish(commandQueue);
		}
		
	}
	
	private static final int BLOCK_SIZE = 64;
	private static final int TR_BLOCK_SIZE = 16;
	
	public static final String[] kernelNames =
			new String[] {"vectorScalarSet",
			"vectorScalarAdd",
			"vectorScalarAddi",
			"vectorScalarMul",
			"vectorScalarMuli",
			"vectorLog",
			"vectorLogi",
			"vectorExp",
			"vectorExpi",
			"vectorSign",
			"vectorSigni",
			"vectorAbs",
			"vectorAbsi",
			"vectorDiv",
			"vectorDivi",
			"vectorMul",
			"vectorMuli",
			"vectorComb",
			"vectorCombi",
			"vectorMax",
			"vectorMaxi",
			"vectorMin",
			"vectorMini",
			"vectorPow",
			"vectorPowi",
			"vectorSqr",
			"vectorSqri",
			"vectorSqrt",
			"vectorSqrti",
			"transpose",
			"rowMul",
			"rowMuli",
			"colMul",
			"colMuli",
			"rowDiv",
			"rowDivi",
			"colDiv",
			"colDivi"}
			;
	
	public static final String kernels =
			
			
			"__kernel void vectorScalarSet(__global float* A, __const float alpha, __const int numElements)\n"+
					"{\n"+
					"    int i = get_global_id(0);\n"+
					"    if (i < numElements)\n"+
					"    {\n"+
					"        A[i] = alpha;\n"+
					"    }\n"+
					"}\n"+
					
			
			"__kernel void vectorScalarAdd(__global float const* __restrict__ A, __global float* B, __const float alpha, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        B[i] = A[i] + alpha;\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorScalarAddi(__global float* A, __const float alpha, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = A[i] + alpha;\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorScalarMul(__global float const* __restrict__ A, __global float* B, __const float alpha, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        B[i] = A[i] * alpha;\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorScalarMuli(__global float* A, __const float alpha, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = A[i] * alpha;\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorLog(__global float const* __restrict__ A, __global float* B, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        B[i] = log(A[i]);\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorLogi(__global float* A, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = log(A[i]);\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorExp(__global float const* __restrict__ A, __global float* B, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        B[i] = exp(A[i]);\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorExpi(__global float* A, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = exp(A[i]);\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorSign(__global float const* __restrict__ A, __global float* B, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        B[i] = (A[i] > 0.0 ? 1.0 : -1.0);\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorSigni(__global float* A, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = (A[i] > 0.0 ? 1.0 : -1.0);\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorAbs(__global float const* __restrict__ A, __global float* B, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        B[i] = fabs(A[i]);\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorAbsi(__global float* A, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = fabs(A[i]);\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorDiv(__global float const* __restrict__ A, __global float const* __restrict__ B, __global float* C, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        C[i] = A[i] / B[i];\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorDivi(__global float* A, __global float const* __restrict__ B, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = A[i] / B[i];\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorMul(__global float const* __restrict__ A, __global float const* __restrict__ B, __global float* C, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        C[i] = A[i] * B[i];\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorMuli(__global float* A, __global float const* __restrict__ B, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = A[i] * B[i];\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorComb(__global float const* __restrict__ A, __global float const* __restrict__ B, __global float* C, __const float a, __const float b, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        C[i] = a * A[i] + b * B[i];\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorCombi(__global float* A, __global float const* __restrict__ B, __const float a, __const float b, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = a * A[i] + b * B[i];\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorMax(__global float const* __restrict__ A, __global float* B, __const float val, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        B[i] = fmax(A[i], val);\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorMaxi(__global float* A, __const float val, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = fmax(A[i], val);\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorMin(__global float const* __restrict__ A, __global float* B, __const float val, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        B[i] = fmin(A[i], val);\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorMini(__global float* A, __const float val, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = fmin(A[i], val);\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorPow(__global float const* __restrict__ A, __global float* B, __const float val, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        B[i] = pow(A[i], val);\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorPowi(__global float* A, __const float val, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = pow(A[i], val);\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorSqr(__global float const* __restrict__ A, __global float* B, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    float val;\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        val = A[i];\n"+
			"        B[i] = val*val;\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorSqri(__global float* A, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    float val;\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        val = A[i];\n"+
			"        A[i] = val*val;\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorSqrt(__global float const* __restrict__ A, __global float* B, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        B[i] = sqrt(A[i]);\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void vectorSqrti(__global float* A, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = sqrt(A[i]);\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void transpose(__global float *odata, __global float *idata, int offset, int width, int height, __local float* block) {\n"+
			"	unsigned int xIndex = get_global_id(0);\n"+
			"	unsigned int yIndex = get_global_id(1);\n"+
			"	if((xIndex + offset < width) && (yIndex < height)){\n"+
			"		unsigned int index_in = yIndex * width + xIndex + offset;\n"+
			"		block[get_local_id(1)*("+TR_BLOCK_SIZE+"+1)+get_local_id(0)] = idata[index_in];\n"+
			"	}\n"+
			"	barrier(CLK_LOCAL_MEM_FENCE);\n"+
			"	xIndex = get_group_id(1) * "+TR_BLOCK_SIZE+" + get_local_id(0);\n"+
			"	yIndex = get_group_id(0) * "+TR_BLOCK_SIZE+" + get_local_id(1);\n"+
			"	if((xIndex < height) && (yIndex + offset < width)) {\n"+
			"		unsigned int index_out = yIndex * height + xIndex;\n"+
			"		odata[index_out] = block[get_local_id(0)*("+TR_BLOCK_SIZE+"+1)+get_local_id(1)];\n"+
			"	}\n"+
			"}\n"+
			
			"__kernel void colMul(__global float const* __restrict__ A, __global float const* __restrict__ x, __global float* B, __const int numRows, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        B[i] = A[i] * x[i % numRows];\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void colMuli(__global float* A, __global float const* __restrict__ x, __const int numRows, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = A[i] * x[i % numRows];\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void rowMul(__global float const* __restrict__ A, __global float const* __restrict__ x, __global float* B, __const int numRows, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        B[i] = A[i] * x[i / numRows];\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void rowMuli(__global float* A, __global float const* __restrict__ x, __const int numRows, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = A[i] * x[i / numRows];\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void colDiv(__global float const* __restrict__ A, __global float const* __restrict__ x, __global float* B, __const int numRows, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        B[i] = A[i] / x[i % numRows];\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void colDivi(__global float* A, __global float const* __restrict__ x, __const int numRows, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = A[i] / x[i % numRows];\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void rowDiv(__global float const* __restrict__ A, __global float const* __restrict__ x, __global float* B, __const int numRows, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        B[i] = A[i] / x[i / numRows];\n"+
			"    }\n"+
			"}\n"+
			
			"__kernel void rowDivi(__global float* A, __global float const* __restrict__ x, __const int numRows, __const int numElements)\n"+
			"{\n"+
			"    int i = get_global_id(0);\n"+
			"    if (i < numElements)\n"+
			"    {\n"+
			"        A[i] = A[i] / x[i / numRows];\n"+
			"    }\n"+
			"}\n"
			
			;
	
	public static void main(String[] args) {
		
		JOCLBlasUtil.startup(1);
		
	    Random rand = new Random(1);
		float[][] Aarray = a.randFloat(2, 3, rand);
		float[][] Barray = a.randFloat(2, 3, rand);
		
		
	    {
	    	System.out.println("\n\nCPU");
	    	FloatMatrix A = new FloatMatrix(Aarray);
	    	System.out.println(a.toString(A.transpose().toArray2()));
	    	FloatMatrix B = new FloatMatrix(Barray);
	    	FloatMatrix C = A.add(B.mul(-2.0f));
	    	System.out.println(a.toString(C.toArray2()));
	    }
		
	    {
	    	System.out.println("\n\nGPU");
	    	Matrix A = Matrix.build(Aarray);
	    	System.out.println(a.toString(A.transpose().toArray2()));
	    	Matrix B = Matrix.build(Barray);
	    	Matrix C = A.comb(1.0f, -2.0f, B);
	    	System.out.println(a.toString(C.toArray2()));
	    	A.free();
	    	B.free();
	    	C.free();
	    }
	    
	    Aarray = new float[][] {{1, 2, 3}, {-1, -2.5f, 1.0f}};
	    Barray = new float[][] {{4, 0, -1}, {0, -2.5f, 1}, {9, -10, -0.5f}};
	    float[][] Carray = new float[][] {{1, 2, 3}, {1, 2, 3}};
	    
	    {
	    	System.out.println("\n\nCPU");
	    	FloatMatrix A = new FloatMatrix(Aarray);
	    	FloatMatrix B = new FloatMatrix(Barray);
	    	FloatMatrix C = new FloatMatrix(Carray);
	    	FloatMatrix D = A.mmul(B.transpose()).add(FloatMatrix.ones(2, 3)).muli(2.0f).mul(C);
	    	D.maxi(-68.0f);
	    	System.out.println(a.toString(D.toArray2()));
	    	System.out.println(D.norm1());
	    	System.out.println(D.norm2());
	    	FloatMatrix E = D.div(A);
	    	System.out.println(a.toString(E.toArray2()));
	    	System.out.println(E.norm1());
	    	System.out.println(E.norm2());
	    }
	    
	    {
	    	System.out.println("\n\nGPU");
	    	Matrix A = Matrix.build(Aarray);
	    	Matrix B = Matrix.build(Barray);
	    	Matrix C = Matrix.build(Carray);
	    	Matrix D = (A.mmul(B.transpose()).add(Matrix.ones(2, 3)).muli(2.0f)).mul(C);
	    	D.maxi(-68.0f);
	    	System.out.println(a.toString(D.toArray2()));
	    	System.out.println(D.norm1());
	    	System.out.println(D.norm2());
	    	Matrix E = D.div(A);
	    	System.out.println(a.toString(E.toArray2()));
	    	System.out.println(E.norm1());
	    	System.out.println(E.norm2());
	    	A.free();
	    	B.free();
	    	C.free();
	    	D.free();
	    }
	    
	    {
	    	System.out.println("\n\n"+a.toString(Matrix.ones(300, 3).toArray2()));
	    	System.out.println("\n\n"+a.toString(Matrix.ones(300, 3).toArray2()));
	    	System.out.println("\n\n"+a.toString(Matrix.ones(300, 3).toArray2()));
	    }
		
	    // Misc tests
	    ////////////////////////////////////////////////
	    
	    {
	    	Matrix A = Matrix.ones(3, 3);
	    	A.muli(4.0f);
	    	System.out.println(a.toString(A.toArray2()));
	    	System.out.println(a.toString(A.sqrt().toArray2()));
	    	System.out.println(a.toString(A.toArray2()));
	    	A.sqrti();
	    	System.out.println(a.toString(A.toArray2()));
	    	A.free();
	    }	   
	    
	    {
	    	System.out.println("\n\nCPU");
	    	FloatMatrix A = new FloatMatrix(Aarray);
	    	System.out.println(a.toString(A.rowSums().toArray2()));
	    	System.out.println(a.toString(A.columnSums().toArray2()));
	    }
		
	    {
	    	System.out.println("\n\nGPU");
	    	Matrix A = Matrix.build(Aarray);
	    	System.out.println(a.toString(A.colSum().toArray2()));
	    	System.out.println(a.toString(A.rowSum().toArray2()));
	    	A.free();
	    }
	    
	    {
	    	System.out.println("\n\nCPU");
	    	FloatMatrix A = new FloatMatrix(Aarray);
	    	FloatMatrix col = FloatMatrix.ones(2, 1);
	    	col.put(0, 0, 2.0f);
	    	FloatMatrix row = FloatMatrix.ones(1, 3);
	    	row.put(0, 0, 2.0f);
	    	System.out.println(a.toString(A.addColumnVector(col).toArray2()));
	    	System.out.println(a.toString(A.addRowVector(row).toArray2()));
	    }
		
	    {
	    	System.out.println("\n\nGPU");
	    	Matrix A = Matrix.build(Aarray);
	    	float[][] col = a.onesFloat(2, 1);
	    	col[0][0] = 2.0f;
	    	float[][] row = a.onesFloat(1, 3);
	    	row[0][0] = 2.0f;
	    	System.out.println(a.toString(A.colAdd(Matrix.build(col)).toArray2()));
	    	System.out.println(a.toString(A.rowAdd(Matrix.build(row)).toArray2()));
	    	A.free();
	    }
	    
	    {
	    	System.out.println("\n\nGPU");
	    	Matrix A = Matrix.rand(5, 7, rand);
	    	System.out.println(a.toString(A.toArray2()));
	    	System.out.println(a.toString(A.copySubmatrix(1, 3, 2, 4).toArray2()));
	    	A.setSubmatrix(Matrix.ones(2,2), 1, 2);
	    	System.out.println(a.toString(A.toArray2()));
	    	A.setSubmatrix(a.onesFloat(2,2), 1, 0);
	    	System.out.println(a.toString(A.toArray2()));
	    	A.set(4, 3, 5.0f);
	    	System.out.println(a.toString(A.toArray2()));
	    	Matrix B = Matrix.rand(2, 3, rand);
	    	System.out.println(a.toString(B.toArray2()));
	    	A.setSubmatrix(1, 1, B, 1, 2, 1, 3);
	    	System.out.println(a.toString(A.toArray2()));
	    	A.free();
	    }
	    
	    {
	    	System.out.println("\n\nCPU");
	    	FloatMatrix A = new FloatMatrix(Aarray);
	    	FloatMatrix col = FloatMatrix.ones(2, 1);
	    	col.put(0, 0, 2.0f);
	    	FloatMatrix row = FloatMatrix.ones(1, 3);
	    	row.put(0, 0, 2.0f);
	    	System.out.println(a.toString(A.toArray2()));
	    	System.out.println(a.toString(A.mulColumnVector(col).toArray2()));
	    	System.out.println(a.toString(A.mulRowVector(row).toArray2()));
	    	System.out.println(a.toString(A.divColumnVector(col).toArray2()));
	    	System.out.println(a.toString(A.divRowVector(row).toArray2()));
	    }
		
	    {
	    	System.out.println("\n\nGPU");
	    	Matrix A = Matrix.build(Aarray);
	    	float[][] col = a.onesFloat(2, 1);
	    	col[0][0] = 2.0f;
	    	float[][] row = a.onesFloat(1, 3);
	    	row[0][0] = 2.0f;
	    	System.out.println(a.toString(A.toArray2()));
	    	System.out.println(a.toString(A.colMul(Matrix.build(col)).toArray2()));
	    	System.out.println(a.toString(A.rowMul(Matrix.build(row)).toArray2()));
	    	System.out.println(a.toString(A.colDiv(Matrix.build(col)).toArray2()));
	    	System.out.println(a.toString(A.rowDiv(Matrix.build(row)).toArray2()));
	    	A.rowDivi(Matrix.build(row));
	    	System.out.println(a.toString(A.toArray2()));
	    	
	    	A.free();
	    }
	    
	    {
	    	System.out.println("\n\nGPU");
	    	Matrix A = Matrix.ones(5, 5);
	    	System.out.println(a.toString(A.toArray2()));
	    	System.out.println(a.toString(A.diagAdd(10.0f).toArray2()));
	    	A.diagAddi(1.0f);
	    	System.out.println(a.toString(A.toArray2()));
	    	
	    	A.free();
	    }
	    
	    {
	    	System.out.println("\n\nGPU");
	    	Matrix A = Matrix.ones(5, 5);
	    	A.muli(-0.2f);
	    	System.out.println(a.toString(A.toArray2()));
	    	A.powi(1.0f);
	    	System.out.println(a.toString(A.toArray2()));
	    	
	    	A.free();
	    }
	    
		float[][] x = a.randFloat(3, 2, rand);
		System.out.println(a.toString(x));
		Matrix X = Matrix.build(x);
		X.colMuli(Matrix.build(3, 1, new float[] {1.0f, 2.0f, 3.0f}));
		System.out.println(a.toString(X.toArray2()));
		
		System.out.println(a.toString(Matrix.ones(2, 1).mmul(Matrix.ones(1, 3)).toArray2()));
	    
	    JOCLBlasUtil.shutdown();
	}

}
