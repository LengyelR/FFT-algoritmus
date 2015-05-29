#include <stdio.h>	
#include <stdlib.h>	
#include <math.h>	
#include <time.h>

#ifdef __APPLE__	
#include <OpenCL/opencl.h>	
#else	
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS 
#include <CL/cl.h>	
#endif	

#include "pgm.h"	

#define PI 3.14159265358979	

#define MAX_SOURCE_SIZE (0x100000)	

#define AMP(a, b) (sqrt((a)*(a)+(b)*(b)))	

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue cmdQueue = NULL;
cl_program program = NULL;

enum Mode {
	forward = 0,
	inverse = 1
};

int setWorkSize(size_t* gws, size_t* lws, cl_int x, cl_int y)
{
	switch (y) {
	case 1:
		gws[0] = x;
		gws[1] = 1;
		lws[0] = 1;
		lws[1] = 1;
		break;
	default:
		gws[0] = x;
		gws[1] = y;
		lws[0] = 1;
		lws[1] = 1;
		break;
	}

	return 0;
}

int fftCore(cl_mem dst, cl_mem src, cl_mem spin, cl_int m, enum Mode direction)
{
	cl_int ret;

	cl_int iter;
	cl_uint flag;

	cl_int n = 1 << m;

	cl_event kernelDone;

	cl_kernel brev = NULL;
	cl_kernel bfly = NULL;
	cl_kernel norm = NULL;

	brev = clCreateKernel(program, "bitReverse", &ret);
	bfly = clCreateKernel(program, "butterfly", &ret);
	norm = clCreateKernel(program, "norm", &ret);

	size_t gws[2];
	size_t lws[2];

	switch (direction) {
	case forward:flag = 0x00000000; break;
	case inverse:flag = 0x80000000; break;
	}

	ret = clSetKernelArg(brev, 0, sizeof(cl_mem), (void *)&dst);
	ret = clSetKernelArg(brev, 1, sizeof(cl_mem), (void *)&src);
	ret = clSetKernelArg(brev, 2, sizeof(cl_int), (void *)&m);
	ret = clSetKernelArg(brev, 3, sizeof(cl_int), (void *)&n);

	ret = clSetKernelArg(bfly, 0, sizeof(cl_mem), (void *)&dst);
	ret = clSetKernelArg(bfly, 1, sizeof(cl_mem), (void *)&spin);
	ret = clSetKernelArg(bfly, 2, sizeof(cl_int), (void *)&m);
	ret = clSetKernelArg(bfly, 3, sizeof(cl_int), (void *)&n);
	ret = clSetKernelArg(bfly, 5, sizeof(cl_uint), (void *)&flag);

	ret = clSetKernelArg(norm, 0, sizeof(cl_mem), (void *)&dst);
	ret = clSetKernelArg(norm, 1, sizeof(cl_int), (void *)&n);
	
	/* Bitreverzió */
	setWorkSize(gws, lws, n, n);
	ret = clEnqueueNDRangeKernel(cmdQueue, brev, 2, NULL, gws, lws, 0, NULL, NULL);

	/* A pillangó műveletek, fft*/
	setWorkSize(gws, lws, n / 2, n);
	for (iter = 1; iter <= m; iter++){
		ret = clSetKernelArg(bfly, 4, sizeof(cl_int), (void *)&iter);
		ret = clEnqueueNDRangeKernel(cmdQueue, bfly, 2, NULL, gws, lws, 0, NULL, &kernelDone);
		ret = clWaitForEvents(1, &kernelDone);
	}

	if (direction == inverse) {
		setWorkSize(gws, lws, n, n);
		ret = clEnqueueNDRangeKernel(cmdQueue, norm, 2, NULL, gws, lws, 0, NULL, &kernelDone);
		ret = clWaitForEvents(1, &kernelDone);
	}

	ret = clReleaseKernel(bfly);
	ret = clReleaseKernel(brev);
	ret = clReleaseKernel(norm);

	return 0;
}

int main()
{
	cl_mem xmobj = NULL;
	cl_mem rmobj = NULL;
	cl_mem wmobj = NULL;
	cl_kernel sfac = NULL;
	cl_kernel trns = NULL;
	cl_kernel hpfl = NULL;

	cl_int ret;

	cl_float2 *xm;
	cl_float2 *rm;
	cl_float2 *wm;

	pgm_t ipgm;
	pgm_t opgm;

	FILE *fp;
	errno_t err;
	const char fileName[] = "kernel.cl";
	size_t source_size;
	char *source_str;
	cl_int i, j;
	cl_int n;
	cl_int m;

	size_t gws[2];
	size_t lws[2];
	
	//Inforámciós változók (device)
	char *vendor;					//CL_DEVICE_VENDOR
	char *device_name;			 //CL_DEVICE_NAME
	char *open_cl_c_version;			//CL_DEVICE_OPENCL_C_VERSION  ---- openCL C
	char *open_cl_version;			// CL_DEVICE_VERSION --- openCL
	size_t *max_workgroup;		// CL_DEVICE_MAX_WORK_GROUP_SIZE
	
	//Információs változók (platform)
	char *profile = NULL;
	char *platform_version = NULL;

	/* Kernel beolvasása*/
	err = fopen_s(&fp, fileName, "r");
	if (err) {
		printf("HIBA: Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char *)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	/* Kép beolvasása */
	readPGM(&ipgm, "lena.pgm");

	n = ipgm.width;
	m = (cl_int)(log((double)n) / log(2.0));

	xm = (cl_float2 *)malloc(n * n * sizeof(cl_float2));
	rm = (cl_float2 *)malloc(n * n * sizeof(cl_float2));
	wm = (cl_float2 *)malloc(n / 2 * sizeof(cl_float2));

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			((float*)xm)[(2 * n*j) + 2 * i + 0] = (float)ipgm.buf[n*j + i];
			((float*)xm)[(2 * n*j) + 2 * i + 1] = (float)0;
		}
	}



	/* Device és Platform beállítása*/
	cl_int status;
	cl_uint numPlatforms;
	cl_uint numDevices;

	size_t size;

	cl_platform_id *platforms;
	cl_device_id *devices;


	clGetPlatformIDs(0, NULL, &numPlatforms);
	printf("Number of Platforms: %d\n\n", numPlatforms);

	platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
	clGetPlatformIDs(numPlatforms, platforms, NULL);

	clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);

	devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
	clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	cmdQueue = clCreateCommandQueue(context, devices[1], 0, &status);


	/*Platform információk*/
	clGetPlatformInfo(platforms[0], CL_PLATFORM_PROFILE, NULL, profile, &size);

	profile = (char*)malloc(size);
	clGetPlatformInfo(platforms[0], CL_PLATFORM_PROFILE, size, profile, NULL);

	clGetPlatformInfo(platforms[0], CL_PLATFORM_VERSION, NULL, platform_version, &size);

	platform_version = (char*)malloc(size);
	clGetPlatformInfo(platforms[0], CL_PLATFORM_VERSION, size, platform_version, NULL);

	printf("Platform Informations:\n");
	printf("(Status now: %d)\n", status);
	printf("Profle: %s\n", profile);
	printf("Platform Version: %s\n", platform_version);

	/*Device információk*/

	clGetDeviceInfo(devices[0], CL_DEVICE_VENDOR, NULL, NULL, &size);
	vendor = (char*)malloc(sizeof(char)*size);
	clGetDeviceInfo(devices[0], CL_DEVICE_VENDOR, size, vendor, NULL);

	clGetDeviceInfo(devices[0], CL_DEVICE_NAME, NULL, NULL, &size);
	device_name = (char*)malloc(sizeof(char)*size);
	clGetDeviceInfo(devices[0], CL_DEVICE_NAME, size, device_name, NULL);

	clGetDeviceInfo(devices[0], CL_DEVICE_OPENCL_C_VERSION, NULL, NULL, &size);
	open_cl_c_version = (char*)malloc(sizeof(char)*size);
	clGetDeviceInfo(devices[0], CL_DEVICE_OPENCL_C_VERSION, size, open_cl_c_version, NULL);

	clGetDeviceInfo(devices[0], CL_DEVICE_VERSION, NULL, NULL, &size);
	open_cl_version = (char*)malloc(sizeof(char)*size);
	clGetDeviceInfo(devices[0], CL_DEVICE_VERSION, size, open_cl_version, NULL);

	clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, NULL, NULL, &size);
	max_workgroup = (size_t*)malloc(sizeof(size_t)*size);
	clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, size, &max_workgroup, NULL);

	clGetDeviceInfo(devices[0], CL_DEVICE_MAX_COMPUTE_UNITS, NULL, NULL, &size);
	max_cu = (cl_uint*)malloc(sizeof(cl_uint)*size);
	clGetDeviceInfo(devices[0], CL_DEVICE_MAX_COMPUTE_UNITS, size, &max_cu, NULL);

	printf("\n\nDevice Informations:\n");
	printf("Vendor: %s\n", vendor);
	printf("Device: %s\n", device_name);
	printf("openCL C: %s\n", open_cl_c_version);
	printf("openCL: %s\n", open_cl_version);
	printf("Max WorkGroup: %d\n", max_workgroup);
	printf("Max ComputeUnits: : %d\n\n", max_cu);


	/* Bufferek */
	xmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float2), NULL, &ret);
	rmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float2), NULL, &ret);
	wmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, (n / 2)*sizeof(cl_float2), NULL, &ret);

	/*Az adatok bufferbe töltése*/
	ret = clEnqueueWriteBuffer(cmdQueue, xmobj, CL_TRUE, 0, n*n*sizeof(cl_float2), xm, 0, NULL, NULL);

	/* Kernel elkészítése*/
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

	/* Kernel buildelése */
	clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

	/* OpenCL Kernel elkészítése */
	sfac = clCreateKernel(program, "spinFact", &ret);
	trns = clCreateKernel(program, "transpose", &ret);
	hpfl = clCreateKernel(program, "highPassFilter", &ret);

	/* Twiddle factor elkészítése */
	ret = clSetKernelArg(sfac, 0, sizeof(cl_mem), (void *)&wmobj);
	ret = clSetKernelArg(sfac, 1, sizeof(cl_int), (void *)&n);
	setWorkSize(gws, lws, n / 2, 1);
	ret = clEnqueueNDRangeKernel(cmdQueue, sfac, 1, NULL, gws, lws, 0, NULL, NULL);

	//tic 
	clock_t t;

	t = clock();
	/* fft */
	fftCore(rmobj, xmobj, wmobj, m, forward);

	/* Transzponálás */
	ret = clSetKernelArg(trns, 0, sizeof(cl_mem), (void *)&xmobj);
	ret = clSetKernelArg(trns, 1, sizeof(cl_mem), (void *)&rmobj);
	ret = clSetKernelArg(trns, 2, sizeof(cl_int), (void *)&n);
	setWorkSize(gws, lws, n, n);
	ret = clEnqueueNDRangeKernel(cmdQueue, trns, 2, NULL, gws, lws, 0, NULL, NULL);

	/* fft */
	fftCore(rmobj, xmobj, wmobj, m, forward);

	t = clock() - t;
	//toc
	double time_taken = ((double)t) / CLOCKS_PER_SEC; 
	printf("Elapsed time: %f\n\n", time_taken);

	/*  high-pass filter */
	cl_int radius = n / 128;
	ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&rmobj);
	ret = clSetKernelArg(hpfl, 1, sizeof(cl_int), (void *)&n);
	ret = clSetKernelArg(hpfl, 2, sizeof(cl_int), (void *)&radius);
	setWorkSize(gws, lws, n, n);
	ret = clEnqueueNDRangeKernel(cmdQueue, hpfl, 2, NULL, gws, lws, 0, NULL, NULL);

	/* inverz fft */

	/* ifft */
	fftCore(xmobj, rmobj, wmobj, m, inverse);

	/* Transzponálás */
	ret = clSetKernelArg(trns, 0, sizeof(cl_mem), (void *)&rmobj);
	ret = clSetKernelArg(trns, 1, sizeof(cl_mem), (void *)&xmobj);
	setWorkSize(gws, lws, n, n);
	ret = clEnqueueNDRangeKernel(cmdQueue, trns, 2, NULL, gws, lws, 0, NULL, NULL);

	/* ifft */
	fftCore(xmobj, rmobj, wmobj, m, inverse);

	/* Adatok olvasása a bufferből*/
	ret = clEnqueueReadBuffer(cmdQueue, xmobj, CL_TRUE, 0, n*n*sizeof(cl_float2), xm, 0, NULL, NULL);

	float* ampd;
	ampd = (float*)malloc(n*n*sizeof(float));
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			ampd[n*((i)) + ((j))] = (AMP(((float*)xm)[(2 * n*i) + 2 * j], ((float*)xm)[(2 * n*i) + 2 * j + 1]));
		}
	}
	opgm.width = n;
	opgm.height = n;
	normalizeF2PGM(&opgm, ampd);
	free(ampd);

	/* OUTPUT kép */
	writePGM(&opgm, "output.pgm");

	/* destruktor feladatai*/
	ret = clFlush(cmdQueue);
	ret = clFinish(cmdQueue);
	ret = clReleaseKernel(hpfl);
	ret = clReleaseKernel(trns);
	ret = clReleaseKernel(sfac);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(xmobj);
	ret = clReleaseMemObject(rmobj);
	ret = clReleaseMemObject(wmobj);
	ret = clReleaseCommandQueue(cmdQueue);
	ret = clReleaseContext(context);

	destroyPGM(&ipgm);
	destroyPGM(&opgm);

	free(source_str);
	free(wm);
	free(rm);
	free(xm);

	return 0;
}
