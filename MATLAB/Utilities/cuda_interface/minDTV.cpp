/*
/*-------------------------------------------------------------------------
 *
 * MATLAB MEX gateway for Total variation minimization via Steepest descend
 *
 * This file gets the data from MATLAB, checks it for errors and then 
 * parses it to C and calls the relevant C/CUDA fucntions.
 *
 * CODE by       Alexey Smirnov and Ander Biguri
 *
---------------------------------------------------------------------------
---------------------------------------------------------------------------
--------------------------------------------------------------------------- 
 */





#include <math.h>
#include <string.h>
#include <tmwtypes.h>
#include <mex.h>
#include <matrix.h>
#include <CUDA/POCS_TV2.hpp>
#include <CUDA/GpuIds.hpp>
#include <CUDA/gpuUtils.hpp>
void mexFunction(int  nlhs , mxArray *plhs[],
        int nrhs, mxArray const *prhs[])
{
///////// First check if the amount of inputs is right.    
    int maxIter;
    int APGMiter;
    float mu;
    float alpha;
    GpuIds gpuids;
    if (nrhs==5) {
        size_t iM = mxGetM(prhs[4]);
        if (iM != 1) {
            mexErrMsgIdAndTxt( "CBCT:MEX:Ax:unknown","4th parameter must be a row vector.");
            return;
        }
        size_t uiGpuCount = mxGetN(prhs[4]);
        if (uiGpuCount == 0) {
            mexErrMsgIdAndTxt( "CBCT:MEX:Ax:unknown","4th parameter must be a row vector.");
            return;
        }
        int* piGpuIds = (int*)mxGetData(prhs[4]);
        gpuids.SetIds(uiGpuCount, piGpuIds);
    } else {
        int iGpuCount = GetGpuCount();
        int* piDev = (int*)malloc(iGpuCount * sizeof(int));
        for (int iI = 0; iI < iGpuCount; ++iI) {
            piDev[iI] = iI;
        }
        gpuids.SetIds(iGpuCount, piDev);
        free(piDev); piDev = 0;
    }    
    if (nrhs==1){
        maxIter=100;
        alpha=15.0f;
    } else if (nrhs==2){
       mexErrMsgIdAndTxt("err", "Only 1 POCS hyperparemter inputed");
    } else if (nrhs==4 || nrhs==5){
        size_t mrows = mxGetM(prhs[1]);
        size_t ncols = mxGetN(prhs[1]);
        if (mrows!=1 || ncols !=1) {
            mexErrMsgIdAndTxt("err", "POCS parameters should be 1x1");
        }
        mrows = mxGetM(prhs[2]);
        ncols = mxGetN(prhs[2]);
        if (mrows!=1 || ncols !=1) {
            mexErrMsgIdAndTxt("err", "POCS parameters should be 1x1");
        }
        alpha= (float)(mxGetScalar(prhs[1]));
        maxIter=(int)floor(mxGetScalar(prhs[2])+0.5);
    } else {
       mexErrMsgIdAndTxt("err", "Too many imput argumets");
    }
    float delta=(float)(mxGetScalar(prhs[3]));
    mu=1.01;
    APGMiter = 10;

////////////////////////// First input.
    // First input should be x from (Ax=b), or the image.
    mxArray const * const image = prhs[0];
    mwSize const numDims = mxGetNumberOfDimensions(image);
    mwSize third_dim = 1;
    
    // Now that input is ok, parse it to C data types.
    float  *  img = static_cast<float  *>(mxGetData(image));
    const mwSize *size_img= mxGetDimensions(image); //get size of image

    // Image should be dim 3
    if (numDims==3){
        third_dim = size_img[2];
    }
    
    // Allocte output image
    plhs[0] = mxCreateNumericArray(numDims, size_img, mxSINGLE_CLASS, mxREAL);
    float *imgout =(float*) mxGetPr(plhs[0]);
    // call C function with the CUDA denoising
  
    const long imageSize[3]={size_img[0], size_img[1], third_dim };
    
    dtv_pocs(img,imgout, alpha, imageSize, maxIter, APGMiter, mu, delta, gpuids); 
    
    //prepareotputs
}
