/*-------------------------------------------------------------------------
 *
 * CUDA functions for Steepest descend in POCS-type algorithms.
 *
 * This file will iteratively minimize by stepest descend the total variation
 * of the input image, with the parameters given, using GPUs.
 *
 * CODE by       Ander Biguri
 *
 * ---------------------------------------------------------------------------
 * ---------------------------------------------------------------------------
 * Copyright (c) 2015, University of Bath and CERN- European Organization for
 * Nuclear Research
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ---------------------------------------------------------------------------
 *
 * Contact: tigre.toolbox@gmail.com
 * Codes  : https://github.com/CERN/TIGRE
 * ---------------------------------------------------------------------------
 */







#define MAXTHREADS 1024
#define MAX_BUFFER 60

#include "POCS_TV.hpp"




#define cudaCheckErrors(msg) \
do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
                mexPrintf("%s \n",msg);\
                cudaDeviceReset();\
                mexErrMsgIdAndTxt("CBCT:CUDA:POCS_TV",cudaGetErrorString(__err));\
        } \
} while (0)
    
// CUDA kernels
//https://stackoverflow.com/questions/21332040/simple-cuda-kernel-optimization/21340927#21340927
    __global__ void divideArrayScalar(float* vec,float scalar,const size_t n){
        unsigned long long i = (blockIdx.x * blockDim.x) + threadIdx.x;
        for(; i<n; i+=gridDim.x*blockDim.x) {
            vec[i]/=scalar;
        }
    }
    __global__ void multiplyArrayScalar(float* vec,float scalar,const size_t n)
    {
        unsigned long long i = (blockIdx.x * blockDim.x) + threadIdx.x;
        for(; i<n; i+=gridDim.x*blockDim.x) {
            vec[i]*=scalar;
        }
    }
    __global__ void substractArrays(float* vec,float* vec2,const size_t n)
    {
        unsigned long long i = (blockIdx.x * blockDim.x) + threadIdx.x;
        for(; i<n; i+=gridDim.x*blockDim.x) {
            vec[i]-=vec2[i];
        }
    }
    
    __device__ __inline__
            void gradient(const float* u, float* grad,
            long z, long y, long x,
            long depth, long rows, long cols){
        unsigned long size2d = rows*cols;
        unsigned long long idx = z * size2d + y * cols + x;
        
        float uidx = u[idx];
        
        if ( z - 1 >= 0 && z<depth) {
            grad[0] = (uidx-u[(z-1)*size2d + y*cols + x]) ;
        }
        
        if ( y - 1 >= 0 && y<rows){
            grad[1] = (uidx-u[z*size2d + (y-1)*cols + x]) ;
        }
        
        if ( x - 1 >= 0 && x<cols) {
            grad[2] = (uidx-u[z*size2d + y*cols + (x-1)]);
        }
    }
    __device__ __inline__
            void prox_mu_sigma_g(float* u, const float* Lop, const float sgm, const float mu){
        
        float t[6] = {0.f,0.f,0.f,0.f,0.f,0.f};
        float musgm = mu*sgm;

        t[0] = u[0]-sgm*Lop[0]-(u[0]-sgm*Lop[0])/fmaxf(musgm*sqrtf(u[0]*u[0]+u[1]*u[1]),1);
        t[1] = u[1]-sgm*Lop[1]-(u[1]-sgm*Lop[1])/fmaxf(musgm*sqrtf(u[0]*u[0]+u[1]*u[1]),1);
        t[2] = u[2]-sgm*Lop[2]-(u[2]-sgm*Lop[2])/fmaxf(musgm*sqrtf(u[2]*u[2]+u[3]*u[3]),1);
        t[3] = u[3]-sgm*Lop[3]-(u[3]-sgm*Lop[3])/fmaxf(musgm*sqrtf(u[2]*u[2]+u[3]*u[3]),1);
        t[4] = u[4]-sgm*Lop[4]-(u[4]-sgm*Lop[4])/fmaxf(musgm*sqrtf(u[4]*u[4]+u[5]*u[5]),1);
        t[5] = u[5]-sgm*Lop[5]-(u[5]-sgm*Lop[5])/fmaxf(musgm*sqrtf(u[4]*u[4]+u[5]*u[5]),1);

        u[0] = t[0];
        u[1] = t[1];
        u[2] = t[2];
        u[3] = t[3];
        u[4] = t[4];
        u[5] = t[5];

    }
        // Define the opL function as a device inline function
    __device__ __inline__ void opL(const float* df, const float* dfi, const float* dfj,
            const float* dfl, const float* dfk, const float* dfp, const float* dfq,
            const float* vadj, const float* vadj_i, const float* vadj_j, 
            const float* vadj_l, const float* vadj_k, const float* vadj_p, const float* vadj_q,
            const float* f1, const float* f2, float mu, float* t,
            long z, long y, long x,
            long depth, long rows, long cols){
        unsigned long size2d = rows*cols;
        unsigned long long idx = z * size2d + y * cols + x;

        float u1 = df[0]+vadj[0]-f1[idx]/mu;
        float u2 = df[0]+vadj[1]-f2[idx]/mu;
        float u1i = 0.f;
        float u2j = 0.f;
        float u1l = 0.f;
        float u2k = 0.f;
        float u1p = 0.f;
        float u2q = 0.f;

        if (x - 1 >= 0 && x<cols) {
            u1i = dfi[0]+vadj_i[0]-f1[idx-1]/mu;
        }
        if (y - 1 >= 0 && y<rows) {
            u2j = dfj[1]+vadj_j[1]-f2[idx-cols]/mu;
        }
        if (y >= 0 && y<rows-1) {
            u1l = dfl[0]+vadj_l[0]-f1[idx+cols]/mu;
        }
        if (x >= 0 && x<cols-1) {
            u2k = dfk[1]+vadj_k[1]-f2[idx+1]/mu;
        }
        if (y >= 0 && y<rows-1 && x - 1 >= 0 && x<cols) {
            u1p = dfp[0]+vadj_p[0]-f1[idx-1+cols]/mu;
        }
        if (x >= 0 && x<cols-1 && y - 1 >= 0 && y<rows) {
            u2q = dfq[1]+vadj_q[1]-f2[idx+1-cols]/mu;
        }

        t[0] = u1; //df[0]+vadj[0]-f1[idx]/mu
        t[1] = 0.25f*(u2+u2j+u2k+u2q);
        t[2] = 0.25f*(u1+u1i+u1l+u1p);
        t[3] = u2;
        t[4] = 0.5f*(u1+u1i);
        t[5] = 0.5f*(u2+u2j);
    }

    __device__ __inline__ void opLadj(const float* u, const float* u11, const float* u12, const float* u21, 
            const float* u22, const float* u31, const float* u32, float* t,
            long z, long y, long x,
            long depth, long rows, long cols){
        unsigned long size2d = rows*cols;
        unsigned long long idx = z * size2d + y * cols + x;
        
        t[0] = u[0];
        t[1] = u[3];

        t[0] += u21[idx]*0.25f;
        t[1] += u12[idx]*0.25f;

        if (y - 1 >= 0 && y<rows) {
            t[0] += u21[idx - cols]*0.25f;
        }
        if (y >= 0 && y<rows-1) {
            t[1] += u12[idx + cols]*0.25f;
        }
        if (x >= 0 && x<cols-1) {
            t[0] += u21[idx + 1]*0.25f;
        }
        if (x - 1 >= 0 && x<cols) {
            t[1] += u12[idx - 1]*0.25f;
        }
        if (y - 1 >= 0 && y<rows && x >= 0 && x<cols-1) {
            t[0] += u21[idx + 1 - cols]*0.25f;
        }
        if (x - 1 >= 0 && x<cols && y >= 0 && y<rows-1) {
            t[1] += u12[idx - 1 + cols]*0.25f;
        }
        
        t[0] += u31[idx]*0.5f;
        t[1] += u32[idx]*0.5f;

        if (x >= 0 && x<cols-1) {
            t[0] += u31[idx + 1]*0.5f;
        }
        if (y >= 0 && y<rows-1) {
            t[1] += u32[idx + cols]*0.5f;
        }
    }

    __global__ void APGM(const float* f, float* f1, float* f2, float* v11, float* v21, float* v31, float* v12, float* v22, float* v32,
            long depth, long rows, long cols,const float delta, const float mu0, int maxIter){
        unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned long z = threadIdx.z + blockIdx.z * blockDim.z;
        unsigned long long idx = z * rows * cols + y * cols + x;
        if ( x >= cols || y >= rows || z >= depth )
            return;
        
        float sgm = 0.33f;
        float mu = mu0;
        float v[6]     = {0.f,0.f,0.f,0.f,0.f,0.f};
        float vadj[2]  = {0.f,0.f};
        float vadjx[2] = {0.f,0.f};
        float vadjy[2] = {0.f,0.f};
        float vadjl[2] = {0.f,0.f};
        float vadjk[2] = {0.f,0.f};
        float vadjp[2] = {0.f,0.f};
        float vadjq[2] = {0.f,0.f};
        float Lop[6]   = {0.f,0.f,0.f,0.f,0.f,0.f};

        float df[3]  ={0.f,0.f,0.f};
        float dfi[3] ={0.f,0.f,0.f};
        float dfj[3] ={0.f,0.f,0.f};
        float dfl[3] ={0.f,0.f,0.f};
        float dfk[3] ={0.f,0.f,0.f};
        float dfp[3] ={0.f,0.f,0.f};
        float dfq[3] ={0.f,0.f,0.f};

        //need Du only at one point to intialize the loop to find the dual test function
        gradient(f,df   ,z  ,y   ,x  , depth,rows,cols);
        gradient(f,dfi  ,z  ,y   ,x-1 , depth,rows,cols);
        gradient(f,dfj  ,z  ,y-1 ,x  , depth,rows,cols);
        gradient(f,dfl  ,z  ,y   ,x+1 , depth,rows,cols);
        gradient(f,dfk  ,z  ,y-1 ,x+1  , depth,rows,cols);
        gradient(f,dfp  ,z  ,y+1 ,x   , depth,rows,cols);
        gradient(f,dfq  ,z  ,y+1 ,x-1  , depth,rows,cols);

        //iitialize vector field
        v[0] = v11[idx];
        v[1] = v12[idx];
        v[2] = v21[idx];
        v[3] = v22[idx];
        v[4] = v31[idx];
        v[5] = v32[idx];
        vadj[0] = v[0];
        vadj[1] = v[3];

        //APGM loop
        for(int i = 0; i < maxIter; i++){
            opLadj(v, v11, v12, v21, v22, v31, v32, vadjx  ,z  ,y   ,x-1 , depth,rows,cols);
            opLadj(v, v11, v12, v21, v22, v31, v32, vadjy  ,z  ,y-1 ,x  , depth,rows,cols);
            opLadj(v, v11, v12, v21, v22, v31, v32, vadjl  ,z  ,y   ,x+1 , depth,rows,cols);
            opLadj(v, v11, v12, v21, v22, v31, v32, vadjk  ,z  ,y-1 ,x+1  , depth,rows,cols);
            opLadj(v, v11, v12, v21, v22, v31, v32, vadjp  ,z  ,y+1 ,x , depth,rows,cols);
            opLadj(v, v11, v12, v21, v22, v31, v32, vadjq  ,z  ,y+1 ,x-1 , depth,rows,cols);
            opL(df, dfi, dfj, dfl, dfk, dfp, dfq, vadj, vadjx, vadjy, vadjl, vadjk, vadjp, vadjq, f1, f2, mu, Lop, z  ,y  ,x  , depth,rows,cols); 
            prox_mu_sigma_g(v,Lop,sgm,mu);
            opLadj(v, v11, v12, v21, v22, v31, v32, vadj  ,z  ,y  ,x  , depth,rows,cols);
            f1[idx] = f2[idx]-(-df[0]+vadj[0])*mu;
            f2[idx] = f2[idx]-(-df[1]+vadj[1])*mu;
            mu *= 0.99f;
        }
        return;
        
    }

    __global__ void gradientDTV(const float* f1, const float* f2, float* dftv,
            long depth, long rows, long cols,const float delta){
        unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned long z = threadIdx.z + blockIdx.z * blockDim.z;
        unsigned long long idx = z * rows * cols + y * cols + x;
        unsigned long long idy = z * rows * cols + y * cols + x-1;
        unsigned long long idx2 = z * rows * cols + (y-1) * cols + x;
        if ( x -1 <= 0 || y-1 <= 0 || z > depth )
            return;
        
        dftv[idx]=-f1[idx]+f1[idx2]-f2[idx]+f2[idy];
        
        return;
        
    }

    __global__ void gradientTV(const float* f, float* dftv,
            long depth, long rows, long cols,const float delta){
        unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned long z = threadIdx.z + blockIdx.z * blockDim.z;
        unsigned long long idx = z * rows * cols + y * cols + x;
        if ( x >= cols || y >= rows || z >= depth )
            return;
        
        
        float df[3] ={0.f,0.f,0.f};
        float dfi[3]={0.f,0.f,0.f}; // dfi== \partial f_{i+1,j,k}
        float dfj[3]={0.f,0.f,0.f};
        float dfk[3]={0.f,0.f,0.f};
        gradient(f,df  ,z  ,y  ,x  , depth,rows,cols);
        gradient(f,dfi ,z  ,y  ,x+1, depth,rows,cols);
        gradient(f,dfj ,z  ,y+1,x  , depth,rows,cols);
        gradient(f,dfk ,z+1,y  ,x  , depth,rows,cols);
        float eps=0.00000001; //% avoid division by zero
        
        float wx=__expf(-(df[0]/delta)*(df[0]/delta));
        float wy=__expf(-(df[1]/delta)*(df[1]/delta));
        float wz=__expf(-(df[2]/delta)*(df[2]/delta));
        
        float wxi=__expf(-(dfi[0]/delta)*(dfi[0]/delta));
        float wyi=__expf(-(dfi[1]/delta)*(dfi[1]/delta));
        float wzi=__expf(-(dfi[2]/delta)*(dfi[2]/delta));
        
        float wxj=__expf(-(dfj[0]/delta)*(dfj[0]/delta));
        float wyj=__expf(-(dfj[1]/delta)*(dfj[1]/delta));
        float wzj=__expf(-(dfj[2]/delta)*(dfj[2]/delta));
        
        float wxk=__expf(-(dfk[0]/delta)*(dfk[0]/delta));
        float wyk=__expf(-(dfk[1]/delta)*(dfk[1]/delta));
        float wzk=__expf(-(dfk[2]/delta)*(dfk[2]/delta));

        
        // this hsould do the trick I think
        
        dftv[idx]=(wx*df[0]+wy*df[1]+wz*df[2])/(sqrt(wx*df[0] *df[0] +wy*df[1] *df[1] +wz*df[2] *df[2])+eps)
        -wzi*dfi[2]/(sqrt(wxi*dfi[0]*dfi[0]+wyi*dfi[1]*dfi[1]+wzi*dfi[2]*dfi[2]) +eps)     // I wish I coudl precompute this, but if I do then Id need to recompute the gradient.
        -wyj*dfj[1]/(sqrt(wxj*dfj[0]*dfj[0]+wyj*dfj[1]*dfj[1]+wzj*dfj[2]*dfj[2]) +eps)
        -wxk*dfk[0]/(sqrt(wxk*dfk[0]*dfk[0]+wyk*dfk[1]*dfk[1]+wzk*dfk[2]*dfk[2]) +eps);
        
    
        return;
        
    }
    
    __device__ void warpReduce(volatile float *sdata, size_t tid) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
    
    __global__ void  reduceNorm2(float *g_idata, float *g_odata, size_t n){
        extern __shared__ volatile float sdata[];
        //http://stackoverflow.com/a/35133396/1485872
        size_t tid = threadIdx.x;
        size_t i = blockIdx.x*blockDim.x + tid;
        size_t gridSize = blockDim.x*gridDim.x;
        float mySum = 0;
        float value=0;
        while (i < n) {
            value=g_idata[i]; //avoid reading twice
            mySum += value*value;
            i += gridSize;
        }
        sdata[tid] = mySum;
        __syncthreads();
        
        if (tid < 512)
            sdata[tid] += sdata[tid + 512];
        __syncthreads();
        if (tid < 256)
            sdata[tid] += sdata[tid + 256];
        __syncthreads();
        
        if (tid < 128)
            sdata[tid] += sdata[tid + 128];
        __syncthreads();
        
        if (tid <  64)
            sdata[tid] += sdata[tid + 64];
        __syncthreads();
        
        
#if (__CUDART_VERSION >= 9000)
        if ( tid < 32 )
        {
            mySum = sdata[tid] + sdata[tid + 32];
            for (int offset = warpSize/2; offset > 0; offset /= 2) {
                mySum += __shfl_down_sync(0xFFFFFFFF, mySum, offset,32);
            }
        }
#else
        if (tid < 32) {
            warpReduce(sdata, tid);
            mySum = sdata[0];
        }
#endif
        if (tid == 0) g_odata[blockIdx.x] = mySum;
    }
    
    __global__ void  reduceSum(float *g_idata, float *g_odata, size_t n){
        extern __shared__ volatile float sdata[];
        //http://stackoverflow.com/a/35133396/1485872
        size_t tid = threadIdx.x;
        size_t i = blockIdx.x*blockDim.x + tid;
        size_t gridSize = blockDim.x*gridDim.x;
        float mySum = 0;
        // float value=0;
        while (i < n) {
            mySum += g_idata[i];
            i += gridSize;
        }
        sdata[tid] = mySum;
        __syncthreads();
        
        if (tid < 512)
            sdata[tid] += sdata[tid + 512];
        __syncthreads();
        if (tid < 256)
            sdata[tid] += sdata[tid + 256];
        __syncthreads();
        
        if (tid < 128)
            sdata[tid] += sdata[tid + 128];
        __syncthreads();
        
        if (tid <  64)
            sdata[tid] += sdata[tid + 64];
        __syncthreads();
        
        
#if (__CUDART_VERSION >= 9000)
        if ( tid < 32 )
        {
            mySum = sdata[tid] + sdata[tid + 32];
            for (int offset = warpSize/2; offset > 0; offset /= 2) {
                mySum += __shfl_down_sync(0xFFFFFFFF, mySum, offset,32);
            }
        }
#else
        if (tid < 32) {
            warpReduce(sdata, tid);
            mySum = sdata[0];
        }
#endif
        if (tid == 0) g_odata[blockIdx.x] = mySum;
    }
    
    
    
    
// main function
void aw_pocs_tv(float* img,float* dst,float alpha,const long* image_size, int maxIter,const float delta, const GpuIds& gpuids){
        // Prepare for MultiGPU
        int deviceCount = gpuids.GetLength();
        cudaCheckErrors("Device query fail");
        if (deviceCount == 0) {
            mexErrMsgIdAndTxt("minimizeDTV:POCS_TV2:GPUselect","There are no available device(s) that support CUDA\n");
        }
        //
        // CODE assumes
        // 1.-All available devices are usable by this code
        // 2.-All available devices are equal, they are the same machine (warning thrown)
        // Check the available devices, and if they are the same
        if (!gpuids.AreEqualDevices()) {
            mexWarnMsgIdAndTxt("minimizeDTV:POCS_TV2:GPUselect","Detected one (or more) different GPUs.\n This code is not smart enough to separate the memory GPU wise if they have different computational times or memory limits.\n First GPU parameters used. If the code errors you might need to change the way GPU selection is performed.");
        }
        int dev;
        
        // We don't know if the devices are being used. lets check that. and only use the amount of memory we need.
        // check free memory
        size_t mem_GPU_global;
        checkFreeMemory(gpuids, &mem_GPU_global);

        
        
        // %5 of free memory should be enough, we have almost no variables in these kernels
        size_t total_pixels              = image_size[0] * image_size[1]  * image_size[2] ;
        size_t mem_slice_image           = sizeof(float)* image_size[0] * image_size[1]  ;
        size_t mem_size_image            = sizeof(float)* total_pixels;
        size_t mem_auxiliary             = sizeof(float)* (total_pixels + MAXTHREADS - 1) / MAXTHREADS;
        
        // Decide how are we handling the distribution of computation
        size_t mem_img_each_GPU;
        
        unsigned int buffer_length=2;
        //Does everything fit in the GPU?
        unsigned int slices_per_split;
        
        // if it is a thin problem (no need to split), just use one GPU
        if (image_size[2]<4){deviceCount=1;}
        
        unsigned int splits=1; // if the number does not fit in an uint, you have more serious trouble than this.
        if(mem_GPU_global> 3*mem_size_image+3*(deviceCount-1)*mem_slice_image*buffer_length+mem_auxiliary) {
            // We only need to split if we have extra GPUs
            slices_per_split=(image_size[2]+deviceCount-1)/deviceCount;
            mem_img_each_GPU=mem_slice_image*((slices_per_split+buffer_length*2));
        }else{
            // As mem_auxiliary is not expected to be a large value (for a 2000^3 image is around 28Mbytes), lets for now assume we need it all
            size_t mem_free=mem_GPU_global-mem_auxiliary;
            
            splits=(unsigned int)(ceil(((float)(3*mem_size_image)/(float)(deviceCount))/mem_free));
            // Now, there is an overhead here, as each splits should have 2 slices more, to account for overlap of images.
            // lets make sure these 2 slices fit, if they do not, add 1 to splits.
            slices_per_split=(image_size[2]+deviceCount*splits-1)/(deviceCount*splits);
            mem_img_each_GPU=(mem_slice_image*(slices_per_split+buffer_length*2));
            
            // if the new stuff does not fit in the GPU, it means we are in the edge case where adding that extra slice will overflow memory
            if (mem_GPU_global< 3*mem_img_each_GPU+mem_auxiliary){
                // one more split should do the job, as its an edge case.
                splits++;
                //recompute for later
                slices_per_split=(image_size[2]+deviceCount*splits-1)/(deviceCount*splits); // amount of slices that fit on a GPU. Later we add 2 to these, as we need them for overlap
                mem_img_each_GPU=(mem_slice_image*(slices_per_split+buffer_length*2));
            }


            // How many EXTRA buffer slices should be able to fit in here??!?!
            // Only do it if there are splits needed. 
            if(splits>1){
                mem_free=mem_GPU_global-(3*mem_img_each_GPU+mem_auxiliary);
                unsigned int extra_buff=(mem_free/mem_slice_image); 
                buffer_length=(extra_buff/2)/3; // we need double whatever this results in, rounded down.
                buffer_length=max(buffer_length,2);// minimum 2
                buffer_length=min(MAX_BUFFER,buffer_length);

                mem_img_each_GPU=mem_slice_image*(slices_per_split+buffer_length*2);
                
            }else{
                buffer_length=2;
            }

            // Assert
            if (mem_GPU_global< 3*mem_img_each_GPU+mem_auxiliary){
                mexErrMsgIdAndTxt("minimizeDTV:POCS_TV2:GPU","Assertion Failed. Logic behind spliting flawed! Please tell: ander.biguri@gmail.com\n");
            }
        }
        
        
         // Assert
       
        if ((slices_per_split+buffer_length*2)*image_size[0]*image_size[1]* sizeof(float)!= mem_img_each_GPU){
            mexErrMsgIdAndTxt("minimizeDTV:POCS_TV2:GPU","Assertion Failed. Memory needed calculation broken! Please tell: ander.biguri@gmail.com\n");
        }
        
        
        
        
        
        
        float** d_image=    (float**)malloc(deviceCount*sizeof(float*));
        float** d_dimgTV=   (float**)malloc(deviceCount*sizeof(float*));
        float** d_norm2aux= (float**)malloc(deviceCount*sizeof(float*));
        float** d_norm2=    (float**)malloc(deviceCount*sizeof(float*));
         
        // allocate memory in each GPU
        for (dev = 0; dev < deviceCount; dev++){
            cudaSetDevice(gpuids[dev]);
            
            cudaMalloc((void**)&d_image[dev]    , mem_img_each_GPU);
            cudaMemset(         d_image[dev],0  , mem_img_each_GPU);
            cudaMalloc((void**)&d_dimgTV[dev]   , mem_img_each_GPU);
            cudaMemset(         d_dimgTV[dev],0 , mem_img_each_GPU);
            cudaMalloc((void**)&d_norm2[dev]    , slices_per_split*mem_slice_image);
            cudaMemset(         d_norm2[dev],0  , slices_per_split*mem_slice_image);
            cudaMalloc((void**)&d_norm2aux[dev]   , mem_auxiliary);
            cudaMemset(         d_norm2aux[dev],0 , mem_auxiliary);
            cudaCheckErrors("Malloc  error");
            
            
        }
       unsigned long long buffer_pixels=buffer_length*image_size[0]*image_size[1];
        float* buffer;
        if(splits>1){
            mexWarnMsgIdAndTxt("minimizeDTV:POCS_TV2:Image_split","Your image can not be fully split between the available GPUs. The computation of minTV will be significantly slowed due to the image size.\nApproximated mathematics turned on for computational speed.");
        }else{
            cudaMallocHost((void**)&buffer,buffer_length*image_size[0]*image_size[1]*sizeof(float));
        }
        
        
        
        // Lets try to make the host memory pinned:
        // We laredy queried the GPU and assuemd they are the same, thus should have the same attributes.
        int isHostRegisterSupported = 0;
#if CUDART_VERSION >= 9020
        cudaDeviceGetAttribute(&isHostRegisterSupported,cudaDevAttrHostRegisterSupported,gpuids[0]);
#endif
        // splits>2 is completely empirical observation
        if (isHostRegisterSupported & splits>2){
            cudaHostRegister(img ,image_size[2]*image_size[1]*image_size[0]*sizeof(float),cudaHostRegisterPortable);
            cudaHostRegister(dst ,image_size[2]*image_size[1]*image_size[0]*sizeof(float),cudaHostRegisterPortable);
        }
        cudaCheckErrors("Error pinning memory");

        
        
                // Create streams
        int nStream_device=2;
        int nStreams=deviceCount*nStream_device;
        cudaStream_t* stream=(cudaStream_t*)malloc(nStreams*sizeof(cudaStream_t));
        
        for (dev = 0; dev < deviceCount; dev++){
            cudaSetDevice(gpuids[dev]);
            for (int i = 0; i < nStream_device; ++i){
                cudaStreamCreate(&stream[i+dev*nStream_device]);
            }
        }
        cudaCheckErrors("Stream creation fail");

        
        // For the reduction

        double totalsum_prev;
        double totalsum;
        float sum_curr_spl;
        float * sumnorm2;
        cudaMallocHost((void**)&sumnorm2,deviceCount*sizeof(float));
        
        unsigned int curr_slices;
        unsigned long long curr_pixels;
        size_t linear_idx_start;
        unsigned long long* offset_device=(unsigned long long*)malloc(deviceCount*sizeof(unsigned long long));
        unsigned long long* offset_host  =(unsigned long long*)malloc(deviceCount*sizeof(unsigned long long));
        unsigned long long* bytes_device =(unsigned long long*)malloc(deviceCount*sizeof(unsigned long long));
        bool is_first_chunk;
        bool is_last_chunk;
        for(unsigned int i=0;i<maxIter;i+=(buffer_length-1)){
            if(splits>1){
                totalsum_prev=0;
            }
            for(unsigned int sp=0;sp<splits;sp++){
                
                // For each iteration we need to compute all the image. The ordering of these loops
                // need to be like this due to the bounding layers between splits. If more than 1 split is needed
                // for each GPU then there is no other way that taking the entire memory out of GPU and putting it back.
                // If the memory can be shared between GPUs fully without extra splits, then there is an easy way of synchronizing the memory
                
                // Copy image to memory
                for (dev = 0; dev < deviceCount; dev++){
                    curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                    curr_pixels=curr_slices*image_size[0]*image_size[1];
                    linear_idx_start=image_size[0]*image_size[1]*slices_per_split*(sp*deviceCount+dev);
                    
                    // Check if its the first or last chunck
                    is_last_chunk=!((sp*deviceCount+dev)<deviceCount*splits-1);
                    is_first_chunk=!(sp*deviceCount+dev);
                    
                    // lets compute where we start copyes and how much. This avoids 3 calls to Memcpy
                    offset_device[dev]=buffer_pixels*is_first_chunk;
                    offset_host[dev]=linear_idx_start-buffer_pixels*!is_first_chunk;
                    bytes_device[dev]=curr_pixels+buffer_pixels*!is_first_chunk+buffer_pixels*!is_last_chunk;
                }

                if(i==0){
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        
                        cudaMemcpyAsync(d_image[dev]+offset_device[dev], img+offset_host[dev]  , bytes_device[dev]*sizeof(float), cudaMemcpyHostToDevice,stream[dev*nStream_device+1]);
                        
                        
                    }
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaDeviceSynchronize();
                    }
                }
                // if we need to split and its not the first iteration, then we need to copy from Host memory the previosu result.
                if (splits>1 & i>0){
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaMemcpyAsync(d_image[dev]+offset_device[dev], dst+offset_host[dev]  , bytes_device[dev]*sizeof(float), cudaMemcpyHostToDevice,stream[dev*nStream_device+1]);
                        
                        
                    }
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaDeviceSynchronize();
                    }
                }
                cudaCheckErrors("Memcpy failure on multi split");
                
                for(unsigned int ib=0;  (ib<(buffer_length-1)) && ((i+ib)<maxIter);  ib++){
                    
                    // For the gradient
                    dim3 blockGrad(10, 10, 10);
                    dim3 gridGrad((image_size[0]+blockGrad.x-1)/blockGrad.x, (image_size[1]+blockGrad.y-1)/blockGrad.y, (curr_slices+buffer_length*2+blockGrad.z-1)/blockGrad.z);
                    
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        // Compute the gradient of the TV norm
                        
                        // I don't understand why I need to store 2 layers to compute correctly with 1 buffer. The bounding checks should
                        // be enough but they are not.
                        gradientTV<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(d_image[dev],d_dimgTV[dev],(long)(curr_slices+buffer_length*2-1), image_size[1],image_size[0],delta);
                        
                    }
                    
                    
                    
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        // no need to copy the 2 aux slices here
                        cudaStreamSynchronize(stream[dev*nStream_device]);
                        cudaMemcpyAsync(d_norm2[dev], d_dimgTV[dev]+buffer_pixels, image_size[0]*image_size[1]*curr_slices*sizeof(float), cudaMemcpyDeviceToDevice,stream[dev*nStream_device+1]);
                    }
                    
                    
                    // Compute the L2 norm of the gradient. For that, reduction is used.
                    //REDUCE
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        
                        size_t dimblockRed = MAXTHREADS;
                        size_t dimgridRed = (total_pixels + MAXTHREADS - 1) / MAXTHREADS;
                        
                        cudaStreamSynchronize(stream[dev*nStream_device+1]);
                        reduceNorm2 << <dimgridRed, dimblockRed, MAXTHREADS*sizeof(float),stream[dev*nStream_device]>> >(d_norm2[dev], d_norm2aux[dev], total_pixels);
                        
                    }
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        size_t dimblockRed = MAXTHREADS;
                        size_t dimgridRed = (total_pixels + MAXTHREADS - 1) / MAXTHREADS;

                        if (dimgridRed > 1) {
                            reduceSum << <1, dimblockRed, MAXTHREADS*sizeof(float),stream[dev*nStream_device] >> >(d_norm2aux[dev], d_norm2[dev], dimgridRed);
                            cudaStreamSynchronize(stream[dev*nStream_device]);
                            cudaMemcpyAsync(&sumnorm2[dev], d_norm2[dev], sizeof(float), cudaMemcpyDeviceToHost,stream[dev*nStream_device+1]);
                        }
                        else {
                            cudaStreamSynchronize(stream[dev*nStream_device]);
                            cudaMemcpyAsync(&sumnorm2[dev], d_norm2aux[dev], sizeof(float), cudaMemcpyDeviceToHost,stream[dev*nStream_device+1]);
                        }
                    }
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaDeviceSynchronize();
                     }
                    cudaCheckErrors("Reduction error");
                    
                    
                    // Accumulate the norm accross devices
                    sum_curr_spl=0;
                    // this is CPU code
                    for (dev = 0; dev < deviceCount; dev++){
                        sum_curr_spl+=sumnorm2[dev];
                    }
                    sum_curr_spl+=0.0000001f; // avoid division by zero
                    
                    // If we have more than one splits, lets use the result from prior calls
                    if(i>0 && splits>1){
                        // this is already stored:
                        //totalsum=totalsum_prev; 
                    }else{
                        totalsum=sum_curr_spl;
                    }
                    
                    
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        //NORMALIZE
                        //in a Tesla, maximum blocks =15 SM * 4 blocks/SM
                        divideArrayScalar  <<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(d_dimgTV[dev]+buffer_pixels,(float)sqrt(totalsum),total_pixels);
                        //MULTIPLY HYPERPARAMETER
                        multiplyArrayScalar<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(d_dimgTV[dev]+buffer_pixels,alpha,   total_pixels);
                    }
                     for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaDeviceSynchronize();
                     }
                    cudaCheckErrors("Scalar operations error");
                    
                    //SUBSTRACT GRADIENT
                    //////////////////////////////////////////////
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        
                        substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(d_image[dev]+buffer_pixels,d_dimgTV[dev]+buffer_pixels, total_pixels);
                    }
                }

                // Synchronize mathematics, make sure bounding pixels are correct
                 for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaDeviceSynchronize();
                     }
                
                if(splits==1){
                    for(dev=0; dev<deviceCount;dev++){
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        if (dev<deviceCount-1){
                            cudaSetDevice(gpuids[dev+1]);
                            cudaMemcpy(buffer, d_image[dev+1], buffer_pixels*sizeof(float), cudaMemcpyDeviceToHost);
                            cudaSetDevice(gpuids[dev]);
                            cudaMemcpy(d_image[dev]+total_pixels+buffer_pixels,buffer, buffer_pixels*sizeof(float), cudaMemcpyHostToDevice); 
                        }
                        cudaDeviceSynchronize();
                        if (dev>0){
                            cudaSetDevice(gpuids[dev-1]);
                            cudaMemcpyAsync(buffer, d_image[dev-1]+total_pixels+buffer_pixels, buffer_pixels*sizeof(float), cudaMemcpyDeviceToHost);
                            cudaSetDevice(gpuids[dev]);
                            cudaMemcpyAsync(d_image[dev],buffer, buffer_pixels*sizeof(float), cudaMemcpyHostToDevice);
                        }
                    }
                }else{
                    
                    // We need to take it out :(
                    for(dev=0; dev<deviceCount;dev++){
                        cudaSetDevice(gpuids[dev]);
                        
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        linear_idx_start=image_size[0]*image_size[1]*slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        cudaMemcpyAsync(&dst[linear_idx_start], d_image[dev]+buffer_pixels,total_pixels*sizeof(float), cudaMemcpyDeviceToHost,stream[dev*nStream_device+1]);
                    }
                }
                
                for (dev = 0; dev < deviceCount; dev++){
                    cudaSetDevice(gpuids[dev]);
                    cudaDeviceSynchronize();
                }
                cudaCheckErrors("Memory gather error");

                totalsum_prev+=sum_curr_spl;
            }
            totalsum=totalsum_prev;
        }
        // If there has not been splits, we still have data in memory
        if(splits==1){
            for(dev=0; dev<deviceCount;dev++){
                cudaSetDevice(gpuids[dev]);
                
                curr_slices=((dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*dev;
                total_pixels=curr_slices*image_size[0]*image_size[1];
                cudaMemcpy(dst+slices_per_split*image_size[0]*image_size[1]*dev, d_image[dev]+buffer_pixels,total_pixels*sizeof(float), cudaMemcpyDeviceToHost);
            }
        }
        cudaCheckErrors("Copy result back");
        
        for(dev=0; dev<deviceCount;dev++){
            cudaSetDevice(gpuids[dev]);
            cudaFree(d_image[dev]);
            cudaFree(d_norm2aux[dev]);
            cudaFree(d_dimgTV[dev]);
            cudaFree(d_norm2[dev]);
        }
        if (splits==1){
            cudaFreeHost(buffer);
        }
        
        if (isHostRegisterSupported& splits>2){
            cudaHostUnregister(img);
            cudaHostUnregister(dst);
        }
        for (int i = 0; i < nStreams; ++i)
           cudaStreamDestroy(stream[i]) ;
        cudaCheckErrors("Memory free");
//         cudaDeviceReset();
    }

void dtv_pocs(float* img,float* dst,float alpha,const long* image_size, int maxIter, int APGMiter, float mu, const float delta, const GpuIds& gpuids){
        // Prepare for MultiGPU
        int deviceCount = gpuids.GetLength();
        cudaCheckErrors("Device query fail");
        if (deviceCount == 0) {
            mexErrMsgIdAndTxt("minimizeDTV:POCS_TV2:GPUselect","There are no available device(s) that support CUDA\n");
        }
        //
        // CODE assumes
        // 1.-All available devices are usable by this code
        // 2.-All available devices are equal, they are the same machine (warning thrown)
        // Check the available devices, and if they are the same
        if (!gpuids.AreEqualDevices()) {
            mexWarnMsgIdAndTxt("minimizeDTV:POCS_TV2:GPUselect","Detected one (or more) different GPUs.\n This code is not smart enough to separate the memory GPU wise if they have different computational times or memory limits.\n First GPU parameters used. If the code errors you might need to change the way GPU selection is performed.");
        }
        int dev;
        
        // We don't know if the devices are being used. lets check that. and only use the amount of memory we need.
        // check free memory
        size_t mem_GPU_global;
        checkFreeMemory(gpuids, &mem_GPU_global);

        
        
        // %5 of free memory should be enough, we have almost no variables in these kernels
        size_t total_pixels              = image_size[0] * image_size[1]  * image_size[2] ;
        size_t mem_slice_image           = sizeof(float)* image_size[0] * image_size[1]  ;
        size_t mem_size_image            = sizeof(float)* total_pixels;
        size_t mem_auxiliary             = sizeof(float)* (total_pixels + MAXTHREADS - 1) / MAXTHREADS;
        
        // Decide how are we handling the distribution of computation
        size_t mem_img_each_GPU;
        
        unsigned int buffer_length=2;
        //Does everything fit in the GPU?
        unsigned int slices_per_split;
        
        // if it is a thin problem (no need to split), just use one GPU
        if (image_size[2]<4){deviceCount=1;}
        
        unsigned int splits=1; // if the number does not fit in an uint, you have more serious trouble than this.
        if(mem_GPU_global> 3*mem_size_image+3*(deviceCount-1)*mem_slice_image*buffer_length+mem_auxiliary) {
            // We only need to split if we have extra GPUs
            slices_per_split=(image_size[2]+deviceCount-1)/deviceCount;
            mem_img_each_GPU=mem_slice_image*((slices_per_split+buffer_length*2));
        }else{
            // As mem_auxiliary is not expected to be a large value (for a 2000^3 image is around 28Mbytes), lets for now assume we need it all
            size_t mem_free=mem_GPU_global-mem_auxiliary;
            
            splits=(unsigned int)(ceil(((float)(3*mem_size_image)/(float)(deviceCount))/mem_free));
            // Now, there is an overhead here, as each splits should have 2 slices more, to account for overlap of images.
            // lets make sure these 2 slices fit, if they do not, add 1 to splits.
            slices_per_split=(image_size[2]+deviceCount*splits-1)/(deviceCount*splits);
            mem_img_each_GPU=(mem_slice_image*(slices_per_split+buffer_length*2));
            
            // if the new stuff does not fit in the GPU, it means we are in the edge case where adding that extra slice will overflow memory
            if (mem_GPU_global< 3*mem_img_each_GPU+mem_auxiliary){
                // one more split should do the job, as its an edge case.
                splits++;
                //recompute for later
                slices_per_split=(image_size[2]+deviceCount*splits-1)/(deviceCount*splits); // amount of slices that fit on a GPU. Later we add 2 to these, as we need them for overlap
                mem_img_each_GPU=(mem_slice_image*(slices_per_split+buffer_length*2));
            }


            // How many EXTRA buffer slices should be able to fit in here??!?!
            // Only do it if there are splits needed. 
            if(splits>1){
                mem_free=mem_GPU_global-(3*mem_img_each_GPU+mem_auxiliary);
                unsigned int extra_buff=(mem_free/mem_slice_image); 
                buffer_length=(extra_buff/2)/3; // we need double whatever this results in, rounded down.
                buffer_length=max(buffer_length,2);// minimum 2
                buffer_length=min(MAX_BUFFER,buffer_length);

                mem_img_each_GPU=mem_slice_image*(slices_per_split+buffer_length*2);
                
            }else{
                buffer_length=2;
            }

            // Assert
            if (mem_GPU_global< 3*mem_img_each_GPU+mem_auxiliary){
                mexErrMsgIdAndTxt("minimizeDTV:POCS_TV2:GPU","Assertion Failed. Logic behind spliting flawed! Please tell: ander.biguri@gmail.com\n");
            }
        }
        
        
         // Assert
       
        if ((slices_per_split+buffer_length*2)*image_size[0]*image_size[1]* sizeof(float)!= mem_img_each_GPU){
            mexErrMsgIdAndTxt("minimizeDTV:POCS_TV2:GPU","Assertion Failed. Memory needed calculation broken! Please tell: ander.biguri@gmail.com\n");
        }
        

        float** d_image=    (float**)malloc(deviceCount*sizeof(float*));
        float** p1_=    (float**)malloc(deviceCount*sizeof(float*)); 
        float** p2_=    (float**)malloc(deviceCount*sizeof(float*)); 
        float** p1_image=    (float**)malloc(deviceCount*sizeof(float*)); 
        float** p2_image=    (float**)malloc(deviceCount*sizeof(float*));
        float** v11_image=    (float**)malloc(deviceCount*sizeof(float*));
        float** v21_image=    (float**)malloc(deviceCount*sizeof(float*));
        float** v31_image=    (float**)malloc(deviceCount*sizeof(float*));
        float** v12_image=    (float**)malloc(deviceCount*sizeof(float*));
        float** v22_image=    (float**)malloc(deviceCount*sizeof(float*));
        float** v32_image=    (float**)malloc(deviceCount*sizeof(float*));
        float** d_dimgTV=   (float**)malloc(deviceCount*sizeof(float*));
        float** d_norm2aux= (float**)malloc(deviceCount*sizeof(float*));
        float** d_norm2=    (float**)malloc(deviceCount*sizeof(float*));
         
        // allocate memory in each GPU
        for (dev = 0; dev < deviceCount; dev++){
            cudaSetDevice(gpuids[dev]);
            
            cudaMalloc((void**)&d_image[dev]    , mem_img_each_GPU);
            cudaMemset(         d_image[dev],0  , mem_img_each_GPU);
            cudaMalloc((void**)&p1_[dev]    , mem_img_each_GPU);
            cudaMemset(         p1_[dev],0  , mem_img_each_GPU);
            cudaMalloc((void**)&p2_[dev]    , mem_img_each_GPU);
            cudaMemset(         p2_[dev],0  , mem_img_each_GPU);
            cudaMalloc((void**)&p1_image[dev]    , mem_img_each_GPU);
            cudaMemset(         p1_image[dev],0  , mem_img_each_GPU);
            cudaMalloc((void**)&p2_image[dev]    , mem_img_each_GPU);
            cudaMemset(         p2_image[dev],0  , mem_img_each_GPU);
            cudaMalloc((void**)&v11_image[dev]    , mem_img_each_GPU);
            cudaMemset(         v11_image[dev],0  , mem_img_each_GPU);
            cudaMalloc((void**)&v21_image[dev]    , mem_img_each_GPU);
            cudaMemset(         v21_image[dev],0  , mem_img_each_GPU);
            cudaMalloc((void**)&v31_image[dev]    , mem_img_each_GPU);
            cudaMemset(         v31_image[dev],0  , mem_img_each_GPU);
            cudaMalloc((void**)&v12_image[dev]    , mem_img_each_GPU);
            cudaMemset(         v12_image[dev],0  , mem_img_each_GPU);
            cudaMalloc((void**)&v22_image[dev]    , mem_img_each_GPU);
            cudaMemset(         v22_image[dev],0  , mem_img_each_GPU);
            cudaMalloc((void**)&v32_image[dev]    , mem_img_each_GPU);
            cudaMemset(         v32_image[dev],0  , mem_img_each_GPU);
            cudaMalloc((void**)&d_dimgTV[dev]   , mem_img_each_GPU);
            cudaMemset(         d_dimgTV[dev],0 , mem_img_each_GPU);
            cudaMalloc((void**)&d_norm2[dev]    , slices_per_split*mem_slice_image);
            cudaMemset(         d_norm2[dev],0  , slices_per_split*mem_slice_image);
            cudaMalloc((void**)&d_norm2aux[dev]   , mem_auxiliary);
            cudaMemset(         d_norm2aux[dev],0 , mem_auxiliary);
            cudaCheckErrors("Malloc  error");
            
            
        }
       unsigned long long buffer_pixels=buffer_length*image_size[0]*image_size[1];
        float* buffer;
        if(splits>1){
            mexWarnMsgIdAndTxt("minimizeDTV:POCS_TV2:Image_split","Your image can not be fully split between the available GPUs. The computation of minTV will be significantly slowed due to the image size.\nApproximated mathematics turned on for computational speed.");
        }else{
            cudaMallocHost((void**)&buffer,buffer_length*image_size[0]*image_size[1]*sizeof(float));
        }
        
        
        
        // Lets try to make the host memory pinned:
        // We laredy queried the GPU and assuemd they are the same, thus should have the same attributes.
        int isHostRegisterSupported = 0;
#if CUDART_VERSION >= 9020
        cudaDeviceGetAttribute(&isHostRegisterSupported,cudaDevAttrHostRegisterSupported,gpuids[0]);
#endif
        // splits>2 is completely empirical observation
        if (isHostRegisterSupported & splits>2){
            cudaHostRegister(img ,image_size[2]*image_size[1]*image_size[0]*sizeof(float),cudaHostRegisterPortable);
            cudaHostRegister(dst ,image_size[2]*image_size[1]*image_size[0]*sizeof(float),cudaHostRegisterPortable);
        }
        cudaCheckErrors("Error pinning memory");

        
        
                // Create streams
        int nStream_device=2;
        int nStreams=deviceCount*nStream_device;
        cudaStream_t* stream=(cudaStream_t*)malloc(nStreams*sizeof(cudaStream_t));
        
        for (dev = 0; dev < deviceCount; dev++){
            cudaSetDevice(gpuids[dev]);
            for (int i = 0; i < nStream_device; ++i){
                cudaStreamCreate(&stream[i+dev*nStream_device]);
            }
        }
        cudaCheckErrors("Stream creation fail");

        
        // For the reduction

        double totalsum_prev;
        double totalsum;
        float sum_curr_spl;
        float * sumnorm2;
        cudaMallocHost((void**)&sumnorm2,deviceCount*sizeof(float));
        
        unsigned int curr_slices;
        unsigned long long curr_pixels;
        size_t linear_idx_start;
        unsigned long long* offset_device=(unsigned long long*)malloc(deviceCount*sizeof(unsigned long long));
        unsigned long long* offset_host  =(unsigned long long*)malloc(deviceCount*sizeof(unsigned long long));
        unsigned long long* bytes_device =(unsigned long long*)malloc(deviceCount*sizeof(unsigned long long));
        bool is_first_chunk;
        bool is_last_chunk;
        for(unsigned int i=0;i<maxIter;i+=(buffer_length-1)){
            if(splits>1){
                totalsum_prev=0;
            }
            for(unsigned int sp=0;sp<splits;sp++){
                
                // For each iteration we need to compute all the image. The ordering of these loops
                // need to be like this due to the bounding layers between splits. If more than 1 split is needed
                // for each GPU then there is no other way that taking the entire memory out of GPU and putting it back.
                // If the memory can be shared between GPUs fully without extra splits, then there is an easy way of synchronizing the memory
                
                // Copy image to memory
                for (dev = 0; dev < deviceCount; dev++){
                    curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                    curr_pixels=curr_slices*image_size[0]*image_size[1];
                    linear_idx_start=image_size[0]*image_size[1]*slices_per_split*(sp*deviceCount+dev);
                    
                    // Check if its the first or last chunck
                    is_last_chunk=!((sp*deviceCount+dev)<deviceCount*splits-1);
                    is_first_chunk=!(sp*deviceCount+dev);
                    
                    // lets compute where we start copyes and how much. This avoids 3 calls to Memcpy
                    offset_device[dev]=buffer_pixels*is_first_chunk;
                    offset_host[dev]=linear_idx_start-buffer_pixels*!is_first_chunk;
                    bytes_device[dev]=curr_pixels+buffer_pixels*!is_first_chunk+buffer_pixels*!is_last_chunk;
                }

                if(i==0){
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        
                        cudaMemcpyAsync(d_image[dev]+offset_device[dev], img+offset_host[dev]  , bytes_device[dev]*sizeof(float), cudaMemcpyHostToDevice,stream[dev*nStream_device+1]);
                        
                        
                    }
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaDeviceSynchronize();
                    }
                }
                // if we need to split and its not the first iteration, then we need to copy from Host memory the previosu result.
                if (splits>1 & i>0){
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaMemcpyAsync(d_image[dev]+offset_device[dev], dst+offset_host[dev]  , bytes_device[dev]*sizeof(float), cudaMemcpyHostToDevice,stream[dev*nStream_device+1]);
                        
                        
                    }
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaDeviceSynchronize();
                    }
                }
                cudaCheckErrors("Memcpy failure on multi split");
                
                for(unsigned int ib=0;  (ib<(buffer_length-1)) && ((i+ib)<maxIter);  ib++){
                    
                    dim3 blockGrad(10, 10, 10);
                    dim3 gridGrad((image_size[0]+blockGrad.x-1)/blockGrad.x, (image_size[1]+blockGrad.y-1)/blockGrad.y, (curr_slices+buffer_length*2+blockGrad.z-1)/blockGrad.z);
                    
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        
                        //COMPUTE p1_image and p2_image using ADMM then use them to compute the dTV gradient
                        APGM<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(d_image[dev],p1_image[dev],p2_image[dev],v11_image[dev],v21_image[dev],v31_image[dev],v12_image[dev],v22_image[dev],v32_image[dev],(long)(curr_slices+buffer_length*2-1), image_size[1],image_size[0],delta,mu,APGMiter);

                    }

                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        // no need to copy the 2 aux slices here
                        cudaStreamSynchronize(stream[dev*nStream_device]);
                        cudaMemcpyAsync(p1_[dev], p1_image[dev]+buffer_pixels, image_size[0]*image_size[1]*curr_slices*sizeof(float), cudaMemcpyDeviceToDevice,stream[dev*nStream_device+1]);
                        cudaMemcpyAsync(p2_[dev], p2_image[dev]+buffer_pixels, image_size[0]*image_size[1]*curr_slices*sizeof(float), cudaMemcpyDeviceToDevice,stream[dev*nStream_device+1]);
                    }
                    
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        
                        // Compute the gradient of the TV norm
                        // I don't understand why I need to store 2 layers to compute correctly with 1 buffer. The bounding checks should
                        // be enough but they are not.
                        gradientDTV<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(p1_[dev],p2_[dev],d_dimgTV[dev],(long)(curr_slices+buffer_length*2-1), image_size[1],image_size[0],delta);
                        
                    }
                    
                    
                    
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        // no need to copy the 2 aux slices here
                        cudaStreamSynchronize(stream[dev*nStream_device]);
                        cudaMemcpyAsync(d_norm2[dev], d_dimgTV[dev]+buffer_pixels, image_size[0]*image_size[1]*curr_slices*sizeof(float), cudaMemcpyDeviceToDevice,stream[dev*nStream_device+1]);
                    }
                    
                    
                    // Compute the L2 norm of the gradient. For that, reduction is used.
                    //REDUCE
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        
                        size_t dimblockRed = MAXTHREADS;
                        size_t dimgridRed = (total_pixels + MAXTHREADS - 1) / MAXTHREADS;
                        
                        cudaStreamSynchronize(stream[dev*nStream_device+1]);
                        reduceNorm2 << <dimgridRed, dimblockRed, MAXTHREADS*sizeof(float),stream[dev*nStream_device]>> >(d_norm2[dev], d_norm2aux[dev], total_pixels);
                        
                    }
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        size_t dimblockRed = MAXTHREADS;
                        size_t dimgridRed = (total_pixels + MAXTHREADS - 1) / MAXTHREADS;

                        if (dimgridRed > 1) {
                            reduceSum << <1, dimblockRed, MAXTHREADS*sizeof(float),stream[dev*nStream_device] >> >(d_norm2aux[dev], d_norm2[dev], dimgridRed);
                            cudaStreamSynchronize(stream[dev*nStream_device]);
                            cudaMemcpyAsync(&sumnorm2[dev], d_norm2[dev], sizeof(float), cudaMemcpyDeviceToHost,stream[dev*nStream_device+1]);
                        }
                        else {
                            cudaStreamSynchronize(stream[dev*nStream_device]);
                            cudaMemcpyAsync(&sumnorm2[dev], d_norm2aux[dev], sizeof(float), cudaMemcpyDeviceToHost,stream[dev*nStream_device+1]);
                        }
                    }
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaDeviceSynchronize();
                     }
                    cudaCheckErrors("Reduction error");
                    
                    
                    // Accumulate the norm accross devices
                    sum_curr_spl=0;
                    // this is CPU code
                    for (dev = 0; dev < deviceCount; dev++){
                        sum_curr_spl+=sumnorm2[dev];
                    }
                    sum_curr_spl+=0.0000001f; // avoid division by zero
                    
                    // If we have more than one splits, lets use the result from prior calls
                    if(i>0 && splits>1){
                        // this is already stored:
                        //totalsum=totalsum_prev; 
                    }else{
                        totalsum=sum_curr_spl;
                    }
                    
                    
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        //NORMALIZE
                        //in a Tesla, maximum blocks =15 SM * 4 blocks/SM
                        divideArrayScalar  <<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(d_dimgTV[dev]+buffer_pixels,(float)sqrt(totalsum),total_pixels);
                        //MULTIPLY HYPERPARAMETER
                        multiplyArrayScalar<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(d_dimgTV[dev]+buffer_pixels,alpha,   total_pixels);
                    }
                     for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaDeviceSynchronize();
                     }
                    cudaCheckErrors("Scalar operations error");
                    
                    //SUBSTRACT GRADIENT
                    //////////////////////////////////////////////
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        
                        substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(d_image[dev]+buffer_pixels,d_dimgTV[dev]+buffer_pixels, total_pixels);
                    }
                }

                // Synchronize mathematics, make sure bounding pixels are correct
                 for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaDeviceSynchronize();
                     }
                
                if(splits==1){
                    for(dev=0; dev<deviceCount;dev++){
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        if (dev<deviceCount-1){
                            cudaSetDevice(gpuids[dev+1]);
                            cudaMemcpy(buffer, d_image[dev+1], buffer_pixels*sizeof(float), cudaMemcpyDeviceToHost);
                            cudaSetDevice(gpuids[dev]);
                            cudaMemcpy(d_image[dev]+total_pixels+buffer_pixels,buffer, buffer_pixels*sizeof(float), cudaMemcpyHostToDevice); 
                        }
                        cudaDeviceSynchronize();
                        if (dev>0){
                            cudaSetDevice(gpuids[dev-1]);
                            cudaMemcpyAsync(buffer, d_image[dev-1]+total_pixels+buffer_pixels, buffer_pixels*sizeof(float), cudaMemcpyDeviceToHost);
                            cudaSetDevice(gpuids[dev]);
                            cudaMemcpyAsync(d_image[dev],buffer, buffer_pixels*sizeof(float), cudaMemcpyHostToDevice);
                        }
                    }
                }else{
                    
                    // We need to take it out :(
                    for(dev=0; dev<deviceCount;dev++){
                        cudaSetDevice(gpuids[dev]);
                        
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        linear_idx_start=image_size[0]*image_size[1]*slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        cudaMemcpyAsync(&dst[linear_idx_start], d_image[dev]+buffer_pixels,total_pixels*sizeof(float), cudaMemcpyDeviceToHost,stream[dev*nStream_device+1]);
                    }
                }
                
                for (dev = 0; dev < deviceCount; dev++){
                    cudaSetDevice(gpuids[dev]);
                    cudaDeviceSynchronize();
                }
                cudaCheckErrors("Memory gather error");

                totalsum_prev+=sum_curr_spl;
            }
            totalsum=totalsum_prev;
        }
        // If there has not been splits, we still have data in memory
        if(splits==1){
            for(dev=0; dev<deviceCount;dev++){
                cudaSetDevice(gpuids[dev]);
                
                curr_slices=((dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*dev;
                total_pixels=curr_slices*image_size[0]*image_size[1];
                cudaMemcpy(dst+slices_per_split*image_size[0]*image_size[1]*dev, d_image[dev]+buffer_pixels,total_pixels*sizeof(float), cudaMemcpyDeviceToHost);
            }
        }
        cudaCheckErrors("Copy result back");
        
        for(dev=0; dev<deviceCount;dev++){
            cudaSetDevice(gpuids[dev]);
            cudaFree(d_image[dev]);
            cudaFree(d_norm2aux[dev]);
            cudaFree(d_dimgTV[dev]);
            cudaFree(d_norm2[dev]);
        }
        if (splits==1){
            cudaFreeHost(buffer);
        }
        
        if (isHostRegisterSupported& splits>2){
            cudaHostUnregister(img);
            cudaHostUnregister(dst);
        }
        for (int i = 0; i < nStreams; ++i)
           cudaStreamDestroy(stream[i]) ;
        cudaCheckErrors("Memory free");
//         cudaDeviceReset();
    }
        
void checkFreeMemory(const GpuIds& gpuids, size_t *mem_GPU_global){
        size_t memfree;
        size_t memtotal;
        const int deviceCount = gpuids.GetLength();
        for (int dev = 0; dev < deviceCount; dev++){
            cudaSetDevice(gpuids[dev]);
            cudaMemGetInfo(&memfree,&memtotal);
            if(dev==0) *mem_GPU_global=memfree;
            if(memfree<memtotal/2){
                mexErrMsgIdAndTxt("tvDenoise:tvdenoising:GPU","One (or more) of your GPUs is being heavily used by another program (possibly graphics-based).\n Free the GPU to run TIGRE\n");
            }
            cudaCheckErrors("Check mem error");
            
            *mem_GPU_global=(memfree<*mem_GPU_global)?memfree:*mem_GPU_global;
        }
        *mem_GPU_global=(size_t)((double)*mem_GPU_global*0.95);
        
        //*mem_GPU_global= insert your known number here, in bytes.
}
