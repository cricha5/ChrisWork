import numpy
import math
import time
import pyopencl as cl
import pyopencl.array as clarray

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'


kernel_source = """
    __kernel void nextStep(__global float *vec_V,__global const float *vec_Constants_dev, const int test)
    {
    //float dS = vec_Constants_dev[0];
    //float dt = vec_Constants_dev[1];
    //float Int_Rate = vec_Constants_dev[2];
    //float Vol = vec_Constants_dev[3];
    int nTimeSteps = (int)vec_Constants_dev[4];
    int NASSETSTEPS = (int)vec_Constants_dev[5];
    
    //float tst = 5;
    //float x = vec_V[0,get_global_id(0)];
    
    //vec_V[0,get_global_id(0)] = x+1.f;
    //vec_V[get_global_id(0)] = vec_Constants_dev[get_global_id(0)%4];
    
    __local int t;
    int global_id = get_global_id(0);
    
    if (global_id == 0) {
        t = 0;
    }
    barrier (CLK_LOCAL_MEM_FENCE);

    while(t < nTimeSteps){
        vec_V[(t*NASSETSTEPS)+global_id] = test;
        
        //atomic_add(&t,1);
        if (global_id == 0) {
            t += 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        
    }
    }
    """

RATE = 0.05
PTYPE = 1.0
STRIKE = 100.0
EXPIRATION = 1.0
NASSETSTEPS = 20
VOLATILITY = 0.2

dS = 2.0 * STRIKE / NASSETSTEPS

dt = 0.9 / VOLATILITY**2 / NASSETSTEPS**2

nTimeSteps = math.ceil(EXPIRATION / dt)

#dt = EXPIRATION / nTimeSteps

platforms = cl.get_platforms()
for platform in platforms:
    for device in platform.get_devices():
        print(device)
        print("local memory size",device.local_mem_size)
        print("global memory size",device.global_mem_size)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

vec_V = numpy.zeros((nTimeSteps,NASSETSTEPS),numpy.float32)
vec_Constants = numpy.array([dS,dt,RATE,VOLATILITY,nTimeSteps,NASSETSTEPS],numpy.float32)

vec_V_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vec_V)
vec_Constants_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vec_Constants)

testint = numpy.int32(3)
#testint_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=testint)

for i in range(NASSETSTEPS):
    vec_V[0,i] = max(PTYPE * (i*dS - STRIKE),0.0)

print(vec_V)

#cl.enqueue_write_buffer(queue, vec_V_dev)
#cl.enqueue_write_buffer(queue, vec_Constants_dev, vec_Constants)

prg = cl.Program(ctx, kernel_source).build()
print(vec_V[0].shape)
print(vec_Constants)



prg.nextStep(queue, vec_V[0].shape, None, vec_V_dev,vec_Constants_dev, testint)

result = numpy.zeros_like(vec_V)
#cl.enqueue_read_buffer(queue, vec_V_dev, result ).wait()

cl.enqueue_copy(queue, result, vec_V_dev).wait()

print(result)
print(nTimeSteps)
