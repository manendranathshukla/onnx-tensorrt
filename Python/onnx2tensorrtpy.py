import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda 
import time
import sys



TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
def build_engine(model_path):
    with trt.Builder(TRT_LOGGER) as builder,builder.create_network(EXPLICIT_BATCH) as network,trt.OnnxParser(network, TRT_LOGGER) as parser: 
        builder.max_workspace_size = 1<<20
        builder.max_batch_size = 1
        with open(model_path, "rb") as f:
            parser.parse(f.read())
        engine = builder.build_cuda_engine(network)
    return engine
def inference(engine, context, inputs,h_input, h_output, d_input, d_output,stream):
    cuda.memcpy_htod_async(d_input, h_input, stream)
	  # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
		# Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    return h_output

def alloc_buf(engine):
    
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream=cuda.Stream()

    return h_input, h_output, d_input, d_output, stream
if __name__ == "__main__":
    model_path = sys.argv[1] #"/content/gdrive/My Drive/UNIT_Working/onnx-tensorrt-master/result.onnx"
    input_size = int(sys.argv[2])
    output_file=sys.argv[3]
    mode=sys.argv[4]
    if(mode==32):
        inputs = np.random.random((1, 3, input_size, input_size)).astype(np.float32)
        engine = build_engine(model_path)
        print("Engine Created :",type(engine))
        context = engine.create_execution_context()
        print("Context executed ",type(context))
        serialized_engine = engine.serialize()
        t1 = time.time()
        #in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
        h_input, h_output, d_input, d_output,stream=alloc_buf(engine)
        res = inference(engine, context, inputs.reshape(-1), h_input, h_output, d_input, d_output, stream)
        #print(type(res))
        #print("using fp32 mode:")
        #print("cost time: ", time.time()-t1)
        #engine_path="FLtask.trt"
        with open(output_file,"wb") as f:
          f.write(serialized_engine)
          print("model converted successfully!")
    if(mode==16):
        inputs = np.random.random((1, 3, input_size, input_size)).astype(np.float16)
        engine = build_engine(model_path)
        print("Engine Created :",type(engine))
        context = engine.create_execution_context()
        print("Context executed ",type(context))
        serialized_engine = engine.serialize()
        t1 = time.time()
        #in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
        h_input, h_output, d_input, d_output,stream=alloc_buf(engine)
        res = inference(engine, context, inputs.reshape(-1), h_input, h_output, d_input, d_output, stream)
        #print(type(res))
        #print("using fp16 mode:")
        #print("cost time: ", time.time()-t1)
        with open(output_file,"wb") as f:
          f.write(serialized_engine)
          print("model converted successfully!")
    if(mode==8):
        inputs = np.random.random((1, 3, input_size, input_size)).astype(np.int8)
        engine = build_engine(model_path)
        print("Engine Created :",type(engine))
        context = engine.create_execution_context()
        print("Context executed ",type(context))
        serialized_engine = engine.serialize()
        t1 = time.time()
        #in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
        h_input, h_output, d_input, d_output,stream=alloc_buf(engine)
        res = inference(engine, context, inputs.reshape(-1), h_input, h_output, d_input, d_output, stream)
        #print(type(res))
        #print("using int8 mode:")
        #print("cost time: ", time.time()-t1)
        with open(output_file,"wb") as f:
          f.write(serialized_engine)
          print("model converted successfully!")
    
