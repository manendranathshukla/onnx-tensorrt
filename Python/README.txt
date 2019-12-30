Using python you can convert any onnx model to a tensorrt engine or model easily.

Steps to be followed:

1-> First install the Tensorrt7.0.0.11 by downloading from Nvidia official site link is here:
  https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.0/7.0.0.11/tars/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz
  
  
2-> Extract the downloaded file and Install the following library from the downloaded Tensorrt7:

  pip3 install tensorrt-7.0.0.11-cp36-none-linux_x86_64.whl
  pip3 install uff-0.6.5-py2.py3-none-any.whl
  pip3 install graphsurgeon-0.4.1-py2.py3-none-any.whl
  sudo apt-get install -y --no-install-recommends libnvinfer5=5.1.2-1+cuda10.0
  sudo apt-get install -y --no-install-recommends libnvinfer-dev=5.1.2-1+cuda10.0
  sudo apt-get install python3-libnvinfer-dev
  
3-> Install the following required library:
  pip install pycuda
  pip install torchfile
  pip install onnx==1.6.0
  
4-> How to use the script?
  First download the script and then execute it.
  
  python onnx2tensorrtpy.py input_onnx_model_name.onnx input_size output_model_name.trt mode_value
  
  i.e. mode_value=32 or 16 or 8
  32 means fp32, 16 means fp16, 8 means Int8
  these 3 inference modes are available. 
  
  for example:
  
  python onnx2tensorrtpy.py result.onnx 256 mymodel.trt 16
  
  
