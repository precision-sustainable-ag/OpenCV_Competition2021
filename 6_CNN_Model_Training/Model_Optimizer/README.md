# Model optimizer

For running the model optimizer use the [Jupyter notebook](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/blob/master/6_CNN_Model_Training/Model_Optimizer/Optimizer_Benchmark-BenchBot.ipynb) with Python 3 (OpenVINO 2020.3.2 LTS) environment.

![](https://lh4.googleusercontent.com/q8GmA_jSUmxk7_1zc9UXgb_u-RI1a6ZxpCYGPrAnYCzDAyOJOZYJR81vieTYIFYF6K2qBvlyvEe3QJYWJYE6rrc1HhcW2ZMf12PY1712d9cPVloyHt_tfsooWDZWm2nPFcnSZfNb)

---

# Model compiler

For creating blob model file to load into Myridax VPU use this [Luxonis Tool](http://luxonis.com:8080/) with this settings.

![](https://lh3.googleusercontent.com/OJkuXAX2JJnbfEhFosDeKJbp8BiQPK2ZzWIgf25pkWO3WjzraQg8OvNpvlc8coqsVlPqWV9eIMBhyPi6OulKhPyk0K6JLvtziXi4ehdsBrBIw6L2rw426RnAKBr7Eg-_q9ih1S8k)

> OpenVINO Version: 2021.3
>> Model source: OpenVINO Model
>> 
>> Upload .xml and .bin files using the interface
>> 
>> MyriadX compile params: `-ip U8 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4`
>> 
>> Download the blob file and save it ito the RPi


