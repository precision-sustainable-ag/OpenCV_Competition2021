# CNN complete pipeline:

```mermaid
graph TD
A(Annotated Structures) --> X((Circle))
A --> Y[Synthetic Bench Images] --> Z
X[Synthetic Natural Images] --> Z
Z(DeepLabV3+ Tensorflow Training) --> M[Small Model]
Z --> L[Large Model]
IR(Intermediate Representation)
M -- FP16 --> IR 
L -- FP16 --> IR 
IR --> D{Best Performance?}
D --> O[Model Compiler - blob]
```
---

In this section you could find the next: 
1- DeepLabV3+ Modeling process
2- Model Opitmizer or Intermediate Representation Process
3- Script to measure the model performance in different instances.

---

For more details visit our [wiki](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/wiki/6.-CNN-Model-Training) page 

