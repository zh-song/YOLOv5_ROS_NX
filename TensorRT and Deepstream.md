# TensorRT 
## Create Engine file by these steps as follows
![image](https://user-images.githubusercontent.com/67272893/196578999-529ee9ee-6919-4487-820f-f70ef6f82dc4.png)

## pt 2 wts 2 engine
Tensorrtx yolov5.cpp wts $\rightarrow$ engine
1. create modelstream and allocate memory for it.
```
    IHostMemory* modelStream{ nullptr };
```
2. create engine and then serialize it into modelstream.
```
    (*modelStream) = engine->serialize();
```

 ```
    How create network and then build engine
    
    // 1. create network with INetworkDefinition* type through builder
    INetworkDefinition* network = builder->createNetworkV2(0U);
     
    // 2. Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    
    // 3. load weight to create each layer of network
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    
    // 4. add layers of network 
    
    // 5. set BatchSize and Workspace memory
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
     
     
 ```
 3. use ofstream variable p to save modelstream to device, which can avoid create and build engine for each time during running.

```
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
```
4. read engine_file by binary mode and write this binary engine to value trtModelStream with char type
```
    // read engine_file by function from ifstream class
    std::ifstream file(engine_name, std::ios::binary);

    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    
    // save content to trtModelStream with size memory
    file.read(trtModelStream, size);
    file.close();

```
5. read image file

```
read_files_in_dir(img_dir.c_str(), file_names)
```
6. deserialize the content in trtModelStream to create engine, create context to be ready to inference
```
    // predicted and size of it
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    
    // deserialize
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    
    // create context
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
```
7. create input and output 
```
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
```
8. move data to memory
```
    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
```
9. read image
```
    cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
    
    imgs_buffer[b] = img;
    size_t  size_image = img.cols * img.rows * 3;
    size_t  size_image_dst = INPUT_H * INPUT_W * 3;
    
    // copy data with size_image from img to img_host
    memcpy(img_host,img.data,size_image);
    
    // copy data from host to device memory
    CUDA_CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
    
    // preprocess for image
    preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);       
    buffer_idx += size_image_dst;
```
10. inference
```
    // start clock
    auto start = std::chrono::system_clock::now();
    // inference
    doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
    // end clock
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

```
```
    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize) {
	    // infer on the batch asynchronously, and DMA output back to host

	    // TensorRT execution is typically asynchronous, so enqueue the kernels on CUDA stream
	    // input: buffer[0], output: buffer[1]
	    context.enqueue(batchSize, buffers, stream, nullptr);

	    // move results(buffer[1]) from device to host in value output
	    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	    cudaStreamSynchronize(stream);
    }

```
11. NMS
```
    // conduct NMS to prob and save output of nms to res
    nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
```
12. rectengle predict bbox and draw and write
```
    for (int b = 0; b < fcount; b++) {
	    auto& res = batch_res[b];
	    cv::Mat img = imgs_buffer[b];
	    for (size_t j = 0; j < res.size(); j++) {

		    // [c_x,c_y,w,h] to [x1,y1,w,h]
		    cv::Rect r = get_rect(img, res[j].bbox);

		    // draw box and text
		    cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
		    cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
	    }
	    cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
    }
```



# Deepstream
it is the framework including capture input, pre-processing, detect or tracking, post-processing, release the results to several different device e.g. cloud.

![image](https://user-images.githubusercontent.com/67272893/198487240-23adafad-7d8b-43ca-8e99-c05963bd4fa6.png)

 TensorRT is the part of detect stage.

![image](https://user-images.githubusercontent.com/67272893/198488438-955d3ee2-7dbc-43d2-adbd-8ed8cbfae6d9.png)
## Capture
*CIS camera > USB camera > Web camera*

1. CIS camera: 需要专用的高速接口，带宽大，传图清晰度高，延迟低，CPU专用率低，可支持底层访问和控制
2. USB camera: 占用CPU使用率，灵活，USB带宽会影响传输，以及压缩编解码限制效率
3. Web camera: 灵活，但受网络带宽影响，压缩率等因素延迟大，图像质量较差

## Output

```
zhsong@ubuntu:~/code/DeepStream-Yolo$ deepstream-app -c deepstream_app_config.txt

Using winsys: x11 
WARNING: Deserialize engine failed because file path: /home/zhsong/code/DeepStream-Yolo/model_b1_gpu0_fp32.engine open error
0:00:03.155849537 44335 0xaaab006e8730 WARN                 nvinfer gstnvinfer.cpp:643:gst_nvinfer_logger:<primary_gie> NvDsInferContext[UID 1]: Warning from NvDsInferContextImpl::deserializeEngineAndBackend() <nvdsinfer_context_impl.cpp:1897> [UID = 1]: deserialize engine from file :/home/zhsong/code/DeepStream-Yolo/model_b1_gpu0_fp32.engine failed
0:00:03.208012177 44335 0xaaab006e8730 WARN                 nvinfer gstnvinfer.cpp:643:gst_nvinfer_logger:<primary_gie> NvDsInferContext[UID 1]: Warning from NvDsInferContextImpl::generateBackendContext() <nvdsinfer_context_impl.cpp:2002> [UID = 1]: deserialize backend context from engine from file :/home/zhsong/code/DeepStream-Yolo/model_b1_gpu0_fp32.engine failed, try rebuild
0:00:03.208114678 44335 0xaaab006e8730 INFO                 nvinfer gstnvinfer.cpp:646:gst_nvinfer_logger:<primary_gie> NvDsInferContext[UID 1]: Info from NvDsInferContextImpl::buildModel() <nvdsinfer_context_impl.cpp:1923> [UID = 1]: Trying to create engine from model files
WARNING: [TRT]: The implicit batch dimension mode has been deprecated. Please create the network with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag whenever possible.

Loading pre-trained weights
Loading weights of yolov5s complete
Total weights read: 7254397
Building YOLO network

        Layer                         Input Shape         Output Shape        WeightPtr
(0)     conv_silu                     [3, 640, 640]       [32, 320, 320]      3584
(1)     conv_silu                     [32, 320, 320]      [64, 160, 160]      22272
(2)     conv_silu                     [64, 160, 160]      [32, 160, 160]      24448
(3)     route: 1                      -                   [64, 160, 160]      -
(4)     conv_silu                     [64, 160, 160]      [32, 160, 160]      26624
(5)     conv_silu                     [32, 160, 160]      [32, 160, 160]      27776
(6)     conv_silu                     [32, 160, 160]      [32, 160, 160]      37120
(7)     shortcut_add_linear: 4        [32, 160, 160]      [32, 160, 160]      -
(8)     route: 7, 2                   -                   [64, 160, 160]      -
(9)     conv_silu                     [64, 160, 160]      [64, 160, 160]      41472
(10)    conv_silu                     [64, 160, 160]      [128, 80, 80]       115712
(11)    conv_silu                     [128, 80, 80]       [64, 80, 80]        124160
(12)    route: 10                     -                   [128, 80, 80]       -
(13)    conv_silu                     [128, 80, 80]       [64, 80, 80]        132608
(14)    conv_silu                     [64, 80, 80]        [64, 80, 80]        136960
(15)    conv_silu                     [64, 80, 80]        [64, 80, 80]        174080
(16)    shortcut_add_linear: 13       [64, 80, 80]        [64, 80, 80]        -
(17)    conv_silu                     [64, 80, 80]        [64, 80, 80]        178432
(18)    conv_silu                     [64, 80, 80]        [64, 80, 80]        215552
(19)    shortcut_add_linear: 16       [64, 80, 80]        [64, 80, 80]        -
(20)    route: 19, 11                 -                   [128, 80, 80]       -
(21)    conv_silu                     [128, 80, 80]       [128, 80, 80]       232448
(22)    conv_silu                     [128, 80, 80]       [256, 40, 40]       528384
(23)    conv_silu                     [256, 40, 40]       [128, 40, 40]       561664
(24)    route: 22                     -                   [256, 40, 40]       -
(25)    conv_silu                     [256, 40, 40]       [128, 40, 40]       594944
(26)    conv_silu                     [128, 40, 40]       [128, 40, 40]       611840
(27)    conv_silu                     [128, 40, 40]       [128, 40, 40]       759808
(28)    shortcut_add_linear: 25       [128, 40, 40]       [128, 40, 40]       -
(29)    conv_silu                     [128, 40, 40]       [128, 40, 40]       776704
(30)    conv_silu                     [128, 40, 40]       [128, 40, 40]       924672
(31)    shortcut_add_linear: 28       [128, 40, 40]       [128, 40, 40]       -
(32)    conv_silu                     [128, 40, 40]       [128, 40, 40]       941568
(33)    conv_silu                     [128, 40, 40]       [128, 40, 40]       1089536
(34)    shortcut_add_linear: 31       [128, 40, 40]       [128, 40, 40]       -
(35)    route: 34, 23                 -                   [256, 40, 40]       -
(36)    conv_silu                     [256, 40, 40]       [256, 40, 40]       1156096
(37)    conv_silu                     [256, 40, 40]       [512, 20, 20]       2337792
(38)    conv_silu                     [512, 20, 20]       [256, 20, 20]       2469888
(39)    route: 37                     -                   [512, 20, 20]       -
(40)    conv_silu                     [512, 20, 20]       [256, 20, 20]       2601984
(41)    conv_silu                     [256, 20, 20]       [256, 20, 20]       2668544
(42)    conv_silu                     [256, 20, 20]       [256, 20, 20]       3259392
(43)    shortcut_add_linear: 40       [256, 20, 20]       [256, 20, 20]       -
(44)    route: 43, 38                 -                   [512, 20, 20]       -
(45)    conv_silu                     [512, 20, 20]       [512, 20, 20]       3523584
(46)    conv_silu                     [512, 20, 20]       [256, 20, 20]       3655680
(47)    maxpool                       [256, 20, 20]       [256, 20, 20]       -
(48)    maxpool                       [256, 20, 20]       [256, 20, 20]       -
(49)    maxpool                       [256, 20, 20]       [256, 20, 20]       -
(50)    route: 46, 47, 48, 49         -                   [1024, 20, 20]      -
(51)    conv_silu                     [1024, 20, 20]      [512, 20, 20]       4182016
(52)    conv_silu                     [512, 20, 20]       [256, 20, 20]       4314112
(53)    upsample                      [256, 20, 20]       [256, 40, 40]       -
(54)    route: 53, 36                 -                   [512, 40, 40]       -
(55)    conv_silu                     [512, 40, 40]       [128, 40, 40]       4380160
(56)    route: 54                     -                   [512, 40, 40]       -
(57)    conv_silu                     [512, 40, 40]       [128, 40, 40]       4446208
(58)    conv_silu                     [128, 40, 40]       [128, 40, 40]       4463104
(59)    conv_silu                     [128, 40, 40]       [128, 40, 40]       4611072
(60)    route: 59, 55                 -                   [256, 40, 40]       -
(61)    conv_silu                     [256, 40, 40]       [256, 40, 40]       4677632
(62)    conv_silu                     [256, 40, 40]       [128, 40, 40]       4710912
(63)    upsample                      [128, 40, 40]       [128, 80, 80]       -
(64)    route: 63, 21                 -                   [256, 80, 80]       -
(65)    conv_silu                     [256, 80, 80]       [64, 80, 80]        4727552
(66)    route: 64                     -                   [256, 80, 80]       -
(67)    conv_silu                     [256, 80, 80]       [64, 80, 80]        4744192
(68)    conv_silu                     [64, 80, 80]        [64, 80, 80]        4748544
(69)    conv_silu                     [64, 80, 80]        [64, 80, 80]        4785664
(70)    route: 69, 65                 -                   [128, 80, 80]       -
(71)    conv_silu                     [128, 80, 80]       [128, 80, 80]       4802560
(72)    conv_silu                     [128, 80, 80]       [128, 40, 40]       4950528
(73)    route: 72, 62                 -                   [256, 40, 40]       -
(74)    conv_silu                     [256, 40, 40]       [128, 40, 40]       4983808
(75)    route: 73                     -                   [256, 40, 40]       -
(76)    conv_silu                     [256, 40, 40]       [128, 40, 40]       5017088
(77)    conv_silu                     [128, 40, 40]       [128, 40, 40]       5033984
(78)    conv_silu                     [128, 40, 40]       [128, 40, 40]       5181952
(79)    route: 78, 74                 -                   [256, 40, 40]       -
(80)    conv_silu                     [256, 40, 40]       [256, 40, 40]       5248512
(81)    conv_silu                     [256, 40, 40]       [256, 20, 20]       5839360
(82)    route: 81, 52                 -                   [512, 20, 20]       -
(83)    conv_silu                     [512, 20, 20]       [256, 20, 20]       5971456
(84)    route: 82                     -                   [512, 20, 20]       -
(85)    conv_silu                     [512, 20, 20]       [256, 20, 20]       6103552
(86)    conv_silu                     [256, 20, 20]       [256, 20, 20]       6170112
(87)    conv_silu                     [256, 20, 20]       [256, 20, 20]       6760960
(88)    route: 87, 83                 -                   [512, 20, 20]       -
(89)    conv_silu                     [512, 20, 20]       [512, 20, 20]       7025152
(90)    route: 71                     -                   [128, 80, 80]       -
(91)    conv_logistic                 [128, 80, 80]       [255, 80, 80]       7058047
(92)    yolo                          [255, 80, 80]       -                   -
(93)    route: 80                     -                   [256, 40, 40]       -
(94)    conv_logistic                 [256, 40, 40]       [255, 40, 40]       7123582
(95)    yolo                          [255, 40, 40]       -                   -
(96)    route: 89                     -                   [512, 20, 20]       -
(97)    conv_logistic                 [512, 20, 20]       [255, 20, 20]       7254397
(98)    yolo                          [255, 20, 20]       -                   -

Output YOLO blob names: 
yolo_93
yolo_96
yolo_99

Total number of YOLO layers: 260

Building YOLO network complete
Building the TensorRT Engine

NOTE: letter_box is set in cfg file, make sure to set maintain-aspect-ratio=1 in config_infer file to get better accuracy

Building complete

0:02:32.545228485 44335 0xaaab006e8730 INFO                 nvinfer gstnvinfer.cpp:646:gst_nvinfer_logger:<primary_gie> NvDsInferContext[UID 1]: Info from NvDsInferContextImpl::buildModel() <nvdsinfer_context_impl.cpp:1955> [UID = 1]: serialize cuda engine to file: /home/zhsong/code/DeepStream-Yolo/model_b1_gpu0_fp32.engine successfully
INFO: [Implicit Engine Info]: layers num: 5
0   INPUT  kFLOAT data            3x640x640       
1   OUTPUT kFLOAT num_detections  1               
2   OUTPUT kFLOAT detection_boxes 25200x4         
3   OUTPUT kFLOAT detection_scores 25200           
4   OUTPUT kFLOAT detection_classes 25200           

0:02:32.607943788 44335 0xaaab006e8730 INFO                 nvinfer gstnvinfer_impl.cpp:328:notifyLoadModelStatus:<primary_gie> [UID 1]: Load new model:/home/zhsong/code/DeepStream-Yolo/config_infer_primary_yoloV5.txt sucessfully

Runtime commands:
	h: Print this help
	q: Quit

	p: Pause
	r: Resume

NOTE: To expand a source in the 2D tiled display and view object details, left-click on the source.
      To go back to the tiled display, right-click anywhere on the window.


**PERF:  FPS 0 (Avg)	
**PERF:  0.00 (0.00)	
** INFO: <bus_callback:194>: Pipeline ready

Opening in BLOCKING MODE 
NvMMLiteOpen : Block : BlockType = 261 
NVMEDIA: Reading vendor.tegra.display-size : status: 6 
NvMMLiteBlockCreate : Block : BlockType = 261 
** INFO: <bus_callback:180>: Pipeline running

**PERF:  50.07 (49.94)	
**PERF:  50.69 (50.31)	
**PERF:  50.79 (50.55)	
**PERF:  50.68 (50.56)	
**PERF:  50.75 (50.61)	
**PERF:  50.77 (50.64)	
**PERF:  50.76 (50.64)	
**PERF:  50.60 (50.66)	
**PERF:  50.84 (50.67)	
**PERF:  50.85 (50.68)	
**PERF:  50.74 (50.69)	
**PERF:  50.79 (50.70)	
**PERF:  50.89 (50.71)	
**PERF:  50.80 (50.72)	
**PERF:  50.67 (50.72)	
** INFO: <bus_callback:217>: Received EOS. Exiting ...

Quitting
App run successful
```
