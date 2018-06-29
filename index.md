# Semantic Segmentation for Road Detection
In this project, we will be developing a Fully Convolutional Network (FCN) based on VGG16, as described in [this paper](fcn.pdf). Specifically, at the core of this design, we use VGG as an encoder, and then upsample the encoded features using transpose convolutions and skip layers from the VGG network. For the VGG portion, we use the [pretrained VGG network provided by Udacity](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip). Testing and training images are obtained from the  [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) available [here](http://www.cvlibs.net/download.php?file=data_road.zip). 

To train your own FCN, you simply run `main.py`:
```
python main.py
```
In this file, the main code that does the training is called `run`. At the top of that code snippet, you will find several options: 

- `batch_size`: the number of images passed to the optimizer during each training iteration
- `learning_rate_val`: the learning rate given to the `AdamOptimizer` 
- `num_classes`: the number of classes used for semantic segmentation
- `epochs`: the number of training epochs, that is, how many times the optimizer will see each image during training
- `keep_prob_val`: the "keep" probability for the dropout layers in VGG to be used when training

## Discussion

- I tried training the aforementioned FCN on a GTX 1080. Due to the memory bandwidth, I could only use batch sizes up to 4 for training. In the end however, I seemed to be capable of achieving the best possible performance with batches of 2. For larger batches, my GPU would randomly crash. So this choice was partially made for me simply by the hardware at my disposal. 
- The learning rate appeared to be fairly arbitrary. As long as you don't set this value to something unreasonably large, the FCN would eventually converge to a stable solution. A value of 1e-3 seems like a reasonable choice with which I was getting decent results.
- In our dataset, the primary road is labelled as magenta, any secondary roads are labelled black, and the background is red. If you set `num_classes` to 3, then the FCN will try to discern all 3 of these classes. If you set this value to 2, then the FCN will only look for the primary road and the background. I did not notice any phenomenal difference in prediction accuracy when trying to pull off all 3 classes, so I did that. 
- I chose to optimize over 10 epochs. I played around with a variety of other settings, and 10 epochs always seemed to be enough for the FCN to converge to a solution. running for more than 10 epochs did not seem to improve the performance for any of the hyperparameters that I tried. 
- I attempted to use keep probabilities of 0.2, 0.5, and 0.8. The lowest keep probability seemed to produce the best validation results. Therefore the results below use a keep probability of 0.2.
- Many people suggested that using regularization was important for getting the FCN to converge. I did not find this to be true. To the contrary, I observed that using a kernel regularizer hurt performance. Specifically, I found that the cost would plateau at a relatively high value, and achieve poor performance when using a kernel regularizer. If you want to try for yourself, you can uncomment the kernel regularizer option in `main.layers`. 

 ### More background comments
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information. 

- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 

- To get a better idea of what the VGG graph looks like, we have created a Jupyter notebook `visualize_vgg_graph.ipynb`. You may need to adjust the `vgg_path` in the final cell. 

  
## Results

There are many ways to understand whether the FCN is converging during training. Typically, you would do this by plotting and monitoring the training and validation loss functions. However, in our case, I found that the most constructive way was to simply view the prediction road pixels on a test image while training. For instance, the following video shows the predicted road pixels on a random test image while training: 

<video controls="controls" width=100%>
  <source type="video/mp4" src="videos/2.2_10_3.mp4"></source>
</video>

After 10 epochs, we can see that the FCN is converged. Therefore, to get an idea of how well it performed, we can show the predicted road pixels on all of our test images:

<video controls="controls" width=100%>
  <source type="video/mp4" src="images/2.2_10_3.mp4"></source>
</video>

If you want to see all of the predicted frames, you can simply navigate to the `images` directory. 