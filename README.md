# Medical-Image-Synthesis
It describes MRI to CT conversion, 3T to 7T conversion using Generative Adversarial Network (GAN).

Due to multiple considerations such as cost and radiation dose, the acquisition of certain image modalities may be limited. Thus, medical image synthesis can be of great benefit by estimating a desired imaging modality without incurring an actual scan.

Here proposed a generative adversarial approach to address this challenging problem. Specifically,trained a fully convolutional network (FCN) to generate a target image given a source image. To better model a nonlinear mapping from source to target and to produce more realistic target images, proposed to use the adversarial learning strategy to better model the FCN.

Moreover, the FCN is designed to incorporate an image-gradient-difference-based loss function to avoid generating blurry target images

Evaluated this method on three datasets, to address the tasks of generating CT from MRI and generating 7T MRI from 3T MRI images.

# DATA SET
BTW, you can download a real medical image synthesis dataset for reconstructing standard-dose PET from low-dose PET via this link: https://www.aapm.org/GrandChallenge/LowDoseCT/

Also, there are some MRI synthesis datasets available: http://brain-development.org/ixi-dataset/

