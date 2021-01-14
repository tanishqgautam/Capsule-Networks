#### Pytorch implementation of Capsule Network architecture from paper ["Dynamic Routing Between Capsules"](https://arxiv.org/abs/1710.09829)   
## Overview   
Let’s begin with the human visual recognition where when we see an object (any real object is made up of smaller objects) our eyes make some fixation points and the relative positions of these fixation points helps our brain in recognizing that object. In this way, our brain does not need to process every minute detail, it can combine hierarchical information together to help identify it.   

The assumption behind Capsule Networks or simply put Capsnet is that there are capsules (a groups of neurons) that tell whether certain objects are present in an image. Corresponding to that object, there is a capsule which gives us the probability that the entity exists along with the instantiation parameters of that entity.
Therefore in a simple definition, a CapsNet takes input an image and finds what kind of object is present in it and what are the instantiation parameters(rotation, thickness, etc).   

#### Check out my article [A Deep Dive into Capsule Networks](https://medium.com/analytics-vidhya/a-deep-dive-into-capsule-networks-dad85d3eed2b) on Medium

