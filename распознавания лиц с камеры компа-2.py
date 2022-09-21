#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Automatic processing of sizes and definition of boot images filtering
#Guys please review the code
#The basic principle is based on the GAN network
#argparse module
#Utils encapsulates common functions in interfaces
#The numpy module is used to handle matrix operations
#Tensorflow module creates model networks, training tests, etc.
#tqdm is a library that displays a loop progress bar


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
return image
def tf_box_filter(x, r):
    k_size = int(2*r+1)
    ch = x.get_shape().as_list()[-1]
    weight = 1/(k_size**2)
    box_kernel = weight*np.ones((k_size, k_size, ch, 1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    output = tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')
    return output
def guided_filter(x, y, r, eps=1e-2):
    x_shape = tf.shape(x)
    #y_shape = tf.shape(y)
    N = tf_box_filter(tf.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype), r)
    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x  = tf_box_filter(x * x, r) / N - mean_x * mean_x
    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x
    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N
    output = mean_A * x + mean_b
   return output
def fast_guided_filter(lr_x, lr_y, hr_x, r=1, eps=1e-8):
    #assert lr_x.shape.ndims == 4 and lr_y.shape.ndims == 4 and hr_x.shape.ndims == 4
    lr_x_shape = tf.shape(lr_x)
    #lr_y_shape = tf.shape(lr_y)
    hr_x_shape = tf.shape(hr_x)
    N = tf_box_filter(tf.ones((1, lr_x_shape[1], lr_x_shape[2], 1), dtype=lr_x.dtype), r)
    mean_x = tf_box_filter(lr_x, r) / N
    mean_y = tf_box_filter(lr_y, r) / N
    cov_xy = tf_box_filter(lr_x * lr_y, r) / N - mean_x * mean_y
    var_x  = tf_box_filter(lr_x * lr_x, r) / N - mean_x * mean_x
    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x
    mean_A = tf.image.resize_images(A, hr_x_shape[1: 3])
    mean_b = tf.image.resize_images(b, hr_x_shape[1: 3])
    output = mean_A * hr_x + mean_b
    return output


# In[4]:


def cartoonize(load_folder, save_folder, model_path):
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)
    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    name_list = os.listdir(load_folder)
    for name in tqdm(name_list):
        try:
            load_path = os.path.join(load_folder, name)
            save_path = os.path.join(save_folder, name)
            image = cv2.imread(load_path)
            image = resize_crop(image)
            batch_image = image.astype(np.float32)/127.5 - 1
            batch_image = np.expand_dims(batch_image, axis=0)
            output = sess.run(final_out, feed_dict={input_photo: batch_image})
            output = (np.squeeze(output)+1)*127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            cv2.imwrite(save_path, output)
        except:
            print('cartoonize {} failed'.format(load_path))


# In[5]:


model_path = 'saved_models'
    load_folder = 'test_images'
    save_folder = 'cartoonized_images'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
cartoonize(load_folder, save_folder, model_path)    


# In[ ]:





# In[ ]:




