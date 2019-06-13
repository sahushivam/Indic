from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
import os
import re
import numpy as np


import numpy.random as rd
import tensorflow as tf
import numpy as np
import multiprocessing #for parallel read of TFRecord files and other CPU based parallel processing tasks


from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPooling2D
from tensorflow import estimator
from tensorflow.python.keras.optimizers import RMSprop, Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.regularizers import l2


# In[97]:

import multiprocessing as mp
num_of_cores = mp.cpu_count()
print(num_of_cores)

# Write in tensorflow 

# In[98]:




# In[99]:


#Distributed Training with Cloud ML Engine
import os

PROJECT = 'indic2019' # REPLACE WITH YOUR PROJECT ID
BUCKET = 'indic2019' # REPLACE WITH YOUR BUCKET NAME
REGION = 'us-central1' # REPLACE WITH YOUR BUCKET REGION e.g. us-central1

os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION


# Defining the variables

# In[159]:


batch_size = 10
nb_boxes=1
grid_w=2
grid_h=2
cell_w=14
cell_h=14
img_w=28
img_h=28
img_channels = 1
input_shape = (None, img_w, img_h, img_channels)


# In[129]:


class FLAGS():
  pass


FLAGS.batch_size = 200
FLAGS.max_steps = 1000
FLAGS.eval_steps = 100
FLAGS.save_checkpoints_steps = 100
FLAGS.tf_random_seed = 19851211
FLAGS.model_name = 'cnn-model-02'
FLAGS.use_checkpoint = False


# In[160]:


model_dir = 'gs://indic2019/model_dir/run'


# Extract function takes data record and return images and data

# In[161]:


def extract_fn(data_record):
    features = {
      'image': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([5], tf.float32)
    }
    data = tf.parse_single_example(data_record, features)
    img1 = tf.decode_raw(data['image'], tf.float32)
    img1 = tf.reshape(img1, (img_w, img_h, img_channels)) #reshape to 28 x 28 x 1
    img1 = tf.image.per_image_standardization(img1) #Standardizing
    return img1, data['label']


# In[162]:


files_pattern = 'gs://indic2019/indic2019/*.tfrecord' #file pattern for tf record
test_pattern = 'gs://indic2019/indic2019/*.tfrecord' #file pattern for testing it on the test pattern


# <h3> Input function</h3>
#   <ul>
#     <li>Input parameters are files pattern(for taking data from the input), batch_size and mode(predict,eval and train)</li>
#   <li>Returns features and label in case of eval and train and features in case of predict</li>
#     </ul>
# 

# In[163]:


def input_fn(files_pattern,batch_size, mode):
  print(batch_size)
  files = tf.data.Dataset.list_files(files_pattern, shuffle=True)
  dataset=tf.data.TFRecordDataset(filenames=files)
  #dataset = files.apply(tf.contrib.data.parallel_interleave( lambda filename: tf.data.TFRecordDataset(filename), cycle_length=num_of_cores)) 
  #T parallel_interleave-HIS FUNCTION IS DEPRECATED. (not exactly know)
    
  #three variables for three mode
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  is_eval = (mode == tf.estimator.ModeKeys.EVAL)
  is_predict = (mode== tf.estimator.ModeKeys.PREDICT)
  
  buffer_size = batch_size * 2 + 1
  dataset = dataset.shuffle(buffer_size=buffer_size)

  # Transformation
  dataset = dataset.map(extract_fn)
  
  if is_training or is_predict:
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2 * batch_size)
    
  if is_eval:
    buffer_size = batch_size * 10
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10 * batch_size)

  image, label = dataset.make_one_shot_iterator().get_next()
  features = {'images': image}
  
  if is_training or is_eval:
    return features, label
  
  if is_predict:
    return features 


# <h3> Custom loss </h3>

# In[164]:



def custom_loss(labels,logits):
  
  print(logits)
  
  true_confidence = labels[:,0]
  true_x=labels[:,1]
  true_y=labels[:,2]
  true_w=labels[:,3]
  true_h=labels[:,4]

  predict_confidence=logits[:,0]
  predict_x=logits[:,1]
  predict_y=logits[:,2]
  predict_w=logits[:,3]
  predict_h=logits[:,4]	

  xy_loss= K.square(true_x-predict_x) + K.square(true_y-predict_y)
  wh_loss= K.square(K.sqrt(true_w)-K.sqrt(predict_w))+ K.square(K.sqrt(true_h)-K.sqrt(predict_h))

  con_loss=K.square(true_confidence-predict_confidence)

  loss= xy_loss + wh_loss + con_loss
  return tf.reduce_mean(loss)


# <h3>Bounding Box</h3>
# Returns coordinates of bounding box

# In[165]:


#Shape of y : conf x y w h

def convert_to_coord(y):
  coord= []
  bb_box_width = y[:,3] * img_w
  bb_box_height = y[:,4] * img_h
  center_x = y[:,1] * img_w
  center_y = y[:,2] * img_h
  coord.append((center_x - (bb_box_width / 2)))
  coord.append((center_y - (bb_box_height / 2)))
  coord.append((center_x + (bb_box_width / 2)))
  coord.append((center_y + (bb_box_height / 2)))
  
  return coord
  


# <h3>IOU Loss</h3>

# In[166]:


def iou_loss(labels,logits):
  #Convert the arrays to absolute coordinates
  coord_labels = convert_to_coord(labels)
  coord_logits = convert_to_coord(logits)
  
  # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
  xi1 = tf.maximum((coord_labels[0]), (coord_logits[0]))
  yi1 = tf.maximum((coord_labels[1]), (coord_logits[1]))
  xi2 = tf.minimum((coord_labels[2]), (coord_logits[2]))
  yi2 = tf.minimum((coord_labels[3]), (coord_logits[3]))
  inter_area = tf.maximum((yi2-yi1), 0) * tf.maximum((xi2-xi1), 0)
  
  # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
  box1_area = (coord_labels[3] - coord_labels[1])*(coord_labels[2]- coord_labels[0])
  box2_area = (coord_logits[3] - coord_logits[1])*(coord_logits[2]- coord_logits[0])
  union_area = (box1_area + box2_area) - inter_area
  # compute the IoU
  
  iou =inter_area / union_area

  return iou
  


# <h3>Feature Columns</h3>
# The feature columns is an intermediaries between raw data and Estimators.<br/>
# Feature columns bridge raw data with the data your model needs.
# 

# In[167]:


def get_feature_columns():
  feature_columns = {'images': tf.feature_column.numeric_column('images', (28, 28, 1))}
  return feature_columns


# <h3>Base Model</h3>

# In[168]:


def base_model(input,batch_size, input_shape):
  
  #print(input.shape)
  #print(batch_size)
  
  
  #Conv Layer - 1
  x = tf.keras.layers.Conv2D(16,(5,5), input_shape = input_shape , name="Conv_1",
                             use_bias=True, kernel_initializer='glorot_uniform', 
                             bias_initializer='zeros')(input)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.activations.relu(x)
  x= tf.keras.layers.MaxPooling2D()(x)
  
  x = tf.keras.layers.Conv2D(16,(5,5), input_shape = input_shape , name="Conv_2",
                             use_bias=True, kernel_initializer='glorot_uniform', 
                             bias_initializer='zeros')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.activations.relu(x)
  
  #print(x)
  
  #Flatten it out
  x = tf.keras.layers.Flatten(name="Flatten_1")(x)
  #print("After flatten")
  #print(x)
  
  #Dense layer
  x = tf.keras.layers.Dense(1024, activation="sigmoid", name="Dense1")(x)
  #print(x)
  x = tf.keras.layers.Dense(5, activation='sigmoid')(x)
  #print(x)
  # x = tf.keras.layers.Reshape((2*2, (1*5)), name= 'model_final_reshape')(x)
  
  return x


# <h3>Model Function</h3>
# <ul>
#   <li>input is features, labels, mode and parameters</li>
# </ul>
# 
# 

# In[169]:


def model_fn(features, labels, mode, params):
                                                                                                
  feature_columns = list(get_feature_columns().values())
  images = tf.feature_column.input_layer(features=features, feature_columns=feature_columns)
  images = tf.reshape(images, shape=(-1, 28, 28, 1),name='my_reshape')


  # Calculate logits through CNN                                                                                                            
  logits = base_model(images,batch_size, input_shape)

  # Create the input layers from the features 
  if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
    global_step = tf.train.get_or_create_global_step()#create default graph
    loss=custom_loss(labels,logits) #loss 
    iou = iou_loss(labels,logits) 
    tf.summary.scalar('IOU', tf.reduce_mean(iou))
    tf.summary.scalar('Loss', loss)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {'coordinates': logits}
    export_outputs = {'coordinates': tf.estimator.export.PredictOutput(predictions)}
    return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)
  
  if mode == tf.estimator.ModeKeys.EVAL:
    iou = iou_loss(labels,logits)
    eval_metric_ops = {'iou_eval': tf.metrics.mean(iou)}
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam')
    train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
  


# Preprocess is used in serving_input_fn

# In[110]:


# def preprocess(image):
#   img1 = tf.decode_raw(image, tf.float32)
#   img1 = tf.reshape(image, (img_w, img_h, img_channels))
#   img1 = tf.image.per_image_standardization(img1)
#   return img1

#Edited by Shivam
def preprocess(image,img_w,img_h,img_channels):
  #print(image)
  print(image)
  img_decoded = tf.image.decode_jpeg(image, channels=1)
  #print(img_decoded)
  img_expanded = tf.expand_dims(img_decoded, 0)
  #print(img_expanded)
  img_resize = tf.image.resize_bilinear(img_expanded,(img_w, img_h))
  #print(img_resize)
  img_squeezed = tf.squeeze(img_resize,0)
  #print(img_squeezed)
  img_standardized = tf.image.per_image_standardization(img_squeezed)
  #print(img_standardized)
  img_expanded = tf.expand_dims(img_standardized, 0)
  return img_expanded


# <h3>Serving function</h3>

# In[111]:

#Edited by Shivam
# def serving_input_fn():
#   receiver_tensor = {'images_bytes': tf.placeholder(dtype=tf.float32, shape=[None])}
#   features = {'images': tf.map_fn(preprocess, receiver_tensor['images'])}
#   return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)

  #(Shivam)code to read the image from url
# def read_and_preprocess(filename, augment=False):
#     # decode the image file starting from the filename
#     # end up with pixel values that are in the -1, 1 range
#     image_contents = tf.read_file(filename)
#     image = tf.image.decode_jpeg(image_contents, channels=1)
#     image = tf.image.convert_image_dtype(image, dtype=tf.float32) # 0-1
#     image = tf.expand_dims(image, 0) # resize_bilinear needs batches
#     image = tf.image.resize_bilinear(image, [28,28], align_corners=False)
#     #image = tf.image.per_image_whitening(image)  # useful if mean not important
#     image = tf.subtract(image, 0.5)
#     image = tf.multiply(image, 2.0) # -1 to 1
#     return image

def serving_input_fn():
    # inputs = {'imageurl': tf.placeholder(tf.string, shape=())}
    # filename = tf.squeeze(inputs['imageurl']) # make it a scalar
    # image = read_and_preprocess(filename)
    # # make the outer dimension unknown (and not 1)
    # image = tf.placeholder_with_default(image, shape=[None, img_w,img_h,img_channels])

    # features = {'image' : image}
    # return tf.estimator.export.ServingInputReceiver(features, inputs)
    receiver_tensor = {'images': tf.placeholder(dtype=tf.string, shape=())}
    image = receiver_tensor['images']
    image_standardized = preprocess(image,img_w,img_h,img_channels)
    # make the outer dimension unknown (and not 1)
    image = tf.placeholder_with_default(image_standardized, shape=[None, img_w,img_h,img_channels])
    features = {'images': image}

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)

# 
# 
# 
# 
# 

# In[183]:


run_config = tf.estimator.RunConfig(model_dir=model_dir, save_summary_steps=10, save_checkpoints_secs = 300, keep_checkpoint_max = 5)
estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

# There is another Exporter named FinalExporter which exports the serving graph and checkpoints at the end.

exporter = tf.estimator.LatestExporter(
  name='Serve',
  serving_input_receiver_fn=serving_input_fn,
  assets_extra=None,
  as_text=False,
  exports_to_keep=5)

train_spec = tf.estimator.TrainSpec(input_fn= lambda:input_fn(files_pattern, batch_size, mode=tf.estimator.ModeKeys.TRAIN),max_steps=4000)

eval_spec = tf.estimator.EvalSpec(lambda: input_fn(files_pattern,  batch_size , mode=tf.estimator.ModeKeys.EVAL),steps=10, exporters=exporter)


# In[184]:


# writer.add_graph(tf.get_default_graph())
# writer.flush()


# In[185]:


tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# In[123]:


#estimator.train(lambda: input_fn(files_pattern,  batch_size , mode=tf.estimator.ModeKeys.TRAIN), steps=100)/


# In[124]:


#estimator.evaluate(lambda: input_fn(files_pattern,  batch_size , mode=tf.estimator.ModeKeys.EVAL), steps=100)


# In[186]:


l = estimator.predict(lambda: input_fn(test_pattern,  1 , mode=tf.estimator.ModeKeys.PREDICT))


# In[191]:


def plot_bb(preds):
  height_image = 28
  width_image = 28
  bb_box_width = preds[3] * width_image
  bb_box_height = preds[4] * height_image
  center_x = preds[1] * width_image
  center_y = preds[2] * height_image
  x_min = (center_x - (bb_box_width / 2))
  y_min = (center_y - (bb_box_height / 2))
  print(x_min)
  print(y_min)
  print(bb_box_width)
  print(bb_box_height)
  
  
  with tf.Session() as sess:
    sess.graph._unsafe_unfinalize()

  dataset = tf.data.TFRecordDataset(['gs://indic2019/indic2019/2.tfrecord'])
  dataset = dataset.map(extract_fn)
  image, label = dataset.make_one_shot_iterator().get_next()




# In[192]:


count = 0
for it in l:
  preds = it['coordinates']
  if count == 0:
    break
print(preds)


# In[193]:


plot_bb(preds)


# Save the model

# In[178]:


#estimator.export_savedmodel('saved_model', serving_input_fn)


# In[194]:



export_dir = 'saved_model/1559719620/saved_model.pb'
estimator.export_savedmodel(export_dir, serving_input_fn,
                            strip_default_attrs=True)
