from xmumodel.model import Model
import tensorlayer.layers as tl
import tensorflow as tf
from xmuutil import utils
from xmuutil.scalelayer import ScaleLayer
from xmuutil.relulayer import ReluLayer
from xmuutil.transposed_conv2d_layer import TransposedConv2dLayer
from tqdm import tqdm
import shutil
import os
import numpy as np
"""
EDSR upsample-layer
subpixel -> deconv 
"""
class EDSR(Model):
    def buildModel(self):
        print("Building EDSR...")
        self.norm_input = utils.normalize_color_tf(self.input)
        self.norm_target = utils.normalize_color_tf(self.target)
        self.op = []
        #input layer
        x = tl.InputLayer(self.norm_input, name='inputlayer')

        # One convolution before res blocks and to convert to required feature depth
        x = tl.Conv2d(x, self.feature_size, [3, 3], name='c')

        # Store the output of the first convolution to add later
        conv_1 = x

        scaling_factor = 0.1
        # Add the residual blocks to the model
        for i in range(self.num_layers):
            x = self.__resBlock(x, self.feature_size, scale=scaling_factor,layer=i)
        # One more convolution, and then we add the output of our first conv layer
        x = tl.Conv2d(x, self.feature_size, [3, 3], act = None, name = 'm1')
        x = tl.ElementwiseLayer([conv_1,x],tf.add, name='res_add')

        x = TransposedConv2dLayer(x, self.feature_size-self.prunedlist[16], [5,5], [2,2], name='deconv_1')
        self.op.append(x.outputs)
        x = tl.Conv2d(x, self.feature_size, [3, 3], act=tf.nn.relu, name='deconv_conv_1')
        x = TransposedConv2dLayer(x, self.feature_size-self.prunedlist[17], [5, 5], [2, 2], name='deconv_2')
        self.op.append(x.outputs)

        x = tl.Conv2d(x, self.feature_size, [3, 3], act=tf.nn.relu, name='deconv_conv_2')
        x = TransposedConv2dLayer(x, self.feature_size-self.prunedlist[18], [5, 5], [2, 2], name='deconv_3')
        self.op.append(x.outputs)
        # One final convolution on the upsampling output
        output = tl.Conv2d(x,self.output_channels,[1,1],act=tf.nn.relu, name='lastLayer')
        self.output = output.outputs
        self.output = tf.clip_by_value(output.outputs, 0.0, 1.0, name='output')

        self.cacuLoss()

        # Tensorflow graph setup... session, saver, etc.
        session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
        session_conf.gpu_options.allocator_type = 'BFC'
        self.sess = tf.Session(config=session_conf)
        self.saver = tf.train.Saver(var_list = tf.trainable_variables(),max_to_keep=100)
        print("Done building!")

    def __resBlock(self, x, channels = 64, kernel_size = (3, 3), scale = 1.0,layer = 0):
        nn = ReluLayer(x, name='res%d/ru1'%(layer))
        nn = tl.Conv2d(nn, channels-self.prunedlist[layer], kernel_size, act=tf.nn.relu, name='res%d/c1'%(layer))
        self.op.append(nn.outputs)
        nn = tl.Conv2d(nn, channels, kernel_size, act=None, name='res%d/c2'%(layer))
        nn = ScaleLayer(nn,scale, name='res%d/scale'%(layer))
        n = tl.ElementwiseLayer([x,nn],tf.add, name='res%d/res_add'%(layer))
        return n

    def cacuLoss(self):
        self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.norm_target, self.output))
        PSNR = utils.psnr_tf(self.norm_target, self.output, is_norm=True)

        # Scalar to keep track for loss
        summary_loss = tf.summary.scalar("loss", self.loss)
        summary_psnr = tf.summary.scalar("PSNR", PSNR)

        streaming_loss, self.streaming_loss_update = tf.contrib.metrics.streaming_mean(self.loss)
        streaming_loss_scalar = tf.summary.scalar('loss',streaming_loss)

        streaming_psnr, self.streaming_psnr_update = tf.contrib.metrics.streaming_mean(PSNR)
        streaming_psnr_scalar = tf.summary.scalar('PSNR',streaming_psnr)

        self.train_merge = tf.summary.merge([summary_loss,summary_psnr])
        self.test_merge = tf.summary.merge([streaming_loss_scalar,streaming_psnr_scalar])

    def train(self,batch_size= 10, iterations=1000, lr_init=1e-4, lr_decay=0.5, decay_every=2e5,
              save_dir="saved_models",reuse=False,reuse_dir=None,reuse_step=None, log_dir="log",prune=False,prunesize=0,prune_method='',prune_dec=True):
        #create the save directory if not exist
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.mkdir(log_dir)
        #Make new save directory
        os.mkdir(save_dir)

        with tf.variable_scope('learning_rate'):
            lr_v = tf.Variable(lr_init, trainable=False)
        # Using adam optimizer as mentioned in the paper
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_v)
        # optimizer = tf.train.RMSPropOptimizer(lr_v, decay=0.95, momentum=0.9, epsilon=1e-8)
        # This is the train operation for our objective
        #self.train_op = optimizer.minimize(self.loss)
        self.comp_op = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(self.comp_op)

        #Operation to initialize all variables
        init = tf.global_variables_initializer()
        print("Begin training...")
        with self.sess as sess:
            #Initialize all variables
            sess.run(init)
            if reuse:
                self.resume(reuse_dir, global_step=reuse_step)
            if prune:
                if(prune_method == 'taylor'):
                    self.taylor_prune(batch_size,sess,prunesize, iterations,save_dir,prune_dec)
                    self.save(save_dir, iterations)
                    return
                if(prune_method == 'thinet'):
                    self.thinet_prune(batch_size,sess,prunesize, iterations,save_dir, prune_dec)
                    self.save(save_dir, iterations)
                    return
                 
            #create summary writer for train
            train_writer = tf.summary.FileWriter(log_dir+"/train",sess.graph)

            #If we're using a test set, include another summary writer for that
            test_writer = tf.summary.FileWriter(log_dir+"/test",sess.graph)
            test_feed = []
            while True:
                test_x,test_y = self.data.get_test_set(batch_size)
                if test_x!=None and test_y!=None:
                    test_feed.append({
                            self.input:test_x,
                            self.target:test_y
                    })
                else:
                    break

            sess.run(tf.assign(lr_v, lr_init))
            #This is our training loop
            for i in tqdm(range(iterations)):
                if i != 0 and (i % decay_every == 0):
                    new_lr_decay = lr_decay ** (i // decay_every)
                    sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                #Use the data function we were passed to get a batch every iteration
                x,y = self.data.get_batch(batch_size)
                #Create feed dictionary for the batch
                feed = {
                    self.input:x,
                    self.target:y
                }
                #Run the train op and calculate the train summary
                summary,_ = sess.run([self.train_merge,self.train_op],feed)
                #Write train summary for this step
                train_writer.add_summary(summary,i)

                # Test every 500 iterations; Save our trained model
                if (i!=0 and i % 50 == 0) or (i+1 == iterations):
                    sess.run(tf.local_variables_initializer())
                    for j in range(len(test_feed)):
                        # sess.run([self.streaming_loss_update],feed_dict=test_feed[j])
                        sess.run([self.streaming_loss_update, self.streaming_psnr_update], feed_dict=test_feed[j])
                    streaming_summ = sess.run(self.test_merge)
                    # Write test summary
                    test_writer.add_summary(streaming_summ, i)

                    self.save(save_dir, i)
            #experimental_create_eval_graph
            grap = tf.contrib.quantize.create_training_graph(sess.graph)
            #grap = tf.contrib.quantize.experimental_create_eval_graph(sess.graph)
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, grap,output_node_names=['output'])
            with tf.gfile.FastGFile('cele_t.pb', mode='wb') as f:
                f.write(output_graph_def.SerializeToString())
            #builder = tf.saved_model.builder.SavedModelBuilder("tmp/model" ) 
            #builder.add_meta_graph_and_variables( sess, [tf.saved_model.tag_constants.SERVING] )
            #builder.save()
            test_writer.close()
            train_writer.close()
    def taylor_prune(self,batch_size,sess,prunesize, its,save_dir,prune_dec=True):
        grad_sum = []
        for tmp in range(19):
            grad_sum.append([])
            if(tmp<16):
                grad_sum[tmp] = np.zeros(shape=(3,3,self.feature_size, self.feature_size-self.prunedlist[0]))
            else:
                grad_sum[tmp] = np.zeros(shape=(5,5,self.feature_size-self.prunedlist[0], self.feature_size))
        aim = np.zeros(shape=(19,self.feature_size-self.prunedlist[0]))
        candidate = [] 
        for tmp in range(its):
            print("sample:"+str(tmp))
            x,y = self.data.get_batch(batch_size)
            #Create feed dictionary for the batch
            feed = {
                self.input:x,
                self.target:y
            }
            gradients = sess.run([self.comp_op],feed)
            for index,element in enumerate(gradients[0]):
                real_index = int((index-2)/4)
                if(index>=2 and index%4==2 and index<63):
                    grad_sum[real_index] = grad_sum[real_index]+np.fabs(np.array(element[0])*np.array(element[1]))
                if(prune_dec):
                    if(index == 68):
                        grad_sum[16] = grad_sum[16]+np.fabs(np.array(element[0])*np.array(element[1]))
                    if(index == 72):
                        grad_sum[17] = grad_sum[17]+np.fabs(np.array(element[0])*np.array(element[1]))
                    if(index == 76):
                        grad_sum[18] = grad_sum[18]+np.fabs(np.array(element[0])*np.array(element[1]))
        for index in range(19):
            if(index>=16):
                if(prune_dec):
                    aim[index] = np.sum(np.sum(np.sum(grad_sum[index],-1),0),0)
                    candidate.append(aim[index].argsort()[0:prunesize])
                    candidate[index] = sorted(candidate[index])
                else:
                    candidate.append([])
            else:
                aim[index] = np.sum(np.sum(np.sum(grad_sum[index],0),0),0)
                candidate.append(aim[index].argsort()[0:prunesize])
                candidate[index] = sorted(candidate[index])
      
        self.do_prune(sess, candidate,save_dir,prune_dec) 

    def thinet_prune(self,batch_size,sess, prunesize, its,save_dir, prune_dec=True):
        samp_x = []
        samp_y = []
        for i in range(19):
            samp_x.append([])
            samp_y.append([])
        for tmp in range(its):
            print("sample:"+str(tmp))
            x,y = self.data.get_batch(batch_size)
                        #Create feed dictionary for the batch
            feed = {
                self.input:x,
                self.target:y
            }
                                   #Run the train op and calculate the train summary
            out = sess.run([self.op[:]],feed)
            out = out[0]
            for layer,ele in enumerate(out):
                for index,ele in enumerate(ele):
                    nn_np = np.array(ele)
                    nn_len1 = len(nn_np)
                    nn_len2 = len(nn_np[0])
                    nn_len3 = len(nn_np[0][0])#input_channels
                    random1 = np.random.randint(0,nn_len1-3+1)
                    random2 = np.random.randint(0,nn_len2-3+1)
                    input_sample = nn_np[random1:random1+3,random2:random2+3,:]


                                #from c2 kernel get a rand input [3,3,input_channels]
                    if(layer<18):
                        random3 = np.random.randint(0,self.feature_size-self.prunedlist[layer])
                    else:
                        random3 = np.random.randint(0,3)
                    if(layer<16):
                        var_tensor = tl.get_variables_with_name("res"+str(layer)+"/"+"c2")
                    if(layer == 16):
                        var_tensor = tl.get_variables_with_name("deconv_conv_1")
                    if(layer == 17):
                        var_tensor = tl.get_variables_with_name("deconv_conv_2")
                    if(layer == 18):
                        var_tensor = tl.get_variables_with_name("lastLayer")
                    w = var_tensor[0].eval()
                    w = w[:,:,:,random3]
                    b = var_tensor[1].eval()
                    b = b[random3]
                                #get y = [x1,x2,x3...x_inputchannels]
                    x_n = input_sample*w
                    x_n = np.sum(np.sum(x_n,0),0)#x^
                    samp_x[layer].append(x_n)
                    y = np.sum(np.sum(x_n),0)-b#y^
                    samp_y[layer].append(y)
                               #sample ready then do choose
        candidate = []
        for layer in range(19):
            candidate.append([])
            x = samp_x[layer]
            y = samp_y[layer]
            for i in range(prunesize):
                min_value = float('inf')
                min_i = None
                for l in range(self.feature_size-self.prunedlist[layer]):
                    if(l in candidate[layer]):
                        continue
                    value = 0
                    for x_one in x:
                        tmp_v = sum([x_one[p] for p in candidate[layer]])+x_one[l]
                        value = value + tmp_v*tmp_v
                    if(value < min_value):
                        min_value = value
                        min_i = l
                candidate[layer].append(min_i)
        self.do_prune(sess, candidate,save_dir,prune_dec)
  
    def do_prune(self, sess, candidate,save_dir,prune_dec=True):
        print(candidate)
        if(prune_dec == False):
            candidate[16:18] = []
        deconv_var = tl.get_variables_with_name("Conv2d_transpose")
        for index,element in enumerate(candidate):
            if(prune_dec == False and index>15):
                continue
            if(index < 16):
                var_tensor = tl.get_variables_with_name("res"+str(index)+"/"+"c1")
                w_next = tl.get_variables_with_name("res"+str(index)+"/"+"c2")[0]
                w = var_tensor[0]
                b = var_tensor[1]
            if(index == 16):
                w = deconv_var[0]
                b = deconv_var[1]
                w_next = tl.get_variables_with_name("deconv_conv_1")[0]
            if(index == 17):
                w = deconv_var[2]
                b = deconv_var[3]
                w_next = tl.get_variables_with_name("deconv_conv_2")[0]
            if(index == 18):
                w = deconv_var[4]
                b = deconv_var[5]
                w_next = tl.get_variables_with_name("lastLayer")[0]

            w_np = w.eval()
            b_np = b.eval()
            w_next_np = w_next.eval()
            if(index>=16):
                w_np = np.delete(w_np, element, -2)
            else:
                w_np = np.delete(w_np, element, -1)
            b_np = np.delete(b_np, element, 0)
            w_next_np = np.delete(w_next_np, element, -2)
  
            w_new = tf.convert_to_tensor(w_np)
            b_new = tf.convert_to_tensor(b_np)
            w_next_new = tf.convert_to_tensor(w_next_np)

            sess.run(tf.assign(w, w_new,False))
            sess.run(tf.assign(b, b_new,False))
            sess.run(tf.assign(w_next, w_next_new,False))
        save_prunedlist = []
        for i,e in enumerate(self.prunedlist):
            if(prune_dec==False and i>15):
                save_prunedlist.append(int(e))
            else:
                save_prunedlist.append(int(e+len(candidate[i])))
        np.savetxt(save_dir+"/prunedlist",np.array(save_prunedlist),fmt="%d",delimiter=",")
