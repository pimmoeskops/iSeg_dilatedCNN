import theano
from theano import tensor as T
import lasagne
from lasagne.layers import dnn
import cPickle

def softmax(x):
    e_x = theano.tensor.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
    
def categorical_crossentropy_eps(predictions, targets):
    eps = 10e-15
    predictions = T.clip(predictions, eps, 1.0 - eps)
    return lasagne.objectives.categorical_crossentropy(predictions, targets)  

def getmodel(X1,X2,X3,X4,seg,p_drop_hidden,inputsize,inputsize3D,nrofinputs, classes):             
    
    #67x67 orthogonal input
    segnetwork1 = lasagne.layers.InputLayer(shape=(None, nrofinputs, inputsize, inputsize),input_var=X1)
    segnetwork2 = lasagne.layers.InputLayer(shape=(None, nrofinputs, inputsize, inputsize),input_var=X2)
    segnetwork3 = lasagne.layers.InputLayer(shape=(None, nrofinputs, inputsize, inputsize),input_var=X3)
    #25x25x25 input
    segnetwork4 = lasagne.layers.InputLayer(shape=(None, nrofinputs, inputsize3D, inputsize3D,inputsize3D),input_var=X4)
    
    
    segnetwork1 = lasagne.layers.DilatedConv2DLayer(segnetwork1, num_filters=32, filter_size=(3,3), dilation=(1,1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork1 = lasagne.layers.batch_norm(segnetwork1)    
    segnetwork2 = lasagne.layers.DilatedConv2DLayer(segnetwork2, num_filters=32, filter_size=(3,3), dilation=(1,1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork2 = lasagne.layers.batch_norm(segnetwork2)    
    segnetwork3 = lasagne.layers.DilatedConv2DLayer(segnetwork3, num_filters=32, filter_size=(3,3), dilation=(1,1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork3 = lasagne.layers.batch_norm(segnetwork3)
    
    print segnetwork1.output_shape
    
    segnetwork1 = lasagne.layers.DilatedConv2DLayer(segnetwork1, num_filters=32, filter_size=(3,3), dilation=(1,1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork1 = lasagne.layers.batch_norm(segnetwork1)
    segnetwork2 = lasagne.layers.DilatedConv2DLayer(segnetwork2, num_filters=32, filter_size=(3,3), dilation=(1,1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork2 = lasagne.layers.batch_norm(segnetwork2)
    segnetwork3 = lasagne.layers.DilatedConv2DLayer(segnetwork3, num_filters=32, filter_size=(3,3), dilation=(1,1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork3 = lasagne.layers.batch_norm(segnetwork3)
    
    print segnetwork1.output_shape
    
    segnetwork1 = lasagne.layers.DilatedConv2DLayer(segnetwork1, num_filters=32, filter_size=(3,3), dilation=(2,2), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork1 = lasagne.layers.batch_norm(segnetwork1)
    segnetwork2 = lasagne.layers.DilatedConv2DLayer(segnetwork2, num_filters=32, filter_size=(3,3), dilation=(2,2), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork2 = lasagne.layers.batch_norm(segnetwork2)
    segnetwork3 = lasagne.layers.DilatedConv2DLayer(segnetwork3, num_filters=32, filter_size=(3,3), dilation=(2,2), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork3 = lasagne.layers.batch_norm(segnetwork3)
    
    print segnetwork1.output_shape
    
    segnetwork1 = lasagne.layers.DilatedConv2DLayer(segnetwork1, num_filters=32, filter_size=(3,3), dilation=(4,4), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork1 = lasagne.layers.batch_norm(segnetwork1)
    segnetwork2 = lasagne.layers.DilatedConv2DLayer(segnetwork2, num_filters=32, filter_size=(3,3), dilation=(4,4), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork2 = lasagne.layers.batch_norm(segnetwork2)
    segnetwork3 = lasagne.layers.DilatedConv2DLayer(segnetwork3, num_filters=32, filter_size=(3,3), dilation=(4,4), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork3 = lasagne.layers.batch_norm(segnetwork3)
   
    print segnetwork1.output_shape
    
    segnetwork1 = lasagne.layers.DilatedConv2DLayer(segnetwork1, num_filters=32, filter_size=(3,3), dilation=(8,8), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork1 = lasagne.layers.batch_norm(segnetwork1)
    segnetwork2 = lasagne.layers.DilatedConv2DLayer(segnetwork2, num_filters=32, filter_size=(3,3), dilation=(8,8), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork2 = lasagne.layers.batch_norm(segnetwork2)
    segnetwork3 = lasagne.layers.DilatedConv2DLayer(segnetwork3, num_filters=32, filter_size=(3,3), dilation=(8,8), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork3 = lasagne.layers.batch_norm(segnetwork3)
   
    print segnetwork1.output_shape    
    
    segnetwork1 = lasagne.layers.DilatedConv2DLayer(segnetwork1, num_filters=32, filter_size=(3,3), dilation=(16,16), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork1 = lasagne.layers.batch_norm(segnetwork1)
    segnetwork2 = lasagne.layers.DilatedConv2DLayer(segnetwork2, num_filters=32, filter_size=(3,3), dilation=(16,16), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork2 = lasagne.layers.batch_norm(segnetwork2)
    segnetwork3 = lasagne.layers.DilatedConv2DLayer(segnetwork3, num_filters=32, filter_size=(3,3), dilation=(16,16), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork3 = lasagne.layers.batch_norm(segnetwork3)
    
    print segnetwork1.output_shape
    
    segnetwork1 = lasagne.layers.DilatedConv2DLayer(segnetwork1, num_filters=32, filter_size=(3,3), dilation=(1,1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork1 = lasagne.layers.batch_norm(segnetwork1)
    segnetwork2 = lasagne.layers.DilatedConv2DLayer(segnetwork2, num_filters=32, filter_size=(3,3), dilation=(1,1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork2 = lasagne.layers.batch_norm(segnetwork2)
    segnetwork3 = lasagne.layers.DilatedConv2DLayer(segnetwork3, num_filters=32, filter_size=(3,3), dilation=(1,1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    segnetwork3 = lasagne.layers.batch_norm(segnetwork3)
    
    print segnetwork1.output_shape
    
    for i in xrange(int((inputsize3D-1)/2)):
        segnetwork4 = dnn.Conv3DDNNLayer(segnetwork4, num_filters=32, filter_size=(3,3,3), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
        segnetwork4 = lasagne.layers.batch_norm(segnetwork4)
        print segnetwork4.output_shape
        
    segnetwork4 = lasagne.layers.ReshapeLayer(segnetwork4, ((-1, 32, 1, 1)))
    print segnetwork4.output_shape
    
    segnetwork = lasagne.layers.concat((segnetwork1,segnetwork2,segnetwork3,segnetwork4))
    
    print segnetwork.output_shape
    
    segnetwork = lasagne.layers.dropout(segnetwork, p=p_drop_hidden)
        
    segnetwork = lasagne.layers.DilatedConv2DLayer(segnetwork, num_filters=classes, filter_size=(1, 1), dilation=(1,1), nonlinearity=softmax, W=lasagne.init.GlorotUniform()) 
            
    return segnetwork1, segnetwork2, segnetwork3, segnetwork4, segnetwork   
    
def save_weights(filename,network):
    with open(filename, 'wb') as f:
        cPickle.dump(lasagne.layers.get_all_param_values(network), f)
        
def load_weights(filename, network):
    with open(filename, 'rb') as f:
        lasagne.layers.set_all_param_values(network, cPickle.load(f))
    

X1 = T.tensor4()
X2 = T.tensor4()
X3 = T.tensor4()
ftensor5 = T.TensorType('float32', (False,)*5)
X4 = ftensor5()
Y = T.tensor4()

convmodel1, convmodel2, convmodel3, convmodel4, model = getmodel(X1, X2, X3, X4, Y, 0.5, 67, 25, 2, 3)
outputtrain = lasagne.layers.get_output(model) 

cost = T.mean(categorical_crossentropy_eps(outputtrain, Y))
params = lasagne.layers.get_all_params(model, trainable=True)
updates = lasagne.updates.adam(cost, params, learning_rate=0.001)

train = theano.function(inputs=[X1, X2, X3, X4, Y], outputs=cost, updates=updates, allow_input_downcast=True)

outputtest = lasagne.layers.get_output(model, deterministic=True)
predict = theano.function(inputs=[X1, X2, X3, X4], outputs=outputtest, allow_input_downcast=True)
