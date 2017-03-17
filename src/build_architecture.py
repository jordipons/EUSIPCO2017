import lasagne

def buildNet(input_var,config):
    if config['type'] == 'smallSquared':
        return smallSquared(input_var, config)

    elif config['type'] == 'proposed':
        return proposeD(input_var, config)

    elif config['type'] == 'proposed2':
        return proposeD2(input_var, config)

    elif config['type'] == 'proposed4':
        return proposeD4(input_var, config)

def proposeD4(input_var, config): # (1,3,5,7)x100,(1,3,5,7)x75,(1,3,5,7)x25
    network={}
    # input layer
    network["1"] = lasagne.layers.InputLayer(shape=(None,1, int(config['patches_params']['xInput']), int(config['patches_params']['yInput'])),input_var=input_var)
    i=4
    # convolutional layer: block nx100
    network["2_1x100"] = lasagne.layers.Conv2DLayer(network["1"], num_filters=(int(i*10)), filter_size=(1,int(100)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_1x100_mp"] = lasagne.layers.MaxPool2DLayer(network["2_1x100"], pool_size=(4,network["2_1x100"].output_shape[3]))

    network["in_2_3x100"] = lasagne.layers.PadLayer(network["1"], (1,0), val=0, batch_ndim=2)
    network["2_3x100"] = lasagne.layers.Conv2DLayer(network["in_2_3x100"], num_filters=(int(i*6)), filter_size=(3,int(100)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_3x100_mp"] = lasagne.layers.MaxPool2DLayer(network["2_3x100"], pool_size=(4,network["2_3x100"].output_shape[3]))

    network["in_2_5x100"] = lasagne.layers.PadLayer(network["1"], (2,0), val=0, batch_ndim=2)
    network["2_5x100"] = lasagne.layers.Conv2DLayer(network["in_2_5x100"], num_filters=(int(i*3)), filter_size=(5,int(100)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_5x100_mp"] = lasagne.layers.MaxPool2DLayer(network["2_5x100"], pool_size=(4,network["2_5x100"].output_shape[3]))

    network["in_2_7x100"] = lasagne.layers.PadLayer(network["1"], (3,0), val=0, batch_ndim=2)
    network["2_7x100"] = lasagne.layers.Conv2DLayer(network["in_2_7x100"], num_filters=(int(i*3)), filter_size=(7,int(100)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_7x100_mp"] = lasagne.layers.MaxPool2DLayer(network["2_7x100"], pool_size=(4,network["2_7x100"].output_shape[3]))

    # convolutional layer: block nx75
    network["2_1x75"] = lasagne.layers.Conv2DLayer(network["1"], num_filters=(int(i*15)), filter_size=(1,int(75)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_1x75_mp"] = lasagne.layers.MaxPool2DLayer(network["2_1x75"], pool_size=(4,network["2_1x75"].output_shape[3]))

    network["in_2_3x75"] = lasagne.layers.PadLayer(network["1"], (1,0), val=0, batch_ndim=2)
    network["2_3x75"] = lasagne.layers.Conv2DLayer(network["in_2_3x75"], num_filters=(int(i*10)), filter_size=(3,int(75)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_3x75_mp"] = lasagne.layers.MaxPool2DLayer(network["2_3x75"], pool_size=(4,network["2_3x75"].output_shape[3]))

    network["in_2_5x75"] = lasagne.layers.PadLayer(network["1"], (2,0), val=0, batch_ndim=2)
    network["2_5x75"] = lasagne.layers.Conv2DLayer(network["in_2_5x75"], num_filters=(int(i*5)), filter_size=(5,int(75)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_5x75_mp"] = lasagne.layers.MaxPool2DLayer(network["2_5x75"], pool_size=(4,network["2_5x75"].output_shape[3]))

    network["in_2_7x75"] = lasagne.layers.PadLayer(network["1"], (3,0), val=0, batch_ndim=2)
    network["2_7x75"] = lasagne.layers.Conv2DLayer(network["in_2_7x75"], num_filters=(int(i*5)), filter_size=(7,int(75)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_7x75_mp"] = lasagne.layers.MaxPool2DLayer(network["2_7x75"], pool_size=(4,network["2_7x75"].output_shape[3]))

    # convolutional layer: block nx25
    network["2_1x25"] = lasagne.layers.Conv2DLayer(network["1"], num_filters=(int(i*15)), filter_size=(1,int(25)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_1x25_mp"] = lasagne.layers.MaxPool2DLayer(network["2_1x25"], pool_size=(4,network["2_1x25"].output_shape[3]))

    network["in_2_3x25"] = lasagne.layers.PadLayer(network["1"], (1,0), val=0, batch_ndim=2)
    network["2_3x25"] = lasagne.layers.Conv2DLayer(network["in_2_3x25"], num_filters=(int(i*10)), filter_size=(3,int(25)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_3x25_mp"] = lasagne.layers.MaxPool2DLayer(network["2_3x25"], pool_size=(4,network["2_3x25"].output_shape[3]))

    network["in_2_5x25"] = lasagne.layers.PadLayer(network["1"], (2,0), val=0, batch_ndim=2)
    network["2_5x25"] = lasagne.layers.Conv2DLayer(network["in_2_5x25"], num_filters=(int(i*5)), filter_size=(5,int(25)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_5x25_mp"] = lasagne.layers.MaxPool2DLayer(network["2_5x25"], pool_size=(4,network["2_5x25"].output_shape[3]))

    network["in_2_7x25"] = lasagne.layers.PadLayer(network["1"], (3,0), val=0, batch_ndim=2)
    network["2_7x25"] = lasagne.layers.Conv2DLayer(network["in_2_7x25"], num_filters=(int(i*5)), filter_size=(7,int(25)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_7x25_mp"] = lasagne.layers.MaxPool2DLayer(network["2_7x25"], pool_size=(4,network["2_7x25"].output_shape[3]))

    network["2"] = lasagne.layers.ConcatLayer([network["2_1x100_mp"], network["2_3x100_mp"],network["2_5x100_mp"],network["2_7x100_mp"],\
                                               network["2_1x75_mp"], network["2_3x75_mp"],network["2_5x75_mp"],network["2_7x75_mp"],\
                                               network["2_1x25_mp"], network["2_3x25_mp"],network["2_5x25_mp"],network["2_7x25_mp"],\
                                               ], axis=1, cropping=None)
    print('\nConv1')
    print network["2"].output_shape
    network["3"]=lasagne.layers.ReshapeLayer(network["2"],([0],[3],[2],[1]))
    print('\nreshape')
    print network["3"].input_shape
    print network["3"].output_shape
    # convolutional layer
    network["4"] = lasagne.layers.Conv2DLayer(network["3"], num_filters=int(i*32), filter_size=(8,network["3"].output_shape[3]),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    print('\nConv2')
    print network["4"].input_shape
    print network["4"].output_shape
    # pooling layer
    network["5pre"] = lasagne.layers.MaxPool2DLayer(network["4"], pool_size=(4,1))
    network["5"]=lasagne.layers.ReshapeLayer(network["5pre"],([0],[3],[2],[1]))
    # feed-forward layer
    network["6"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["5"], p=0.5),num_units=100,nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    # output layer
    network["7"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["6"], p=0.5),num_units=int(config['setup_params']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.sigmoid)

    return network["7"],network

def proposeD(input_var, config): # (1,3,5,7)x100,(1,3,5,7)x75,(1,3,5,7)x25
    network={}
    # input layer
    network["1"] = lasagne.layers.InputLayer(shape=(None,1, int(config['patches_params']['xInput']), int(config['patches_params']['yInput'])),input_var=input_var)
    i=1
    # convolutional layer: block nx100
    network["2_1x100"] = lasagne.layers.Conv2DLayer(network["1"], num_filters=(int(i*10)), filter_size=(1,int(100)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_1x100_mp"] = lasagne.layers.MaxPool2DLayer(network["2_1x100"], pool_size=(4,network["2_1x100"].output_shape[3]))

    network["in_2_3x100"] = lasagne.layers.PadLayer(network["1"], (1,0), val=0, batch_ndim=2)
    network["2_3x100"] = lasagne.layers.Conv2DLayer(network["in_2_3x100"], num_filters=(int(i*6)), filter_size=(3,int(100)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_3x100_mp"] = lasagne.layers.MaxPool2DLayer(network["2_3x100"], pool_size=(4,network["2_3x100"].output_shape[3]))

    network["in_2_5x100"] = lasagne.layers.PadLayer(network["1"], (2,0), val=0, batch_ndim=2)
    network["2_5x100"] = lasagne.layers.Conv2DLayer(network["in_2_5x100"], num_filters=(int(i*3)), filter_size=(5,int(100)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_5x100_mp"] = lasagne.layers.MaxPool2DLayer(network["2_5x100"], pool_size=(4,network["2_5x100"].output_shape[3]))

    network["in_2_7x100"] = lasagne.layers.PadLayer(network["1"], (3,0), val=0, batch_ndim=2)
    network["2_7x100"] = lasagne.layers.Conv2DLayer(network["in_2_7x100"], num_filters=(int(i*3)), filter_size=(7,int(100)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_7x100_mp"] = lasagne.layers.MaxPool2DLayer(network["2_7x100"], pool_size=(4,network["2_7x100"].output_shape[3]))

    # convolutional layer: block nx75
    network["2_1x75"] = lasagne.layers.Conv2DLayer(network["1"], num_filters=(int(i*15)), filter_size=(1,int(75)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_1x75_mp"] = lasagne.layers.MaxPool2DLayer(network["2_1x75"], pool_size=(4,network["2_1x75"].output_shape[3]))

    network["in_2_3x75"] = lasagne.layers.PadLayer(network["1"], (1,0), val=0, batch_ndim=2)
    network["2_3x75"] = lasagne.layers.Conv2DLayer(network["in_2_3x75"], num_filters=(int(i*10)), filter_size=(3,int(75)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_3x75_mp"] = lasagne.layers.MaxPool2DLayer(network["2_3x75"], pool_size=(4,network["2_3x75"].output_shape[3]))

    network["in_2_5x75"] = lasagne.layers.PadLayer(network["1"], (2,0), val=0, batch_ndim=2)
    network["2_5x75"] = lasagne.layers.Conv2DLayer(network["in_2_5x75"], num_filters=(int(i*5)), filter_size=(5,int(75)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_5x75_mp"] = lasagne.layers.MaxPool2DLayer(network["2_5x75"], pool_size=(4,network["2_5x75"].output_shape[3]))

    network["in_2_7x75"] = lasagne.layers.PadLayer(network["1"], (3,0), val=0, batch_ndim=2)
    network["2_7x75"] = lasagne.layers.Conv2DLayer(network["in_2_7x75"], num_filters=(int(i*5)), filter_size=(7,int(75)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_7x75_mp"] = lasagne.layers.MaxPool2DLayer(network["2_7x75"], pool_size=(4,network["2_7x75"].output_shape[3]))

    # convolutional layer: block nx25
    network["2_1x25"] = lasagne.layers.Conv2DLayer(network["1"], num_filters=(int(i*15)), filter_size=(1,int(25)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_1x25_mp"] = lasagne.layers.MaxPool2DLayer(network["2_1x25"], pool_size=(4,network["2_1x25"].output_shape[3]))

    network["in_2_3x25"] = lasagne.layers.PadLayer(network["1"], (1,0), val=0, batch_ndim=2)
    network["2_3x25"] = lasagne.layers.Conv2DLayer(network["in_2_3x25"], num_filters=(int(i*10)), filter_size=(3,int(25)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_3x25_mp"] = lasagne.layers.MaxPool2DLayer(network["2_3x25"], pool_size=(4,network["2_3x25"].output_shape[3]))

    network["in_2_5x25"] = lasagne.layers.PadLayer(network["1"], (2,0), val=0, batch_ndim=2)
    network["2_5x25"] = lasagne.layers.Conv2DLayer(network["in_2_5x25"], num_filters=(int(i*5)), filter_size=(5,int(25)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_5x25_mp"] = lasagne.layers.MaxPool2DLayer(network["2_5x25"], pool_size=(4,network["2_5x25"].output_shape[3]))

    network["in_2_7x25"] = lasagne.layers.PadLayer(network["1"], (3,0), val=0, batch_ndim=2)
    network["2_7x25"] = lasagne.layers.Conv2DLayer(network["in_2_7x25"], num_filters=(int(i*5)), filter_size=(7,int(25)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_7x25_mp"] = lasagne.layers.MaxPool2DLayer(network["2_7x25"], pool_size=(4,network["2_7x25"].output_shape[3]))

    network["2"] = lasagne.layers.ConcatLayer([network["2_1x100_mp"], network["2_3x100_mp"],network["2_5x100_mp"],network["2_7x100_mp"],\
                                               network["2_1x75_mp"], network["2_3x75_mp"],network["2_5x75_mp"],network["2_7x75_mp"],\
                                               network["2_1x25_mp"], network["2_3x25_mp"],network["2_5x25_mp"],network["2_7x25_mp"],\
                                               ], axis=1, cropping=None)
    print('\nConv1')
    print network["2"].output_shape
    network["3"]=lasagne.layers.ReshapeLayer(network["2"],([0],[3],[2],[1]))
    print('\nreshape')
    print network["3"].input_shape
    print network["3"].output_shape
    # convolutional layer
    network["4"] = lasagne.layers.Conv2DLayer(network["3"], num_filters=32, filter_size=(8,network["3"].output_shape[3]),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    print('\nConv2')
    print network["4"].input_shape
    print network["4"].output_shape
    # pooling layer
    network["5pre"] = lasagne.layers.MaxPool2DLayer(network["4"], pool_size=(4,1))
    network["5"]=lasagne.layers.ReshapeLayer(network["5pre"],([0],[3],[2],[1]))
    # feed-forward layer
    network["6"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["5"], p=0.5),num_units=100,nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    # output layer
    network["7"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["6"], p=0.5),num_units=int(config['setup_params']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.sigmoid)

    return network["7"],network

def proposeD2(input_var, config): # (1,3,5,7)x100,(1,3,5,7)x75,(1,3,5,7)x25
    network={}
    # input layer
    network["1"] = lasagne.layers.InputLayer(shape=(None,1, int(config['patches_params']['xInput']), int(config['patches_params']['yInput'])),input_var=input_var)
    i=2
    # convolutional layer: block nx100
    network["2_1x100"] = lasagne.layers.Conv2DLayer(network["1"], num_filters=(int(i*10)), filter_size=(1,int(100)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_1x100_mp"] = lasagne.layers.MaxPool2DLayer(network["2_1x100"], pool_size=(4,network["2_1x100"].output_shape[3]))

    network["in_2_3x100"] = lasagne.layers.PadLayer(network["1"], (1,0), val=0, batch_ndim=2)
    network["2_3x100"] = lasagne.layers.Conv2DLayer(network["in_2_3x100"], num_filters=(int(i*6)), filter_size=(3,int(100)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_3x100_mp"] = lasagne.layers.MaxPool2DLayer(network["2_3x100"], pool_size=(4,network["2_3x100"].output_shape[3]))

    network["in_2_5x100"] = lasagne.layers.PadLayer(network["1"], (2,0), val=0, batch_ndim=2)
    network["2_5x100"] = lasagne.layers.Conv2DLayer(network["in_2_5x100"], num_filters=(int(i*3)), filter_size=(5,int(100)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_5x100_mp"] = lasagne.layers.MaxPool2DLayer(network["2_5x100"], pool_size=(4,network["2_5x100"].output_shape[3]))

    network["in_2_7x100"] = lasagne.layers.PadLayer(network["1"], (3,0), val=0, batch_ndim=2)
    network["2_7x100"] = lasagne.layers.Conv2DLayer(network["in_2_7x100"], num_filters=(int(i*3)), filter_size=(7,int(100)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_7x100_mp"] = lasagne.layers.MaxPool2DLayer(network["2_7x100"], pool_size=(4,network["2_7x100"].output_shape[3]))

    # convolutional layer: block nx75
    network["2_1x75"] = lasagne.layers.Conv2DLayer(network["1"], num_filters=(int(i*15)), filter_size=(1,int(75)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_1x75_mp"] = lasagne.layers.MaxPool2DLayer(network["2_1x75"], pool_size=(4,network["2_1x75"].output_shape[3]))

    network["in_2_3x75"] = lasagne.layers.PadLayer(network["1"], (1,0), val=0, batch_ndim=2)
    network["2_3x75"] = lasagne.layers.Conv2DLayer(network["in_2_3x75"], num_filters=(int(i*10)), filter_size=(3,int(75)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_3x75_mp"] = lasagne.layers.MaxPool2DLayer(network["2_3x75"], pool_size=(4,network["2_3x75"].output_shape[3]))

    network["in_2_5x75"] = lasagne.layers.PadLayer(network["1"], (2,0), val=0, batch_ndim=2)
    network["2_5x75"] = lasagne.layers.Conv2DLayer(network["in_2_5x75"], num_filters=(int(i*5)), filter_size=(5,int(75)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_5x75_mp"] = lasagne.layers.MaxPool2DLayer(network["2_5x75"], pool_size=(4,network["2_5x75"].output_shape[3]))

    network["in_2_7x75"] = lasagne.layers.PadLayer(network["1"], (3,0), val=0, batch_ndim=2)
    network["2_7x75"] = lasagne.layers.Conv2DLayer(network["in_2_7x75"], num_filters=(int(i*5)), filter_size=(7,int(75)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_7x75_mp"] = lasagne.layers.MaxPool2DLayer(network["2_7x75"], pool_size=(4,network["2_7x75"].output_shape[3]))

    # convolutional layer: block nx25
    network["2_1x25"] = lasagne.layers.Conv2DLayer(network["1"], num_filters=(int(i*15)), filter_size=(1,int(25)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_1x25_mp"] = lasagne.layers.MaxPool2DLayer(network["2_1x25"], pool_size=(4,network["2_1x25"].output_shape[3]))

    network["in_2_3x25"] = lasagne.layers.PadLayer(network["1"], (1,0), val=0, batch_ndim=2)
    network["2_3x25"] = lasagne.layers.Conv2DLayer(network["in_2_3x25"], num_filters=(int(i*10)), filter_size=(3,int(25)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_3x25_mp"] = lasagne.layers.MaxPool2DLayer(network["2_3x25"], pool_size=(4,network["2_3x25"].output_shape[3]))

    network["in_2_5x25"] = lasagne.layers.PadLayer(network["1"], (2,0), val=0, batch_ndim=2)
    network["2_5x25"] = lasagne.layers.Conv2DLayer(network["in_2_5x25"], num_filters=(int(i*5)), filter_size=(5,int(25)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_5x25_mp"] = lasagne.layers.MaxPool2DLayer(network["2_5x25"], pool_size=(4,network["2_5x25"].output_shape[3]))

    network["in_2_7x25"] = lasagne.layers.PadLayer(network["1"], (3,0), val=0, batch_ndim=2)
    network["2_7x25"] = lasagne.layers.Conv2DLayer(network["in_2_7x25"], num_filters=(int(i*5)), filter_size=(7,int(25)),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["2_7x25_mp"] = lasagne.layers.MaxPool2DLayer(network["2_7x25"], pool_size=(4,network["2_7x25"].output_shape[3]))

    network["2"] = lasagne.layers.ConcatLayer([network["2_1x100_mp"], network["2_3x100_mp"],network["2_5x100_mp"],network["2_7x100_mp"],\
                                               network["2_1x75_mp"], network["2_3x75_mp"],network["2_5x75_mp"],network["2_7x75_mp"],\
                                               network["2_1x25_mp"], network["2_3x25_mp"],network["2_5x25_mp"],network["2_7x25_mp"],\
                                               ], axis=1, cropping=None)
    print('\nConv1')
    print network["2"].output_shape
    network["3"]=lasagne.layers.ReshapeLayer(network["2"],([0],[3],[2],[1]))
    print('\nreshape')
    print network["3"].input_shape
    print network["3"].output_shape
    # convolutional layer
    network["4"] = lasagne.layers.Conv2DLayer(network["3"], num_filters=int(i*32), filter_size=(8,network["3"].output_shape[3]),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    print('\nConv2')
    print network["4"].input_shape
    print network["4"].output_shape
    # pooling layer
    network["5pre"] = lasagne.layers.MaxPool2DLayer(network["4"], pool_size=(4,1))
    network["5"]=lasagne.layers.ReshapeLayer(network["5pre"],([0],[3],[2],[1]))
    # feed-forward layer
    network["6"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["5"], p=0.5),num_units=100,nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    # output layer
    network["7"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["6"], p=0.5),num_units=int(config['setup_params']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.sigmoid)

    return network["7"],network

def smallSquared(input_var, config):
    network={}
    # input layer
    network["1"] = lasagne.layers.InputLayer(shape=(None,1, int(config['patches_params']['xInput']), int(config['patches_params']['yInput'])),input_var=input_var)
    # convolutional layer
    network["2"] = lasagne.layers.Conv2DLayer(lasagne.layers.dropout(network["1"], p=0.25), num_filters=32, filter_size=(3,3),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    network["bn2"] = lasagne.layers.batch_norm(network["2"])
    # pooling layer
    network["3"] = lasagne.layers.MaxPool2DLayer(network["bn2"], pool_size=(4,1))
    # convolutional layer
    network["4"] = lasagne.layers.Conv2DLayer(lasagne.layers.dropout(network["3"], p=0.25), num_filters=32, filter_size=(3,3),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    print network['4'].input_shape
    print network['4'].output_shape
    network["bn4"] = lasagne.layers.batch_norm(network["4"])
    # pooling layer
    network["5"] = lasagne.layers.MaxPool2DLayer(network["bn4"], pool_size=(1,4))
    # convolutional layer
    network["6"] = lasagne.layers.Conv2DLayer(lasagne.layers.dropout(network["5"], p=0.25), num_filters=64, filter_size=(3,3),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    print network['6'].input_shape
    print network['6'].output_shape
    network["bn6"] = lasagne.layers.batch_norm(network["6"])
    # pooling layer
    network["7"] = lasagne.layers.MaxPool2DLayer(network["bn6"], pool_size=(2,2))
    # convolutional layer
    network["8"] = lasagne.layers.Conv2DLayer(lasagne.layers.dropout(network["7"], p=0.25), num_filters=32, filter_size=(3,3),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    print network['8'].input_shape
    print network['8'].output_shape
    network["bn8"] = lasagne.layers.batch_norm(network["8"])
    # pooling layer
    network["9"] = lasagne.layers.MaxPool2DLayer(network["bn8"], pool_size=(2,2))
    # convolutional layer
    network["10"] = lasagne.layers.Conv2DLayer(lasagne.layers.dropout(network["9"], p=0.25), num_filters=16, filter_size=(3,3),nonlinearity=lasagne.nonlinearities.elu,W=lasagne.init.HeUniform())
    print network['10'].input_shape
    print network['10'].output_shape
    network["bn10"] = lasagne.layers.batch_norm(network["10"])
    # output layer
    network["11"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["bn10"], p=0.5),num_units=int(config['setup_params']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.sigmoid)

    return network["11"],network