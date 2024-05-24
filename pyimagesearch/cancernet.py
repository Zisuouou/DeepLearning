class CancerNet:
    @staticmethod # 클래스를 부르면 자동으로 실행 되라고 
    def build(width, height, depth, classes):
        # initalize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
            
        # Conv => RELU => POOL
        model.add(SeparableConv2D(32, (3, 3), padding = "same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))
        # (CONV => RELU => POOL) * 2
        model.add(SeparableConv2D(64, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(SeparableConv2D(64, (3, 3), padding = "smae"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))
        # (CONV => RELU => POOL) * 3
        model.add(SeparableConv2D(128, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(SeparableConv2D(128, (3, 3), padding = "smae"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))   
        model.add(SeparableConv2D(128, (3, 3), padding = "smae"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))  
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))       
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        # return the constucted network architecture
        return model