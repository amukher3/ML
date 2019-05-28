def discriminator(img_shape):
    inp = Input(shape=[*img_shape, 1], name="image")
    
    x = Reshape([*img_shape, 1], name="disc_reshape")(inp)
    
    x = Conv2D(64, 5, strides=2, padding="same", name="disc_conv_1")(x)
    x = LeakyReLU(alpha=0.01, name="disc_act_1")(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(128, 5, strides=2, padding="same", name="disc_conv_2")(x)
    x = LeakyReLU(alpha=0.01, name="disc_act_2")(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(256, 5, strides=2, padding="same", name="disc_conv_3")(x)
    x = LeakyReLU(alpha=0.01, name="disc_act_3")(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(512, 5, strides=1, padding="same", name="disc_conv_4")(x)
    x = LeakyReLU(alpha=0.01, name="disc_act_4")(x)
    x = Dropout(0.4)(x)
x = Flatten()(x)
out = Dense(1, activation="sigmoid", name="disc_out")(x)
return Model(inp, out, name="discriminator")
