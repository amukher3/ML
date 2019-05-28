def generative_adversarial_model(img_shape, z_dim):
    gen = generator(img_shape, z_dim)
    disc = discriminator(img_shape)
    
    inp = Input(shape=(*gen.input_shape[1:],), name="image")
    img = gen(inp)
    # NOTE: We do not want the discriminator to be trainable
    disc.trainable = False
    prediction = disc(img)
    gan = Model(inp, prediction, name="gan")
    gan.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return gan
