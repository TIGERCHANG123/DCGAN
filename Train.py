import tensorflow as tf

#1. soft label
#2. fake->1, real->0
def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss, real_loss, fake_loss

def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)

class train_one_epoch():
    def __init__(self, model, train_dataset, optimizers, metrics, noise_dim):
        self.generator, self.discriminator = model
        self.generator_optimizer, self.discriminator_optimizer = optimizers
        self.gen_loss, self.disc_loss = metrics
        self.train_dataset = train_dataset
        self.noise_dim = noise_dim
        self.real_loss = 0
        self.fake_loss = 0

    def train_discriminator_step(self, noise, images):
        with tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            disc_loss, self.real_loss, self.fake_loss = discriminator_loss(real_output, fake_output)
        self.disc_loss(disc_loss)
        # print('disc: {}, {}'.format(disc_loss, self.disc_loss.result()))
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    def train_generator_step(self, noise):
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = generator_loss(fake_output)
        self.gen_loss(gen_loss)
        # print('gen: {}, {}'.format(gen_loss, self.gen_loss.result()))
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

    def train(self, epoch,  pic):
        self.gen_loss.reset_states()
        self.disc_loss.reset_states()
        k = 0
        gen_times = 1
        disc_times = 1
        for (batch, (images, labels)) in enumerate(self.train_dataset):
            if k < gen_times:
                k = k + 1
                noise = tf.random.normal([images.shape[0], self.noise_dim])
                self.train_generator_step(noise)
            elif k < (gen_times + disc_times):
                k = k+1
                noise = tf.random.normal([images.shape[0], self.noise_dim])
                self.train_discriminator_step(noise, images)
                pic.add([self.gen_loss.result().numpy(), self.disc_loss.result().numpy()])
                pic.save()
            else:
                k = 0
            if (batch + 1) % 100 == 0:
                print('epoch: {}, gen loss: {}, disc loss: {}, real loss: {}, fake loss{}'
                      .format(epoch, self.gen_loss.result(), self.disc_loss.result(), self.real_loss, self.fake_loss))