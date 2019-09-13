# Tensorflow Deterministic Solution

* Set `PYTHONHASHSEED` before another imports:
    

    os.environ['PYTHONHASHSEED'] = str(settings.RANDOM_SEED)
    
* Set random seed:


    import random
    random.seed(RANDOM_SEED)
    
* Set NumPy random seed


    import numpy as np
    np.random.seed(RANDOM_SEED)
    
* Use `set_random_seed` tensorflow method before any tensorflow's imports
    
    

* Set keras backend (tenserflow) session with TF config (intra_op_parallelism_threads=1, inter_op_parallelism_threads=1) and set Tensorflow random seed:
 
 
    import tensorflow as tf
    tf.reset_default_graph()
    tf.set_random_seed(RANDOM_SEED)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    tf.keras.backend.set_session(sess)
    tf.set_random_seed(RANDOM_SEED)
    
* Set **random seed** and enable **shuffle** for each `tensorflow.keras.preprocessing.image.ImageDataGenerator.flow_from_directory`  call

Example:

    flow_from_directory(train_dir,
                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                        batch_size=BATCH_SIZE,
                        seed=RANDOM_SEED,  # <<-- This
                        shuffle=True,      # <<-- And this
                        class_mode='categorical')
                        
* Set **random seed** fpr each initializer in all layers (like Convolution2D, Dense, etc.).

Example:

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Convolution2D(filters=56, kernel_size=(3, 3), activation='relu',
                                            input_shape=train_generator.image_shape,
                                            kernel_initializer=tf.keras.initializers.glorot_uniform(
                                                seed=RANDOM_SEED)))  # <<-- This
                                                
                                                
* Disable **shuffle** for `fit_generator`

Example:

    model.fit_generator(train_generator,
                        epochs=EPOCHS,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        shuffle=False,     # <<-- This
                        callbacks=[])