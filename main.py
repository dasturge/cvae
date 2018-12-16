import tensorflow as tf

import inputs
from optimization import create_model

# define shorthands
TensorBoard = tf.keras.callbacks.TensorBoard


def main():
    batch_size = 16
    record_files = ['./model/train.tfrecord']
    test_record = ['./model/test.tfrecord']
    pickup_where_i_left_off = False

    params = (7, 18, 28, 2, 800)

    model = create_model(
        *params
    )

    train, train2 = inputs.image_input_fn(filenames=record_files, train=True, batch_size=batch_size)
    test, test2 = inputs.image_input_fn(filenames=test_record, train=False, batch_size=batch_size)
    dirname = './cvae/main_runner'
    callback_log = TensorBoard(
        log_dir=dirname, histogram_freq=0, batch_size=batch_size,
        write_graph=True, write_grads=False, write_images=False)

    if pickup_where_i_left_off:
        old_model = tf.keras.models.load_model('./main_model.keras', custom_objects={'loss': tf.keras.losses.mean_squared_error})
        for new_layer, old_layer in zip(model.layers, old_model.layers):
            new_layer.set_weights(old_layer.get_weights())
        del old_model

    for i in range(12):

        history = model.fit(x=train, y=train2, epochs=25, validation_data=(test, test2),
                            steps_per_epoch=int(2445 * .9 / batch_size) + 1, callbacks=[callback_log],
                            validation_steps=int(2445 * .1 / batch_size) + 1)

        print(history.history['val_mean_squared_error'][-1])

        model.save('./main_model.keras')


if __name__ == '__main__':
    main()
