import argparse 
import importlib
import tensorflow as tf
import pathlib


def build_model(model_name, model, base_trainable, image_size, num_of_output, weights):
    if (model_name in ['MobileNetV3Small', 'MobileNetV3Large']):
        base_architecture = model(
            include_top=False,
            input_shape=image_size + (3, ),
            pooling='avg',
            minimalistic=True)
    else:
        base_architecture = model(
            include_top=False,
            input_shape=image_size + (3, ),
            pooling='avg')

    base_architecture.trainable = base_trainable
    outputs = tf.keras.layers.Dense(num_of_output, activation="softmax")(base_architecture.output)

    model = tf.keras.Model(base_architecture.input, outputs)

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    model.load_weights(weights)

    return model


def main(args):
    module = importlib.import_module("tensorflow.keras.applications")
    backbone = getattr(module, args.model_name)
    num_output = 102 if args.cfg == "oxford_flowers102" else 100

    model = build_model(model=backbone,
                        model_name=args.model_name,
                        base_trainable=False,
                        image_size=(224, 224),
                        num_of_output=num_output,
                        weights=args.weights)

    model.save(f'generated/{args.cfg}/{args.model_name}/saved-{args.num}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name')
    parser.add_argument('--cfg')
    parser.add_argument('--weights')
    parser.add_argument('--num')
    
    args = parser.parse_args()
    main(args)
