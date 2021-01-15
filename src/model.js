const tf = require('@tensorflow/tfjs-node');

class Model {
  constructor() {
    this.model=null;
  }
  init(labelsNum){
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
      inputShape: [24, 32, 1],
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
    }));
    model.add(tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
    model.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu',
    }));
    model.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu',
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dropout({rate: 0.25}));
    model.add(tf.layers.dense({units: 512, activation: 'relu'}));
    model.add(tf.layers.dropout({rate: 0.5}));
    model.add(tf.layers.dense({units: labelsNum, activation: 'softmax'}));
    
    const optimizer = 'rmsprop';
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
    return model
  }
  
}


module.exports = new Model();