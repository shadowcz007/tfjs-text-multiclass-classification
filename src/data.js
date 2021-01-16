const tf = require('@tensorflow/tfjs-node');
const fs = require('fs'),
    path = require('path');


// bert.predict("中文")

//  data constants:
const TEXT_FLAT_SIZE = 768;

/** Helper class to handle loading training and test data. */
class TextDataset {
    constructor() {
        this.dataset = null;
        this.trainSize = 0;
        this.testSize = 0;
        this.trainBatchIndex = 0;
        this.testBatchIndex = 0;
    }

    /** Loads training and test data. */
    async loadData(dataset = []) {


        dataset = this.initLabels(dataset);

        tf.util.shuffle(dataset);

        this.trainSize = parseInt(0.8 * dataset.length);

        dataset = tf.data.array(dataset);
        let trainData = await (dataset.take(this.trainSize)).toArray();
        let testData = await (dataset.skip(this.trainSize)).toArray();

        this.testSize = testData.length;

        this.dataset = [trainData, testData];
    }

    initLabels(dataset) {
        let labels = {};
        let labelsIndex = {};
        dataset.forEach(d => {
            if (!labels[d.label]) {
                labels[d.label] = (Object.keys(labels)).length;
                labelsIndex[labels[d.label]] = d.label;
            };
        });

        dataset = Array.from(dataset, d => {
            d.label = labels[d.label];
            return d
        });
        this.labels = labelsIndex;
        this.labelSize = (Object.keys(labels)).length;

        return dataset
    }

    label2text(index) {
        // console.log('this.labels:',this.labels)
        index = parseInt(index);
        return this.labels[index]
    }

    getTrainData() {
        return this.getData_(true);
    }

    getTestData() {
        return this.getData_(false);
    }

    getData_(isTrainingData) {
        let textsIndex;

        if (isTrainingData) {
            textsIndex = 0;
        } else {
            textsIndex = 1;
        }

        const data = this.dataset[textsIndex];
        const size = data.length;
        // console.log("size::::::::", size)

        // Only create one big array to hold batch of texts.
        const texts = [];
        const labels = [];

        for (let index = 0; index < size; index++) {
            const t = data[index];
            //console.log(t.text.slice(0,500))
            texts.push(t.text);

            // texts.push(tf.tensor(v));
            labels.push(t.label);
        }

        return {
            texts: texts,
            labels: labels,
            size: size
        };
    }
    newData(data = []) {

        const size = data.length;
        const texts = [];

        for (let index = 0; index < size; index++) {
            const t = data[index];
            texts.push(t.text);
        }
        return {
            texts: texts,
            size: size
        };

    }
}


module.exports = new TextDataset();