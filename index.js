const path = require("path"),
    fs = require("fs");
const embedding = require("./src/embedding");
const data = require('./src/data');
const knn = require('./src/knn');
const textModel = require('./src/model');
const predict = require('./src/predict');

const Train = require('./train');

function createDataset() {

    let dataset = fs.readFileSync(path.join(__dirname, "dataset/test.txt"), "utf-8");
    // console.log(dataset)
    dataset = dataset.trim();
    let newData = [];
    dataset = Array.from(dataset.split("\n"), d => {
        let ds = d.split("___");
        if (ds[0].trim() != "undefined") {
            return {
                text: ds[1].replace(/\<.*\>|\s/ig, "").trim(),
                label: ds[0].trim()
            }
        } else {
            newData.push({
                text: ds[1].replace(/\<.*\>|\s/ig, "").trim(),
                label: ds[0].trim()
            })
        }

    }).filter(f => f);
    return dataset
}




async function train(dataset, modelSavePath) {

    await data.loadData(dataset);

    let { texts: trainTexts, labels: trainLabels } = data.getTrainData();
    // console.log(trainTexts.length)
    trainTexts = embedding.getBatch(trainTexts);
    // console.log('=', trainTexts[0])
    knn.train(trainTexts, trainLabels);

    // model.summary();

    // let epochBeginTime;
    // let millisPerStep;
    // const validationSplit = 0.1;
    // const numTrainExamplesPerEpoch =trainTexts.shape[0] * (1 - validationSplit);
    // const numTrainBatchesPerEpoch =
    //     Math.ceil(numTrainExamplesPerEpoch / batchSize);

    // await model.fit(trainTexts, trainLabels, {
    //   epochs,
    //   batchSize,
    //   validationSplit
    // });


    // const evalOutput = model.evaluate(testTexts, testLabels);

    // console.log(
    //     `\nEvaluation result:\n` +
    //     `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
    //     `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

    // if (modelSavePath != null) {
    //     await model.save(`file://${modelSavePath}`);
    //     console.log(`Saved model to path: ${modelSavePath}`);

    //     // console.log(fs.existsSync(`${modelSavePath}`))
    //     fs.writeFileSync(`${path.join(modelSavePath,'/label.json')}`,JSON.stringify(data.labels));
    //     console.log(`Saved labels to path: ${path.join(modelSavePath,'/label.json')}`);
    // };
    // await predict.init();
    // let {texts:newTexts}=data.newData(Array.from(newData,d=>d.text));
    // let predictRes=predict.run(newTexts);
    // predictRes=Array.from(newData,(d,i)=>{
    //   console.log(i,d,predictRes[i])
    //     return {
    //       text:d,
    //       result:predictRes[i]
    //     }
    // });
    //console.log(JSON.stringify(predictRes,null,2))
}

async function test() {
    const { texts: testTexts, labels: testLabels } = data.getTestData();
    for (let index = 0; index < testTexts.length; index++) {
        const t = testTexts[index];
        let v = embedding.get(t);
        let res = await knn.predict(v);
        // console.log(res.label)

        console.log(v, data.label2text(res.label), data.label2text(testLabels[index]))
    }
}

async function run(modelLocalPath) {
    let dataset = createDataset();
    await embedding.init(modelLocalPath);
    await train(dataset)
    await test();
}

// run(path.join(__dirname, "model/bert_zh_L-12_H-768_A-12_2"))
module.exports = {
    textTrain: new Train()
};