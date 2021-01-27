const fs = require("fs");
const embedding = require("./src/embedding");
const data = require('./src/data');
const knn = require('./src/knn');

class Train {
    constructor() {}

    async train(dataset, modelSavePath) {

        await data.loadData(dataset);

        let { texts: trainTexts, labels: trainLabels } = data.getTrainData();
        // console.log(trainTexts.length)
        trainTexts = embedding.getBatch(trainTexts);
        // console.log('=', trainTexts[0])
        knn.train(trainTexts, trainLabels);


        return knn
    }

    async start(dataset, modelLocalPath, bertModel, saveLocalPath) {
        await embedding.init(modelLocalPath, bertModel);
        let model = await this.train(dataset);
        let modelRes = model.export2str();
        let json = {
            labels: data.labels,
            model: modelRes
        };
        fs.writeFileSync(saveLocalPath, JSON.stringify(json));
        return json
    }
}

class TextModel {
    constructor(modelLocalPath) {
        let jsonStr = fs.readFileSync(modelLocalPath, "utf-8");
        let json = JSON.parse(jsonStr);
        this.labels = json.labels;
        knn.load(json.model)
    }
    async predict(text) {
        let v = embedding.get(text);
        let res = await knn.predict(v);
        return this.labels[res.label];
    }
}

// run(path.join(__dirname, "model/bert_zh_L-12_H-768_A-12_2"))
module.exports = {
    textTrain: new Train(),
    TextModel: TextModel
};