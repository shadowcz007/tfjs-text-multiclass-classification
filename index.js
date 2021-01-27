const fs = require("fs");
const embedding = require("./src/embedding");
const data = require('./src/data');
const knn = require('./src/knn');

class Train {
    constructor() {}

    async train(dataset) {

        await data.loadData(dataset);

        let { texts: trainTexts, labels: trainLabels } = data.getTrainData();
        // console.log(trainTexts.length)
        trainTexts = embedding.getBatch(trainTexts);
        // console.log('=', trainTexts[0])
        knn.train(trainTexts, trainLabels);

        return knn
    }

    async start(dataset, saveLocalPath) {
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

async function initEmbedding(modelLocalPath, bertModel) {
    await embedding.init(modelLocalPath, bertModel);
}

class TextModel {
    constructor(modelLocalPath) {
        let jsonStr = fs.readFileSync(modelLocalPath, "utf-8");
        let json = JSON.parse(jsonStr);
        this.labels = json.labels;
        let initRes = knn.load(json.model);
        console.log('模型初始化：', initRes)
    }
    async predict(text) {
        let v = embedding.get(text);
        // console.log(v)
        let res = await knn.predict(v);
        return this.labels[res.label];
    }
}

// run(path.join(__dirname, "model/bert_zh_L-12_H-768_A-12_2"))
module.exports = {
    initEmbedding: initEmbedding,
    textTrain: new Train(),
    TextModel: TextModel
};