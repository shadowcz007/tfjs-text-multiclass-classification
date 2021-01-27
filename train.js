const path = require("path"),
    fs = require("fs");
const embedding = require("./src/embedding");
const data = require('./src/data');
const knn = require('./src/knn');
const textModel = require('./src/model');
const predict = require('./src/predict');


//run(path.join(__dirname, "model/bert_zh_L-12_H-768_A-12_2"))


module.exports = Train;