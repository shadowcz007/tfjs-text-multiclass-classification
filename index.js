const path=require("path"),fs=require("fs");
const data = require('./src/data');
const knn = require('./src/knn');
const textModel = require('./src/model');
const predict=require('./src/predict');

let dataset=fs.readFileSync(path.join(__dirname,"dataset/test.txt"),"utf-8");
// console.log(dataset)
dataset=dataset.trim();
let newData=[];
dataset=Array.from(dataset.split("\n"),d=>{
  let ds=d.split("___");
  if(ds[0].trim()!="undefined"){
    return {
      text:ds[1].replace(/\<.*\>|\s/ig,"").trim(),
      label:ds[0].trim()
    }
  }else{
    newData.push({
      text:ds[1].replace(/\<.*\>|\s/ig,"").trim(),
      label:ds[0].trim()
    })
  }

}).filter(f=>f);


async function run(epochs, batchSize, modelSavePath) {
  await data.loadData(dataset);
  // let model=textModel.init(data.labelSize);
  const {texts: trainTexts, labels: trainLabels} = data.getTrainData();
  // console.log(trainTexts)
  knn.train(trainTexts,trainLabels);

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

  const {texts: testTexts, labels: testLabels} = data.getTestData();
  for (let index = 0; index < testTexts.length; index++) {
    const t = testTexts[index];
    let res=await knn.predict(t);
    // console.log(res.label)

    console.log(data.label2text(res.label),data.label2text(testLabels[index]))
  }
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

run(100,10,path.join(__dirname,"model/text_model"))
