const tf = require('@tensorflow/tfjs-node');
const path=require("path"),fs=require("fs");


class Predict {
    constructor() {
        this.modelFilePath =path.join(__dirname,"../model/text_model");
        this.model=null;
        this.labels={};
    }
    async init(){
        this.labels=JSON.parse(fs.readFileSync(`${this.modelFilePath}/label.json`,"utf-8"));
        // console.log(this.labels)
        this.model = await tf.loadLayersModel(`file://${this.modelFilePath}/model.json`);
        console.log('loaded model:',this.model);
        const zeros = tf.zeros([1, 24, 32, 1]);
        let res=this.run(zeros);
        console.log(res)
    }
    getResult(predictRes){
        console.log(predictRes.arraySync())
        predictRes=Array.from(predictRes.arraySync(),res=>{
            
            return Array.from(res,(r,i)=>{
                return {
                    label:this.labels[i],
                    score:r
                }
            }).sort((a,b)=>{return b.score-a.score})
        });
        return predictRes
    }
    run(textTensor){
        let predictRes=this.model.predict(textTensor);
        return this.getResult(predictRes);
    }
}



// let {texts:newTexts}=data.newData();
// let predictRes=model.predict(newTexts);
// console.log(await tf.data.array(predictRes.dataSync()).take(data.labelSize).toArray())

module.exports =new Predict();