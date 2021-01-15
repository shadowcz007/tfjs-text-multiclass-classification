const knnClassifier=require("@tensorflow-models/knn-classifier");

class Knn{
    constructor(){
        this.knn = knnClassifier.create();
        this.topk=20;
    }

    add(tensor,className){
        // console.log('+===',tensor,className)
        this.knn.addExample(tensor, className);
    }

    train(tensors=[],classNames=[]){
        for (let index = 0; index < tensors.length; index++) {
            const t = tensors[index];
            this.add(t,classNames[index]);
        }
    }

   async predict(tensor){
        return await this.knn.predictClass(tensor, this.topk);
    }
    
    export2str(){
        let dataset = this.knn.getClassifierDataset();
        var datasetObj = {};
        Object.keys(dataset).forEach((key) => {
            let data = dataset[key].dataSync();
            var shape = dataset[key].shape,
                dtype = dataset[key].dtype;
            datasetObj[key] = {
                tensor: Array.from(data),
                shape: shape,
                dtype: dtype
            };
        });

        let jsonModel=JSON.stringify(datasetObj)
        //localStorage.setItem("easyteach_model",jsonModel);
        return jsonModel;
    }

}


module.exports = new Knn();