const { Bert } = require('bert');
// console.log(__dirname)

class Embedding {
    constructor() {}
    async init(modelLocalPath) {
        // console.log(modelLocalPath)
        if (modelLocalPath) {
            this.bert = new Bert({
                modelLocalPath: modelLocalPath
            });
            await this.bert.init()
        };
    }
    get(text) {
        //bert的长度是500
        // console.log(this.bert)
        if (this.bert) return this.bert.predictAndStore(text.slice(0, 500))


    }
    getBatch(texts) {
        return Array.from(texts, (t, i) => this.get(t));
    }
}


module.exports = new Embedding();