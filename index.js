const express = require("express");
const { json } = express;
const cors = require('cors');
const bodyParser = require('body-parser');
const http = require("http");
const swaggerUi = require('swagger-ui-express');
const swaggerDocument = require('./swagger.json');
const app = express();

const { urlencoded } = bodyParser;

app.use(json());

app.use(cors());

app.use(urlencoded({ extended: false }));
app.use(
    '/api-docs',
    swaggerUi.serve,
    swaggerUi.setup(swaggerDocument)
);

const tf = require('@tensorflow/tfjs');

// Define the logistic regression model
function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 4, inputShape: [3] }));
    model.add(tf.layers.activation({ activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 3 }));
    model.add(tf.layers.activation({ activation: 'softmax' }));
    model.compile({ optimizer: tf.train.adam(), loss: 'categoricalCrossentropy' });
    return model;
}

// Prepare the data
const data = tf.tensor2d([
    [3.565647252, 75.7286512, 234695],
    [2.361176978, 108.8976072, 36168],
    [1.661886078, 12.44229468, 44229],
    [1.400158009, 58.51313451, 17207],
    [1.150561798, 51.6494382, 2941],
    [1.434367542, 39.92840095, 1643]
]);

const labels = tf.tensor2d([
    [1, 0, 0], // High possibility
    [1, 0, 0], // High possibility
    [0, 1, 0], // Medium possibility
    [0, 1, 0], // Medium possibility
    [0, 0, 1], // Low possibility
    [0, 1, 0]  // Medium possibility
]);


app.get("/", async function (req, res) {
    res.json({ status: true })
})

app.post("/check", async function (req, res) {
    // // Train the model
    const { engagementTime, eventCount, viewRate } = req.body
    if (engagementTime <= 0 || eventCount <= 0 || viewRate <= 0) {
        res.json({ status: 500, message: "Invalid request format, Please enter valid values" })
    } else {
        const model = createModel();
        model.fit(data, labels, { epochs: 100 }).then(() => {
            // Use the model to make predictions
            const prediction = model.predict(tf.tensor2d([[engagementTime, eventCount, viewRate]], [1, 3]));
            const finalOutput = prediction.dataSync();
            console.log(Math.round(finalOutput[0]), Math.round(finalOutput[1]), Math.round(finalOutput[2]), finalOutput);

            if (Math.round(finalOutput[0]) && Math.round(finalOutput[1]) && Math.round(finalOutput[2])) {
                res.json({ status: 200, message: "High Possibility", possibilityShare: { high: finalOutput[0] * 100, medium: finalOutput[1] * 100, low: finalOutput[2] * 100 } })
            }
            else if ((Math.round(finalOutput[0]) && Math.round(finalOutput[1])) || (Math.round(finalOutput[0]) && Math.round(finalOutput[2])) || (Math.round(finalOutput[1]) && Math.round(finalOutput[2]))) {
                res.json({ status: 200, message: "Medium Possibility", possibilityShare: { high: finalOutput[0] * 100, medium: finalOutput[1] * 100, low: finalOutput[2] * 100 } })
            }
            else {
                res.json({ status: 200, message: "Low Possibility", possibilityShare: { high: finalOutput[0] * 100, medium: finalOutput[1] * 100, low: finalOutput[2] * 100 } });
            }



        });
    }

});

const serverPort = http.Server(app);

var server = serverPort.listen(3000, () => {

    console.log("server is running on port", server.address().port);

});
