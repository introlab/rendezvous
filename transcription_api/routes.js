module.exports = function(app) {
    // Add always enabled routes.
    let controllers = {
        indexController: require('./src/controllers/index'),
        transcriptionController: require('./src/controllers/transcription')
    };

    Object.keys(controllers).forEach(function (controllerName) {
        app.use(controllers[controllerName]);
    });
};
