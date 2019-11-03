module.exports = function(app) {
    // Add always enabled routes.
    let controllers = {
        indexController: require('./src/controllers/index'),
        usersController: require('./src/controllers/users')
    };

    Object.keys(controllers).forEach(function (controllerName) {
        app.use(controllers[controllerName]);
    });
};
