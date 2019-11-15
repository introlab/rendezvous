let express = require('express');
let router = express.Router();
let path = require('path');

let httpErrors = require('../utils/HttpError');

/* GET home page. */
router.get('/', function(req, res, next) {
  if (process.env.NODE_ENV === 'production') {
    return res.sendStatus(httpErrors.OK);
  }

  res.sendFile(path.join(__dirname, '../../public', 'index.html'));
});

module.exports = router;
