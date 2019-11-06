/**
 * Currently used error codes, feel free to add some.
 */
const httpError = {
    // 2xx SUCCESS
    OK: 200,
    CREATED: 201,

    // 3xx REDIRECTION

    // 4xx CLIENT ERRORS
    BAD_REQUEST: 400,
    UNAUTHORIZED: 401,
    FORBIDDEN: 403,
    NOT_FOUND: 404,
    NOT_ALLOWED: 405,
    CONFLICT: 409,

    // 5xx SERVER ERRORS
    INTERNAL_SERVER_ERROR: 500,
    NOT_IMPLEMENTED: 501
};

module.exports = httpError;
