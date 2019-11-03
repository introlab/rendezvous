if (process.env.NODE_ENV === 'production') {
    require('dotenv').config({ path: 'config/production/.env' });
    process.stdout.write('production mode...\n');
} else {
    require('dotenv').config({ path: 'config/development/.env' });
    process.stdout.write('development mode..\n');
}