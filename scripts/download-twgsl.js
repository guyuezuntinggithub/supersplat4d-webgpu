/**
 * 尝试下载 twgsl.js 与 twgsl.wasm 到 static/lib/twgsl/
 * npm 包 twgsl@0.0.10 已 404，本脚本会尝试从 Wayback Machine 等源拉取。
 */
const fs = require('fs');
const path = require('path');
const https = require('https');

const OUT_DIR = path.join(__dirname, '..', 'static', 'lib', 'twgsl');

const SOURCES = [
    'https://web.archive.org/web/20240000000000/https://unpkg.com/twgsl@0.0.10/dist/twgsl.js',
    'https://web.archive.org/web/20230000000000/https://unpkg.com/twgsl@0.0.10/dist/twgsl.js'
];

const WASM_SOURCES = [
    'https://web.archive.org/web/20240000000000/https://unpkg.com/twgsl@0.0.10/dist/twgsl.wasm',
    'https://web.archive.org/web/20230000000000/https://unpkg.com/twgsl@0.0.10/dist/twgsl.wasm'
];

function fetch(url) {
    return new Promise((resolve, reject) => {
        https.get(url, { headers: { 'User-Agent': 'Node-download-twgsl' } }, (res) => {
            if (res.statusCode !== 200) {
                reject(new Error(`${url} → ${res.statusCode}`));
                return;
            }
            const chunks = [];
            res.on('data', (c) => chunks.push(c));
            res.on('end', () => resolve(Buffer.concat(chunks)));
        }).on('error', reject);
    });
}

async function tryDownload(sources, filename) {
    for (const url of sources) {
        try {
            const buf = await fetch(url);
            if (buf && buf.length > 100) {
                fs.mkdirSync(OUT_DIR, { recursive: true });
                fs.writeFileSync(path.join(OUT_DIR, filename), buf);
                console.log(`OK: ${filename} (${buf.length} bytes) from ${url.slice(0, 60)}...`);
                return true;
            }
        } catch (e) {
            console.warn(`Skip: ${url} - ${e.message}`);
        }
    }
    return false;
}

async function main() {
    console.log('Downloading twgsl to static/lib/twgsl/ ...');
    const okJs = await tryDownload(SOURCES, 'twgsl.js');
    const okWasm = await tryDownload(WASM_SOURCES, 'twgsl.wasm');
    if (okJs && okWasm) {
        console.log('Done. twgsl.js and twgsl.wasm are in place.');
    } else {
        console.log('Could not download from archive. Put twgsl.js and twgsl.wasm in static/lib/twgsl/ manually (see that folder README).');
        process.exitCode = 1;
    }
}

main();
