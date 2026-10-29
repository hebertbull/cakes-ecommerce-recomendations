// ====================================================================
// 🗄️ Servidor do projeto (Aula 7 - opção B, versão leve)
// ====================================================================
// Faz duas coisas na mesma porta:
//  1. Serve os arquivos do app (index.html, src/, data/ ...)
//  2. Expõe um "banco vetorial" de verdade: SQLite + extensão sqlite-vec
//
//  POST /api/index   { vectors: [{ id, vector: [...] }] }  → regrava a tabela de produtos
//  POST /api/search  { vector: [...], k }                   → devolve os k produtos mais próximos
//  GET  /api/health                                         → { indexed, dimensions }
//
// O navegador não fala direto com um banco: o worker chama este servidor.
// Requer Node >= 22.13 (node:sqlite sem flag). O node:sqlite ainda é "experimental".
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';
import * as sqliteVec from 'sqlite-vec';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PORT = Number(process.env.PORT) || 3000;
const DB_FILE = process.env.VECTOR_DB || path.join(ROOT, 'server', 'vectors.db');
const MAX_BODY_BYTES = 10 * 1024 * 1024;
const MAX_DIMENSIONS = 4096;
const MAX_K = 1000;

// ---------- banco ----------
const db = new DatabaseSync(DB_FILE, { allowExtension: true });
sqliteVec.load(db);
db.exec('CREATE TABLE IF NOT EXISTS vector_meta (key TEXT PRIMARY KEY, value TEXT)');

const getMeta = key => db.prepare('SELECT value FROM vector_meta WHERE key = ?').get(key)?.value;
const setMeta = (key, value) => db.prepare('INSERT OR REPLACE INTO vector_meta(key, value) VALUES (?, ?)').run(key, String(value));

// float32 em bytes: é o formato que o sqlite-vec espera
const toBlob = vector => new Uint8Array(new Float32Array(vector).buffer);

function indexVectors(vectors) {
    const dimensions = vectors[0].vector.length;
    if (!Number.isInteger(dimensions) || dimensions < 1 || dimensions > MAX_DIMENSIONS) {
        throw new HttpError(400, `dimensão inválida: ${dimensions}`);
    }
    const seen = new Set();
    for (const { id, vector } of vectors) {
        if (!Number.isSafeInteger(id) || id < 1) throw new HttpError(400, `id inválido: ${id}`);
        if (seen.has(id)) throw new HttpError(400, `id repetido: ${id}`);
        seen.add(id);
        if (!isNumericVector(vector, dimensions)) throw new HttpError(400, `vetor do id ${id} precisa ter ${dimensions} números`);
    }

    // distance_metric=cosine: distância = 1 - similaridade do cosseno
    // `dimensions` já foi validado como inteiro acima, então é seguro no SQL
    db.exec('BEGIN');
    try {
        db.exec('DROP TABLE IF EXISTS produtos');
        db.exec(`CREATE VIRTUAL TABLE produtos USING vec0(embedding float[${dimensions}] distance_metric=cosine)`);
        const insert = db.prepare('INSERT INTO produtos(rowid, embedding) VALUES (?, ?)');
        for (const { id, vector } of vectors) insert.run(BigInt(id), toBlob(vector));
        setMeta('dimensions', dimensions);
        setMeta('count', vectors.length);
        db.exec('COMMIT');
    } catch (error) {
        db.exec('ROLLBACK');
        throw error;
    }
    return { indexed: vectors.length, dimensions };
}

function searchVectors(vector, k) {
    const dimensions = Number(getMeta('dimensions'));
    if (!dimensions) throw new HttpError(409, 'nenhum vetor indexado ainda: chame /api/index primeiro');
    if (!isNumericVector(vector, dimensions)) throw new HttpError(400, `o vetor precisa ter ${dimensions} números`);
    // vetor zerado não tem direção: o cosseno não faz sentido
    if (!vector.some(v => v !== 0)) throw new HttpError(400, 'vetor zerado: a busca por cosseno não faz sentido');
    if (!Number.isInteger(k) || k < 1 || k > MAX_K) throw new HttpError(400, `k precisa ser inteiro entre 1 e ${MAX_K}`);

    const rows = db
        .prepare('SELECT rowid, distance FROM produtos WHERE embedding MATCH ? AND k = ? ORDER BY distance')
        .all(toBlob(vector), k);
    return rows.map(row => ({
        id: Number(row.rowid),
        distance: row.distance,
        similarity: 1 - row.distance
    }));
}

function isNumericVector(vector, dimensions) {
    return Array.isArray(vector) && vector.length === dimensions && vector.every(v => typeof v === 'number' && Number.isFinite(v));
}

// ---------- HTTP ----------
class HttpError extends Error {
    constructor(status, message) { super(message); this.status = status; }
}

const sendJson = (res, status, body) => {
    res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
    res.end(JSON.stringify(body));
};

function readJson(req) {
    return new Promise((resolve, reject) => {
        const chunks = [];
        let size = 0;
        req.on('data', chunk => {
            size += chunk.length;
            if (size > MAX_BODY_BYTES) { reject(new HttpError(413, 'corpo grande demais')); req.destroy(); return; }
            chunks.push(chunk);
        });
        req.on('end', () => {
            try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
            catch { reject(new HttpError(400, 'JSON inválido')); }
        });
        req.on('error', reject);
    });
}

async function handleApi(req, res, pathname) {
    if (pathname === '/api/health' && req.method === 'GET') {
        return sendJson(res, 200, { indexed: Number(getMeta('count') || 0), dimensions: Number(getMeta('dimensions') || 0) });
    }
    if (pathname === '/api/index' && req.method === 'POST') {
        const body = await readJson(req);
        if (!Array.isArray(body?.vectors) || !body.vectors.length) throw new HttpError(400, 'envie { vectors: [{ id, vector }] }');
        const startedAt = performance.now();
        const result = indexVectors(body.vectors);
        return sendJson(res, 200, { ...result, ms: Number((performance.now() - startedAt).toFixed(1)) });
    }
    if (pathname === '/api/search' && req.method === 'POST') {
        const body = await readJson(req);
        const startedAt = performance.now();
        const results = searchVectors(body?.vector, body?.k);
        return sendJson(res, 200, { results, ms: Number((performance.now() - startedAt).toFixed(2)) });
    }
    throw new HttpError(404, 'rota não encontrada');
}

// ---------- arquivos estáticos ----------
const MIME = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.png': 'image/png',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
};

function serveStatic(req, res, pathname) {
    if (req.method !== 'GET' && req.method !== 'HEAD') throw new HttpError(405, 'método não permitido');
    let relative;
    try { relative = decodeURIComponent(pathname); } catch { throw new HttpError(400, 'caminho inválido'); }
    if (relative.endsWith('/')) relative += 'index.html';

    const absolute = path.resolve(ROOT, '.' + relative);
    const fromRoot = path.relative(ROOT, absolute);
    const parts = fromRoot.split(path.sep);
    // nada fora do projeto, nada escondido (.git, .env), nem node_modules/ ou server/ (onde fica o banco)
    const blocked = fromRoot.startsWith('..') || path.isAbsolute(fromRoot) || !fromRoot
        || parts.some(part => part.startsWith('.') || part === 'node_modules' || part === 'server');
    if (blocked) throw new HttpError(404, 'não encontrado');

    let stat;
    try { stat = fs.statSync(absolute); } catch { throw new HttpError(404, 'não encontrado'); }
    if (!stat.isFile()) throw new HttpError(404, 'não encontrado');

    res.writeHead(200, {
        'Content-Type': MIME[path.extname(absolute).toLowerCase()] || 'application/octet-stream',
        'Content-Length': stat.size,
        'Cache-Control': 'no-cache',
    });
    if (req.method === 'HEAD') return res.end();
    fs.createReadStream(absolute).pipe(res);
}

const server = http.createServer(async (req, res) => {
    try {
        const { pathname } = new URL(req.url, 'http://localhost');
        if (pathname.startsWith('/api/')) return await handleApi(req, res, pathname);
        return serveStatic(req, res, pathname);
    } catch (error) {
        const status = error instanceof HttpError ? error.status : 500;
        if (status === 500) console.error(error);
        if (res.headersSent) return res.destroy();
        sendJson(res, status, { error: status === 500 ? 'erro interno' : error.message });
    }
});

// só aceita conexões da própria máquina
server.on('error', error => {
    if (error.code === 'EADDRINUSE') {
        console.error(`A porta ${PORT} já está em uso (talvez o browser-sync antigo ou outro "npm start"). Pare o outro processo ou rode com outra porta: PORT=3001 npm start`);
    } else {
        console.error(`Não foi possível iniciar o servidor: ${error.message}`);
    }
    process.exit(1);
});
server.listen(PORT, '127.0.0.1', () => {
    console.log(`Servidor em http://localhost:${PORT}  (banco vetorial: ${path.relative(ROOT, DB_FILE) || DB_FILE})`);
});
