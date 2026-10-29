import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';
let _globalCtx = {};
let _model = null

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1,
};


// 🔢 Normalize continuous values (price, age) to 0–1 range
// Why? Keeps all features balanced so no one dominates training
// Formula: (val - min) / (max - min)
// Example: price=129.99, minPrice=39.99, maxPrice=199.99 → 0.56
const normalize = (value, min, max) => (value - min) / ((max - min) || 1)

function makeContext(products, users) {
    const ages = users.map(u => u.age)
    const prices = products.map(p => p.price)

    const minAge = Math.min(...ages)
    const maxAge = Math.max(...ages)

    const minPrice = Math.min(...prices)
    const maxPrice = Math.max(...prices)

    const colors = [...new Set(products.map(p => p.color))]
    const categories = [...new Set(products.map(p => p.category))]

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index]
        }))
    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index]
        }))

    // Computar a média de idade dos comprados por produto
    // (ajuda a personalizar)
    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1
        })
    })

    const productAvgAgeNorm = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] :
                midAge

            return [product.name, normalize(avg, minAge, maxAge)]
        })
    )

    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        productAvgAgeNorm,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        // price + age + colors + categories
        dimentions: 2 + categories.length + colors.length
    }
}

const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight)

function encodeProduct(product, context) {
    // normalizando dados para ficar de 0 a 1 e
    // aplicar o peso na recomendação
    const price = tf.tensor1d([
        normalize(
            product.price,
            context.minPrice,
            context.maxPrice
        ) * WEIGHTS.price
    ])

    const age = tf.tensor1d([
        (
            context.productAvgAgeNorm[product.name] ?? 0.5
        ) * WEIGHTS.age
    ])

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    )

    const color = oneHotWeighted(
        context.colorsIndex[product.color],
        context.numColors,
        WEIGHTS.color
    )

    return tf.concat1d(
        [price, age, category, color]
    )
}

function encodeUser(user, context) {
    if (user.purchases.length) {
        return tf.stack(
            user.purchases.map(
                product => encodeProduct(product, context)
            )
        )
            .mean(0)
            .reshape([
                1,
                context.dimentions
            ])
    }

    return tf.concat1d(
        [
            tf.zeros([1]), // preço é ignorado,
            tf.tensor1d([
                normalize(user.age, context.minAge, context.maxAge)
                * WEIGHTS.age
            ]),
            tf.zeros([context.numCategories]), // categoria ignorada,
            tf.zeros([context.numColors]), // color ignorada,

        ]
    ).reshape([1, context.dimentions])
}

function createTrainingData(context) {
    const inputs = []
    const labels = []
    context.users
        .filter(u => u.purchases.length)
        .forEach(user => {
            const userVector = encodeUser(user, context).dataSync()
            context.products.forEach(product => {
                const productVector = encodeProduct(product, context).dataSync()

                const label = user.purchases.some(
                    purchase => purchase.name === product.name ?
                        1 :
                        0
                )
                // combinar user + product
                inputs.push([...userVector, ...productVector])
                labels.push(label)

            })
        })

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimention: context.dimentions * 2
        // tamanho = userVector + productVector
    }
}

// ====================================================================
// 📌 Exemplo de como um usuário é ANTES da codificação
// ====================================================================
/*
const exampleUser = {
    id: 201,
    name: 'Rafael Souza',
    age: 27,
    purchases: [
        { id: 1, name: 'Bolo de Chocolate Belga', category: 'chocolate', price: 89.99, color: 'marrom' },
        { id: 6, name: 'Bolo de Morango com Chantilly', category: 'frutas', price: 84.99, color: 'rosa' }
    ]
};
*/

// ====================================================================
// 📌 Após a codificação, o modelo NÃO vê nomes ou palavras.
// Ele vê um VETOR NUMÉRICO (todos normalizados entre 0–1).
// Exemplo: [preço_normalizado, idade_normalizada, cat_one_hot..., cor_one_hot...]
//
// Suponha categorias = ['chocolate', 'frutas', 'tradicional']
// Suponha cores      = ['marrom', 'rosa', 'branco']
//
// Para Rafael (idade 27, categoria: chocolate, cores: marrom/rosa),
// o vetor poderia ficar assim:
//
// [
//   0.45,            // peso do preço normalizado
//   0.60,            // idade normalizada
//   1, 0, 0,         // one-hot de categoria (chocolate = ativo)
//   1, 0, 0          // one-hot de cores (marrom e rosa ativos, branco inativo)
// ]
//
// São esses números que vão para a rede neural.
// ====================================================================



// ====================================================================
// 🧠 Configuração e treinamento da rede neural
// ====================================================================
async function configureNeuralNetAndTrain(trainData) {

    const model = tf.sequential()
    // Camada de entrada
    // - inputShape: Número de features por exemplo de treino (trainData.inputDim)
    //   Exemplo: Se o vetor produto + usuário = 20 números, então inputDim = 20
    // - units: 128 neurônios (muitos "olhos" para detectar padrões)
    // - activation: 'relu' (mantém apenas sinais positivos, ajuda a aprender padrões não-lineares)
    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimention],
            units: 128,
            activation: 'relu'
        })
    )
    // Camada oculta 1
    // - 64 neurônios (menos que a primeira camada: começa a comprimir informação)
    // - activation: 'relu' (ainda extraindo combinações relevantes de features)
    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    )

    // Camada oculta 2
    // - 32 neurônios (mais estreita de novo, destilando as informações mais importantes)
    //   Exemplo: De muitos sinais, mantém apenas os padrões mais fortes
    // - activation: 'relu'
    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    )
    // Camada de saída
    // - 1 neurônio porque vamos retornar apenas uma pontuação de recomendação
    // - activation: 'sigmoid' comprime o resultado para o intervalo 0–1
    //   Exemplo: 0.9 = recomendação forte, 0.1 = recomendação fraca
    model.add(
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
    )

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    })

    return model
}
async function trainModel({ users }) {
    console.log('Training model with users:', users);
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 1 } });
    const products = await (await fetch('/data/products.json')).json()

    const context = makeContext(products, users)
    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: { ...product },
            vector: encodeProduct(product, context).dataSync()
        }
    })

    _globalCtx = context

    // 🗄️ Grava os vetores dos produtos no banco vetorial (server/server.js).
    //    Retreinou? O contexto (min/max, idade média) muda e os vetores mudam
    //    junto, então o índice inteiro é regravado.
    await indexProductVectors(context.productVectors)
    const trainData = createTrainingData(context)
    _model = await configureNeuralNetAndTrain(trainData)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

// ====================================================================
// 🔎 "Mini banco vetorial" (Aula 7 - opção A)
// ====================================================================
// Em vez de rodar a rede em TODOS os produtos, primeiro buscamos os N mais
// parecidos com o vetor do usuário (similaridade do cosseno) e só nesses
// rodamos o predict. É o que um banco vetorial faz (Pinecone, ChromaDB,
// pgvector), só que aqui a busca é exata e acontece na memória do worker.
//
// Se TOP_N_CANDIDATES >= número de produtos, o comportamento é igual ao antigo.
const TOP_N_CANDIDATES = 10

// cosseno = (a · b) / (|a| * |b|): 1 = mesma direção, 0 = nada em comum
function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    // vetor zerado não tem direção: evita dividir por zero (NaN)
    if (!normA || !normB) return 0
    return dot / (Math.sqrt(normA) * Math.sqrt(normB))
}

// devolve os produtos ordenados do mais para o menos parecido com o vetor
function searchBySimilarity(queryVector, productVectors) {
    return productVectors
        .map(item => ({ item, similarity: cosineSimilarity(queryVector, item.vector) }))
        .sort((a, b) => b.similarity - a.similarity)
}

// ====================================================================
// 🗄️ Banco vetorial de verdade (Aula 7 - opção B): SQLite + sqlite-vec
// ====================================================================
// O worker não fala direto com o banco: ele chama o server/server.js.
// Se o servidor/banco não estiver disponível (ex.: npm run start:static),
// o app continua funcionando com a busca local (searchBySimilarity).
let _dbReady = false

async function indexProductVectors(productVectors) {
    _dbReady = false
    try {
        const response = await fetch('/api/index', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                vectors: productVectors.map(({ meta, vector }) => {
                    return { id: meta.id, vector: Array.from(vector) }
                })
            })
        })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const result = await response.json()
        _dbReady = true
        console.log(`banco vetorial: ${result.indexed} produtos indexados (${result.dimensions} dimensões) em ${result.ms} ms`)
    } catch (error) {
        console.warn(`banco vetorial indisponível (${error.message}): usando a busca local`)
    }
}

// Devolve { ranked: [{ item, similarity }], source }, igual à busca local.
// Do banco vêm só os k mais próximos; os demais entram depois, na ordem do catálogo.
async function searchCandidates(queryVector, productVectors, k) {
    if (_dbReady) {
        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ vector: Array.from(queryVector), k })
            })
            if (!response.ok) throw new Error(`HTTP ${response.status}`)
            const { results } = await response.json()

            const byId = new Map(productVectors.map(item => [item.meta.id, item]))
            const found = results
                .filter(result => byId.has(result.id))
                .map(result => ({ item: byId.get(result.id), similarity: result.similarity }))
            const foundIds = new Set(found.map(({ item }) => item.meta.id))
            const rest = productVectors
                .filter(item => !foundIds.has(item.meta.id))
                .map(item => ({ item, similarity: null }))

            return { ranked: [...found, ...rest], source: 'banco' }
        } catch (error) {
            console.warn(`falha na busca no banco (${error.message}): usando a busca local`)
        }
    }
    return { ranked: searchBySimilarity(queryVector, productVectors), source: 'local' }
}

async function recommend({ user }) {
    if (!_model) return;
    const context = _globalCtx
    // 1️⃣ Converta o usuário fornecido no vetor de features codificadas
    //    (preço ignorado, idade normalizada, categorias ignoradas)
    //    Isso transforma as informações do usuário no mesmo formato numérico
    //    que foi usado para treinar o modelo.

    const userVector = encodeUser(user, context).dataSync()

    // Em aplicações reais:
    //  Armazene todos os vetores de produtos em um banco de dados vetorial (como Postgres, Neo4j ou Pinecone)
    //  Consulta: Encontre os 200 produtos mais próximos do vetor do usuário
    //  Execute _model.predict() apenas nesses produtos

    // 🔎 Busca vetorial: pega só os N produtos mais próximos do usuário.
    //    Usuário SEM compras: o vetor dele só tem a idade (categoria e cor
    //    zeradas), então "parecido" não significa nada. Nesse caso avalia todos.
    const startedAt = performance.now()
    const hasHistory = user.purchases.length > 0
    const { ranked, source } = hasHistory ?
        await searchCandidates(userVector, context.productVectors, TOP_N_CANDIDATES) :
        { ranked: context.productVectors.map(item => ({ item, similarity: null })), source: 'sem busca' }
    const candidates = hasHistory ? ranked.slice(0, TOP_N_CANDIDATES) : ranked
    const notEvaluated = hasHistory ? ranked.slice(TOP_N_CANDIDATES) : []

    // 2️⃣ Crie pares de entrada: para cada produto, concatene o vetor do usuário
    //    com o vetor codificado do produto.
    //    Por quê? O modelo prevê o "score de compatibilidade" para cada par (usuário, produto).


    const inputs = candidates.map(({ item: { vector } }) => {
        return [...userVector, ...vector]
    })

    // 3️⃣ Converta todos esses pares (usuário, produto) em um único Tensor.
    //    Formato: [numProdutos, inputDim]
    const inputTensor = tf.tensor2d(inputs)

    // 4️⃣ Rode a rede neural treinada em todos os pares (usuário, produto) de uma vez.
    //    O resultado é uma pontuação para cada produto entre 0 e 1.
    //    Quanto maior, maior a probabilidade do usuário querer aquele produto.
    const predictions = _model.predict(inputTensor)

    // 5️⃣ Extraia as pontuações para um array JS normal.
    const scores = predictions.dataSync()
    const recommendations = candidates.map(({ item, similarity }, index) => {
        return {
            ...item.meta,
            name: item.name,
            score: scores[index], // previsão do modelo para este produto
            similarity // semelhança com o usuário na busca vetorial (null se não houve busca)
        }
    })

    const sortedItems = recommendations
        .sort((a, b) => b.score - a.score)

    // Os produtos que ficaram de fora da busca não passaram pela rede (sem score).
    // Entram no final (na ordem da busca local, ou do catálogo quando a busca veio
    // do banco) para o catálogo continuar completo na tela.
    const skippedItems = notEvaluated.map(({ item, similarity }) => {
        return { ...item.meta, name: item.name, score: null, similarity }
    })

    console.log(
        `recommend: a rede avaliou ${candidates.length} de ${context.productVectors.length} produtos em ${(performance.now() - startedAt).toFixed(1)} ms (busca: ${source})`
    )

    // 8️⃣ Envie a lista ordenada de produtos recomendados
    //    para a thread principal (a UI pode exibi-los agora).
    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: [...sortedItems, ...skippedItems]
    });

}
const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: recommend,
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
