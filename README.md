# Recomendação de Bolos para E-commerce

Projeto da **Aula 7 da Unidade 3**: o mesmo sistema de recomendação do `exemplo-01`
(parte 05), aplicado a um **catálogo de bolos**. Roda 100% no navegador com TensorFlow.js,
treinando a rede em um Web Worker.

## O que mudou em relação ao exemplo-01

- `data/products.json`: 20 bolos (5 categorias: chocolate, frutas, tradicional, gourmet, fit; 8 cores).
- `data/users.json`: 15 clientes (19 a 62 anos), com 3 a 4 compras cada.
- Interface traduzida para português.
- Comentário de exemplo do `modelTrainingWorker.js` adaptado para bolos.
- O código da rede (contexto, vetores, treino e recomendação) **não foi alterado**.

## Estrutura

- `index.html` - página principal
- `src/index.js` - ponto de entrada (cria o cliente de teste "Josézin da Silva", sem compras)
- `src/view/` - views e templates
- `src/controller/` - controllers que ligam views e services
- `src/service/` - leitura dos dados e `sessionStorage`
- `src/events/` - eventos da interface e do worker
- `src/workers/modelTrainingWorker.js` - rede neural (TensorFlow.js)
- `data/` - bolos e clientes

## Como rodar

```
npm install
npm start
```

Abra `http://localhost:3000`. O `npm start` sobe o `server/server.js`, que serve o app **e** o banco
vetorial (veja abaixo). Não abra o `index.html` com duplo clique, porque o worker e os services usam
caminhos como `/data/products.json`. É preciso internet: TensorFlow.js, tfjs-vis e Bootstrap vêm de CDN.

Requer **Node 22.13 ou mais novo** (usa o SQLite embutido, `node:sqlite`, que ainda é experimental e
imprime um aviso no terminal).

- `npm run start:static`: só o `browser-sync` (com recarga automática), **sem banco**. O app
  funciona igual, com a busca local em memória.
- Se aparecer "A porta 3000 já está em uso", pare o outro servidor (por exemplo o `browser-sync`
  antigo) ou use `PORT=3001 npm start`.

## Como usar

1. Aguarde o treino terminar (o botão "Gerar Recomendações" é habilitado).
2. Escolha um cliente. Os bolos são reordenados do mais para o menos recomendado.
3. Clique em **Comprar** para simular uma compra e gere as recomendações de novo.
4. Escolha o **Josézin da Silva** (sem compras) para ver a recomendação só pela idade.

## Busca vetorial (Aula 7, opção A)

O `recommend()` do worker agora usa um **mini banco vetorial** em memória:

1. Calcula a **similaridade do cosseno** entre o vetor do cliente e o vetor de cada bolo.
2. Pega os `TOP_N_CANDIDATES` mais parecidos (padrão: **10**, no `modelTrainingWorker.js`).
3. Roda o `predict` da rede **só nesses**. Os demais bolos vão para o fim da lista, sem score,
   para o catálogo continuar completo na tela.
4. **Cliente sem compras** (ex.: Josézin): o vetor dele só tem a idade, então "parecido" não faz
   sentido. Nesse caso a rede avalia **todos** os bolos, como antes.

Com `TOP_N_CANDIDATES` maior ou igual ao número de bolos, o comportamento é o mesmo de antes.
O console do worker mostra quantos produtos a rede avaliou e quanto tempo levou.

Num banco vetorial de verdade (pgvector, ChromaDB, Pinecone), o passo 1 e 2 seriam uma consulta ao
banco, e os vetores teriam que ser regravados sempre que o modelo for retreinado (o contexto de
min/max muda).

## Banco vetorial de verdade (Aula 7, opção B)

O `server/server.js` guarda os vetores dos bolos em um arquivo SQLite (`server/vectors.db`, ignorado
pelo git) com a extensão **sqlite-vec**, e faz a busca por cosseno em SQL:

```
navegador (worker) ──fetch──▶ server/server.js ──▶ SQLite + sqlite-vec
```

| Rota | O que faz |
|---|---|
| `POST /api/index` | recebe `{ vectors: [{ id, vector }] }` e **regrava** a tabela de produtos |
| `POST /api/search` | recebe `{ vector, k }` e devolve os `k` bolos mais próximos (`id`, `distance`, `similarity`) |
| `GET /api/health` | quantos vetores estão indexados e com quantas dimensões |

Fluxo no worker:

1. Ao fim de cada treino, os vetores são enviados para `/api/index`. Retreinou, o contexto
   (min/max, idade média por produto) muda e os vetores mudam, então a tabela inteira é regravada.
2. No `recommend()`, a busca dos `TOP_N_CANDIDATES` mais parecidos é feita no banco (`/api/search`).
3. Se o servidor ou o banco não responderem, o worker avisa no console e **cai para a busca local**.
4. Cliente sem compras não consulta o banco (o vetor dele só tem a idade).

Limites: o sqlite-vec compara com **todos** os vetores (não tem índice ANN/HNSW), então aqui o ganho é
entender a arquitetura, não velocidade. O servidor só aceita conexões da própria máquina (`127.0.0.1`).
