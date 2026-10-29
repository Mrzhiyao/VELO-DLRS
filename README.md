# VELO-DLRS

Cloud-edge LLM request scheduling with a diffusion-based multi-agent PPO policy.

Each edge server keeps a vector cache of past LLM requests in Milvus. For each incoming request, the agent on that server either answers from the cache (action 0) or forwards the request to the cloud LLM (action 1). There is one agent per edge server. The policy is MAPPO with a diffusion-model actor.

## Layout

```
velo_config.py          addresses, keys and paths (read from environment or .env)
single_node/            three edge servers simulated on one machine
  module_train_vec.py   entry point
multi_node/trainer/     central training server
  start.py              entry point
  node_receive.py       receives experience from one edge node
multi_node/edge/        code for each edge server
  train1.py             entry point
download_data.py        downloads the data from Google Drive
```

## Requirements

- Python 3.8, a CUDA GPU is recommended
- `pip install -r requirements.txt`
- Milvus 2.4, one instance per edge server
- Towhee + Triton embedding service (gpt-neo-1.3B, 768-dim), default `localhost:8010`
- A cloud LLM behind an OpenAI-compatible API. We use Qwen-7B quantized to int8.

## Data

Download and extract the data:

```
python download_data.py
```

The script downloads `velo_data.zip` ([Google Drive](https://drive.google.com/file/d/1KLYnrl_WvWkDe53fpSAdqZXyr3tXW6nr/view?usp=sharing), 173 MB, 9.3 GB after extraction) and extracts it into the repository:

```
data/tasks/single_node/    task_{que,ans}_node{1,2,3}_{0..2}.npz, 5001 pairs per node and shard
data/tasks/multi_node/     task_{que,ans}_node{1,2,3}_{0..6}.npz, shard 0 has 10001 pairs per node, the others 7501
data/expert/single_node/   data_greedy_sim_*{300..1500}.npz, Greedy experience used in the first updates
data/expert/multi_node/    same, for the multi-node trainer
```

Training reads the shards in order and moves to the next one when the current shard is used up. If the next shard is missing, it goes back to shard 0.

The requests come from [oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) (Apache-2.0). We took the highest-ranked dialogues and translated each item into English, Spanish, German, Chinese and Russian with Qwen-14B. The test set uses Qwen-14B paraphrases of the training questions.

## Configuration

```
cp .env.example .env
```

Every value in `velo_config.py` can be set through an environment variable with the `VELO_` prefix. Relative paths are resolved from the repository root.

Create the Milvus collections before training. Each one uses an `IVF_FLAT` index on `vec` (nlist 128, L2), and searches use nprobe 10.

Cache collections:
- single_node: `two_input{1,2,3}_diffusion` and `two_input_nodes{1,2,3}1_diffusion`
- multi_node/edge: `VELO_EDGE_COLLECTION` and `VELO_EDGE_SEND_COLLECTION`

Cache fields: `id` (INT64, auto), `message_id`, `created_date`, `text`, `vec` (FLOAT_VECTOR 768), `role`, `lang`, `review_count` (INT64), `rank`, `text_answer`, `text_answerid`.

`oasst1_llm_result` stores previously generated LLM answers together with their latency. Its fields are `id` (INT64, auto), `father_id`, `message_id`, `created_date`, `text`, `vec` (FLOAT_VECTOR 768) and `time_consume` (DOUBLE). When a request is forwarded to the LLM, the stored answer is used if there is one; otherwise the LLM is called.

## Running

Single machine:

```
cd single_node
python module_train_vec.py
```

Multiple machines. On the trainer:

```
cd multi_node/trainer
python node_receive.py --node 1 &
python node_receive.py --node 2 &
python node_receive.py --node 3 &
python start.py
```

On edge node k, set `VELO_NODE_ID=k` and run an FTP server that writes to `VELO_EDGE_WEIGHT_PATH`, then:

```
cd multi_node/edge
python train1.py
```

Every 300 steps, each edge sends its experience over TCP to port `VELO_TRAINER_BASE_PORT + k`. When all three nodes have reported, the trainer updates the policy and sends the new weights to every edge over FTP. Each edge reloads the weights when the file changes. If `weights/pretrained.pth` exists, an edge starts from it; otherwise it starts from random weights.

The TCP receiver and the FTP upload do not use authentication. Run them only on a trusted network.

## Hyper-parameters

- PPO: K_epochs 50, eps_clip 0.2, gamma 0.99, lr_actor 3e-4, lr_critic 1e-3, update every 300 steps
- Diffusion actor: 10 steps, linear beta schedule, MLP hidden size 256 (Mish), time embedding size 16
- Request encoder: TransformerEncoderLayer (d_model 768, 6 heads) followed by an MLP from 768 to 30. The actor input is 60-dim. The critic input is 180-dim, with hidden size 384.
- Reward: (-distance - 0.1 * latency) * 10. When the LLM is used, 0.6 is also subtracted before the scaling. `distance` is the L2 distance between the embeddings of the returned answer and the reference answer.
- Cache eviction: every 2400 steps, delete entries whose rank is below the average
- Expert data: used while 1500 / update_times > 200

## License

Apache-2.0. `*/diffusion/` is adapted from [Diffusion-QL](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL) (Apache-2.0, Copyright 2022 Twitter, Inc and Zhendong Wang).
