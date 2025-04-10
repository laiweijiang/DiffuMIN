#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import tensorflow as tf
from handle_layer.handle_lib.handle_base import InputBase
import numpy as np

class MultiHeadSelfAttention(object):
    def __init__(self, d_model, heads, variable_scope):
        self.d_model = d_model
        self.num_heads = heads
        self.variable_scope = variable_scope
        assert 0 == d_model % heads
        self.depth = d_model // heads  # 每个头的维度
    def __call__(self, q,k,v):
        # q[b,4,24]
        # k,v[b,1,24]
        with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
            b, n_q, d = q.get_shape().as_list()
            b, n_kv, d= k.get_shape().as_list()

            h, dh = self.num_heads, self.depth

            q = tf.layers.dense(q, self.d_model, activation=None, use_bias=False,
                                   name="q_linear")
            k = tf.layers.dense(k, self.d_model, activation=None, use_bias=False,
                                name="k_linear")
            v = tf.layers.dense(v, self.d_model, activation=None, use_bias=False,
                                name="v_linear")

            q = tf.transpose(tf.reshape(q, [-1, n_q, h, dh]), [0, 2, 1, 3])  # [b, h, n, dh]
            k = tf.transpose(tf.reshape(k, [-1, n_kv, h, dh]), [0, 2, 3, 1])  # [b, h, dh, n]
            v = tf.transpose(tf.reshape(v, [-1, n_kv, h, dh]), [0, 2, 1, 3])  # [b, h, n, dh]

            #[b,2,4,12] [b,2,12,1]
            # 求相似度矩阵并缩放
            attn = tf.matmul(q, k)
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            attn = attn / tf.math.sqrt(dk)

            attn = tf.nn.softmax(attn, axis=-1)
            attn_out = tf.matmul(attn, v)  # [b, h, n, dh]
            attn_out = tf.reshape(tf.transpose(attn_out, [0, 2, 1, 3]), [-1, n_q, d])
            attn_out = tf.layers.dense(attn_out, self.d_model, activation=None, use_bias=False, name="o_linear")
            return attn_out

class FFN(object):
    def __init__(self, d_model, dff_factor, variable_scope):
        self.variable_scope = variable_scope
        self.dff_factor = dff_factor
        self.d_model = d_model

    def __call__(self, x):
        with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(x, self.d_model*self.dff_factor, activation='relu', use_bias=True, name="ffn1")
            x = tf.layers.dense(x, self.d_model, activation=None, use_bias=True, name="ffn2")
            return x

class Diffusion_Net(object):
    def __init__(self, d_model, head, dff_factor, variable_scope):
        self.attention1 = MultiHeadSelfAttention(d_model, head, variable_scope + "self")
        self.attention2 = MultiHeadSelfAttention(d_model, head, variable_scope + "cross")

        self.ffn = FFN(d_model, dff_factor, variable_scope + "_ffn")
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.variable_scope = variable_scope

    def __call__(self, x, target):
        x_output = self.attention1(x,x,x)
        x = self.norm1(x_output + x)

        x_output = self.attention2(x, target, target)
        x = self.norm2(x_output + x)

        x_output = self.ffn(x)
        x = self.norm3(x_output + x)
        return x

class Transformer(object):
    def __init__(self, d_model, head, dff_factor, variable_scope):
        self.attention1 = MultiHeadSelfAttention(d_model, head, variable_scope + "self")

        self.ffn = FFN(d_model, dff_factor, variable_scope + "_ffn")
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.variable_scope = variable_scope

    def __call__(self, x, target):
        x_output = self.attention1(x,x,x)
        x = self.norm1(x_output + x)
        x_output = self.ffn(x)
        x = self.norm2(x_output + x)
        return x

class MLP(object):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn, variable_scope):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation_fn = activation_fn
        self.variable_scope = variable_scope

    def __call__(self, x):
        with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
            for i, hidden_dim in enumerate(self.hidden_dims):
                x = tf.layers.dense(x, hidden_dim, activation=self.activation_fn, name="hidden_layer_{}".format(i))
            output = tf.layers.dense(x, self.output_dim, activation=None, name="output_layer")
        return output



class Long_term_DiffuMIN(InputBase):
    def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
        super(Long_term_DiffuMIN, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)

        # Base configuration
        self.seq_len = self.params['seq_len']
        self.subseq_len = self.params['subseq_len']
        self.embed_dim = self.params['embed_dim']

        # Example feature lists
        self.target_poi = ["target_poi"]
        self.target_third_id = ["target_third_id"]
        self.target_aor_id = ["target_aor_id"]
        self.item_id_list = ["item_id_list[%d]" % _ for _ in range(self.seq_len)]
        self.third_id_list = ["third_id_list[%d]" % _ for _ in range(self.seq_len)]
        self.aor_id_list = ["aor_id_list[%d]" % _ for _ in range(self.seq_len)]
        self.cat_list = [self.target_poi, self.target_third_id, self.target_aor_id,
                         self.item_id_list, self.third_id_list, self.aor_id_list]

        self.channel_num = self.params['channel_num']
        self.feature_num = self.params['feature_num']  # Diffusion feature number

        # Time MLP
        self.timesteps = self.params['timesteps']
        self.infer_timesteps = self.params['infer_timesteps']
        self.beta_sche = self.params['beta_sche']
        self.beta_start = self.params['beta_start']
        self.beta_end = self.params['beta_end']

        def swish(x):
            return x * tf.sigmoid(x)

        self.time_embed = tf.keras.Sequential([
            tf.keras.layers.Dense(self.embed_dim * self.feature_num * 4),
            tf.keras.layers.Activation(swish),
            tf.keras.layers.Dense(self.embed_dim * self.feature_num)]
        )

        # Beta schedule configuration
        if self.beta_sche == 'linear':
            self.betas = self.linear_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
        elif self.beta_sche == 'exp':  # Exponential
            self.betas = self.exp_beta_schedule(self.timesteps)
        elif self.beta_sche == 'cosine':
            self.betas = self.cosine_beta_schedule(timesteps=self.timesteps)

        self.alphas = 1. - self.betas  # [alpha_1, alpha_2, ..., alpha_t]
        self.alphas_cumprod = tf.cumprod(
            self.alphas, axis=0)  # Cumulative product of alphas [alpha_cumprod_1, alpha_cumprod_2, ..., alpha_cumprod_t]
        self.alphas_cumprod_prev = tf.pad(self.alphas_cumprod[:-1], [(1, 0)], constant_values=1.0)  # Padding for cumulative alpha
        self.sqrt_recip_alphas = tf.sqrt(1.0 / self.alphas)  # Reciprocal square root

        # Diffusion calculations
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = tf.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = tf.sqrt(1. / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = self.betas * tf.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * tf.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


        # Diffusion and transformer models
        self.n_layers = self.params['n_layers']
        self.num_heads = self.params['num_heads']
        self.dff_factor = self.params['dff_factor']
        self.diffusion_blocks = [
            Diffusion_Net(self.embed_dim * self.feature_num, self.num_heads, self.dff_factor,
                          "DiffusionBlocks_{}".format(i)) for i in range(self.n_layers)]
        self.transformer_blocks = [
            Transformer(self.embed_dim * self.feature_num, self.num_heads, self.dff_factor,
                          "TransformerBlocks_{}".format(i)) for i in range(self.n_layers)]

        mlp_hidden_units_factor = [4]
        mlp_hidden_units = [unit * self.embed_dim * self.feature_num for unit in mlp_hidden_units_factor]
        self.ProjectHeads1 = MLP(self.feature_num * self.embed_dim, mlp_hidden_units,
                                 self.feature_num * self.embed_dim, activation_fn='relu',
                                 variable_scope='ProjectHeads1')
        self.ProjectHeads2 = MLP(self.feature_num * self.embed_dim, mlp_hidden_units,
                                 self.feature_num * self.embed_dim, activation_fn='relu',
                                 variable_scope='ProjectHeads2')
        self.lamda1 = self.params['lamda1']
        self.lamda2 = self.params['lamda2']
        self.temp = self.params['temp']

    def recieve_gather_features(self, cat_feas, dense_feas):
        # Dense features
        self.dense_fea_split = dense_feas
        # Categorical features
        self.cat_fea_split = cat_feas
        self.cat_fea_emb = self.ptable_lookup(list_ids=cat_feas, v_name=self.__class__.__name__)

    def __call__(self, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
        # Concat target and history category embeddings
        target_cate = tf.concat(self.cat_fea_emb[:self.feature_num], -1)
        his_cate = tf.concat(self.cat_fea_emb[self.feature_num:], -1)

        # Process target category
        target_cate = tf.layers.dense(
            target_cate,
            self.feature_num * self.embed_dim * self.channel_num,
            activation=None,
        )
        reshaped_embedding = tf.reshape(target_cate, [-1, self.feature_num * self.embed_dim, self.channel_num])
        reshaped_embedding += 1e-8 * tf.random.normal(tf.shape(reshaped_embedding))

        # QR decomposition
        orthogonal_channels, r = tf.qr(reshaped_embedding)  # Orthogonal [batch, channel_num, embedding]

        # Compute similarity between each user action and each channel
        similarities = tf.matmul(his_cate, orthogonal_channels)  # [batch, seq_len, channel_num]

        # Find the channel index with the highest similarity for each action
        max_channel_indices = tf.argmax(similarities, axis=-1, output_type=tf.int32)  # [batch, seq_len]

        # Generate One-Hot mask, retaining only the highest similarity channel
        channel_mask = tf.one_hot(max_channel_indices, depth=self.channel_num, dtype=tf.float32)

        # Second stage: Find top-k actions for each channel
        channel_similarities = tf.transpose(similarities, [0, 2, 1])

        def generate_topk_mask_v2(similarities, topk):
            """Optimized mask generation"""
            # Get dynamic shapes
            batch_size = tf.shape(similarities)[0]
            channel_num = tf.shape(similarities)[1]
            seq_len = tf.shape(similarities)[2]

            # Get Top-K indices
            _, topk_indices = tf.math.top_k(similarities, k=topk)

            # Create scatter indices
            batch_indices = tf.tile(
                tf.reshape(tf.range(batch_size), [batch_size, 1, 1, 1]),
                [1, channel_num, topk, 1]
            )
            channel_indices = tf.tile(
                tf.reshape(tf.range(channel_num), [1, channel_num, 1, 1]),
                [batch_size, 1, topk, 1]
            )
            indices = tf.concat([
                batch_indices,
                channel_indices,
                tf.expand_dims(topk_indices, -1)
            ], axis=-1)

            # Generate sparse mask
            updates = tf.ones_like(topk_indices, dtype=tf.float32)
            shape = tf.stack([batch_size, channel_num, seq_len])
            topk_mask = tf.scatter_nd(indices=tf.reshape(indices, [-1, 3]), updates=tf.reshape(updates, [-1]), shape=shape)

            return tf.minimum(topk_mask, 1.0)  # Ensure values do not exceed 1

        # Get top-k indices for each channel
        topk_mask = generate_topk_mask_v2(channel_similarities, topk=100)

        # Transpose back to original dimensions
        topk_mask = tf.transpose(topk_mask, [0, 2, 1])

        # Combine two masks using logical AND
        combined_mask = channel_mask * topk_mask

        # Apply combined mask
        masked_similarities = similarities * combined_mask

        # Aggregate user behavior representations
        activated_behaviors = tf.expand_dims(his_cate, 2) * tf.expand_dims(masked_similarities, -1)
        aggregated_repr = tf.reduce_mean(activated_behaviors, axis=1)

        aggregated_repr_norm = tf.nn.l2_normalize(aggregated_repr, axis=-1)

        orthogonal_channels = tf.transpose(orthogonal_channels, [0, 2, 1])
        orthogonal_channels = tf.nn.l2_normalize(orthogonal_channels, axis=-1)

        variable_scope = "Diffusion"
        with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
            if self.is_training:
                self.loss = []
                for i in range(self.channel_num):
                    # Random timestep selection
                    ts = tf.random_uniform(shape=(tf.shape(orthogonal_channels)[0],), minval=0, maxval=self.params['timesteps'], dtype=tf.int32)
                    h_target = tf.expand_dims(orthogonal_channels[:, i, :], 1)

                    loss_i, predicted_x_i = self.p_losses(aggregated_repr_norm, i, h_target, ts, loss_type='l2')
                    self.loss.append(loss_i)

            predicted_x = []
            for i in range(self.channel_num):
                h_target = tf.expand_dims(orthogonal_channels[:, i, :], 1)
                x_i = self.sample(aggregated_repr_norm, i, h_target)
                predicted_x.append(x_i)
            predicted_x = tf.concat(predicted_x, 1)
        predicted_x = tf.Print(predicted_x, ["LWJ:predicted_x", predicted_x], first_n=-1, summarize=-1)

        if self.is_training:
            self.contrastive_loss = self.contrastive(aggregated_repr_norm, predicted_x, self.temp)

        variable_scope = "Transformer"
        with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
            hidden_state = tf.concat([his_cate[:, :self.subseq_len, :], target_cate], axis=1)
            for i, block in enumerate(self.diffusion_blocks):
                hidden_state = block(hidden_state, h_target)
            transformer_output = hidden_state

        # Return concatenated features including aggregated representation, predicted embeddings, and attention output
        return tf.concat([tf.reshape(aggregated_repr, [-1, 16 * self.channel_num]),
                          tf.reshape(predicted_x, [-1, self.feature_num * self.embed_dim * self.channel_num]),
                          transformer_output[:,-1,:],
                          tf.squeeze(target_cate, axis=1)], -1)

    def get_auxiliary_loss(self):
        # Compute auxiliary loss based on mean of losses
        loss = sum(self.loss) / self.channel_num
        return self.lamda1 * loss, self.lamda2 * self.contrastive_loss

    def linear_beta_schedule(self, timesteps, beta_start, beta_end):
        # Linear beta schedule
        return tf.linspace(beta_start, beta_end, timesteps)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        # Cosine beta schedule
        steps = timesteps + 1
        x = tf.linspace(0, timesteps, steps)
        alphas_cumprod = tf.cos(((x / timesteps) + s) / (1 + s) * tf.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return tf.clip(betas, 0.0001, 0.9999)

    def exp_beta_schedule(self, timesteps, beta_min=0.1, beta_max=10):
        # Exponential beta schedule
        x = tf.linspace(1.0, 2.0 * timesteps + 1.0, timesteps)
        betas = 1 - tf.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps ** 2))
        return betas

    def contrastive(self, orig_embs, gen_embs, temperature=0.07):
        # Contrastive loss computation
        orig_embs = tf.transpose(orig_embs, [1, 0, 2])
        gen_embs = tf.transpose(gen_embs, [1, 0, 2])
        orig_embs = self.ProjectHeads1(orig_embs)
        gen_embs = self.ProjectHeads2(gen_embs)
        orig_embs = tf.nn.l2_normalize(orig_embs, axis=-1)
        gen_embs = tf.nn.l2_normalize(gen_embs, axis=-1)

        # Compute similarity matrix
        similarity = tf.matmul(orig_embs, gen_embs, transpose_b=True) / temperature

        # Positive similarities
        pos_similarity = tf.linalg.diag_part(similarity)

        # Logsumexp with positive samples
        logsumexp = tf.reduce_logsumexp(similarity, axis=2)

        # Loss calculation
        loss = logsumexp - pos_similarity
        return tf.reduce_mean(loss)

    def q_sample(self, x_start, t, noise=None):
        # Forward diffusion process
        if noise is None:
            noise = tf.random.normal(tf.shape(x_start))

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, tf.shape(x_start))
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, tf.shape(x_start))

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def step_mlp(self, inputs):
        # Time embedding for steps
        half_dim = self.embed_dim * self.feature_num // 2  # Half dimension of embeddings
        embeddings = tf.math.log(10000.0) / (half_dim - 1)
        embeddings = tf.math.exp(tf.range(half_dim, dtype=tf.float32) * -embeddings)
        embeddings = tf.cast(inputs[:, None], dtype=tf.float32) * embeddings[None, :]
        embeddings = tf.concat([tf.sin(embeddings), tf.cos(embeddings)], -1)
        x = self.time_embed(embeddings)
        return x

    def extract(self, a, t, x_shape):
        # Extract value corresponding to timesteps from tensor
        batch_size = t.shape[0]
        out = tf.gather(a, t, axis=-1)
        return tf.reshape(out, [batch_size] + [1] * (len(x_shape) - 1))

    def predict_start_from_noise(self, x_t, t, noise):
        # Predict the start state given current state x_t and noise
        return (
                self.extract(self.sqrt_recip_alphas_cumprod, t, tf.shape(x_t)) * x_t -
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, tf.shape(x_t)) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        # Compute posterior distribution for q(x_{t-1} | x_t, x_0)
        posterior_mean = (
                self.extract(self.posterior_mean_coef1, t, tf.shape(x_t)) * x_start +
                self.extract(self.posterior_mean_coef2, t, tf.shape(x_t)) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        self.posterior_log_variance_clipped = tf.log(tf.maximum(self.posterior_variance, 1e-20))
        posterior_log_variance_clipped = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_losses(self, channels, i, h_target, t, loss_type="l2"):
        # Compute loss for the reverse diffusion process
        x_start = channels[:, i, :]
        x_start = tf.expand_dims(x_start, 1)

        noise = tf.random.normal(tf.shape(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if i == 0:
            input = tf.concat([x_noisy, channels[:, i + 1:, :]], axis=1)
        elif i == self.channel_num - 1:
            input = tf.concat([channels[:, :i, :], x_noisy], axis=1)
        else:
            input = tf.concat([channels[:, :i, :], x_noisy, channels[:, i + 1:, :]], axis=1)

        predicted_x = self.forward(input, h_target, t)  # Predicted x_{t-1}
        predicted_x = predicted_x[:, i, :]
        predicted_x = tf.expand_dims(predicted_x, 1)
        loss = tf.losses.mean_squared_error(x_start, predicted_x)  # MSE loss

        return loss, predicted_x

    def forward(self, x, h_target, step):
        # Pass input through diffusion blocks
        t = self.step_mlp(step)
        t = tf.expand_dims(t, axis=1)
        t = tf.tile(t, [1, tf.shape(x)[1], 1])
        x += t

        hidden_state = x
        for i, block in enumerate(self.diffusion_blocks):
            hidden_state = block(hidden_state, h_target)
        transformer_output = hidden_state

        return transformer_output

    def sample(self, channels, i, h_target):
        # Sampling method for channels in diffusion model
        x = channels[:, i, :]
        x = tf.expand_dims(x, axis=1)
        t = tf.random_uniform(shape=(tf.shape(x)[0],), minval=0, maxval=self.params['timesteps'], dtype=tf.int32)
        noise = tf.random.normal(tf.shape(x))
        x = self.q_sample(x_start=x, t=t, noise=noise)

        for n in reversed(range(0, self.infer_timesteps)):
            x = self.p_sample(x, channels, i, h_target, tf.fill([tf.shape(channels)[0]], value=n), n)
        return x

    def p_sample(self, x, channels, i, h, t, t_index):
        # Perform reverse diffusion sampling
        if i == 0:
            input = tf.concat([x, channels[:, 1:, :]], axis=1)
        elif i == self.channel_num - 1:
            input = tf.concat([channels[:, :i, :], x], axis=1)
        else:
            input = tf.concat([channels[:, :i, :], x, channels[:, i + 1:, :]], axis=1)

        pred_noise = self.forward(input, h, t)
        pred_noise = pred_noise[:, i, :]
        pred_noise = tf.expand_dims(pred_noise, axis=1)
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        model_mean, _, model_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        noise = tf.random.normal(tf.shape(x))
        nonzero_mask = tf.reshape(1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1])
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise


class Eleme_DiffuMIN(InputBase):
    def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
        super(Eleme_DiffuMIN, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)


        # Base configuration
        self.seq_len = self.params['seq_len']
        self.subseq_len = self.params['subseq_len']
        self.embed_dim = self.params['embed_dim']

        self.cat_fea_list = ["user_id", "gender", "visit_city", "is_supervip", "ctr_30", "ord_30", "shop_id", "item_id",
                             "city_id", "district_id", "shop_aoi_id", "shop_geohash_6", "shop_geohash_12", "brand_id",
                             "category_1_id", "merge_standard_food_id", "rank_7", "rank_30", "rank_90", "times",
                             "hours", "time_type", "weekdays", "geohash12"]
        self.cat_seq_fea = ["shop_id_list[0]", "shop_id_list[1]", "shop_id_list[2]", "shop_id_list[3]",
                            "shop_id_list[4]", "shop_id_list[5]", "shop_id_list[6]", "shop_id_list[7]",
                            "shop_id_list[8]", "shop_id_list[9]", "shop_id_list[10]", "shop_id_list[11]",
                            "shop_id_list[12]", "shop_id_list[13]", "shop_id_list[14]", "shop_id_list[15]",
                            "shop_id_list[16]", "shop_id_list[17]", "shop_id_list[18]", "shop_id_list[19]",
                            "shop_id_list[20]", "shop_id_list[21]", "shop_id_list[22]", "shop_id_list[23]",
                            "shop_id_list[24]", "shop_id_list[25]", "shop_id_list[26]", "shop_id_list[27]",
                            "shop_id_list[28]", "shop_id_list[29]", "shop_id_list[30]", "shop_id_list[31]",
                            "shop_id_list[32]", "shop_id_list[33]", "shop_id_list[34]", "shop_id_list[35]",
                            "shop_id_list[36]", "shop_id_list[37]", "shop_id_list[38]", "shop_id_list[39]",
                            "shop_id_list[40]", "shop_id_list[41]", "shop_id_list[42]", "shop_id_list[43]",
                            "shop_id_list[44]", "shop_id_list[45]", "shop_id_list[46]", "shop_id_list[47]",
                            "shop_id_list[48]", "shop_id_list[49]",
                            "item_id_list[0]", "item_id_list[1]", "item_id_list[2]", "item_id_list[3]",
                            "item_id_list[4]", "item_id_list[5]", "item_id_list[6]", "item_id_list[7]",
                            "item_id_list[8]", "item_id_list[9]", "item_id_list[10]", "item_id_list[11]",
                            "item_id_list[12]", "item_id_list[13]", "item_id_list[14]", "item_id_list[15]",
                            "item_id_list[16]", "item_id_list[17]", "item_id_list[18]", "item_id_list[19]",
                            "item_id_list[20]", "item_id_list[21]", "item_id_list[22]", "item_id_list[23]",
                            "item_id_list[24]", "item_id_list[25]", "item_id_list[26]", "item_id_list[27]",
                            "item_id_list[28]", "item_id_list[29]", "item_id_list[30]", "item_id_list[31]",
                            "item_id_list[32]", "item_id_list[33]", "item_id_list[34]", "item_id_list[35]",
                            "item_id_list[36]", "item_id_list[37]", "item_id_list[38]", "item_id_list[39]",
                            "item_id_list[40]", "item_id_list[41]", "item_id_list[42]", "item_id_list[43]",
                            "item_id_list[44]", "item_id_list[45]", "item_id_list[46]", "item_id_list[47]",
                            "item_id_list[48]", "item_id_list[49]",
                            "category_1_id_list[0]", "category_1_id_list[1]", "category_1_id_list[2]",
                            "category_1_id_list[3]", "category_1_id_list[4]", "category_1_id_list[5]",
                            "category_1_id_list[6]", "category_1_id_list[7]", "category_1_id_list[8]",
                            "category_1_id_list[9]", "category_1_id_list[10]", "category_1_id_list[11]",
                            "category_1_id_list[12]", "category_1_id_list[13]", "category_1_id_list[14]",
                            "category_1_id_list[15]", "category_1_id_list[16]", "category_1_id_list[17]",
                            "category_1_id_list[18]", "category_1_id_list[19]", "category_1_id_list[20]",
                            "category_1_id_list[21]", "category_1_id_list[22]", "category_1_id_list[23]",
                            "category_1_id_list[24]", "category_1_id_list[25]", "category_1_id_list[26]",
                            "category_1_id_list[27]", "category_1_id_list[28]", "category_1_id_list[29]",
                            "category_1_id_list[30]", "category_1_id_list[31]", "category_1_id_list[32]",
                            "category_1_id_list[33]", "category_1_id_list[34]", "category_1_id_list[35]",
                            "category_1_id_list[36]", "category_1_id_list[37]", "category_1_id_list[38]",
                            "category_1_id_list[39]", "category_1_id_list[40]", "category_1_id_list[41]",
                            "category_1_id_list[42]", "category_1_id_list[43]", "category_1_id_list[44]",
                            "category_1_id_list[45]", "category_1_id_list[46]", "category_1_id_list[47]",
                            "category_1_id_list[48]", "category_1_id_list[49]",
                            "merge_standard_food_id_list[0]", "merge_standard_food_id_list[1]",
                            "merge_standard_food_id_list[2]", "merge_standard_food_id_list[3]",
                            "merge_standard_food_id_list[4]", "merge_standard_food_id_list[5]",
                            "merge_standard_food_id_list[6]", "merge_standard_food_id_list[7]",
                            "merge_standard_food_id_list[8]", "merge_standard_food_id_list[9]",
                            "merge_standard_food_id_list[10]", "merge_standard_food_id_list[11]",
                            "merge_standard_food_id_list[12]", "merge_standard_food_id_list[13]",
                            "merge_standard_food_id_list[14]", "merge_standard_food_id_list[15]",
                            "merge_standard_food_id_list[16]", "merge_standard_food_id_list[17]",
                            "merge_standard_food_id_list[18]", "merge_standard_food_id_list[19]",
                            "merge_standard_food_id_list[20]", "merge_standard_food_id_list[21]",
                            "merge_standard_food_id_list[22]", "merge_standard_food_id_list[23]",
                            "merge_standard_food_id_list[24]", "merge_standard_food_id_list[25]",
                            "merge_standard_food_id_list[26]", "merge_standard_food_id_list[27]",
                            "merge_standard_food_id_list[28]", "merge_standard_food_id_list[29]",
                            "merge_standard_food_id_list[30]", "merge_standard_food_id_list[31]",
                            "merge_standard_food_id_list[32]", "merge_standard_food_id_list[33]",
                            "merge_standard_food_id_list[34]", "merge_standard_food_id_list[35]",
                            "merge_standard_food_id_list[36]", "merge_standard_food_id_list[37]",
                            "merge_standard_food_id_list[38]", "merge_standard_food_id_list[39]",
                            "merge_standard_food_id_list[40]", "merge_standard_food_id_list[41]",
                            "merge_standard_food_id_list[42]", "merge_standard_food_id_list[43]",
                            "merge_standard_food_id_list[44]", "merge_standard_food_id_list[45]",
                            "merge_standard_food_id_list[46]", "merge_standard_food_id_list[47]",
                            "merge_standard_food_id_list[48]", "merge_standard_food_id_list[49]",
                            "brand_id_list[0]", "brand_id_list[1]", "brand_id_list[2]", "brand_id_list[3]",
                            "brand_id_list[4]", "brand_id_list[5]", "brand_id_list[6]", "brand_id_list[7]",
                            "brand_id_list[8]", "brand_id_list[9]", "brand_id_list[10]", "brand_id_list[11]",
                            "brand_id_list[12]", "brand_id_list[13]", "brand_id_list[14]", "brand_id_list[15]",
                            "brand_id_list[16]", "brand_id_list[17]", "brand_id_list[18]", "brand_id_list[19]",
                            "brand_id_list[20]", "brand_id_list[21]", "brand_id_list[22]", "brand_id_list[23]",
                            "brand_id_list[24]", "brand_id_list[25]", "brand_id_list[26]", "brand_id_list[27]",
                            "brand_id_list[28]", "brand_id_list[29]", "brand_id_list[30]", "brand_id_list[31]",
                            "brand_id_list[32]", "brand_id_list[33]", "brand_id_list[34]", "brand_id_list[35]",
                            "brand_id_list[36]", "brand_id_list[37]", "brand_id_list[38]", "brand_id_list[39]",
                            "brand_id_list[40]", "brand_id_list[41]", "brand_id_list[42]", "brand_id_list[43]",
                            "brand_id_list[44]", "brand_id_list[45]", "brand_id_list[46]", "brand_id_list[47]",
                            "brand_id_list[48]", "brand_id_list[49]",
                            "shop_aoi_id_list[0]", "shop_aoi_id_list[1]", "shop_aoi_id_list[2]", "shop_aoi_id_list[3]",
                            "shop_aoi_id_list[4]", "shop_aoi_id_list[5]", "shop_aoi_id_list[6]", "shop_aoi_id_list[7]",
                            "shop_aoi_id_list[8]", "shop_aoi_id_list[9]", "shop_aoi_id_list[10]",
                            "shop_aoi_id_list[11]", "shop_aoi_id_list[12]", "shop_aoi_id_list[13]",
                            "shop_aoi_id_list[14]", "shop_aoi_id_list[15]", "shop_aoi_id_list[16]",
                            "shop_aoi_id_list[17]", "shop_aoi_id_list[18]", "shop_aoi_id_list[19]",
                            "shop_aoi_id_list[20]", "shop_aoi_id_list[21]", "shop_aoi_id_list[22]",
                            "shop_aoi_id_list[23]", "shop_aoi_id_list[24]", "shop_aoi_id_list[25]",
                            "shop_aoi_id_list[26]", "shop_aoi_id_list[27]", "shop_aoi_id_list[28]",
                            "shop_aoi_id_list[29]", "shop_aoi_id_list[30]", "shop_aoi_id_list[31]",
                            "shop_aoi_id_list[32]", "shop_aoi_id_list[33]", "shop_aoi_id_list[34]",
                            "shop_aoi_id_list[35]", "shop_aoi_id_list[36]", "shop_aoi_id_list[37]",
                            "shop_aoi_id_list[38]", "shop_aoi_id_list[39]", "shop_aoi_id_list[40]",
                            "shop_aoi_id_list[41]", "shop_aoi_id_list[42]", "shop_aoi_id_list[43]",
                            "shop_aoi_id_list[44]", "shop_aoi_id_list[45]", "shop_aoi_id_list[46]",
                            "shop_aoi_id_list[47]", "shop_aoi_id_list[48]", "shop_aoi_id_list[49]",
                            "shop_geohash6_list[0]", "shop_geohash6_list[1]", "shop_geohash6_list[2]",
                            "shop_geohash6_list[3]", "shop_geohash6_list[4]", "shop_geohash6_list[5]",
                            "shop_geohash6_list[6]", "shop_geohash6_list[7]", "shop_geohash6_list[8]",
                            "shop_geohash6_list[9]", "shop_geohash6_list[10]", "shop_geohash6_list[11]",
                            "shop_geohash6_list[12]", "shop_geohash6_list[13]", "shop_geohash6_list[14]",
                            "shop_geohash6_list[15]", "shop_geohash6_list[16]", "shop_geohash6_list[17]",
                            "shop_geohash6_list[18]", "shop_geohash6_list[19]", "shop_geohash6_list[20]",
                            "shop_geohash6_list[21]", "shop_geohash6_list[22]", "shop_geohash6_list[23]",
                            "shop_geohash6_list[24]", "shop_geohash6_list[25]", "shop_geohash6_list[26]",
                            "shop_geohash6_list[27]", "shop_geohash6_list[28]", "shop_geohash6_list[29]",
                            "shop_geohash6_list[30]", "shop_geohash6_list[31]", "shop_geohash6_list[32]",
                            "shop_geohash6_list[33]", "shop_geohash6_list[34]", "shop_geohash6_list[35]",
                            "shop_geohash6_list[36]", "shop_geohash6_list[37]", "shop_geohash6_list[38]",
                            "shop_geohash6_list[39]", "shop_geohash6_list[40]", "shop_geohash6_list[41]",
                            "shop_geohash6_list[42]", "shop_geohash6_list[43]", "shop_geohash6_list[44]",
                            "shop_geohash6_list[45]", "shop_geohash6_list[46]", "shop_geohash6_list[47]",
                            "shop_geohash6_list[48]", "shop_geohash6_list[49]",
                            "timediff_list[0]", "timediff_list[1]", "timediff_list[2]", "timediff_list[3]",
                            "timediff_list[4]", "timediff_list[5]", "timediff_list[6]", "timediff_list[7]",
                            "timediff_list[8]", "timediff_list[9]", "timediff_list[10]", "timediff_list[11]",
                            "timediff_list[12]", "timediff_list[13]", "timediff_list[14]", "timediff_list[15]",
                            "timediff_list[16]", "timediff_list[17]", "timediff_list[18]", "timediff_list[19]",
                            "timediff_list[20]", "timediff_list[21]", "timediff_list[22]", "timediff_list[23]",
                            "timediff_list[24]", "timediff_list[25]", "timediff_list[26]", "timediff_list[27]",
                            "timediff_list[28]", "timediff_list[29]", "timediff_list[30]", "timediff_list[31]",
                            "timediff_list[32]", "timediff_list[33]", "timediff_list[34]", "timediff_list[35]",
                            "timediff_list[36]", "timediff_list[37]", "timediff_list[38]", "timediff_list[39]",
                            "timediff_list[40]", "timediff_list[41]", "timediff_list[42]", "timediff_list[43]",
                            "timediff_list[44]", "timediff_list[45]", "timediff_list[46]", "timediff_list[47]",
                            "timediff_list[48]", "timediff_list[49]",
                            "hours_list[0]", "hours_list[1]", "hours_list[2]", "hours_list[3]", "hours_list[4]",
                            "hours_list[5]", "hours_list[6]", "hours_list[7]", "hours_list[8]", "hours_list[9]",
                            "hours_list[10]", "hours_list[11]", "hours_list[12]", "hours_list[13]", "hours_list[14]",
                            "hours_list[15]", "hours_list[16]", "hours_list[17]", "hours_list[18]", "hours_list[19]",
                            "hours_list[20]", "hours_list[21]", "hours_list[22]", "hours_list[23]", "hours_list[24]",
                            "hours_list[25]", "hours_list[26]", "hours_list[27]", "hours_list[28]", "hours_list[29]",
                            "hours_list[30]", "hours_list[31]", "hours_list[32]", "hours_list[33]", "hours_list[34]",
                            "hours_list[35]", "hours_list[36]", "hours_list[37]", "hours_list[38]", "hours_list[39]",
                            "hours_list[40]", "hours_list[41]", "hours_list[42]", "hours_list[43]", "hours_list[44]",
                            "hours_list[45]", "hours_list[46]", "hours_list[47]", "hours_list[48]", "hours_list[49]",
                            "time_type_list[0]", "time_type_list[1]", "time_type_list[2]", "time_type_list[3]",
                            "time_type_list[4]", "time_type_list[5]", "time_type_list[6]", "time_type_list[7]",
                            "time_type_list[8]", "time_type_list[9]", "time_type_list[10]", "time_type_list[11]",
                            "time_type_list[12]", "time_type_list[13]", "time_type_list[14]", "time_type_list[15]",
                            "time_type_list[16]", "time_type_list[17]", "time_type_list[18]", "time_type_list[19]",
                            "time_type_list[20]", "time_type_list[21]", "time_type_list[22]", "time_type_list[23]",
                            "time_type_list[24]", "time_type_list[25]", "time_type_list[26]", "time_type_list[27]",
                            "time_type_list[28]", "time_type_list[29]", "time_type_list[30]", "time_type_list[31]",
                            "time_type_list[32]", "time_type_list[33]", "time_type_list[34]", "time_type_list[35]",
                            "time_type_list[36]", "time_type_list[37]", "time_type_list[38]", "time_type_list[39]",
                            "time_type_list[40]", "time_type_list[41]", "time_type_list[42]", "time_type_list[43]",
                            "time_type_list[44]", "time_type_list[45]", "time_type_list[46]", "time_type_list[47]",
                            "time_type_list[48]", "time_type_list[49]",
                            "weekdays_list[0]", "weekdays_list[1]", "weekdays_list[2]", "weekdays_list[3]",
                            "weekdays_list[4]", "weekdays_list[5]", "weekdays_list[6]", "weekdays_list[7]",
                            "weekdays_list[8]", "weekdays_list[9]", "weekdays_list[10]", "weekdays_list[11]",
                            "weekdays_list[12]", "weekdays_list[13]", "weekdays_list[14]", "weekdays_list[15]",
                            "weekdays_list[16]", "weekdays_list[17]", "weekdays_list[18]", "weekdays_list[19]",
                            "weekdays_list[20]", "weekdays_list[21]", "weekdays_list[22]", "weekdays_list[23]",
                            "weekdays_list[24]", "weekdays_list[25]", "weekdays_list[26]", "weekdays_list[27]",
                            "weekdays_list[28]", "weekdays_list[29]", "weekdays_list[30]", "weekdays_list[31]",
                            "weekdays_list[32]", "weekdays_list[33]", "weekdays_list[34]", "weekdays_list[35]",
                            "weekdays_list[36]", "weekdays_list[37]", "weekdays_list[38]", "weekdays_list[39]",
                            "weekdays_list[40]", "weekdays_list[41]", "weekdays_list[42]", "weekdays_list[43]",
                            "weekdays_list[44]", "weekdays_list[45]", "weekdays_list[46]", "weekdays_list[47]",
                            "weekdays_list[48]", "weekdays_list[49]"
                            ]
        self.user_info_list = ["user_id", "gender", "visit_city", "is_supervip", "ctr_30", "ord_30"]
        self.item_info_list = ["shop_id", "item_id", "city_id", "district_id", "shop_aoi_id", "shop_geohash_6",
                               "shop_geohash_12", "brand_id", "category_1_id", "merge_standard_food_id", "rank_7",
                               "rank_30", "rank_90", "hours"]
        self.context_list = ["times", "hours", "time_type", "weekdays", "geohash12"]

        self.shop_list = ["shop_id_list[%d]" % i for i in range(self.seq_len)]
        self.item_list = ["item_id_list[%d]" % i for i in range(self.seq_len)]
        self.cate_list = ["category_1_id_list[%d]" % i for i in range(self.seq_len)]
        self.aoi_list = ["shop_aoi_id_list[%d]" % i for i in range(self.seq_len)]
        self.hour_list = ["hours_list[%d]" % i for i in range(self.seq_len)]
        self.dense_fea = ["avg_price", "total_amt_30"]
        self.dense_seq_fea = ["price_list[0]", "price_list[1]", "price_list[2]", "price_list[3]", "price_list[4]",
                              "price_list[5]", "price_list[6]", "price_list[7]", "price_list[8]", "price_list[9]",
                              "price_list[10]", "price_list[11]", "price_list[12]", "price_list[13]", "price_list[14]",
                              "price_list[15]", "price_list[16]", "price_list[17]", "price_list[18]", "price_list[19]",
                              "price_list[20]", "price_list[21]", "price_list[22]", "price_list[23]", "price_list[24]",
                              "price_list[25]", "price_list[26]", "price_list[27]", "price_list[28]", "price_list[29]",
                              "price_list[30]", "price_list[31]", "price_list[32]", "price_list[33]", "price_list[34]",
                              "price_list[35]", "price_list[36]", "price_list[37]", "price_list[38]", "price_list[39]",
                              "price_list[40]", "price_list[41]", "price_list[42]", "price_list[43]", "price_list[44]",
                              "price_list[45]", "price_list[46]", "price_list[47]", "price_list[48]", "price_list[49]"]
        self.cat_list = [self.user_info_list, self.item_info_list, self.context_list, self.shop_list, self.item_list,
                         self.cate_list, self.aoi_list, self.hour_list]
        self.dense_list = [self.dense_fea, self.dense_seq_fea]

        self.channel_num = self.params['channel_num']
        self.feature_num = self.params['feature_num']  # Diffusion feature number

        # Time MLP
        self.timesteps = self.params['timesteps']
        self.infer_timesteps = self.params['infer_timesteps']
        self.beta_sche = self.params['beta_sche']
        self.beta_start = self.params['beta_start']
        self.beta_end = self.params['beta_end']

        def swish(x):
            return x * tf.sigmoid(x)

        self.time_embed = tf.keras.Sequential([
            tf.keras.layers.Dense(self.embed_dim * self.feature_num * 4),
            tf.keras.layers.Activation(swish),
            tf.keras.layers.Dense(self.embed_dim * self.feature_num)]
        )

        # Beta schedule configuration
        if self.beta_sche == 'linear':
            self.betas = self.linear_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
        elif self.beta_sche == 'exp':  # Exponential
            self.betas = self.exp_beta_schedule(self.timesteps)
        elif self.beta_sche == 'cosine':
            self.betas = self.cosine_beta_schedule(timesteps=self.timesteps)

        self.alphas = 1. - self.betas  # [alpha_1, alpha_2, ..., alpha_t]
        self.alphas_cumprod = tf.cumprod(
            self.alphas, axis=0)  # Cumulative product of alphas [alpha_cumprod_1, alpha_cumprod_2, ..., alpha_cumprod_t]
        self.alphas_cumprod_prev = tf.pad(self.alphas_cumprod[:-1], [(1, 0)], constant_values=1.0)  # Padding for cumulative alpha
        self.sqrt_recip_alphas = tf.sqrt(1.0 / self.alphas)  # Reciprocal square root

        # Diffusion calculations
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = tf.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = tf.sqrt(1. / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = self.betas * tf.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * tf.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


        # Diffusion and transformer models
        self.n_layers = self.params['n_layers']
        self.num_heads = self.params['num_heads']
        self.dff_factor = self.params['dff_factor']
        self.diffusion_blocks = [
            Diffusion_Net(self.embed_dim * self.feature_num, self.num_heads, self.dff_factor,
                          "DiffusionBlocks_{}".format(i)) for i in range(self.n_layers)]
        self.transformer_blocks = [
            Transformer(self.embed_dim * self.feature_num, self.num_heads, self.dff_factor,
                          "TransformerBlocks_{}".format(i)) for i in range(self.n_layers)]

        mlp_hidden_units_factor = [4]
        mlp_hidden_units = [unit * self.embed_dim * self.feature_num for unit in mlp_hidden_units_factor]
        self.ProjectHeads1 = MLP(self.feature_num * self.embed_dim, mlp_hidden_units,
                                 self.feature_num * self.embed_dim, activation_fn='relu',
                                 variable_scope='ProjectHeads1')
        self.ProjectHeads2 = MLP(self.feature_num * self.embed_dim, mlp_hidden_units,
                                 self.feature_num * self.embed_dim, activation_fn='relu',
                                 variable_scope='ProjectHeads2')
        self.lamda1 = self.params['lamda1']
        self.lamda2 = self.params['lamda2']
        self.temp = self.params['temp']

    def recieve_gather_features(self, cat_feas, dense_feas):
        # cat_feas
        self.cat_fea_split = cat_feas
        self.user_info_list, self.item_info_list, self.context_list, self.shop_list, self.item_list, self.cate_list, self.aoi_list, self.hour_list = self.cat_fea_split
        self.user_info_list_emb, self.item_info_list_emb, self.context_list_emb, self.shop_list_emb, self.item_list_emb, self.cate_list_emb, self.aoi_list_emb, self.hour_list_emb = self.ptable_lookup(
            list_ids=self.cat_fea_split, v_name=self.__class__.__name__)

        self.dense_fea_split = dense_feas
        self.dense_fea, self.dense_seq_fea = self.dense_fea_split


        self.reverse_ts_list = self.hour_list
        self.target_ts = tf.expand_dims(self.item_info_list[:,-1], -1)
        self.mix_list = tf.concat([self.shop_list, tf.expand_dims(self.item_info_list[:,0], -1)], -1)

    def __call__(self, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
        output_dense = tf.layers.batch_normalization(inputs=self.dense_fea, training=self.is_training,
                                                     name='num_fea_batch_norm2')
        tgt_emb = tf.reshape(self.item_info_list_emb, [self.item_info_list_emb.shape[0], 1, -1])
        # Concat target and history category embeddings
        self.target_ts_emb = tgt_emb[:, :, -embed_dim:]
        self.reverse_ts_emb = self.hour_list_emb
        his_cate = tf.concat(
            [self.shop_list_emb, self.item_list_emb, self.cate_list_emb, self.aoi_list_emb, self.hour_list_emb], -1)
        target_cate = tf.layers.dense(tgt_emb, embed_dim * self.feature_num, activation=None, name='tgt_emb')

        # Process target category
        target_cate = tf.layers.dense(
            target_cate,
            self.feature_num * self.embed_dim * self.channel_num,
            activation=None,
        )
        reshaped_embedding = tf.reshape(target_cate, [-1, self.feature_num * self.embed_dim, self.channel_num])
        reshaped_embedding += 1e-8 * tf.random.normal(tf.shape(reshaped_embedding))

        # QR decomposition
        orthogonal_channels, r = tf.qr(reshaped_embedding)  # Orthogonal [batch, channel_num, embedding]

        # Compute similarity between each user action and each channel
        similarities = tf.matmul(his_cate, orthogonal_channels)  # [batch, seq_len, channel_num]

        # Find the channel index with the highest similarity for each action
        max_channel_indices = tf.argmax(similarities, axis=-1, output_type=tf.int32)  # [batch, seq_len]

        # Generate One-Hot mask, retaining only the highest similarity channel
        channel_mask = tf.one_hot(max_channel_indices, depth=self.channel_num, dtype=tf.float32)

        # Second stage: Find top-k actions for each channel
        channel_similarities = tf.transpose(similarities, [0, 2, 1])

        def generate_topk_mask_v2(similarities, topk):
            """Optimized mask generation"""
            # Get dynamic shapes
            batch_size = tf.shape(similarities)[0]
            channel_num = tf.shape(similarities)[1]
            seq_len = tf.shape(similarities)[2]

            # Get Top-K indices
            _, topk_indices = tf.math.top_k(similarities, k=topk)

            # Create scatter indices
            batch_indices = tf.tile(
                tf.reshape(tf.range(batch_size), [batch_size, 1, 1, 1]),
                [1, channel_num, topk, 1]
            )
            channel_indices = tf.tile(
                tf.reshape(tf.range(channel_num), [1, channel_num, 1, 1]),
                [batch_size, 1, topk, 1]
            )
            indices = tf.concat([
                batch_indices,
                channel_indices,
                tf.expand_dims(topk_indices, -1)
            ], axis=-1)

            # Generate sparse mask
            updates = tf.ones_like(topk_indices, dtype=tf.float32)
            shape = tf.stack([batch_size, channel_num, seq_len])
            topk_mask = tf.scatter_nd(indices=tf.reshape(indices, [-1, 3]), updates=tf.reshape(updates, [-1]), shape=shape)

            return tf.minimum(topk_mask, 1.0)  # Ensure values do not exceed 1

        # Get top-k indices for each channel
        topk_mask = generate_topk_mask_v2(channel_similarities, topk=100)

        # Transpose back to original dimensions
        topk_mask = tf.transpose(topk_mask, [0, 2, 1])

        # Combine two masks using logical AND
        combined_mask = channel_mask * topk_mask

        # Apply combined mask
        masked_similarities = similarities * combined_mask

        # Aggregate user behavior representations
        activated_behaviors = tf.expand_dims(his_cate, 2) * tf.expand_dims(masked_similarities, -1)
        aggregated_repr = tf.reduce_mean(activated_behaviors, axis=1)

        aggregated_repr_norm = tf.nn.l2_normalize(aggregated_repr, axis=-1)

        orthogonal_channels = tf.transpose(orthogonal_channels, [0, 2, 1])
        orthogonal_channels = tf.nn.l2_normalize(orthogonal_channels, axis=-1)

        variable_scope = "Diffusion"
        with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
            if self.is_training:
                self.loss = []
                for i in range(self.channel_num):
                    # Random timestep selection
                    ts = tf.random_uniform(shape=(tf.shape(orthogonal_channels)[0],), minval=0, maxval=self.params['timesteps'], dtype=tf.int32)
                    h_target = tf.expand_dims(orthogonal_channels[:, i, :], 1)

                    loss_i, predicted_x_i = self.p_losses(aggregated_repr_norm, i, h_target, ts, loss_type='l2')
                    self.loss.append(loss_i)

            predicted_x = []
            for i in range(self.channel_num):
                h_target = tf.expand_dims(orthogonal_channels[:, i, :], 1)
                x_i = self.sample(aggregated_repr_norm, i, h_target)
                predicted_x.append(x_i)
            predicted_x = tf.concat(predicted_x, 1)
        predicted_x = tf.Print(predicted_x, ["LWJ:predicted_x", predicted_x], first_n=-1, summarize=-1)

        if self.is_training:
            self.contrastive_loss = self.contrastive(aggregated_repr_norm, predicted_x, self.temp)

        variable_scope = "Transformer"
        with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
            hidden_state = tf.concat([his_cate[:, :self.subseq_len, :], target_cate], axis=1)
            for i, block in enumerate(self.diffusion_blocks):
                hidden_state = block(hidden_state, h_target)
            transformer_output = hidden_state


        user_cate = self.user_info_list_emb
        # Return concatenated features including aggregated representation, predicted embeddings, and attention output
        return tf.concat([output_dense, tf.layers.flatten(user_cate),tf.layers.flatten(self.item_info_list_emb),
                            tf.layers.flatten(self.context_list_emb),
                          tf.reshape(aggregated_repr, [-1, self.embed_dim * self.feature_num * self.channel_num]),
                          tf.reshape(predicted_x, [-1, self.feature_num * self.embed_dim * self.channel_num]),
                          transformer_output[:,-1,:],
                          tf.squeeze(target_cate, axis=1)], -1)

    def get_auxiliary_loss(self):
        # Compute auxiliary loss based on mean of losses
        loss = sum(self.loss) / self.channel_num
        return self.lamda1 * loss, self.lamda2 * self.contrastive_loss

    def linear_beta_schedule(self, timesteps, beta_start, beta_end):
        # Linear beta schedule
        return tf.linspace(beta_start, beta_end, timesteps)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        # Cosine beta schedule
        steps = timesteps + 1
        x = tf.linspace(0, timesteps, steps)
        alphas_cumprod = tf.cos(((x / timesteps) + s) / (1 + s) * tf.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return tf.clip(betas, 0.0001, 0.9999)

    def exp_beta_schedule(self, timesteps, beta_min=0.1, beta_max=10):
        # Exponential beta schedule
        x = tf.linspace(1.0, 2.0 * timesteps + 1.0, timesteps)
        betas = 1 - tf.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps ** 2))
        return betas

    def contrastive(self, orig_embs, gen_embs, temperature=0.07):
        # Contrastive loss computation
        orig_embs = tf.transpose(orig_embs, [1, 0, 2])
        gen_embs = tf.transpose(gen_embs, [1, 0, 2])
        orig_embs = self.ProjectHeads1(orig_embs)
        gen_embs = self.ProjectHeads2(gen_embs)
        orig_embs = tf.nn.l2_normalize(orig_embs, axis=-1)
        gen_embs = tf.nn.l2_normalize(gen_embs, axis=-1)

        # Compute similarity matrix
        similarity = tf.matmul(orig_embs, gen_embs, transpose_b=True) / temperature

        # Positive similarities
        pos_similarity = tf.linalg.diag_part(similarity)

        # Logsumexp with positive samples
        logsumexp = tf.reduce_logsumexp(similarity, axis=2)

        # Loss calculation
        loss = logsumexp - pos_similarity
        return tf.reduce_mean(loss)

    def q_sample(self, x_start, t, noise=None):
        # Forward diffusion process
        if noise is None:
            noise = tf.random.normal(tf.shape(x_start))

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, tf.shape(x_start))
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, tf.shape(x_start))

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def step_mlp(self, inputs):
        # Time embedding for steps
        half_dim = self.embed_dim * self.feature_num // 2  # Half dimension of embeddings
        embeddings = tf.math.log(10000.0) / (half_dim - 1)
        embeddings = tf.math.exp(tf.range(half_dim, dtype=tf.float32) * -embeddings)
        embeddings = tf.cast(inputs[:, None], dtype=tf.float32) * embeddings[None, :]
        embeddings = tf.concat([tf.sin(embeddings), tf.cos(embeddings)], -1)
        x = self.time_embed(embeddings)
        return x

    def extract(self, a, t, x_shape):
        # Extract value corresponding to timesteps from tensor
        batch_size = t.shape[0]
        out = tf.gather(a, t, axis=-1)
        return tf.reshape(out, [batch_size] + [1] * (len(x_shape) - 1))

    def predict_start_from_noise(self, x_t, t, noise):
        # Predict the start state given current state x_t and noise
        return (
                self.extract(self.sqrt_recip_alphas_cumprod, t, tf.shape(x_t)) * x_t -
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, tf.shape(x_t)) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        # Compute posterior distribution for q(x_{t-1} | x_t, x_0)
        posterior_mean = (
                self.extract(self.posterior_mean_coef1, t, tf.shape(x_t)) * x_start +
                self.extract(self.posterior_mean_coef2, t, tf.shape(x_t)) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        self.posterior_log_variance_clipped = tf.log(tf.maximum(self.posterior_variance, 1e-20))
        posterior_log_variance_clipped = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_losses(self, channels, i, h_target, t, loss_type="l2"):
        # Compute loss for the reverse diffusion process
        x_start = channels[:, i, :]
        x_start = tf.expand_dims(x_start, 1)

        noise = tf.random.normal(tf.shape(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if i == 0:
            input = tf.concat([x_noisy, channels[:, i + 1:, :]], axis=1)
        elif i == self.channel_num - 1:
            input = tf.concat([channels[:, :i, :], x_noisy], axis=1)
        else:
            input = tf.concat([channels[:, :i, :], x_noisy, channels[:, i + 1:, :]], axis=1)

        predicted_x = self.forward(input, h_target, t)  # Predicted x_{t-1}
        predicted_x = predicted_x[:, i, :]
        predicted_x = tf.expand_dims(predicted_x, 1)
        loss = tf.losses.mean_squared_error(x_start, predicted_x)  # MSE loss

        return loss, predicted_x

    def forward(self, x, h_target, step):
        # Pass input through diffusion blocks
        t = self.step_mlp(step)
        t = tf.expand_dims(t, axis=1)
        t = tf.tile(t, [1, tf.shape(x)[1], 1])
        x += t

        hidden_state = x
        for i, block in enumerate(self.diffusion_blocks):
            hidden_state = block(hidden_state, h_target)
        transformer_output = hidden_state

        return transformer_output

    def sample(self, channels, i, h_target):
        # Sampling method for channels in diffusion model
        x = channels[:, i, :]
        x = tf.expand_dims(x, axis=1)
        t = tf.random_uniform(shape=(tf.shape(x)[0],), minval=0, maxval=self.params['timesteps'], dtype=tf.int32)
        noise = tf.random.normal(tf.shape(x))
        x = self.q_sample(x_start=x, t=t, noise=noise)

        for n in reversed(range(0, self.infer_timesteps)):
            x = self.p_sample(x, channels, i, h_target, tf.fill([tf.shape(channels)[0]], value=n), n)
        return x

    def p_sample(self, x, channels, i, h, t, t_index):
        # Perform reverse diffusion sampling
        if i == 0:
            input = tf.concat([x, channels[:, 1:, :]], axis=1)
        elif i == self.channel_num - 1:
            input = tf.concat([channels[:, :i, :], x], axis=1)
        else:
            input = tf.concat([channels[:, :i, :], x, channels[:, i + 1:, :]], axis=1)

        pred_noise = self.forward(input, h, t)
        pred_noise = pred_noise[:, i, :]
        pred_noise = tf.expand_dims(pred_noise, axis=1)
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        model_mean, _, model_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        noise = tf.random.normal(tf.shape(x))
        nonzero_mask = tf.reshape(1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1])
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise
