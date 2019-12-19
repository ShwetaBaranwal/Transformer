import numpy as np
from layer_normalization import LayerNormalization
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from trig_pos_embd import TrigPosEmbedding
from embeddings import EmbeddingRet, EmbeddingSim
import keras
import keras.backend as K
from tqdm import tqdm

__all__ = [
    'get_custom_objects', 'get_encoders', 'get_decoders', 'get_model', 'decode',
    'attention_builder', 'feed_forward_builder', 'get_encoder_component', 'get_decoder_component',
]


def get_custom_objects():
    return {
        'LayerNormalization': LayerNormalization,
        'MultiHeadAttention': MultiHeadAttention,
        'FeedForward': FeedForward,
        'TrigPosEmbedding': TrigPosEmbedding,
        'EmbeddingRet': EmbeddingRet,
        'EmbeddingSim': EmbeddingSim,
    }


def _wrap_layer(name,
                input_layer,
                build_func,
                dropout_rate=0.0,
                trainable=True,
                use_adapter=False,
                adapter_units=None,
                adapter_activation='relu'):
    """Wrap layers with residual, normalization and dropout.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param build_func: A callable that takes the input tensor and generates the output tensor.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :param use_adapter: Whether to use feed-forward adapters before each residual connections.
    :param adapter_units: The dimension of the first transformation in feed-forward adapter.
    :param adapter_activation: The activation after the first transformation in feed-forward adapter.
    :return: Output layer.
    """
    build_output = build_func(input_layer)
    if dropout_rate > 0.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='%s-Dropout' % name,
        )(build_output)
    else:
        dropout_layer = build_output
    if isinstance(input_layer, list):
        input_layer = input_layer[0]
    if use_adapter:
        adapter = FeedForward(
            units=adapter_units,
            activation=adapter_activation,
            kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
            name='%s-Adapter' % name,
        )(dropout_layer)
        dropout_layer = keras.layers.Add(name='%s-Adapter-Add' % name)([dropout_layer, adapter])
    add_layer = keras.layers.Add(name='%s-Add' % name)([input_layer, dropout_layer])
    normal_layer = LayerNormalization(
        trainable=trainable,
        name='%s-Norm' % name,
    )(add_layer)
    return normal_layer


def attention_builder(name,
                      head_num,
                      activation,
                      history_only,
                      trainable=True):
    """Get multi-head self-attention builder.

    :param name: Prefix of names for internal layers.
    :param head_num: Number of heads in multi-head self-attention.
    :param activation: Activation for multi-head self-attention.
    :param history_only: Only use history data.
    :param trainable: Whether the layer is trainable.
    :return:
    """
    def _attention_builder(x):
        return MultiHeadAttention(
            head_num=head_num,
            activation=activation,
            history_only=history_only,
            trainable=trainable,
            name=name,
        )(x)
    return _attention_builder


def feed_forward_builder(name,
                         hidden_dim,
                         activation,
                         trainable=True):
    """Get position-wise feed-forward layer builder.

    :param name: Prefix of names for internal layers.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param activation: Activation for feed-forward layer.
    :param trainable: Whether the layer is trainable.
    :return:
    """
    def _feed_forward_builder(x):
        return FeedForward(
            units=hidden_dim,
            activation=activation,
            trainable=trainable,
            name=name,
        )(x)
    return _feed_forward_builder


def get_encoder_component(name,
                          input_layer,
                          head_num,
                          hidden_dim,
                          attention_activation=None,
                          feed_forward_activation='relu',
                          dropout_rate=0.0,
                          trainable=True,
                          use_adapter=False,
                          adapter_units=None,
                          adapter_activation='relu'):
    """Multi-head self-attention and feed-forward layer.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :param use_adapter: Whether to use feed-forward adapters before each residual connections.
    :param adapter_units: The dimension of the first transformation in feed-forward adapter.
    :param adapter_activation: The activation after the first transformation in feed-forward adapter.
    :return: Output layer.
    """
    attention_name = '%s-MultiHeadSelfAttention' % name
    feed_forward_name = '%s-FeedForward' % name
    attention_layer = _wrap_layer(
        name=attention_name,
        input_layer=input_layer,
        build_func=attention_builder(
            name=attention_name,
            head_num=head_num,
            activation=attention_activation,
            history_only=False,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
        use_adapter=use_adapter,
        adapter_units=adapter_units,
        adapter_activation=adapter_activation,
    )
    feed_forward_layer = _wrap_layer(
        name=feed_forward_name,
        input_layer=attention_layer,
        build_func=feed_forward_builder(
            name=feed_forward_name,
            hidden_dim=hidden_dim,
            activation=feed_forward_activation,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
        use_adapter=use_adapter,
        adapter_units=adapter_units,
        adapter_activation=adapter_activation,
    )
    return feed_forward_layer


def get_decoder_component(name,
                          input_layer,
                          encoded_layer,
                          head_num,
                          hidden_dim,
                          attention_activation=None,
                          feed_forward_activation='relu',
                          dropout_rate=0.0,
                          trainable=True,
                          use_adapter=False,
                          adapter_units=None,
                          adapter_activation='relu'):
    """Multi-head self-attention, multi-head query attention and feed-forward layer.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param encoded_layer: Encoded layer from encoder.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :param use_adapter: Whether to use feed-forward adapters before each residual connections.
    :param adapter_units: The dimension of the first transformation in feed-forward adapter.
    :param adapter_activation: The activation after the first transformation in feed-forward adapter.
    :return: Output layer.
    """
    self_attention_name = '%s-MultiHeadSelfAttention' % name
    query_attention_name = '%s-MultiHeadQueryAttention' % name
    feed_forward_name = '%s-FeedForward' % name
    self_attention_layer = _wrap_layer(
        name=self_attention_name,
        input_layer=input_layer,
        build_func=attention_builder(
            name=self_attention_name,
            head_num=head_num,
            activation=attention_activation,
            history_only=True,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
        use_adapter=use_adapter,
        adapter_units=adapter_units,
        adapter_activation=adapter_activation,
    )
    query_attention_layer = _wrap_layer(
        name=query_attention_name,
        input_layer=[self_attention_layer, encoded_layer, encoded_layer],
        build_func=attention_builder(
            name=query_attention_name,
            head_num=head_num,
            activation=attention_activation,
            history_only=False,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
        use_adapter=use_adapter,
        adapter_units=adapter_units,
        adapter_activation=adapter_activation,
    )
    feed_forward_layer = _wrap_layer(
        name=feed_forward_name,
        input_layer=query_attention_layer,
        build_func=feed_forward_builder(
            name=feed_forward_name,
            hidden_dim=hidden_dim,
            activation=feed_forward_activation,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
        use_adapter=use_adapter,
        adapter_units=adapter_units,
        adapter_activation=adapter_activation,
    )
    return feed_forward_layer


def get_encoders(encoder_num,
                 input_layer,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu'):
    """Get encoders.

    :param encoder_num: Number of encoder components.
    :param input_layer: Input layer.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :param use_adapter: Whether to use feed-forward adapters before each residual connections.
    :param adapter_units: The dimension of the first transformation in feed-forward adapter.
    :param adapter_activation: The activation after the first transformation in feed-forward adapter.
    :return: Output layer.
    """
    last_layer = input_layer
    for i in range(encoder_num):
        last_layer = get_encoder_component(
            name='Encoder-%d' % (i + 1),
            input_layer=last_layer,
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
            trainable=trainable,
            use_adapter=use_adapter,
            adapter_units=adapter_units,
            adapter_activation=adapter_activation,
        )
    return last_layer


def get_decoders(decoder_num,
                 input_layer,
                 encoded_layer,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu'):
    """Get decoders.

    :param decoder_num: Number of decoder components.
    :param input_layer: Input layer.
    :param encoded_layer: Encoded layer from encoder.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :param use_adapter: Whether to use feed-forward adapters before each residual connections.
    :param adapter_units: The dimension of the first transformation in feed-forward adapter.
    :param adapter_activation: The activation after the first transformation in feed-forward adapter.
    :return: Output layer.
    """
    last_layer = input_layer
    for i in range(decoder_num):
        last_layer = get_decoder_component(
            name='Decoder-%d' % (i + 1),
            input_layer=last_layer,
            encoded_layer=encoded_layer,
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
            trainable=trainable,
            use_adapter=use_adapter,
            adapter_units=adapter_units,
            adapter_activation=adapter_activation,
        )
    return last_layer


def get_model(token_num,
              embed_dim,
              encoder_num,
              decoder_num,
              head_num,
              hidden_dim,
              attention_activation=None,
              feed_forward_activation='relu',
              dropout_rate=0.0,
              use_same_embed=True,
              embed_weights=None,
              embed_trainable=None,
              trainable=True,
              use_adapter=False,
              adapter_units=None,
              adapter_activation='relu'):
    """Get full model without compilation.

    :param token_num: Number of distinct tokens.
    :param embed_dim: Dimension of token embedding.
    :param encoder_num: Number of encoder components.
    :param decoder_num: Number of decoder components.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param use_same_embed: Whether to use the same token embedding layer. `token_num`, `embed_weights` and
                           `embed_trainable` should be lists of two elements if it is False.
    :param embed_weights: Initial weights of token embedding.
    :param embed_trainable: Whether the token embedding is trainable. It will automatically set to False if the given
                            value is None when embedding weights has been provided.
    :param trainable: Whether the layers are trainable.
    :param use_adapter: Whether to use feed-forward adapters before each residual connections.
    :param adapter_units: The dimension of the first transformation in feed-forward adapter.
    :param adapter_activation: The activation after the first transformation in feed-forward adapter.
    :return: Keras model.
    """
    if not isinstance(token_num, list):
        token_num = [token_num, token_num]
    encoder_token_num, decoder_token_num = token_num

    if not isinstance(embed_weights, list):
        embed_weights = [embed_weights, embed_weights]
    encoder_embed_weights, decoder_embed_weights = embed_weights
    if encoder_embed_weights is not None:
        encoder_embed_weights = [encoder_embed_weights]
    if decoder_embed_weights is not None:
        decoder_embed_weights = [decoder_embed_weights]

    if not isinstance(embed_trainable, list):
        embed_trainable = [embed_trainable, embed_trainable]
    encoder_embed_trainable, decoder_embed_trainable = embed_trainable
    if encoder_embed_trainable is None:
        encoder_embed_trainable = encoder_embed_weights is None
    if decoder_embed_trainable is None:
        decoder_embed_trainable = decoder_embed_weights is None

    if use_same_embed:
        encoder_embed_layer = decoder_embed_layer = EmbeddingRet(
            input_dim=encoder_token_num,
            output_dim=embed_dim,
            mask_zero=True,
            weights=encoder_embed_weights,
            trainable=encoder_embed_trainable,
            name='Token-Embedding',
        )
    else:
        encoder_embed_layer = EmbeddingRet(
            input_dim=encoder_token_num,
            output_dim=embed_dim,
            mask_zero=True,
            weights=encoder_embed_weights,
            trainable=encoder_embed_trainable,
            name='Encoder-Token-Embedding',
        )
        decoder_embed_layer = EmbeddingRet(
            input_dim=decoder_token_num,
            output_dim=embed_dim,
            mask_zero=True,
            weights=decoder_embed_weights,
            trainable=decoder_embed_trainable,
            name='Decoder-Token-Embedding',
        )
    encoder_input = keras.layers.Input(shape=(None,), name='Encoder-Input')
    encoder_embed = TrigPosEmbedding(
        name='Encoder-Embedding',
    )(encoder_embed_layer(encoder_input)[0])
    encoded_layer = get_encoders(
        encoder_num=encoder_num,
        input_layer=encoder_embed,
        head_num=head_num,
        hidden_dim=hidden_dim,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=dropout_rate,
        trainable=trainable,
        use_adapter=use_adapter,
        adapter_units=adapter_units,
        adapter_activation=adapter_activation,
    )
    decoder_input = keras.layers.Input(shape=(None,), name='Decoder-Input')
    decoder_embed, decoder_embed_weights = decoder_embed_layer(decoder_input)
    decoder_embed = TrigPosEmbedding(
        name='Decoder-Embedding',
    )(decoder_embed)
    decoded_layer = get_decoders(
        decoder_num=decoder_num,
        input_layer=decoder_embed,
        encoded_layer=encoded_layer,
        head_num=head_num,
        hidden_dim=hidden_dim,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=dropout_rate,
        trainable=trainable,
        use_adapter=use_adapter,
        adapter_units=adapter_units,
        adapter_activation=adapter_activation,
    )
    dense_layer = EmbeddingSim(
        trainable=trainable,
        name='Output',
    )([decoded_layer, decoder_embed_weights])
    return keras.models.Model(inputs=[encoder_input, decoder_input], outputs=dense_layer)


def decode(model,
           tokens,
           dict_token2idx,
           dict_idx2token=None,
           start_token = '<s>',
           end_token = '</s>',
           pad_token = '<pad>',
           max_len=20,
           sp=None):
    """Decode with the given model and input tokens.

    :param model: The trained model.
    :param tokens: The input tokens of encoder.
    :param dict_token2idx: Dictionary for token to id mapping.
    :param dict_idx2token: Dictionary for id to token mapping.
    :param start_token: The token that represents the start of a sentence.
    :param end_token: The token that represents the end of a sentence.
    :param pad_token: The token that represents padding.
    :param max_len: Maximum length of decoded list.
    :param sp: Sentence piece model object.
    """
    is_single = not isinstance(tokens[0], list)
    if is_single:
        tokens = [tokens]
    batch_size = len(tokens)
    encoder_inputs = np.array(tokens)
    decoder_inputs = np.array([[dict_token2idx[start_token]] for _ in range(batch_size)])

    _decoder_inputs = decoder_inputs
    for _ in tqdm(range(max_len-1)):
      logits = model.predict([encoder_inputs, _decoder_inputs])
      pred = np.argmax(logits, axis=-1)
      _decoder_inputs = np.concatenate((decoder_inputs, pred), axis=1)

    output_idx = _decoder_inputs

    raw_output_token_ = None
    if dict_idx2token:
        raw_output_token_ = [' '.join(map(lambda x: dict_idx2token[x], i)) for i in output_idx]

    output_sentence=None
    if sp:
        output_sentence = [sp.DecodeIds(list(map(lambda x: int(x), list(j)))) for j in output_idx]

    return output_idx, raw_output_token_, output_sentence
