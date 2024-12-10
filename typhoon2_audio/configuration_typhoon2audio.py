from transformers import PretrainedConfig, WhisperConfig

class BEATsConfig(PretrainedConfig):
    def __init__(self, cfg=None):
        # update the default values to BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
        self.input_patch_size: int = 16  # path size of patch embedding
        self.embed_dim: int = 512  # patch embedding dimension
        self.conv_bias: bool = False  # include bias in conv encoder

        self.encoder_layers: int = 12  # num encoder layers in the transformer
        self.encoder_embed_dim: int = 768  # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 3072  # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 12  # num encoder attention heads
        self.activation_fn: str = "gelu"  # activation function to use

        self.layer_wise_gradient_decay_ratio: float = 0.6  # ratio for layer-wise gradient decay
        self.layer_norm_first: bool = False  # apply layernorm first in the transformer
        self.deep_norm: bool = True  # apply deep_norm first in the transformer

        # dropouts
        self.dropout: float = 0.0  # dropout probability for the transformer
        self.attention_dropout: float = 0.0  # dropout probability for attention weights
        self.activation_dropout: float = 0.0  # dropout probability after activation in FFN
        self.encoder_layerdrop: float = 0.05  # probability of dropping a tarnsformer layer
        self.dropout_input: float = 0.0  # dropout to apply to the input (after feat extr)

        # positional embeddings
        self.conv_pos: int = 128  # number of filters for convolutional positional embeddings
        self.conv_pos_groups: int = 16  # number of groups for convolutional positional embedding

        # relative position embedding
        self.relative_position_embedding: bool = True  # apply relative position embedding
        self.num_buckets: int = 320  # number of buckets for relative position embedding
        self.max_distance: int = 800  # maximum distance for relative position embedding
        self.gru_rel_pos: bool = True  # apply gated relative position embedding

        # label predictor
        self.finetuned_model: bool = True  # whether the model is a fine-tuned model.
        self.predictor_dropout: float = 0.0  # dropout probability for the predictor
        self.predictor_class: int = 527  # target class number for the predictor

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class Typhoon2AudioConfig(PretrainedConfig):
    model_type = "typhoon2audio"

    def __init__(self, **kwargs):   
        # LLM -- Llama3
        self.llama_base_model = "scb10x/typhoon-2-llama31-8b-instruct-beta-v1"

        # Whisper
        self.whisper_extractor_feature_size=128
        self.whisper = WhisperConfig(
            activation_dropout=0.0,
            activation_function="gelu",
            apply_spec_augment=True,
            attention_dropout=0.0,
            begin_suppress_tokens=[220, 50257],
            bos_token_id=50257,
            d_model=1280,
            decoder_attention_heads=20,
            decoder_ffn_dim=5120,
            decoder_layerdrop=0.0,
            decoder_layers=32,
            decoder_start_token_id=50258,
            dropout=0.0,
            encoder_attention_heads=20,
            encoder_ffn_dim=5120,
            encoder_layerdrop=0.0,
            encoder_layers=32,
            eos_token_id=50257,
            init_std=0.02,
            mask_feature_length=64,
            mask_feature_min_masks=0,
            mask_feature_prob=0.1,
            mask_time_length=10,
            mask_time_min_masks=2,
            mask_time_prob=0.1,
            max_length=448,
            max_source_positions=1500,
            max_target_positions=448,
            median_filter_width=7,
            num_hidden_layers=32,
            num_mel_bins=128,
            pad_token_id=50256,
            scale_embedding=False,
            use_weighted_layer_sum=False,
            vocab_size=51866,
        )
        # BEATs
        self.beats = BEATsConfig()

        # Speech QFormer
        self.speech_qformer_token_num=1
        self.speech_qformer_layer=2
        self.second_per_frame=0.333333
        self.second_stride=0.333333

        # SpeechDecoder CTC
        self.pretraining_tp = 1
        self.ctc_decoder_config='(4,4096,32,11008)'
        self.ctc_upsample_factor=25
        self.ctc_loss_weight=1.0
        self.unit_vocab_size=1000
        self.speech_decoder_ignore_index=-100
        self.attention_bias=False
        self.attention_dropout=0.0
        self.bos_token_id=128000
        self.eos_token_id=128009
        self.head_dim=128
        self.hidden_act="silu"
        self.hidden_size=4096
        self.intermediate_size=14336
        self.max_position_embeddings=131072
        self.mlp_bias=False
        self.num_attention_heads=32
        self.num_hidden_layers=32
        self.num_key_value_heads=8
        self.rms_norm_eps=1e-05
        self.rope_scaling={
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        }
        self.rope_theta=500000.0
        self.vocab_size=128256

        # Unit Vocoder (HiFiGAN)
        self.vocoder_path = {
            'repo_id': 'scb10x/unit-vocoder-gcp-th-v1-00206600',
            'filename': 'checkpoint.pt'
        }
        self.vocoder_config = {
            'resblock': 1,
            'upsample_rates': [5, 4, 4, 2, 2],
            'upsample_kernel_sizes':  [11, 8, 8, 4, 4],
            'upsample_initial_channel': 512,
            'resblock_kernel_sizes': [3, 7, 11],
            'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            'num_embeddings': 1000,
            'embedding_dim': 512,
            'model_in_dim': 512,
            'segment_size': 8960,
            'code_hop_size': 320,
            'num_mels': 80,
            'num_freq': 1025,
            'n_fft': 1024,
            'hop_size': 256,
            'win_size': 1024,
            'sampling_rate': 16000,
            'dur_prediction_weight': 1.0,
            'dur_predictor_params': {
                'encoder_embed_dim': 512, 
                'var_pred_hidden_dim': 512, 
                'var_pred_kernel_size': 3, 
                'var_pred_dropout': 0.5
            } 
        }
        super().__init__(**kwargs)
