"""TFRS two-tower retrieval model with ID and side-feature embeddings."""
from __future__ import annotations
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def build_model(user_ids, item_ids, user_vocabularies=None, item_vocabularies=None, embedding_dim: int = 64,
               learning_rate: float = 1e-3):
    """Create the model lazily so preprocessing/API can run without TensorFlow installed."""
    try:
        import tensorflow as tf
        import tensorflow_recommenders as tfrs
    except ImportError as exc:
        raise RuntimeError("Training requires TensorFlow and tensorflow-recommenders; see docs/SETUP.md") from exc

    user_vocabularies = user_vocabularies or {}
    item_vocabularies = item_vocabularies or {}

    class Encoder(tf.keras.Model):
        def __init__(self, id_vocabulary, side_vocabularies):
            super().__init__()
            self.id_lookup = tf.keras.layers.StringLookup(vocabulary=list(id_vocabulary), mask_token=None)
            self.id_embedding = tf.keras.layers.Embedding(self.id_lookup.vocabulary_size(), embedding_dim)
            self.side = {}
            for name, vocabulary in side_vocabularies.items():
                lookup = tf.keras.layers.StringLookup(vocabulary=list(vocabulary), mask_token=None)
                self.side[name] = (lookup, tf.keras.layers.Embedding(lookup.vocabulary_size(), embedding_dim // 2 or 1))
            self.numeric = tf.keras.layers.Dense(embedding_dim // 2, activation="relu")
            self.projection = tf.keras.Sequential([
                tf.keras.layers.Dense(embedding_dim, activation="relu"),
                tf.keras.layers.Dense(embedding_dim),
                tf.keras.layers.UnitNormalization(),
            ])

        def call(self, features):
            parts = [self.id_embedding(self.id_lookup(features["entity_id"]))]
            for name, (lookup, embedding) in self.side.items():
                parts.append(embedding(lookup(features[name])))
            if "numeric" in features:
                parts.append(self.numeric(tf.cast(features["numeric"], tf.float32)))
            return self.projection(tf.concat(parts, axis=-1))

    class RetrievalModel(tfrs.Model):
        def __init__(self):
            super().__init__()
            self.user_tower = Encoder(user_ids, user_vocabularies)
            self.item_tower = Encoder(item_ids, item_vocabularies)
            self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.logit_scale = self.add_weight(name="logit_scale", shape=(), initializer=tf.keras.initializers.Constant(5.0), trainable=True)

        def compute_loss(self, features, training=False):
            user = self.user_tower(features["user"])
            item = self.item_tower(features["item"])
            # Explicit sampled non-interactions carry label 0; purchases carry 1.
            logits = self.logit_scale * tf.reduce_sum(user * item, axis=1)
            return self.loss_fn(tf.cast(features["label"], tf.float32), logits)

    model = RetrievalModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
    return model