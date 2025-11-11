import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
from sklearn.svm import OneClassSVM
import pickle

class LSTMAutoencoder:
    def __init__(self, sequence_length, n_features, latent_dim=32):
        """
        Initialize LSTM Autoencoder
        
        Args:
            sequence_length: Number of time steps
            n_features: Number of features per time step
            latent_dim: Dimension of latent space
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.model = None
        self.encoder = None
        
    def build_model(self):
        """Build LSTM Autoencoder architecture"""
        
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Encoder
        encoded = layers.LSTM(64, activation='relu', return_sequences=True)(inputs)
        encoded = layers.LSTM(self.latent_dim, activation='relu', return_sequences=False)(encoded)
        
        # Repeat vector for decoder
        decoded = layers.RepeatVector(self.sequence_length)(encoded)
        
        # Decoder
        decoded = layers.LSTM(self.latent_dim, activation='relu', return_sequences=True)(decoded)
        decoded = layers.LSTM(64, activation='relu', return_sequences=True)(decoded)
        
        # Output layer
        outputs = layers.TimeDistributed(layers.Dense(self.n_features))(decoded)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.MeanSquaredError(),  # Use class instead of string
            metrics=[keras.metrics.MeanAbsoluteError()]  # Use class instead of string
        )
        
        # Create encoder model for extracting latent features
        self.encoder = models.Model(inputs=inputs, outputs=encoded)
        
        print("✅ Model built successfully!")
        print(self.model.summary())
        
        return self.model
    
    def train(self, X_train, X_val, epochs=50, batch_size=32):
        """
        Train the autoencoder
        
        Args:
            X_train: Training data (normal traffic only)
            X_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            history: Training history
        """
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        checkpoint = ModelCheckpoint(
            '../results/models/lstm_autoencoder_best.keras',  # Changed to .keras
            monitor='val_loss',
            save_best_only=True
        )
        
        # Train model (input = output for autoencoder)
        history = self.model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        return history
    
    def extract_latent_features(self, X):
        """Extract latent features using encoder"""
        return self.encoder.predict(X, verbose=0)
    
    def reconstruct(self, X):
        """Reconstruct input data"""
        return self.model.predict(X, verbose=0)
    
    def compute_reconstruction_error(self, X):
        """Compute reconstruction error (MSE)"""
        X_reconstructed = self.reconstruct(X)
        mse = np.mean(np.square(X - X_reconstructed), axis=(1, 2))
        return mse
    
    def save_model(self, filepath):
        """Save model in Keras format"""
        # Use .keras extension for new format
        if not filepath.endswith('.keras'):
            filepath = filepath.replace('.h5', '.keras')
        
        self.model.save(filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model in Keras format"""
        # Handle both .h5 and .keras files
        if filepath.endswith('.h5'):
            filepath_keras = filepath.replace('.h5', '.keras')
            if os.path.exists(filepath_keras):
                filepath = filepath_keras
        
        try:
            # Load model
            self.model = keras.models.load_model(filepath)
            
            # Recreate encoder from loaded model
            # Find the latent layer (LSTM with latent_dim units)
            for i, layer in enumerate(self.model.layers):
                if isinstance(layer, layers.LSTM) and not layer.return_sequences:
                    self.encoder = models.Model(
                        inputs=self.model.input,
                        outputs=layer.output
                    )
                    break
            
            print(f"✅ Model loaded from {filepath}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please retrain the model!")
            raise


class OneClassSVMClassifier:
    def __init__(self, kernel='rbf', nu=0.1, gamma='auto'):
        """
        Initialize One-Class SVM
        
        Args:
            kernel: Kernel type
            nu: Upper bound on fraction of training errors
            gamma: Kernel coefficient
        """
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        
    def train(self, latent_features):
        """
        Train OC-SVM on latent features
        
        Args:
            latent_features: Latent features from autoencoder
        """
        print("Training One-Class SVM...")
        self.model.fit(latent_features)
        print("✅ OC-SVM training complete!")
    
    def predict(self, latent_features):
        """
        Predict anomalies
        
        Args:
            latent_features: Latent features
            
        Returns:
            predictions: 1 for normal, -1 for anomaly
        """
        return self.model.predict(latent_features)
    
    def save_model(self, filepath):
        """Save OC-SVM model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ OC-SVM saved to {filepath}")
    
    def load_model(self, filepath):
        """Load OC-SVM model"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✅ OC-SVM loaded from {filepath}")