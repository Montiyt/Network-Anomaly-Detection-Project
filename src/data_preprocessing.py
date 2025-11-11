import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, sequence_length=10):
        """
        Initialize preprocessor
        
        Args:
            sequence_length: Number of time steps for LSTM sequences
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        
    def load_data(self, filepath):
        """Load dataset from CSV"""
        df = pd.read_csv(filepath)
        print(f"Loaded dataset with shape: {df.shape}")
        return df
    
    def separate_normal_attack(self, df, label_column='label'):
        """
        Separate normal and attack traffic
        """
        # Try different possible label column names
        possible_labels = ['label', 'Label', 'class', 'Class', 'target']
        
        label_col = None
        for col in possible_labels:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            print("Available columns:", df.columns.tolist())
            raise ValueError("Could not find label column. Please specify correct column name.")
        
        # Assuming 0 = Normal, 1 = Attack (or similar binary classification)
        normal_df = df[df[label_col] == 0].copy()
        attack_df = df[df[label_col] == 1].copy()
        
        # If no data found, try with string labels
        if len(normal_df) == 0:
            normal_df = df[df[label_col].str.lower().str.contains('normal', na=False)].copy()
            attack_df = df[~df[label_col].str.lower().str.contains('normal', na=False)].copy()
        
        print(f"Normal samples: {len(normal_df)}")
        print(f"Attack samples: {len(attack_df)}")
        
        return normal_df, attack_df
    
    def clean_data(self, df):
        """Remove missing values and infinite values"""
        print(f"Data shape before cleaning: {df.shape}")
        
        # Remove rows with missing values
        df = df.dropna()
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        print(f"Data shape after cleaning: {df.shape}")
        return df
    
    def select_features(self, df, exclude_columns=['label', 'Label', 'attack_type', 'Attack_type', 'timestamp', 'Timestamp']):
        """
        Select only numeric features for training
        """
        # Get label if exists
        label_col = None
        for col in ['label', 'Label', 'class', 'Class']:
            if col in df.columns:
                label_col = col
                break
        
        labels = df[label_col].values if label_col else None
        
        # Select numeric features only
        feature_columns = [col for col in df.columns 
                          if col not in exclude_columns 
                          and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        features = df[feature_columns]
        print(f"Selected {len(feature_columns)} features")
        
        return features, labels, feature_columns
    
    def normalize_features(self, train_data, test_data=None):
        """
        Normalize features using Min-Max scaling
        """
        # Fit scaler on training data
        normalized_train = self.scaler.fit_transform(train_data)
        
        if test_data is not None:
            normalized_test = self.scaler.transform(test_data)
            return normalized_train, normalized_test
        
        return normalized_train
    
    def create_sequences(self, data, labels=None):
        """
        Create sequences for LSTM input
        """
        sequences = []
        sequence_labels = []
        
        for i in range(len(data) - self.sequence_length + 1):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
            
            if labels is not None:
                sequence_labels.append(labels[i + self.sequence_length - 1])
        
        sequences = np.array(sequences)
        
        if labels is not None:
            sequence_labels = np.array(sequence_labels)
            return sequences, sequence_labels
        
        return sequences
    
    def prepare_data(self, filepath, test_size=0.3, random_state=42):
        """
        Complete preprocessing pipeline
        """
        print("="*50)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*50)
        
        # 1. Load data
        df = self.load_data(filepath)
        
        # 2. Clean data
        df = self.clean_data(df)
        
        # 3. Separate normal and attack traffic
        normal_df, attack_df = self.separate_normal_attack(df)
        
        # 4. Select features
        normal_features, normal_labels, feature_columns = self.select_features(normal_df)
        attack_features, attack_labels, _ = self.select_features(attack_df)
        
        # 5. Split normal data into train and validation
        X_train_normal, X_val_normal = train_test_split(
            normal_features, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # 6. Normalize features
        X_train_normal_scaled = self.normalize_features(X_train_normal)
        X_val_normal_scaled = self.scaler.transform(X_val_normal)
        X_attack_scaled = self.scaler.transform(attack_features)
        
        # 7. Create sequences
        print(f"\nCreating sequences with length {self.sequence_length}...")
        X_train_seq = self.create_sequences(X_train_normal_scaled)
        X_val_normal_seq = self.create_sequences(X_val_normal_scaled)
        X_attack_seq = self.create_sequences(X_attack_scaled)
        
        # Create labels (0 for normal, 1 for attack)
        y_train = np.zeros(len(X_train_seq))
        y_val_normal = np.zeros(len(X_val_normal_seq))
        y_attack = np.ones(len(X_attack_seq))
        
        # Combine validation: normal + attack
        X_val_combined = np.vstack([X_val_normal_seq, X_attack_seq])
        y_val_combined = np.hstack([y_val_normal, y_attack])
        
        # Shuffle validation data
        shuffle_idx = np.random.permutation(len(X_val_combined))
        X_val_combined = X_val_combined[shuffle_idx]
        y_val_combined = y_val_combined[shuffle_idx]
        
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETE")
        print("="*50)
        print(f"Training data shape: {X_train_seq.shape}")
        print(f"Validation data shape: {X_val_combined.shape}")
        print(f"Number of features: {len(feature_columns)}")
        print(f"Sequence length: {self.sequence_length}")
        
        return {
            'X_train': X_train_seq,
            'y_train': y_train,
            'X_val': X_val_combined,
            'y_val': y_val_combined,
            'feature_columns': feature_columns,
            'scaler': self.scaler,
            'sequence_length': self.sequence_length
        }