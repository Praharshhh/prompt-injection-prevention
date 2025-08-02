"""Model training and evaluation."""
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, precision_recall_curve
from torch import nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                        Trainer, TrainingArguments)

from ..features.feature_extraction import PromptFeatureExtractor

class InjectionDetector(nn.Module):
    """Neural network for prompt injection detection."""
    
    def __init__(
        self,
        embedding_dim: int = 384,  # Default for all-MiniLM-L6-v2
        syntactic_features: int = 8,
        hidden_dim: int = 256
    ):
        """Initialize the model."""
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim + syntactic_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.layers(x)

def train_model(
    train_texts: list,
    train_labels: list,
    val_texts: list = None,
    val_labels: list = None,
    model_dir: Path = Path("models"),
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-5
) -> Tuple[InjectionDetector, Dict]:
    """Train the injection detection model."""
    
    # Initialize feature extractor
    feature_extractor = PromptFeatureExtractor()
    
    # Extract features
    X_train = feature_extractor.fit_transform(train_texts)
    X_val = feature_extractor.transform(val_texts) if val_texts else None
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(train_labels).reshape(-1, 1)
    
    if val_texts:
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(val_labels).reshape(-1, 1)
    
    # Initialize model
    model = InjectionDetector(
        embedding_dim=X_train.shape[1] - 8,  # Subtract syntactic feature count
        syntactic_features=8
    )
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        history['train_loss'].append(loss.item())
        
        # Validation
        if val_texts:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                history['val_loss'].append(val_loss.item())
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_dir.mkdir(exist_ok=True)
                    torch.save(model.state_dict(), model_dir / "best_model.pt")
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}"
              + (f" - Val Loss: {val_loss.item():.4f}" if val_texts else ""))
    
    return model, history

def evaluate_model(
    model: InjectionDetector,
    test_texts: list,
    test_labels: list,
    threshold: float = 0.5
) -> Dict:
    """Evaluate the model's performance."""
    
    # Extract features
    feature_extractor = PromptFeatureExtractor()
    X_test = feature_extractor.transform(test_texts)
    X_test = torch.FloatTensor(X_test)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        probs = model(X_test).numpy()
    
    # Convert to binary predictions
    preds = (probs >= threshold).astype(int)
    
    # Calculate metrics
    results = classification_report(test_labels, preds, output_dict=True)
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(test_labels, probs)
    
    # Add PR curve data to results
    results['pr_curve'] = {
        'precisions': precisions.tolist(),
        'recalls': recalls.tolist(),
        'thresholds': thresholds.tolist()
    }
    
    return results

if __name__ == "__main__":
    # Example usage
    from ..data.prepare_data import process_raw_data
    
    # Get data
    train_ds, val_ds, test_ds = process_raw_data()
    
    # Train model
    model, history = train_model(
        train_ds['text'],
        train_ds['label'],
        val_ds['text'],
        val_ds['label']
    )
    
    # Evaluate
    results = evaluate_model(model, test_ds['text'], test_ds['label'])
    print(json.dumps(results, indent=2))
