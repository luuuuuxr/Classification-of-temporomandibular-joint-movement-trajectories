class PrototypicalNetwork:
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = None
        self.prototypes = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = 1.0
        
    def _build_encoder(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
        
    def fit(self, X, y, epochs=200, batch_size=32, patience=20):
        X_scaled = self.scaler.fit_transform(X)
        self.model = self._build_encoder().to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        n_samples = len(X_tensor)
        indices = torch.randperm(n_samples)
        split_idx = int(0.8 * n_samples)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
        X_val, y_val = X_tensor[val_indices], y_tensor[val_indices]
        
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_end = min(i + batch_size, len(X_train))
                batch_x = X_train[i:batch_end]
                batch_y = y_train[i:batch_end]
                
                optimizer.zero_grad()
                embeddings = self.model(batch_x)
                prototypes = self._compute_prototypes(embeddings, batch_y)
                loss = self._prototypical_loss(embeddings, batch_y, prototypes)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            self.model.eval()
            with torch.no_grad():
                val_embeddings = self.model(X_val)
                val_prototypes = self._compute_prototypes(val_embeddings, y_val)
                val_loss = self._prototypical_loss(val_embeddings, y_val, val_prototypes)
            self.model.train()
            
            avg_train_loss = total_loss / num_batches
            scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 50 == 0:
                print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        self.model.load_state_dict(best_model_state)
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(X_tensor)
            self.prototypes = self._compute_final_prototypes(embeddings, y_tensor)
        
        return self
        
    def _compute_prototypes(self, embeddings, labels):
        unique_labels = torch.unique(labels)
        prototypes = []
        
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                class_embeddings = embeddings[mask]
                prototype = class_embeddings.mean(dim=0)
                prototypes.append(prototype)
            else:
                prototypes.append(torch.zeros(embeddings.shape[1], device=embeddings.device))
        
        return torch.stack(prototypes)
    
    def _compute_final_prototypes(self, embeddings, labels):
        unique_labels = torch.unique(labels).sort()[0]
        prototypes = []
        
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                class_embeddings = embeddings[mask]
                prototype = class_embeddings.mean(dim=0)
                prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def _prototypical_loss(self, embeddings, labels, prototypes):
        distances = torch.cdist(embeddings, prototypes, p=2)
        logits = -distances / self.temperature
        loss = F.cross_entropy(logits, labels)
        
        if len(prototypes) > 1:
            proto_distances = torch.cdist(prototypes, prototypes, p=2)
            mask = ~torch.eye(len(prototypes), dtype=bool, device=prototypes.device)
            proto_reg = -torch.mean(proto_distances[mask])
            loss += 0.01 * proto_reg
        
        return loss
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(X_tensor)
            distances = torch.cdist(embeddings, self.prototypes, p=2)
            predictions = torch.argmin(distances, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(X_tensor)
            distances = torch.cdist(embeddings, self.prototypes, p=2)
            logits = -distances / self.temperature
            probabilities = F.softmax(logits, dim=1)
        
        return probabilities.cpu().numpy()
        
    def set_training_data(self, X, y):
        self.X_train = X
        self.y_train = y