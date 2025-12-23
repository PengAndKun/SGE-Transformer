# Policy Network Training Module
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class TemporalStateEncoder(nn.Module):
    def __init__(self, num_states=98, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_states, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim)
        self.hidden_dim = hidden_dim  # Save hidden_dim for use in forward method

    def forward(self, state_seq_batch):
        """
        MODIFIED: This forward method now correctly handles a batch of sequences.
        :param state_seq_batch: A list of sequences, e.g., [[-1, 10], [3, 4, 5]]
        :return: A tensor of embeddings for the batch, shape [B, hidden_dim]
        """
        # If tensor data, process in parallel
        if isinstance(state_seq_batch, torch.Tensor):
            # Tensor input: parallel processing
            batch_tensor = state_seq_batch.to(self.embedding.weight.device)
            
            # Get embeddings: [B, T, embed_dim]
            embeddings = self.embedding(batch_tensor)
            # Convert to [T, B, embed_dim] format
            embeddings = embeddings.transpose(0, 1)
            
            # Pass through LSTM: [T, B, embed_dim] -> h_n: [1, B, hidden_dim]
            _, (h_n, _) = self.lstm(embeddings)
            
            # Return the last layer's hidden state: [B, hidden_dim]
            return h_n.squeeze(0)
        
        batch_embeddings = []
        # Iterate through each sequence in the batch
        for state_seq in state_seq_batch:
            # --- Internal logic for processing single sequence ---
            # indices = [i for i in state_seq if i >= 0]  # Filter out padding values
            indices = state_seq

            # Convert to tensor and get embeddings
            input_tensor = torch.tensor(indices, dtype=torch.long, device=self.embedding.weight.device)
            input_emb = self.embedding(input_tensor).unsqueeze(1)  # [T, 1, D]

            # Pass through LSTM
            _, (h_n, _) = self.lstm(input_emb)

            # Add this sequence's result to the batch list
            batch_embeddings.append(h_n.squeeze(0).squeeze(0))

        # Stack all sequence results into a batch tensor and return
        return torch.stack(batch_embeddings)

class EmbeddingBehaviorClone:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, device='cpu'):
        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def learn(self, batch_states, batch_actions, total_loss):
        logits = self.policy(batch_states)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, batch_actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()
        return total_loss

    def predict_action_batch_sample(self, state_batch):
        if state_batch.dim() == 1:
            state_batch = state_batch.unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(state_batch)
            probs = torch.softmax(logits, dim=1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1).tolist()
        return actions, probs.tolist()


class EndToEndModel(nn.Module):
    def __init__(self, num_states, embed_dim, encoder_hidden_dim, policy_hidden_dim, action_dim):
        super().__init__()
        # Create encoder and policy network instances internally
        self.encoder = TemporalStateEncoder(num_states, embed_dim, encoder_hidden_dim)
        self.policy = PolicyNetwork(encoder_hidden_dim, policy_hidden_dim, action_dim)

    def forward(self, raw_state_sequences):
        """
        Define the complete pipeline from raw state sequences to final action logits.
        :param raw_state_sequences: A batch of raw historical trajectories, e.g., [[-1, 10, 20], [3, 4, -1]]
        :return: Action logits, shape [B, action_dim]
        """
        # 1. First pass through encoder to convert raw sequences to state embedding vectors
        state_embeddings = self.encoder(raw_state_sequences)

        # 2. Then input embedding vectors to policy network to get final action logits
        action_logits = self.policy(state_embeddings)

        return action_logits


class EndToEndBehaviorCloning:
    def __init__(self, num_states, embed_dim, encoder_hidden_dim, policy_hidden_dim, action_dim, lr, device='cpu'):
        self.device = device

        # 1. Initialize the complete end-to-end model
        self.model = EndToEndModel(
            num_states, embed_dim, encoder_hidden_dim,
            policy_hidden_dim, action_dim
        ).to(self.device)

        # 2. Key: Optimizer manages all parameters of the combined model
        #    This way, gradients can update both encoder and policy network simultaneously
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, batch_sequences, batch_actions):
        """
        Execute one training step.
        :param batch_sequences: A batch of raw historical trajectories
        :param batch_actions: Corresponding expert actions
        """
        self.model.train()  # Set to training mode

        # Move action data to correct device
        batch_actions = batch_actions.to(self.device)

        # Directly call the combined model for forward propagation
        logits = self.model(batch_sequences)

        # Calculate loss
        loss = self.criterion(logits, batch_actions)

        # Backward propagation and optimization
        self.optimizer.zero_grad()
        loss.backward()  # Gradients flow through the entire EndToEndModel (including encoder and policy)
        self.optimizer.step()

        return loss.item()

    def predict(self, sequence):
        """Predict for a single sequence"""
        self.model.eval()
        with torch.no_grad():
            # Need to add one dimension to batch dimension
            logits = self.model([sequence])
            probs = torch.softmax(logits, dim=1)
            action = torch.argmax(probs, dim=1).item()
        return action

    def predicts(self, sequence):
        """Predict for multiple sequences"""
        self.model.eval()
        with torch.no_grad():
            # Need to add one dimension to batch dimension
            logits = self.model(sequence)
            probs = torch.softmax(logits, dim=1)
            actions = torch.argmax(probs, dim=1)
        return actions

    def save_model(self, file_path):
        """
        保存模型和优化器的状态字典。
        :param file_path: 模型保存路径 (e.g., 'model.pth')

        Save the state dictionary of the model and optimizer.
        :param filepasswth: Path for saving the model (e.g., 'model. pth')
        """
        # print(f"Saving model to {file_path} ...")
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(state, file_path)
        # print("Model saved successfully.")

    def load_model(self, file_path):
        """
        加载模型和优化器的状态字典。
        :param file_path: 模型文件路径 (e.g., 'model.pth')


        Load the state dictionary of the model and optimizer.
        :param file_path: Model file path (e.g., 'model. pth')
        """
        # print(f"Loading model from {file_path} .. ...")
        # 加载状态字典，并确保它被映射到正确的设备上
        # Load the state dictionary and ensure it is mapped to the correct device
        checkpoint = torch.load(file_path, map_location=self.device)
        # 将加载的状态加载到模型和优化器中
        # Load the loaded state into the model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # print("Model loaded successfully.")