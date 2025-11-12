import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from SAnD_Repl.sand_model import SAnD
import copy
from tqdm import tqdm
import joblib
import wandb

class SAnDEnv():
    def __init__(
        self,

        # Model Parameters
        input_features,
        seq_len,
        num_heads,
        factor,
        n_layers,
        d_model,
        dropout_rate,
        n_class,
        attn_window,
        mode,

        # Training Parameters
        optimizer,
        optimizer_config={
            "betas" : (0.9, 0.98),
            "eps" : 1e-08
        }
    ):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SAnD(
            input_features,
            seq_len,
            num_heads,
            factor,
            n_layers,
            d_model,
            dropout_rate,
            n_class,
            attn_window,
            mode=mode
        ).to(self.device)

        self.metadata = {
            "model_parameters" : {
                "input_features": input_features,
                "seq_len": seq_len,
                "num_heads": num_heads,
                "factor": factor,
                "n_layers": n_layers,
                "d_model": d_model,
                "dropout_rate": dropout_rate,
                "n_class": n_class,
                "attn_window": attn_window,
                "mode": mode
            },
            "training_parameters" : {
                "optimizer" : optimizer,
                "optimizer_config" : optimizer_config
            }   
        }

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_config)
        else:
            raise Exception("Optimizer type not supported yet")
        
        if mode == 'classification':
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == 'multiclass':
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == "regression":
            self.loss = nn.MSELoss()
        else:
            raise Exception("Mode not supported yet")
        
    def evaluate_one_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(self.device)
                y = y.float().squeeze(-1).to(self.device)
                logits = self.model(X).squeeze(-1)
                loss = self.loss(logits, y)
                total_loss += loss.item()

        
        return total_loss / len(dataloader)

    
    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for X, y in tqdm(dataloader, total=len(dataloader)):
            X = X.to(self.device)
            y = y.float().squeeze(-1).to(self.device)

            logits = self.model(X).squeeze(-1)

            # Compute loss
            loss = self.loss(logits, y)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)


    def train(
        self,
        dataset,
        eval_dataset=None,
        model_save_dir=".",
        model_name=None,
        num_epochs=50,
        batch_size=4096,
        save_frequency=2,
        num_workers=8,
        dataset_name=""
    ):
        run = wandb.init(
            project='sand-mimic3',
            config={
                "dataset_name" : dataset_name,
                **self.metadata
            }
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        if eval_dataset is not None:
            eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        best_model = None
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch(dataloader)
            
            best_loss = None
            if eval_dataset is not None:
                val_loss = self.evaluate_one_epoch(eval_dataloader)
            
                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} ")
                
                run.log({
                    "train_loss" : train_loss,
                    "val_loss" : val_loss
                })

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = copy.deepcopy(self.model)
                
            else:
                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}")

                run.log({
                    "train_loss" : train_loss
                })

                # If no eval dataset, best_model is always the newest model so don't need to create a deep copy
                best_model = self.model

            # Save model every save_frequency
            if epoch % save_frequency == 0 and model_name is not None:
                save_model(best_model, self.metadata, model_save_dir, model_name)
                
        # Save the model at the end of every training
        if model_name is not None:
            save_model(best_model, self.metadata, model_save_dir, model_name)
        
        run.finish()


    def evaluate(
        self,
        model_name,
        batch_size,
        num_workers,
        dataset_name=''
    )
def save_model(model, metadata, model_save_dir, model_name):
    torch.save(model.state_dict(), f"{model_save_dir}/{model_name}.pth")
    joblib.dump(metadata, f"{model_save_dir}/{model_name}_metadata.joblib")

def load_model(model_save_dir, model_name, **kwargs):
    metadata = joblib.load(f"{model_save_dir}/{model_name}_metadata.joblib") 
    metadata['model_parameters'].update(metadata['training_parameters'])
    env = SAnDEnv(**metadata['model_parameters'], **kwargs)
    env.model.load_state_dict(torch.load(f"{model_save_dir}/{model_name}.pth"))
    return env