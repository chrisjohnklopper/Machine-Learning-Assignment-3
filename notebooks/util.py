import torch
from torch.nn import Module, Linear, Parameter, MSELoss, Dropout
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import tensor, float32
from torch.optim import SGD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==================== Helper functions for RNN ====================

def create_sequences(X, y, sequence_length, id_feature_name_in_X=None):
    X_sequences = []
    y_sequences = []

    # make X and y pandas
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    if id_feature_name_in_X is not None:
        ids = X[id_feature_name_in_X].values
        for i in range(len(X) - sequence_length + 1):
            if len(set(ids[i:i + sequence_length])) == 1:
                X_sequences.append(X.iloc[i:i + sequence_length].values)
                y_sequences.append(y.iloc[i + sequence_length - 1])
    else:
        for i in range(len(X) - sequence_length + 1):
            X_sequences.append(X.iloc[i:i + sequence_length].values)
            y_sequences.append(y.iloc[i + sequence_length - 1])
    
    return np.array(X_sequences), np.array(y_sequences)

# ==================== Functions for graphing ====================

def plot_crosstab(df, bool_col1, target_boolean_feature):
    cross_tab = pd.crosstab(df[bool_col1], df[target_boolean_feature], normalize='index')
    
    
    cross_tab.plot(kind='bar', stacked=True)
    plt.title(f'{target_boolean_feature} by {bool_col1}')
    
    plt.xlabel(bool_col1)
    plt.ylabel('Relative Frequency')
    
    plt.legend(title=target_boolean_feature, loc='upper right', labels=['No Failure', 'Failure'])
    
    plt.show()

def plot_boxplot_numerical_vs_boolean(df, num_col, bool_col):
    df.boxplot(column=num_col, by=bool_col, figsize=(8,5))
    plt.title(f"{num_col} vs {bool_col}")
    plt.suptitle("")
    
    plt.xlabel(bool_col)
    
    plt.ylabel(num_col)
    plt.show()

def plot_test_versus_validation(test_losses, validation_losses, save_path=None, title=None):
    plt.plot(test_losses, label='Test Loss', color='blue')
    plt.plot(validation_losses, label='Validation Loss', color='orange')
    plt.title(title if title else 'Test vs Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_over_time(data, feature, save_path=None, dates=None, shape=(10,5)):
    plt.figure(figsize=shape)
    plt.plot(dates,
             data[feature],
             label=feature,
             color='black',
             linewidth=0.8)
    plt.title(f'{feature} Over Time')
    plt.xlabel('Date')
    plt.ylabel(feature)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.grid()
    
    plt.show()

def plot_boxplot_of_feature(data, feature):
    plt.boxplot(data[feature],
                vert=False,
                medianprops={'color': 'black'})
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)
    plt.yticks([])
    plt.grid()
    plt.show()

def convert_feature_into_cyclic(data, feature, max_value):
    data[f'{feature}Sin'] = np.sin(2 * np.pi * data[feature] / max_value)
    data[f'{feature}Cos'] = np.cos(2 * np.pi * data[feature] / max_value)
    return data

def get_predictions(raw_test, preprocessed_test, results, model_name, target_feature_name, xlabel, ylabel, title, show_plot=False, save_path=None): 
    dates = raw_test['DateTime'].values
    #results = weather_elman_h32_results 
    scaler = results['scaler_X']
    model = results['model']
    preprocessed_test = scaler.transform(preprocessed_test.drop(columns=[target_feature_name]))
    preprocessed_test = torch.tensor(preprocessed_test, dtype=torch.float32).unsqueeze(1)
    preds = model(preprocessed_test)
    preds = preds.detach().numpy()
    preds = preds.squeeze()
    actual = raw_test[target_feature_name].values
    actual = actual[-len(preds):]

    mses = abs(preds - actual)
    prediction_results = pd.DataFrame(mses).describe().T
    prediction_results.drop(columns=['count'], inplace=True)
    prediction_results.rename(columns={'50%': 'Median', 'mean':'Mean', 'std':'Std. Dev.', 'min':'Min', 'max':'Max'}, inplace=True)
    # make row name model name
    prediction_results.index = [model_name]


    # plot
    if show_plot:
        plt.figure(figsize=(15,7))
        plt.plot(dates, actual, label='Actual')
        plt.plot(dates, preds, label='Predicted')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    return prediction_results

# ==================== Torch modules for RNNs ====================

class JordanRNN(Module):
    def __init__(self, input_size, hidden_size, output_size, state_size, dropout=0):
        super().__init__()
        self.dropout = Dropout(dropout)
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.state_size = state_size

        self.W_h = Linear(input_size, hidden_size, bias=False)
        self.U_h = Linear(self.state_size, hidden_size, bias=False)
        self.b_h = Parameter(torch.zeros(hidden_size))

        self.W_y = Linear(hidden_size, output_size, bias=False)
        self.b_y = Parameter(torch.zeros(output_size))

        self.W_s = Linear(self.state_size, state_size, bias=False)
        self.W_sy = Linear(output_size, state_size, bias=False)
        self.b_s = Parameter(torch.zeros(state_size))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        device = x.device

        s_t = torch.zeros(batch_size, self.state_size, device=device)
        y_prev = torch.zeros(batch_size, self.output_size, device=device)

        for t in range(seq_len):
            h_t = torch.tanh(self.W_h(x[:, t, :]) + self.U_h(s_t) + self.b_h)
            h_t = self.dropout(h_t)
            y_t = self.W_y(h_t) + self.b_y
            s_prev = s_t
            s_t = torch.tanh(self.W_s(s_prev) + self.W_sy(y_prev) + self.b_s)
            y_prev = y_t

        return y_t
    
class ElmanRNN(Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = Dropout(dropout)

        self.W_h = Linear(input_size, hidden_size, bias=False)
        self.U_h = Linear(hidden_size, hidden_size, bias=True)
        self.b_h = Parameter(torch.zeros(hidden_size))

        self.W_y = Linear(hidden_size, output_size, bias=True)
        self.b_y = Parameter(torch.zeros(output_size))
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        device = x.device

        h_t = torch.zeros(batch_size, self.hidden_size).to(device)

        for t in range(seq_len):
            h_prev = h_t
            h_t = torch.tanh(self.W_h(x[:, t, :]) + self.U_h(h_prev) + self.b_h)
            h_t = self.dropout(h_t)

        y_t = self.W_y(h_t) + self.b_y

        return y_t


class MultiRecurrentNN(Module):
    def __init__(self, input_size, hidden_size, output_size, memory_config, dropout=0):
        super().__init__()
        self.dropout = Dropout(dropout)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #  [4, 4, 4] was deemed best performance
        # Memory banks parameters

        self.n_input_mem = memory_config[0]
        self.n_hidden_mem = memory_config[1]
        self.n_output_mem = memory_config[2]
        
        if self.n_input_mem > 0:
            self.W_Mih = Linear(input_size * self.n_input_mem, hidden_size, bias=False)
        
        if self.n_hidden_mem > 0:
            self.W_Mhh = Linear(hidden_size * self.n_hidden_mem, hidden_size, bias=False)
        
        if self.n_output_mem > 0:
            self.W_Moh = Linear(output_size * self.n_output_mem, hidden_size, bias=False)

        self.W_ih = Linear(input_size, hidden_size, bias=False)
        self.b_i = Parameter(torch.zeros(hidden_size))
        
        self.W_ho = Linear(hidden_size, output_size, bias=False)
        self.b_h = Parameter(torch.zeros(output_size))
        
        self._init_memory_ratios()
    
    def _init_memory_ratios(self):
        self.input_layer_ratios = []
        self.input_self_ratios = []
        
        for i in range(self.n_input_mem, 0, -1):
            self.input_layer_ratios.append(i / self.n_input_mem)
            self.input_self_ratios.append(1 - i / self.n_input_mem)
        
        self.hidden_layer_ratios = []
        self.hidden_self_ratios = []
        
        for i in range(self.n_hidden_mem, 0, -1):
            self.hidden_layer_ratios.append(i / self.n_hidden_mem)
            self.hidden_self_ratios.append(1 - i / self.n_hidden_mem)
        
        self.output_layer_ratios = []
        self.output_self_ratios = []
        
        for i in range(self.n_output_mem, 0, -1):
            self.output_layer_ratios.append(i / self.n_output_mem)
            self.output_self_ratios.append(1 - i / self.n_output_mem)
    
    def _update_memory(self, layer_output, memory_banks, layer_ratios, self_ratios):
        new_memory = []
        
        # Equation 3.1, 3.2 and 3.3 from paper
        
        for i, (layer_r, self_r) in enumerate(zip(layer_ratios, self_ratios)):
            # layer_r = Layer link ratio and self_r = self link ratio 
            new_mem = layer_r * layer_output + self_r * memory_banks[i]
            new_memory.append(new_mem)

        return new_memory
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        device = x.device

        # Initialize memory banks + states
        
        M_i = [torch.zeros(batch_size, self.input_size, device=device) 
               for _ in range(self.n_input_mem)]
        M_h = [torch.zeros(batch_size, self.hidden_size, device=device) 
               for _ in range(self.n_hidden_mem)]
        M_o = [torch.zeros(batch_size, self.output_size, device=device) 
               for _ in range(self.n_output_mem)]
        
        I_t = torch.zeros(batch_size, self.input_size, device=device)
        H_t = torch.zeros(batch_size, self.hidden_size, device=device)
        O_t = torch.zeros(batch_size, self.output_size, device=device)
        
        for t in range(seq_len):
            if self.n_input_mem > 0:
                M_i = self._update_memory(I_t, M_i, self.input_layer_ratios, self.input_self_ratios)
            if self.n_hidden_mem > 0:
                M_h = self._update_memory(H_t, M_h, self.hidden_layer_ratios, self.hidden_self_ratios)
            if self.n_output_mem > 0:
                M_o = self._update_memory(O_t, M_o, self.output_layer_ratios, self.output_self_ratios)
            
            # Current input at time t

            I_t = x[:, t, :]

            # Eqn 3.4 - weighted sum by linear layers
            
            H_net = self.W_ih(I_t) + self.b_i
            
            if self.n_input_mem > 0:
                H_net = H_net + self.W_Mih(torch.cat(M_i, dim=1))
            if self.n_hidden_mem > 0:
                H_net = H_net + self.W_Mhh(torch.cat(M_h, dim=1))
            if self.n_output_mem > 0:
                H_net = H_net + self.W_Moh(torch.cat(M_o, dim=1))
            
            # Activation function 

            H_t = torch.tanh(H_net)
            H_t = self.dropout(H_t)
            O_t = self.W_ho(H_t) + self.b_h
        
        return O_t

# ==================== Training method for torch modules ====================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return
        
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def training(model,
            data,
            num_epochs=25,
            batch_size=16,
            optimizer_learning_rate=0.002,
            optimizer_weight_decay=0,
            optimizer_b1=0.9,
            optimizer_b2=0.999,
            target_feature_name=None,
            model_name=None,
            splits=8,
            patience=5,
            min_delta=0,
            loss_function=MSELoss(),
            scaler_X=StandardScaler(),
            use_final_fold=False,
            sequence_length=64):
    
    # Prepare data for training using blocked cross validation

    X = data.drop(columns=[target_feature_name]).values
    y = data[target_feature_name].values

    input_shape = X.shape[1]
    output_shape = 1

    print(X.shape)
    print(y.shape)

    X_sequences, y_sequences = create_sequences(X, y, sequence_length=sequence_length)

    print(X_sequences.shape)
    print(y_sequences.shape)

    split = TimeSeriesSplit(n_splits=splits)

    split_number = 1

    cv_scores = []

    for train_indices, validation_indices in split.split(X_sequences):

        if use_final_fold and split_number < splits:
            split_number += 1
            continue
        

        earlystopping = EarlyStopping(patience=patience, min_delta=min_delta)

        test_losses = []
        validation_losses = []

        hidden_size = model.hidden_size
        
        print(f'Train indices from {train_indices[0]} to {train_indices[-1]} and validation indices from {validation_indices[0]} to {validation_indices[-1]}')

        X_train, X_validation = X_sequences[train_indices], X_sequences[validation_indices]
        y_train, y_validation = y_sequences[train_indices], y_sequences[validation_indices]

        # Scale data
        #scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train.reshape(-1, input_shape)).reshape(X_train.shape)
        X_validation = scaler_X.transform(X_validation.reshape(-1, input_shape)).reshape(X_validation.shape)

        # Convert to tensors
        X_train_tensor = tensor(X_train, dtype=float32)
        y_train_tensor = tensor(y_train, dtype=float32).view(-1, 1)
        X_validation_tensor = tensor(X_validation, dtype=float32)
        y_validation_tensor = tensor(y_validation, dtype=float32).view(-1, 1)

        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model, loss function, and optimizer

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=optimizer_learning_rate,
                                     weight_decay=optimizer_weight_decay,
                                     betas=(optimizer_b1, optimizer_b2))


        best_epoch_loss = float('inf')
        best_epoch = -1
        best_model = None

        for epoch in range(num_epochs):
            
            # Training of model
            model.train()
            epoch_train_loss = 0
            for i, (inputs, targets) in enumerate(train_loader):
                # Grads reset
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            # Validation of model
            model.eval()
            epoch_validation_loss = 0
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(validation_loader):
                    outputs = model(inputs)
                    loss = loss_function(outputs, targets)
                    epoch_validation_loss += loss.item()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                f'Loss: {epoch_train_loss / len(train_loader):.4f}, '
                f'Validation Loss: {epoch_validation_loss / len(validation_loader):.4f}, '
                f'Squareroot of Validation Loss: {(epoch_validation_loss / len(validation_loader))**0.5:.4f}')
            
            test_losses.append(epoch_train_loss / len(train_loader))
            validation_losses.append(epoch_validation_loss / len(validation_loader))

            if epoch_validation_loss / len(validation_loader) < best_epoch_loss:
                best_model = model
                best_epoch_loss = epoch_validation_loss / len(validation_loader)
                best_epoch = epoch+1


            earlystopping(epoch_validation_loss / len(validation_loader))
            if earlystopping.stop:
                print(f'Early stopping at epoch {epoch+1}')
                break

        print(f'Best epoch: {best_epoch}, Best validation loss: {best_epoch_loss:.4f}')

        plot_test_versus_validation(test_losses,
                                    validation_losses,
                                    f'../images/{model_name}-hs{hidden_size}-fold{split_number}.pdf',
                                    title=f'Test vs. Validation Performance')
        
        cv_scores.append(best_epoch_loss)

        split_number += 1
    return {'cv_scores': cv_scores, 'model': best_model, 'scaler_X': scaler_X}

