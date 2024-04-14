import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def training_run(model, num_epochs, batch_size, train_dataset, val_dataset):
  """ model to train to give two classification heads"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    training_aspect = []
    training_polarization = []
    validation_aspect = []
    validation_polarization = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_aspect_train_loss = 0.0
        total_pol_train_loss = 0.0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            total_loss, aspect_loss, polarization_loss, aspect_logits, polarization_logits = model(input_ids=input_ids, labels=labels)

            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            total_aspect_train_loss += aspect_loss.item()
            total_pol_train_loss += polarization_loss.item()
            
        
      
            training_aspect.append(aspect_loss)
       
            training_polarization.append(polarization_loss)
        

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_aspect_train_loss = total_aspect_train_loss / len(train_dataloader)
        avg_pol_train_loss = total_pol_train_loss / len(train_dataloader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        total_aspect_val_loss = 0.0
        total_pol_val_loss = 0.0
    

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                total_loss, aspect_loss, polarization_loss, aspect_logits, polarization_logits = model(input_ids=input_ids, labels=labels)
                total_val_loss += total_loss.item()
                total_aspect_val_loss += aspect_loss.item()
                total_pol_val_loss += polarization_loss.item()
                
               
                validation_aspect.append(aspect_loss)
                
                validation_polarization.append(polarization_loss)

        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_aspect_val_loss = total_aspect_val_loss / len(train_dataloader)
        avg_pol_val_loss = total_pol_val_loss / len(train_dataloader)

        print(f'Epoch {epoch + 1}/{num_epochs}: Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')
      
    return training_aspect, training_polarization, validation_aspect, validation_polarization


import torch.nn.functional as F

def testing_run(model, batch_size, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    #print("this is the test dataloader: ",test_dataloader)

    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_aspect_predictions = 0
    correct_polarization_predictions = 0

    
    for batch in test_dataloader:
        #print("coolness")
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        

        print(labels)
        
        with torch.no_grad():

                total_loss, aspect_loss, polarization_loss, aspect_logits, polarization_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            
            # Calculate aspect accuracy
                aspect_predictions = torch.argmax(aspect_logits, dim=1)
                #print("aspect predictions: ",aspect_predictions)
                correct_aspect_predictions += torch.sum(aspect_predictions == labels[:, 0]).item()
                #print("aspect labels: ",labels[:,0])
    
            # Calculate polarization accuracy
                polarization_predictions = torch.argmax(polarization_logits, dim=1)
                #print("polarization predictios: ",polarization_predictions)
                #print("polarization labels: ", labels[:,1])
                correct_polarization_predictions += torch.sum(polarization_predictions == labels[:, 1]).item()

        total_samples += labels.size(0)

    average_loss = total_loss / len(test_dataset)
    aspect_accuracy = correct_aspect_predictions / total_samples
    polarization_accuracy = correct_polarization_predictions / total_samples

    print(f'Average Test Loss: {average_loss}')
    print(f'Aspect Accuracy: {aspect_accuracy}')
    print(f'Polarization Accuracy: {polarization_accuracy}')
