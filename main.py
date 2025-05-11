

# Dummy paths and labels to illustrate usage
front_paths = ['data/front1.jpg', 'data/front2.jpg']
back_paths = ['data/back1.jpg', 'data/back2.jpg']
labels = [13.5, 14.2]

train_loader = get_dataloader(front_paths, back_paths, labels)
val_loader = get_dataloader(front_paths, back_paths, labels, shuffle=False)

model = LateFusionCBAM()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(10):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
