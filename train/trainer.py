
import torch
from collections import defaultdict
from ..utils.metrics import compute_metrics

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler=None, device='cpu'):
        self.model, self.crit, self.opt, self.sch, self.device = model, criterion, optimizer, scheduler, device
        self.model.to(device)
        self.hist = defaultdict(list)

    def _run(self, loader, train):
        self.model.train() if train else self.model.eval()
        losses; preds; targets = [], [], []
        with torch.set_grad_enabled(train):
            for x,y in loader:
                x,y = x.to(self.device), y.to(self.device)
                out,_ = self.model(x)
                loss = self.crit(out, y)
                if train:
                    self.opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                losses.append(loss.item())
                preds.extend(out.argmax(1).cpu().numpy())
                targets.extend(y.cpu().numpy())
        metrics = compute_metrics(targets, preds)
        return sum(losses)/len(losses), metrics

    def fit(self, train_loader, val_loader=None, epochs=1):
        for ep in range(epochs):
            tloss,tmet = self._run(train_loader, True)
            if val_loader:
                vloss,vmet = self._run(val_loader, False)
                print(f'E{ep+1}: train_loss={tloss:.4f} val_loss={vloss:.4f} val_acc={vmet["accuracy"]:.3f}')
            else:
                vloss,vmet={'accuracy':None}, {}
                print(f'E{ep+1}: train_loss={tloss:.4f}')
            if self.sch: self.sch.step()
            self.hist['train_loss'].append(tloss)
            self.hist['val_loss'].append(vloss)
            self.hist['val_acc'].append(vmet.get('accuracy'))
        return self.hist
