"""Train a stopping gate on a synthetic task, in memory without artifact output."""
from pathlib import Path
import torch
import weightlang


def main():
    torch.manual_seed(42)
    model = weightlang.load(Path(__file__).with_name('model.wgt'))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Four input features per sample; synthetic targets illustrate the API only.
    x = torch.randn(32, 4)
    target = 0.5 * x
    initial = model.reasoning.stop.step0.weight.detach().clone()
    for _ in range(30):
        result = model(x)
        per_step_loss = (result['states'] - target[:, None, :]).square().mean(-1, keepdim=True)
        expected_loss = (result['halt_weights'] * per_step_loss).sum(1).mean()
        loss = expected_loss + 0.01 * result['expected_steps'].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    changed = not torch.equal(initial, model.reasoning.stop.step0.weight)
    assert changed, 'the stopping gate must receive a training signal'
    print(f'synthetic loss={loss.item():.6f}; stopping gate updated={changed}')


if __name__ == '__main__':
    main()
